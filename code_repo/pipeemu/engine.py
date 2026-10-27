"""Exact single-GPU emulation of pipeline weight-version semantics.

The model is a sequence of *units* (BERT: embeddings, each encoder layer, MLM head;
ResNet: stem, each residual block, head). Units are assigned to d contiguous pipeline
stages. For every unit we keep two copies of its weights in the emulator:

    cur  = newest version V(b)       (what the optimizer updates)
    prev = previous version V(b-1)   (2BW's "shadow")

and, per unit, a mode that decides which copy each microbatch sees:

    'exact'    vanilla / PipeDream-Flush:   F and B use cur                (no staleness)
    'shadow'   2BW:                         F and B use prev               (delay 1, consistent)
    'noshadow' 1BW (single version):        microbatch j of a batch uses prev for its FORWARD
                                            on stage s iff j <= d - s (it was mid-flight when
                                            stage s updated); its BACKWARD always uses cur.
                                            Everything else uses cur.
    'frozen'   no update; prev == cur, so no shadow is needed and nothing can mismatch

The 'noshadow' rule is the closed form of the simulated 1F1B schedule (verified in
tests/test_schedule.py). The forward/backward mismatch is reproduced by overwriting the
unit's weights in place between forward and backward (verified in tests/test_swap.py).
Requires fp32/TF32 (no autocast): autocast would back-propagate through cached bf16
weight copies and silently ignore the swap.
"""
from __future__ import annotations

import torch

MODES = ("exact", "shadow", "noshadow", "frozen")


class PipelineEmulator:
    def __init__(self, unit_params, stage_of_unit, d, m, optimizer, clip_norm=None):
        """unit_params: list (per unit) of lists of nn.Parameter.
        stage_of_unit: list of 1-indexed stage ids, non-decreasing."""
        self.units = [list(ps) for ps in unit_params]
        self.n = len(self.units)
        self.stage = list(stage_of_unit)
        self.d, self.m = d, m
        self.opt = optimizer
        self.clip_norm = clip_norm
        self.mode = ["exact"] * self.n
        self.batch_idx = 0
        self._versions_allocated = False
        self.working = [None] * self.n          # which copy the live params hold: 'cur' | 'prev'
        self.stats = {"mismatched_microbatch_units": 0, "microbatch_units": 0}

    # ------------------------------------------------------------------ versions
    def _alloc(self):
        with torch.no_grad():
            self.cur = [[p.detach().clone() for p in ps] for ps in self.units]
            self.prev = [[p.detach().clone() for p in ps] for ps in self.units]
        self.working = ["cur"] * self.n
        self._versions_allocated = True

    def _load(self, u, which):
        if self.working[u] == which:
            return
        src = self.cur[u] if which == "cur" else self.prev[u]
        with torch.no_grad():
            for p, s in zip(self.units[u], src):
                p.data.copy_(s)          # .data: in-place change invisible to autograd (intended)
        self.working[u] = which

    def set_modes(self, modes):
        """Change unit modes; only takes effect at a batch (= version) boundary."""
        assert len(modes) == self.n and all(x in MODES for x in modes)
        for u, (old, new) in enumerate(zip(self.mode, modes)):
            if new == old:
                continue
            frozen = new == "frozen"
            for p in self.units[u]:
                p.requires_grad_(not frozen)
                if frozen:
                    p.grad = None
            if frozen and self._versions_allocated:
                # a frozen unit never changes again until thawed: make prev == cur now
                self._load(u, "cur")
                with torch.no_grad():
                    for a, b in zip(self.prev[u], self.cur[u]):
                        a.copy_(b)
        self.mode = list(modes)
        if not self._versions_allocated and any(x in ("shadow", "noshadow") for x in self.mode):
            self._alloc()

    # ------------------------------------------------------------------ one batch
    def fwd_version(self, u, j):
        mode = self.mode[u]
        if mode == "shadow":
            return "prev"
        if mode == "noshadow" and j <= self.d - self.stage[u]:
            return "prev"
        return "cur"

    def step(self, microbatches, loss_fn, after_backward=None):
        """Run one batch = len(microbatches) microbatches, then one optimizer step.
        loss_fn(mb) -> scalar loss for that microbatch (already averaged within it).
        Returns the mean microbatch loss."""
        versioned = self._versions_allocated
        m = len(microbatches)
        total = 0.0
        for j, mb in enumerate(microbatches, 1):
            if versioned:
                for u in range(self.n):
                    if self.mode[u] != "frozen":
                        self._load(u, self.fwd_version(u, j))
            loss = loss_fn(mb)
            if versioned:
                # 1BW straddlers: backward sees the newest weights (overwrite in place)
                for u in range(self.n):
                    if self.mode[u] == "noshadow":
                        self.stats["microbatch_units"] += 1
                        if self.working[u] == "prev":
                            self.stats["mismatched_microbatch_units"] += 1
                            self._load(u, "cur")
            (loss / m).backward()
            total += float(loss.detach())
        if versioned:
            for u in range(self.n):
                if self.mode[u] != "frozen":
                    self._load(u, "cur")
        if after_backward is not None:
            after_backward()                      # e.g. gradient statistics for freezing
        if self.clip_norm:
            params = [p for ps in self.units for p in ps if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, self.clip_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if versioned:
            with torch.no_grad():
                for u in range(self.n):
                    if self.mode[u] == "frozen":
                        continue
                    for pv, cv, p in zip(self.prev[u], self.cur[u], self.units[u]):
                        pv.copy_(cv)            # V(b)   becomes the shadow
                        cv.copy_(p)             # V(b+1) is the new current
        self.batch_idx += 1
        return total / m

    def reset_optimizer_state(self, units):
        for u in units:
            for p in self.units[u]:
                self.opt.state.pop(p, None)
