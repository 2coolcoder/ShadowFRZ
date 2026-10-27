"""Per-unit gradient statistics (PolicyFRZ's GV and AM) and the decisions built on them.

For unit i and each of its parameters p, over a window of `window` optimizer steps:
    GV_p = || sum_t g_t || / sum_t ||g_t||     in [0, 1]; ~1 = consistent direction (still
                                               moving), ~1/sqrt(window) = gradients cancel
    AM_p = mean over steps and elements of |g_t|
Unit values are the mean over the unit's parameters (PolicyFRZ Eqs. 11-14).
AM is scale-dependent, so decisions use AM relative to the unit's first measured window.

Decisions are taken only at window boundaries, which are also weight-version boundaries.
Freezing policies:  off | gvam (threshold rule, PolicyFRZ Eq. 20) | random (replays the
                    number of frozen units recorded by a gvam run) | freezeout (prefix
                    frozen permanently at linearly spaced times, FreezeOut "linear, t0")
Shadow policies:    all | none | gvam (drop the shadow when GV < sel_gv) | random (replay)
Frozen units are re-measured ("thawed" for one window) every `thaw_every` frozen windows,
because a frozen unit has no gradient to measure (PolicyFRZ re-trains all layers for a
window after LR decays for the same reason).
"""
from __future__ import annotations

import json
import math
import random

import torch


class FreezeController:
    def __init__(self, unit_params, *, pipeline, freeze_policy="off", shadow_policy="all",
                 window=50, total_steps=1, warmup_frac=0.1, frz_gv=0.2, frz_am=0.5, sel_gv=0.4, sel_am=None,
                 thaw_every=4, protect=(), replay=None, freezeout_t0=0.5, seed=0, trace_path=None):
        self.units = [list(ps) for ps in unit_params]
        self.n = len(self.units)
        self.pipeline = pipeline                 # 'none' (exact) or 'pipe' (versioned)
        self.freeze_policy, self.shadow_policy = freeze_policy, shadow_policy
        self.window, self.total = window, total_steps
        self.warmup_steps = int(warmup_frac * total_steps)
        self.frz_gv, self.frz_am, self.sel_gv = frz_gv, frz_am, sel_gv
        self.sel_am = sel_am                     # if set: drop shadow when AM/AM(ref) < sel_am
        self.thaw_every = thaw_every
        self.protect = set(protect)
        self.replay = replay                     # {step: {"frozen": k, "noshadow": k}}
        self._step = 0
        self.freezeout_t0 = freezeout_t0
        self.rng = random.Random(seed)
        self.trace_f = open(trace_path, "a") if trace_path else None
        self.state = ["train"] * self.n          # 'train' | 'frozen'
        self.frozen_for = [0] * self.n
        self.am_ref = [None] * self.n
        # 'none' (1BW) drops every shadow from step 0; adaptive policies start safe (shadow kept)
        self.shadow = [shadow_policy != "none"] * self.n
        self.last_gv = [float("nan")] * self.n
        self.last_am = [float("nan")] * self.n
        self._reset_sums()
        self.measuring = freeze_policy == "gvam" or shadow_policy == "gvam" or trace_path is not None

    # ---------------------------------------------------------------- statistics
    def _reset_sums(self):
        # per unit: (param indices, sum of grads, sum of ||g||, sum of mean|g|), all on GPU
        self.acc = [None] * self.n
        self.count = [0] * self.n

    @torch.no_grad()
    def observe(self):
        """Call after the batch gradient is accumulated, before the optimizer step.
        Fused GPU ops only; no host sync until the window closes."""
        if not self.measuring:
            return
        for u, ps in enumerate(self.units):
            if self.state[u] == "frozen":
                continue
            idx = tuple(j for j, p in enumerate(ps) if p.grad is not None)
            if not idx:
                continue
            grads = [ps[j].grad for j in idx]
            norms = torch.stack(torch._foreach_norm(grads))
            l1 = torch.stack(torch._foreach_norm(grads, 1))
            numel = torch.tensor([g.numel() for g in grads], device=norms.device, dtype=norms.dtype)
            a = self.acc[u]
            if a is None or a[0] != idx:
                self.acc[u] = [idx, [g.detach().clone() for g in grads], norms, l1 / numel]
            else:
                torch._foreach_add_(a[1], grads)
                a[2] += norms
                a[3] += l1 / numel
            self.count[u] += 1

    def _unit_stats(self, u):
        a = self.acc[u]
        if a is None or self.count[u] == 0:
            return float("nan"), float("nan")
        sum_norms = torch.stack(torch._foreach_norm(a[1]))
        gv = (sum_norms / a[2].clamp_min(1e-30)).mean()
        am = (a[3] / self.count[u]).mean()
        return float(gv), float(am)

    # ---------------------------------------------------------------- decisions
    def maybe_decide(self, step):
        """Call after optimizer step `step` (1-indexed). Returns a new mode list at a window
        boundary, else None."""
        if step % self.window != 0:
            return None
        frac = step / self.total
        self._step = step
        for u in range(self.n):
            if self.count[u] >= max(1, self.window // 2):
                self.last_gv[u], self.last_am[u] = self._unit_stats(u)
                if self.am_ref[u] is None and step > self.warmup_steps:
                    self.am_ref[u] = self.last_am[u]
        decide = step > self.warmup_steps
        if decide:
            self._decide_freeze(frac)
            self._decide_shadow()
        modes = self.modes()
        if self.trace_f:
            self.trace_f.write(json.dumps({
                "step": step, "frac": round(frac, 5),
                "gv": [None if math.isnan(x) else round(x, 5) for x in self.last_gv],
                "am": [None if math.isnan(x) else float(f"{x:.4e}") for x in self.last_am],
                "am_rel": [None if (self.am_ref[u] in (None, 0) or math.isnan(self.last_am[u]))
                           else round(self.last_am[u] / self.am_ref[u], 5) for u in range(self.n)],
                "modes": modes}) + "\n")
            self.trace_f.flush()
        self._reset_sums()
        return modes

    def _freeze_candidates(self):
        return [u for u in range(self.n) if u not in self.protect]

    def _decide_freeze(self, frac):
        pol = self.freeze_policy
        if pol == "off":
            return
        if pol == "freezeout":
            cand = self._freeze_candidates()
            k = len(cand)
            for i, u in enumerate(cand):          # unit order == depth order
                t_i = self.freezeout_t0 + (1 - self.freezeout_t0) * i / max(k, 1)
                self.state[u] = "frozen" if frac >= t_i else "train"
            return
        # thaw units that have been frozen long enough, so they can be re-measured
        for u in range(self.n):
            if self.state[u] == "frozen":
                self.frozen_for[u] += 1
                if self.frozen_for[u] >= self.thaw_every:
                    self.state[u], self.frozen_for[u] = "train", 0
                    self.last_gv[u] = self.last_am[u] = float("nan")   # stale; needs a new window
        if pol == "gvam":
            for u in self._freeze_candidates():
                if self.state[u] == "frozen" or self.am_ref[u] in (None, 0):
                    continue
                gv, am = self.last_gv[u], self.last_am[u]
                if math.isnan(gv):
                    continue
                if gv < self.frz_gv and am / self.am_ref[u] < self.frz_am:
                    self.state[u], self.frozen_for[u] = "frozen", 0
        elif pol == "random":
            target = self._replay_entry().get("frozen", 0)
            cur = [u for u in range(self.n) if self.state[u] == "frozen"]
            free = [u for u in self._freeze_candidates() if self.state[u] != "frozen"]
            need = max(0, target - len(cur))
            for u in self.rng.sample(free, min(need, len(free))):
                self.state[u], self.frozen_for[u] = "frozen", 0

    def _replay_entry(self):
        return (self.replay or {}).get(self._step, {})

    def _decide_shadow(self):
        pol = self.shadow_policy
        train = [u for u in range(self.n) if self.state[u] != "frozen"]
        if pol == "all":
            self.shadow = [True] * self.n
        elif pol == "none":
            self.shadow = [False] * self.n
        elif pol == "gvam":
            for u in train:
                if self.sel_am is not None:   # AM rule: small updates -> shadow unnecessary
                    am, ref = self.last_am[u], self.am_ref[u]
                    self.shadow[u] = math.isnan(am) or ref in (None, 0) or am / ref >= self.sel_am
                else:                         # GV rule
                    gv = self.last_gv[u]
                    self.shadow[u] = math.isnan(gv) or gv >= self.sel_gv
        elif pol == "random":
            target = self._replay_entry().get("noshadow", 0)
            drop = set(self.rng.sample(train, min(target, len(train))))
            self.shadow = [u not in drop for u in range(self.n)]

    def modes(self):
        out = []
        for u in range(self.n):
            if self.state[u] == "frozen":
                out.append("frozen")
            elif self.pipeline == "none":
                out.append("exact")
            else:
                out.append("shadow" if self.shadow[u] else "noshadow")
        return out


def replay_from_trace(path):
    """Counts of frozen / no-shadow units per decision, from a gvam run's trace."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["step"]] = {"frozen": sum(m == "frozen" for m in r["modes"]),
                              "noshadow": sum(m == "noshadow" for m in r["modes"])}
    return out
