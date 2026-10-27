"""Shared training loop for Tracks B and C: emulator + freezing controller + logging +
cost-model summary.

Variants (rows of the ablation table, slide 22):
    vanilla        no pipeline staleness (== PipeDream-Flush / GPipe / data parallel)
    frz            vanilla + GV/AM freezing                         (row 1, ~PolicyFRZ)
    2bw            all units keep a shadow                           (row 2)
    1bw            no unit keeps a shadow                            (row 3)
    sel            shadow kept only where GV is high, no freezing     (row 4)
    2bw_frz        2BW + GV/AM freezing                              (row 5)
    full           selective shadow + GV/AM freezing                 (row 6, our method)
    full_random    same counts as `full` (replayed), random units    (row 7)
    full_freezeout selective shadow + FreezeOut-style prefix schedule (row 8)
    full_adamreset `full` + optimizer state reset when a unit thaws   (row 9)
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

from .costmodel import batch_time, partition, profile_units, stage_memory
from .engine import PipelineEmulator
from .freeze import FreezeController, replay_from_trace

VARIANTS = {
    #               pipeline  freeze       shadow
    "vanilla":        ("none", "off",       "all"),
    "frz":            ("none", "gvam",      "all"),
    "2bw":            ("pipe", "off",       "all"),
    "1bw":            ("pipe", "off",       "none"),
    "sel":            ("pipe", "off",       "gvam"),
    "2bw_frz":        ("pipe", "gvam",      "all"),
    "full":           ("pipe", "gvam",      "gvam"),
    "full_random":    ("pipe", "random",    "random"),
    "full_freezeout": ("pipe", "freezeout", "gvam"),
    "full_adamreset": ("pipe", "gvam",      "gvam"),
}


def run(net, *, variant, d, m, optimizer, lr_fn, total_steps, get_batch, split, evaluate,
        eval_every, out, opt_slots, profile_batch, ctrl_kw, clip_norm=None, replay_trace=None,
        log_every=50, save_state=None, stop_at=None, canon_key=None):
    os.makedirs(out, exist_ok=True)
    pipeline, fpol, spol = VARIANTS[variant]

    # ---- profile units (also defines the stage partition), cached per run
    prof_path = os.path.join(out, "profile.json")
    shared = (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), "canonical",
                           f"{canon_key}_m{m}_profile.json") if canon_key else None)
    if not os.path.exists(prof_path) and shared and os.path.exists(shared):
        import shutil
        shutil.copy(shared, prof_path)       # skip the memory-hungry profiling pass
    if os.path.exists(prof_path):
        prof = json.load(open(prof_path))
    else:
        prof = profile_units(net, profile_batch)
        net.zero_grad(set_to_none=True)
        json.dump(prof, open(prof_path, "w"))
    # canonical stage partition per (model, d): shared by every run so that 1BW's
    # stage-dependent semantics and the cost model are identical across runs
    canon = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), "canonical", f"{canon_key}.json") if canon_key else None
    if canon and os.path.exists(canon):
        stages = json.load(open(canon))
    else:
        stages = partition([f + b for f, b in zip(prof["t_fwd"], prof["t_bwd_train"])], d)
        if canon:
            os.makedirs(os.path.dirname(canon), exist_ok=True)
            json.dump(stages, open(canon, "w"))

    n = len(net.units)
    emu = PipelineEmulator(net.unit_params(), stages, d, m, optimizer, clip_norm=clip_norm)
    replay = replay_from_trace(replay_trace) if fpol == "random" or spol == "random" else None
    trace_path = os.path.join(out, "trace.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)
    ctrl = FreezeController(net.unit_params(), pipeline=pipeline, freeze_policy=fpol, shadow_policy=spol,
                            total_steps=total_steps, protect=(n - 1,), replay=replay,
                            trace_path=trace_path, **ctrl_kw)
    emu.set_modes(ctrl.modes())
    reset_on_thaw = variant == "full_adamreset"

    logf = open(os.path.join(out, "log.jsonl"), "w")
    t0, run_loss, k = time.time(), 0.0, 0
    for step in range(1, total_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_fn(step - 1) * g.get("lr_scale", 1.0)
        batch = get_batch()
        loss = emu.step(split(batch, m), net.loss, after_backward=ctrl.observe)
        if not math.isfinite(loss):
            logf.write(json.dumps({"step": step, "diverged": True}) + "\n")
            break
        run_loss += loss; k += 1
        new = ctrl.maybe_decide(step)
        if new is not None:
            if reset_on_thaw:
                thawed = [u for u in range(n) if emu.mode[u] == "frozen" and new[u] != "frozen"]
                emu.reset_optimizer_state(thawed)
            emu.set_modes(new)
        if step % log_every == 0 or step == stop_at:
            rec = {"step": step, "train_loss": run_loss / k, "lr": lr_fn(step - 1), "time": time.time() - t0}
            run_loss, k = 0.0, 0
            if step % eval_every == 0 or step == total_steps or step == stop_at:
                rec.update(evaluate())
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if stop_at is not None and step >= stop_at:
            break
    wall = time.time() - t0
    final = evaluate()
    if save_state:
        save_state()

    # ---- cost model over the recorded trace (each window weighted equally)
    summ = summarize_trace(trace_path, prof, stages, d, m, opt_slots, pipeline)
    res = {"variant": variant, "d": d, "m": m, "stages": stages, "wall_s": wall, "steps": step,
           "final": final, "mismatch": emu.stats, **summ}
    json.dump(res, open(os.path.join(out, "final.json"), "w"), indent=1)
    return res


def summarize_trace(trace_path, prof, stages, d, m, opt_slots, pipeline):
    flush = pipeline == "none"
    rows = [json.loads(l) for l in open(trace_path)] if os.path.exists(trace_path) else []
    if not rows:
        return {}
    mem_curve, t_static, t_reb, frozen_frac, noshadow_frac = [], [], [], [], []
    for r in rows:
        modes = r["modes"]
        mem = stage_memory(prof, stages, modes, d, opt_slots)
        mem_curve.append({s: mem[s]["total"] for s in mem})
        t_static.append(batch_time(prof, stages, modes, d, m, flush))
        t_reb.append(batch_time(prof, stages, modes, d, m, flush, rebalance=True))
        frozen_frac.append(sum(x == "frozen" for x in modes) / len(modes))
        noshadow_frac.append(sum(x == "noshadow" for x in modes) / len(modes))
    first = rows[0]["modes"]
    base_mem = stage_memory(prof, stages, first, d, opt_slots)
    avg = lambda v: sum(v) / len(v)
    max_stage_mem = [max(c.values()) for c in mem_curve]
    return {
        "mem_peak_maxstage_MB": max(max_stage_mem) / 2**20,
        "mem_avg_maxstage_MB": avg(max_stage_mem) / 2**20,
        "mem_avg_stage1_MB": avg([c[1] for c in mem_curve]) / 2**20,
        "mem_start_by_stage_MB": {s: base_mem[s]["total"] / 2**20 for s in base_mem},
        "mem_start_breakdown_stage1_MB": {k: v / 2**20 for k, v in base_mem[1].items()},
        "time_static_avg": avg(t_static), "time_rebal_avg": avg(t_reb),
        "time_static_first": t_static[0],
        "frozen_frac_avg": avg(frozen_frac), "noshadow_frac_avg": avg(noshadow_frac),
        "n_windows": len(rows),
    }
