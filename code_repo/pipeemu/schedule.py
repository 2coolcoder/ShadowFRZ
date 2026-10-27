"""1F1B pipeline schedule simulation.

Two uses:
  1. Derive, from an actual simulated schedule, which weight version every (stage,
     microbatch) forward/backward sees under each weight-update policy. The emulator's
     version rules (engine.py) are checked against this in tests/test_schedule.py.
  2. Model throughput: given per-stage forward/backward times, simulate steady-state
     time per batch for flush-based (vanilla / PipeDream-Flush) and flush-free (2BW, 1BW)
     pipelines.
"""
from __future__ import annotations


def _simulate(order, dep, dur):
    """List-schedule per-stage op sequences. dur(stage, kind) -> time.
    Returns {(stage, kind, mb): (start, end)}."""
    done, free = {}, {s: 0.0 for s in order}
    idx = {s: 0 for s in order}
    progress = True
    while progress:
        progress = False
        for s in order:
            while idx[s] < len(order[s]):
                kind, k = order[s][idx[s]]
                deps = dep(s, kind, k)
                if any(d not in done for d in deps):
                    break
                start = max([free[s]] + [done[d][1] for d in deps])
                end = start + dur(s, kind)
                done[(s, kind, k)] = (start, end)
                free[s] = end
                idx[s] += 1
                progress = True
    assert all(idx[s] == len(order[s]) for s in order), "schedule deadlocked"
    return done


def one_f_one_b_order(d, n, start=1):
    """Stage s (1-indexed) runs d-s+1 warm-up forwards, then alternates B, F (PipeDream 1F1B)."""
    order = {}
    for s in range(1, d + 1):
        w = min(d - s + 1, n)
        seq = [("F", start + k) for k in range(w)]
        nf = w
        for nb in range(n):
            seq.append(("B", start + nb))
            if nf < n:
                seq.append(("F", start + nf))
                nf += 1
        order[s] = seq
    return order


def pipeline_dep(d):
    def dep(s, kind, k):
        if kind == "F":
            return [] if s == 1 else [(s - 1, "F", k)]
        return [(d, "F", k)] if s == d else [(s + 1, "B", k)]
    return dep


def simulate_1f1b(d, n, fwd=1.0, bwd=2.0):
    """Continuous (flush-free) 1F1B over n microbatches. fwd/bwd: scalar or per-stage list."""
    f = (lambda s: fwd[s - 1]) if isinstance(fwd, (list, tuple)) else (lambda s: fwd)
    b = (lambda s: bwd[s - 1]) if isinstance(bwd, (list, tuple)) else (lambda s: bwd)
    return _simulate(one_f_one_b_order(d, n), pipeline_dep(d), lambda s, kind: f(s) if kind == "F" else b(s))


def version_table(d, m, n_batches, policy):
    """Weight version seen by every (stage, microbatch) forward and backward.

    policy: 'vanilla' (PipeDream-Flush: flush every batch, versions exact),
            '2bw'     (double-buffered: microbatch k uses max(floor((k-1)/m)-1, 0)),
            '1bw'     (single version, updated in place when a stage finishes its last
                       backward of a batch; read off the simulated schedule).
    Returns {(stage, 'F'|'B', k): version}, k 1-indexed over all microbatches.
    """
    n = m * n_batches
    table = {}
    if policy in ("vanilla", "2bw"):
        for s in range(1, d + 1):
            for k in range(1, n + 1):
                b = (k - 1) // m
                v = b if policy == "vanilla" else max(b - 1, 0)
                table[(s, "F", k)] = table[(s, "B", k)] = v
        return table
    assert policy == "1bw"
    sched = simulate_1f1b(d, n)
    for s in range(1, d + 1):
        upd = sorted(sched[(s, "B", j * m)][1] for j in range(1, n_batches + 1))
        for kind in ("F", "B"):
            for k in range(1, n + 1):
                t0 = sched[(s, kind, k)][0]
                table[(s, kind, k)] = sum(1 for x in upd if x <= t0)
    return table


def straddlers_1bw(d, m):
    """Closed form used by the emulator: microbatch j (1..m) of a batch has its FORWARD on
    stage s with the previous version iff j <= d - s; its backward always sees the newest."""
    return {(s, j): j <= d - s for s in range(1, d + 1) for j in range(1, m + 1)}


def time_per_batch(stage_fwd, stage_bwd, m, flush, n_batches=6):
    """Steady-state time per batch of m microbatches for given per-stage times.
    flush=True: pipeline drains after every batch (vanilla / PipeDream-Flush semantics).
    flush=False: continuous 1F1B (2BW / 1BW)."""
    d = len(stage_fwd)
    if flush:
        sched = simulate_1f1b(d, m, list(stage_fwd), list(stage_bwd))
        return max(e for (_, e) in sched.values())
    sched = simulate_1f1b(d, m * n_batches, list(stage_fwd), list(stage_bwd))
    # completion times of the last backward of each batch on stage 1
    ends = [sched[(1, "B", j * m)][1] for j in range(1, n_batches + 1)]
    return (ends[-1] - ends[1]) / (n_batches - 2)
