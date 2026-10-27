"""Per-unit profiling and the pipeline memory / throughput cost model.

Profile (per unit, at the run's microbatch size):
    param_bytes
    act_train   bytes saved for backward when the unit trains
    act_flbp    bytes saved when the unit is frozen but gradients must flow through it
                (PolicyFRZ's FLBP: a trainable unit sits earlier in the network)
    t_fwd, t_bwd_train, t_bwd_flbp   milliseconds (CUDA events, isolated unit)
A frozen unit with no trainable unit before it (PolicyFRZ's FL) saves nothing and runs no
backward.

Memory per stage s of a d-stage 1F1B pipeline (identical stash depth for Flush, 2BW, 1BW):
    sum_u versions_u*W_u + grads_u + opt_slots*W_u + (d - s + 1) * A_u(regime)
Time: per-stage F and B sums fed to the 1F1B simulator (flush for vanilla, none otherwise).
Communication between stages is not modelled.
"""
from __future__ import annotations

import statistics

import torch
from torch.autograd.graph import saved_tensors_hooks

from .schedule import time_per_batch


# ----------------------------------------------------------------------------- profiling
def _saved_bytes_per_unit(net, run, n_units):
    """run(on_unit) executes a forward; returns bytes saved for backward per unit,
    excluding parameters and counting each storage once."""
    param_ptrs = {p.untyped_storage().data_ptr() for p in net.parameters()}
    seen, per = set(), [0] * n_units
    cur = {"u": 0}

    def pack(t):
        try:
            st = t.untyped_storage()
            key = st.data_ptr()
            if key not in seen and key not in param_ptrs and t.is_cuda:
                seen.add(key)
                per[cur["u"]] += st.nbytes()
        except Exception:
            pass
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        out = run(lambda i: cur.__setitem__("u", i))
    del out
    return per


def _time(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def profile_units(net, batch, reps=20):
    n = len(net.units)
    params = [list(u.parameters()) for u in net.units]
    pbytes = [sum(p.numel() * p.element_size() for p in ps) for ps in params]
    for p in net.parameters():
        p.requires_grad_(True)

    # activations, train regime: plain full forward
    act_train = _saved_bytes_per_unit(net, lambda on: net.loss(batch, on_unit=on), n)

    # activations, FLBP regime: all params frozen, gradient flows from unit 1 on
    def run_flbp(on):
        x, ctx = net.prepare(batch)
        with torch.no_grad():
            on(0)
            x = net.unit_forward(0, x, ctx)
        x = x.detach().requires_grad_(True)
        for i in range(1, n):
            on(i)
            x = net.unit_forward(i, x, ctx)
        return x
    for p in net.parameters():
        p.requires_grad_(False)
    act_flbp = _saved_bytes_per_unit(net, run_flbp, n)
    act_flbp[0] = 0
    for p in net.parameters():
        p.requires_grad_(True)

    # capture each unit's input
    inputs = []
    x, ctx = net.prepare(batch)
    with torch.no_grad():
        for i in range(n):
            inputs.append(x)
            x = net.unit_forward(i, x, ctx)

    t_fwd, t_bt, t_bf = [], [], []
    for i in range(n):
        xin = inputs[i]
        float_in = torch.is_tensor(xin) and xin.is_floating_point()

        def fwd(train_params, grad_in):
            for p in params[i]:
                p.requires_grad_(train_params)
            xi = xin.detach().requires_grad_(grad_in) if float_in else xin
            return net.unit_forward(i, xi, ctx)

        def fb(train_params, grad_in):
            out = fwd(train_params, grad_in)
            if out.requires_grad:
                out.backward(torch.ones_like(out) if out.dim() == 0 else torch.randn_like(out))
            for p in params[i]:
                p.grad = None

        with torch.no_grad():
            pass
        tf = _time(lambda: fwd(True, float_in), reps)
        tft = _time(lambda: fb(True, float_in), reps)
        t_fwd.append(tf)
        t_bt.append(max(tft - tf, 0.0))
        if float_in:
            tff = _time(lambda: fb(False, True), reps)
            t_bf.append(max(tff - tf, 0.0))
        else:
            t_bf.append(0.0)
        for p in params[i]:
            p.requires_grad_(True)
    return {"n_units": n, "param_bytes": pbytes, "act_train": act_train, "act_flbp": act_flbp,
            "t_fwd": t_fwd, "t_bwd_train": t_bt, "t_bwd_flbp": t_bf}


# ----------------------------------------------------------------------------- partition
def partition(costs, d):
    """Contiguous split of unit costs into d stages minimising the max stage cost (DP).
    Returns 1-indexed stage ids per unit."""
    n = len(costs)
    assert n >= d
    pre = [0.0]
    for c in costs:
        pre.append(pre[-1] + c)
    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(d + 1)]
    cut = [[0] * (n + 1) for _ in range(d + 1)]
    best[0][0] = 0.0
    for k in range(1, d + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                v = max(best[k - 1][i], pre[j] - pre[i])
                if v < best[k][j]:
                    best[k][j], cut[k][j] = v, i
    stages, j = [0] * n, n
    for k in range(d, 0, -1):
        i = cut[k][j]
        for u in range(i, j):
            stages[u] = k
        j = i
    return stages


# ----------------------------------------------------------------------------- cost model
def regimes(modes):
    """'train' | 'flbp' | 'fl' per unit, from emulator modes."""
    out, trainable_before = [], False
    for m in modes:
        if m != "frozen":
            out.append("train")
            trainable_before = True
        else:
            out.append("flbp" if trainable_before else "fl")
    return out


def stage_memory(prof, stages, modes, d, opt_slots):
    reg = regimes(modes)
    mem = {s: {"weights": 0, "grads": 0, "opt": 0, "acts": 0} for s in range(1, d + 1)}
    for u, s in enumerate(stages):
        w = prof["param_bytes"][u]
        mem[s]["weights"] += w * (2 if modes[u] == "shadow" else 1)
        mem[s]["grads"] += 0 if modes[u] == "frozen" else w
        mem[s]["opt"] += opt_slots * w
        # a frozen pass-through (FLBP) unit never stores more than when training; the cap
        # removes the profiler's double count of the unit's input (see analysis.md 2.4)
        a = {"train": prof["act_train"][u], "flbp": min(prof["act_flbp"][u], prof["act_train"][u]), "fl": 0}[reg[u]]
        mem[s]["acts"] += (d - s + 1) * a
    for s in mem:
        mem[s]["total"] = sum(mem[s].values())
    return mem


def unit_times(prof, modes):
    reg = regimes(modes)
    f = list(prof["t_fwd"])
    b = [{"train": prof["t_bwd_train"][u], "flbp": prof["t_bwd_flbp"][u], "fl": 0.0}[r]
         for u, r in enumerate(reg)]
    return f, b


def batch_time(prof, stages, modes, d, m, flush, rebalance=False):
    f, b = unit_times(prof, modes)
    if rebalance:
        stages = partition([x + y for x, y in zip(f, b)], d)
    sf = [sum(f[u] for u in range(len(f)) if stages[u] == s) for s in range(1, d + 1)]
    sb = [sum(b[u] for u in range(len(b)) if stages[u] == s) for s in range(1, d + 1)]
    return time_per_batch(sf, sb, m, flush)
