"""V3: a REAL 2-stage, 2-GPU pipeline (1F1B, PipeDream order) vs. the single-GPU emulator.

Rank 0 = stage 1 on GPU 0, rank 1 = stage 2 on GPU 1. Activations/gradients go over a
gloo process group (CPU copies; correctness test, not a speed test): sends are isend, so
1F1B's interleaving cannot deadlock. Semantics:
  vanilla : flush after every batch, update with the batch's gradient     (PipeDream-Flush)
  2bw     : microbatch k uses version max(floor((k-1)/m)-1, 0); 2 versions per stage
  1bw     : one version, updated in place as soon as the stage finishes its last backward
            of a batch (no shadow)
Compares per-step losses and final weights with the emulator, and reports each rank's
peak CUDA memory next to the cost model's prediction.
Run: torchrun --nproc_per_node 2 tests/realpipe_d2.py --variant 1bw
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

from data.loaders import WikiTextMLM
from models.unitnet import BertMLMUnits
from pipeemu.schedule import one_f_one_b_order

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="1bw", choices=["vanilla", "2bw", "1bw"])
ap.add_argument("--size", default="mini")
ap.add_argument("--m", type=int, default=4)
ap.add_argument("--micro", type=int, default=16)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--out", default="results/validation")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def build(dev):
    torch.manual_seed(0)
    net = BertMLMUnits(a.size)
    for mod in net.modules():                 # no dropout: removes RNG divergence between runs
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout_prob"):      # SDPA attention dropout is an attribute, not a module
            mod.dropout_prob = 0.0
    return net.to(dev)


def make_data(dev):
    data = WikiTextMLM(dev, seed=0)
    return [[tuple(t.chunk(a.m)[j] for t in data.batch(a.micro * a.m)) for j in range(a.m)]
            for _ in range(a.steps)]


def stage_split(net):
    n = len(net.units)
    return list(range(0, n // 2)), list(range(n // 2, n))


# ------------------------------------------------------------------------- emulator side
def emulate():
    from pipeemu.engine import PipelineEmulator
    dev = "cuda:0"
    net = build(dev)
    s1, s2 = stage_split(net)
    stages = [1 if u in s1 else 2 for u in range(len(net.units))]
    opt = torch.optim.SGD(net.parameters(), lr=a.lr)
    emu = PipelineEmulator(net.unit_params(), stages, 2, a.m, opt)
    emu.set_modes([{"vanilla": "exact", "2bw": "shadow", "1bw": "noshadow"}[a.variant]] * len(net.units))
    batches = make_data(dev)
    losses = [emu.step(b, net.loss) for b in batches]
    return losses, [p.detach().cpu() for p in net.parameters()]


# ------------------------------------------------------------------------- real pipeline
def real():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    dev = f"cuda:{rank}"
    torch.cuda.set_device(dev)
    net = build(dev)
    s1, s2 = stage_split(net)
    mine = s1 if rank == 0 else s2
    params = [p for u in mine for p in net.units[u].parameters()]
    for u in range(len(net.units)):
        if u not in mine:
            net.units[u].to("meta")            # free the other stage's weights
    # fp32 master weights own the optimizer; working weights are refreshed through .data,
    # so in-flight autograd graphs are not invalidated (a no-stash pipeline must do this)
    master = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt = torch.optim.SGD(master, lr=a.lr)
    batches = make_data(dev)
    m, d, s = a.m, 2, rank + 1
    n = m * a.steps
    flat_mb = [mb for b in batches for mb in b]
    versions = {0: [p.detach().clone() for p in params]}   # 2BW: version index -> weights
    working = {"v": 0}

    def load_version(v):
        if working["v"] != v:
            with torch.no_grad():
                for p, t in zip(params, versions[v]):
                    p.data.copy_(t)
            working["v"] = v

    def fwd_stage(x, mb):
        for u in mine:
            x = net.unit_forward(u, x, mb[1])
        return x

    torch.cuda.reset_peak_memory_stats(dev)
    stash, losses, newest = {}, [], 0
    if a.variant == "vanilla":
        order = []
        for b in range(a.steps):
            order += [(kind, k + b * m) for kind, k in one_f_one_b_order(d, m, start=1)[s]]
    else:
        order = one_f_one_b_order(d, n, start=1)[s]
    pend = []
    for kind, k in order:
        mb = flat_mb[k - 1]
        bidx = (k - 1) // m
        if a.variant == "2bw":
            load_version(max(bidx - 1, 0))                  # same version for F and B
        if kind == "F":
            if rank == 0:
                out = fwd_stage(mb[0], mb)
                pend.append(dist.isend(out.detach().cpu(), 1, tag=k))
                stash[k] = out
            else:
                buf = torch.empty((a.micro, 128, net.cfg.hidden_size))
                dist.recv(buf, 0, tag=k)
                x = buf.to(dev).requires_grad_(True)
                stash[k] = (x, fwd_stage(x, mb))
        else:
            if rank == 0:
                g = torch.empty(stash[k].shape)
                dist.recv(g, 1, tag=10**6 + k)
                stash.pop(k).backward(g.to(dev))      # already scaled by 1/m on stage 2
            else:
                x, loss = stash.pop(k)
                (loss / m).backward()
                losses.append(float(loss.detach()))
                pend.append(dist.isend(x.grad.detach().cpu(), 0, tag=10**6 + k))
            if k % m == 0:                                  # last backward of the batch
                for mp, p in zip(master, params):
                    mp.grad = p.grad; p.grad = None
                opt.step(); opt.zero_grad(set_to_none=True)
                newest += 1
                if a.variant == "2bw":
                    versions[newest] = [mp.detach().clone() for mp in master]
                    for v in [v for v in versions if v < newest - 1]:
                        del versions[v]
                    assert len(versions) <= 2
                else:                                       # vanilla / 1bw: in place, now
                    with torch.no_grad():
                        for p, mp in zip(params, master):
                            p.data.copy_(mp)
    for r in pend:
        r.wait()
    peak = torch.cuda.max_memory_allocated(dev) / 2**20
    res = {"rank": rank, "peak_MB": peak, "losses": losses,
           "params": [mp.detach().cpu() for mp in master]}
    os.makedirs(a.out, exist_ok=True)
    torch.save(res, os.path.join(a.out, f"tmp_realpipe_{a.variant}_rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" in os.environ:
        real()
    else:
        losses_e, params_e = emulate()
        r0 = torch.load(os.path.join(a.out, f"tmp_realpipe_{a.variant}_rank0.pt"))
        r1 = torch.load(os.path.join(a.out, f"tmp_realpipe_{a.variant}_rank1.pt"))
        # rank1 losses are per microbatch; emulator losses are per batch (mean of m)
        real_batch = [sum(r1["losses"][i * a.m:(i + 1) * a.m]) / a.m for i in range(a.steps)]
        dl = max(abs(x - y) for x, y in zip(losses_e, real_batch))
        real_params = r0["params"] + r1["params"]
        dp = max(float((x - y).abs().max()) for x, y in zip(params_e, real_params))
        res = {"variant": a.variant, "max_loss_diff": dl, "max_param_diff": dp,
               "loss_first_last_emul": [losses_e[0], losses_e[-1]],
               "peak_MB_stage1": r0["peak_MB"], "peak_MB_stage2": r1["peak_MB"]}
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, f"V3_{a.variant}.json"), "w"), indent=1)
        print(json.dumps(res))
