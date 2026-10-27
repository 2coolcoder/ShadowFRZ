"""Track A: backprop vs. DDG (Huo et al., ICML'18) on CIFAR ResNets, emulated exactly.

DDG as specified by the paper's Eqs. (6)-(9): the network is split into K modules; at
iteration t the forward pass of sample-batch t is unchanged; module k back-propagates
sample-batch t-(K-k) using the error gradient produced by module k+1 one iteration
earlier, at the weights w^{t-K+k} that its forward pass used, and every module then
applies its delayed gradient to its CURRENT weights. K = 1 is exactly backprop, so the
baseline runs through the same code path.

Weights are kept as plain tensors and replaced (not modified in place) at every update,
so the autograd graphs of in-flight samples still reference the old weights. That makes
the delayed gradients exact, with no recomputation.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call

from data.loaders import CifarGPU
from models.unitnet import ResNetUnits

p = argparse.ArgumentParser()
p.add_argument("--model", default="resnet56")
p.add_argument("--data", default="cifar10")
p.add_argument("--K", type=int, default=1)
p.add_argument("--recipe", default="modern", choices=["modern", "paper"])
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
if os.path.exists(os.path.join(args.out, "final.json")):
    print("already done"); raise SystemExit

# recipes: modern = lr 0.1 cosine 200 ep; paper = DDG Sec. 5.2 (lr 0.01, /10 at 150, 225 of 300)
R = {"modern": dict(lr=0.1, epochs=200, wd=5e-4, sched="cosine"),
     "paper": dict(lr=0.01, epochs=300, wd=5e-4, sched="step")}[args.recipe]
epochs = args.epochs or R["epochs"]

data = CifarGPU(args.data, dev, seed=args.seed)
net = ResNetUnits(args.model, data.classes).to(dev)
n_units = len(net.units)
K = args.K
bounds = [round(i * n_units / K) for i in range(K + 1)]          # equal units per module
modules = [torch.nn.Sequential(*net.units[bounds[k]:bounds[k + 1]]) for k in range(K)]
params = [{n: p.detach().clone() for n, p in m.named_parameters()} for m in modules]
mom = [{n: torch.zeros_like(t) for n, t in ps.items()} for ps in params]
steps_per_epoch = data.steps_per_epoch(args.batch)
total = epochs * steps_per_epoch


def lr_at(step):
    e = step / steps_per_epoch
    if R["sched"] == "cosine":
        return R["lr"] * 0.5 * (1 + math.cos(math.pi * step / total))
    return R["lr"] * (0.1 ** sum(e >= x for x in (150, 225)))


def run_module(k, x, ps):
    return functional_call(modules[k], ps, (x,))


@torch.no_grad()
def evaluate():
    for m in modules:
        m.eval()
    correct = n = 0
    loss = 0.0
    for x, y in data.test_batches():
        h = x
        for k in range(K):
            h = run_module(k, h, params[k])
        loss += float(F.cross_entropy(h, y, reduction="sum"))
        correct += int((h.argmax(1) == y).sum())
        n += y.numel()
    for m in modules:
        m.train()
    return correct / n, loss / n


pending = [dict() for _ in range(K)]      # module k: sample id -> (param tensors, h_in, h_out)
deltas = [dict() for _ in range(K)]       # module k: sample id -> error gradient at its output
log = open(os.path.join(args.out, "log.jsonl"), "a")
json.dump(vars(args) | {"recipe_cfg": R, "bounds": bounds, "n_units": n_units},
          open(os.path.join(args.out, "config.json"), "w"), indent=1)
t0, step, best = time.time(), 0, 0.0
for epoch in range(epochs):
    tr_loss, nb = 0.0, 0
    for x, y in data.train_epoch(args.batch):
        lr = lr_at(step)
        # ---- forward of sample-batch `step` through all modules with current weights
        h, grads = x, [None] * K
        for k in range(K):
            ps = {n: t.requires_grad_(True) for n, t in params[k].items()}
            h_in = h if k == 0 else h.detach().requires_grad_(True)
            h = run_module(k, h_in, ps)
            pending[k][step] = (ps, h_in, h)
        loss = F.cross_entropy(h, y)
        tr_loss += float(loss.detach()); nb += 1
        # ---- backward: module K fresh; module k < K uses the delta queued last iteration
        for k in range(K - 1, -1, -1):
            sid = step - (K - 1 - k)
            if sid < 0 or sid not in pending[k]:
                continue
            ps, h_in, h_out = pending[k].pop(sid)
            if k == K - 1:
                out, gout = loss, None
            else:
                if sid not in deltas[k]:
                    continue
                out, gout = h_out, deltas[k].pop(sid)
            wrt = list(ps.values()) + ([h_in] if k > 0 else [])
            g = torch.autograd.grad(out, wrt, grad_outputs=gout)
            grads[k] = dict(zip(ps.keys(), g[:len(ps)]))
            if k > 0:
                deltas[k - 1][sid] = g[-1].detach()
        # ---- SGD with momentum + weight decay, out-of-place (old tensors stay in old graphs)
        with torch.no_grad():
            for k in range(K):
                new = {}
                for n, t in params[k].items():
                    gk = grads[k][n] if grads[k] is not None else torch.zeros_like(t)
                    buf = mom[k][n].mul_(0.9).add_(gk + R["wd"] * t.detach())
                    new[n] = t.detach() - lr * buf
                params[k] = new
        step += 1
    acc, te_loss = evaluate()
    best = max(best, acc)
    rec = {"epoch": epoch + 1, "step": step, "lr": lr, "train_loss": tr_loss / nb,
           "test_loss": te_loss, "test_acc": acc, "time": time.time() - t0}
    log.write(json.dumps(rec) + "\n"); log.flush()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(json.dumps(rec), flush=True)

json.dump({"final_acc": acc, "best_acc": best, "epochs": epochs, "time_s": time.time() - t0,
           "K": K, "model": args.model, "data": args.data, "recipe": args.recipe, "seed": args.seed},
          open(os.path.join(args.out, "final.json"), "w"), indent=1)
print("done", acc, best)
