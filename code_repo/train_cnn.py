"""Track B: ResNet-50 on CIFAR under each pipeline / freezing variant (emulated)."""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from data.loaders import CifarGPU
from models.unitnet import ResNetUnits
from pipeemu.runner import VARIANTS, run

p = argparse.ArgumentParser()
p.add_argument("--model", default="resnet50")
p.add_argument("--data", default="cifar10")
p.add_argument("--variant", default="vanilla", choices=list(VARIANTS))
p.add_argument("--d", type=int, default=4)
p.add_argument("--m", type=int, default=4)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--epochs", type=int, default=160)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--window", type=int, default=100)
p.add_argument("--frz_gv", type=float, default=0.2)
p.add_argument("--frz_am", type=float, default=0.5)
p.add_argument("--sel_gv", type=float, default=0.4)
p.add_argument("--sel_am", type=float, default=None)
p.add_argument("--thaw_every", type=int, default=4)
p.add_argument("--replay", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
if os.path.exists(os.path.join(a.out, "final.json")):
    print("already done"); raise SystemExit
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(a.seed)
dev = "cuda"
data = CifarGPU(a.data, dev, seed=a.seed)
net = ResNetUnits(a.model, data.classes, zero_init_residual=True).to(dev)
opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
spe = data.steps_per_epoch(a.batch)
total = a.epochs * spe
lr_fn = lambda s: a.lr * 0.5 * (1 + math.cos(math.pi * s / total))

def batches():
    while True:
        yield from data.train_epoch(a.batch)
it = batches()
split = lambda b, m: list(zip(b[0].chunk(m), b[1].chunk(m)))

@torch.no_grad()
def evaluate():
    net.eval()
    c = n = 0; loss = 0.0
    for x, y in data.test_batches():
        z = net.logits(x)
        loss += float(F.cross_entropy(z, y, reduction="sum")); c += int((z.argmax(1) == y).sum()); n += y.numel()
    net.train()
    return {"test_acc": c / n, "test_loss": loss / n}

x0, y0 = next(data.test_batches())
res = run(net, canon_key=f"{a.model}_{a.data}_d{a.d}", variant=a.variant, d=a.d, m=a.m, optimizer=opt, lr_fn=lr_fn, total_steps=total,
          get_batch=lambda: next(it), split=split, evaluate=evaluate, eval_every=spe * 4,
          out=a.out, opt_slots=1, profile_batch=(x0[: a.batch // a.m], y0[: a.batch // a.m]),
          ctrl_kw=dict(window=a.window, frz_gv=a.frz_gv, frz_am=a.frz_am, sel_gv=a.sel_gv, sel_am=a.sel_am,
                       thaw_every=a.thaw_every, seed=a.seed),
          replay_trace=a.replay, log_every=spe // 2)
print(json.dumps({k: res[k] for k in ("final", "wall_s", "frozen_frac_avg", "noshadow_frac_avg")}))
