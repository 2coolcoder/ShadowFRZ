"""Track C: BERT MLM pre-training on WikiText-103 under each variant (emulated pipeline)."""
import argparse
import json
import os

import torch

from data.loaders import WikiTextMLM
from models.unitnet import BertMLMUnits, BertPreLNMLMUnits
from pipeemu.runner import VARIANTS, run

p = argparse.ArgumentParser()
p.add_argument("--size", default="medium")
p.add_argument("--arch", default="preln", choices=["preln", "postln"])
p.add_argument("--stop_at", type=int, default=None, help="stop early; LR schedule still uses --steps")
p.add_argument("--variant", default="vanilla", choices=list(VARIANTS))
p.add_argument("--d", type=int, default=4)
p.add_argument("--m", type=int, default=4)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--warmup", type=float, default=0.06)
p.add_argument("--window", type=int, default=100)
p.add_argument("--frz_gv", type=float, default=0.2)
p.add_argument("--frz_am", type=float, default=0.5)
p.add_argument("--sel_gv", type=float, default=0.4)
p.add_argument("--sel_am", type=float, default=None)
p.add_argument("--thaw_every", type=int, default=4)
p.add_argument("--replay", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save", action="store_true")
p.add_argument("--out", required=True)
a = p.parse_args()
if os.path.exists(os.path.join(a.out, "final.json")):
    print("already done"); raise SystemExit
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(a.seed)
dev = "cuda"
data = WikiTextMLM(dev, seed=a.seed)
net = (BertPreLNMLMUnits(a.size) if a.arch == "preln" else BertMLMUnits(a.size)).to(dev)
decay, no_decay = [], []
for n_, p_ in net.named_parameters():
    (no_decay if p_.ndim == 1 else decay).append(p_)        # biases and LayerNorm: no decay
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.999), eps=1e-6)
W = int(a.warmup * a.steps)
lr_fn = lambda s: a.lr * (s / max(W, 1) if s < W else max(0.0, (a.steps - s) / max(a.steps - W, 1)))
split = lambda b, m: list(zip(b[0].chunk(m), b[1].chunk(m)))

@torch.no_grad()
def evaluate():
    net.eval()
    tot = cor = 0; loss = 0.0
    for ids, lab in data.valid_batches():
        x, _ = net.prepare((ids, lab))
        for i in range(len(net.units) - 1):
            x = net.unit_forward(i, x, None)
        msk = lab != -100
        logits = net.units[-1](x[msk])
        loss += float(torch.nn.functional.cross_entropy(logits, lab[msk], reduction="sum"))
        cor += int((logits.argmax(-1) == lab[msk]).sum()); tot += int(msk.sum())
    net.train()
    return {"valid_loss": loss / tot, "valid_mlm_acc": cor / tot}

save = (lambda: torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))) if a.save else None
res = run(net, canon_key=f"bert_{a.size}_d{a.d}" if a.arch == "preln" else None, variant=a.variant, d=a.d, m=a.m, optimizer=opt, lr_fn=lr_fn, total_steps=a.steps, stop_at=a.stop_at,
          get_batch=lambda: data.batch(a.batch), split=split, evaluate=evaluate, eval_every=1000,
          out=a.out, opt_slots=2, profile_batch=data.batch(a.batch // a.m), clip_norm=1.0,
          ctrl_kw=dict(window=a.window, frz_gv=a.frz_gv, frz_am=a.frz_am, sel_gv=a.sel_gv, sel_am=a.sel_am,
                       thaw_every=a.thaw_every, seed=a.seed),
          replay_trace=a.replay, log_every=100, save_state=save)
print(json.dumps({k: res[k] for k in ("final", "wall_s", "frozen_frac_avg", "noshadow_frac_avg")}))
