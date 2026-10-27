"""GLUE fine-tuning, two modes.

  --from_mlm DIR   downstream check of a Track C pre-training run: load its MLM encoder,
                   fine-tune normally (no pipeline emulation) on SST-2, report dev accuracy.
                   Same recipe for every variant, so differences reflect pre-training only.
  --public NAME    fine-tune a public checkpoint (bert-base-uncased / bert-large-uncased)
                   UNDER a pipeline variant (emulated 2BW / 1BW / freezing during fine-tuning).
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from data.loaders import glue_tensors
from models.unitnet import BertClsUnits, bert_config
from pipeemu.runner import VARIANTS, run

p = argparse.ArgumentParser()
p.add_argument("--task", default="sst2", choices=["sst2", "mnli"])
p.add_argument("--from_mlm", default=None)
p.add_argument("--public", default=None)
p.add_argument("--variant", default="vanilla", choices=list(VARIANTS))
p.add_argument("--d", type=int, default=4)
p.add_argument("--m", type=int, default=4)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--window", type=int, default=50)
p.add_argument("--frz_gv", type=float, default=0.2)
p.add_argument("--frz_am", type=float, default=0.5)
p.add_argument("--sel_gv", type=float, default=0.4)
p.add_argument("--sel_am", type=float, default=None)
p.add_argument("--thaw_every", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
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
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
num_labels = 2 if a.task == "sst2" else 3
dev_split = "validation" if a.task == "sst2" else "validation_matched"
tr = glue_tensors(a.task, "train", tok, dev)
va = glue_tensors(a.task, dev_split, tok, dev)

if a.public:
    net = BertClsUnits(a.public, num_labels).to(dev)
    lr = a.lr or 2e-5
elif json.load(open(os.path.join(a.from_mlm, "config.json"))).get("arch", "postln") == "preln":
    from models.unitnet import BertPreLNClsUnits
    cfg_run = json.load(open(os.path.join(a.from_mlm, "config.json")))
    sd = torch.load(os.path.join(a.from_mlm, "model.pt"), map_location="cpu")
    net = BertPreLNClsUnits(cfg_run["size"], num_labels, sd).to(dev)
    lr = a.lr or 1e-4
else:
    # build a classifier with the pre-trained encoder of a (Post-LN) Track C run
    from transformers import BertForSequenceClassification
    cfg_run = json.load(open(os.path.join(a.from_mlm, "config.json")))
    cfg = bert_config(cfg_run["size"]); cfg.num_labels = num_labels
    hf = BertForSequenceClassification(cfg)
    sd = torch.load(os.path.join(a.from_mlm, "model.pt"), map_location="cpu")
    L = cfg.num_hidden_layers
    enc = {}
    for k, v in sd.items():
        head, rest = k.split(".", 2)[1], k.split(".", 2)[2]
        idx = int(head)
        if idx == 0:
            enc["bert.embeddings." + rest] = v
        elif idx <= L:
            enc[f"bert.encoder.layer.{idx - 1}." + rest] = v
    missing, unexpected = hf.load_state_dict(enc, strict=False)
    assert not unexpected and all(("pooler" in k or "classifier" in k) for k in missing), missing
    net = BertClsUnits.__new__(BertClsUnits)
    torch.nn.Module.__init__(net)
    net.cfg = hf.config
    head = torch.nn.ModuleDict({"pooler": hf.bert.pooler, "dropout": hf.dropout, "classifier": hf.classifier})
    net.units = torch.nn.ModuleList([hf.bert.embeddings, *hf.bert.encoder.layer, head])
    net = net.to(dev)
    lr = a.lr or 1e-4

n_tr = tr[0].shape[0]
spe = n_tr // a.batch
total = int(a.epochs * spe)
decay = [p_ for p_ in net.parameters() if p_.ndim > 1]
nod = [p_ for p_ in net.parameters() if p_.ndim == 1]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nod, "weight_decay": 0.0}],
                        lr=lr, eps=1e-6)
W = int(0.1 * total)
lr_fn = lambda s: lr * (s / max(W, 1) if s < W else max(0.0, (total - s) / max(total - W, 1)))
g = torch.Generator(device=dev).manual_seed(a.seed)
state = {"perm": torch.randperm(n_tr, device=dev, generator=g), "pos": 0}


def get_batch():
    if state["pos"] + a.batch > n_tr:
        state["perm"], state["pos"] = torch.randperm(n_tr, device=dev, generator=g), 0
    idx = state["perm"][state["pos"]:state["pos"] + a.batch]
    state["pos"] += a.batch
    return tuple(t[idx] for t in tr)


split = lambda b, m: list(zip(*[t.chunk(m) for t in b]))


@torch.no_grad()
def evaluate():
    net.eval()
    cor = n = 0
    for i in range(0, va[0].shape[0], 256):
        b = tuple(t[i:i + 256] for t in va)
        cor += int((net.logits(b).argmax(-1) == b[3]).sum()); n += b[3].numel()
    net.train()
    return {"dev_acc": cor / n}


prof_batch = tuple(t[: a.batch // a.m] for t in tr)
res = run(net, canon_key=(f"cls_{a.public}_d{a.d}" if a.public else None), variant=a.variant if a.public else "vanilla", d=a.d, m=a.m if a.public else 1,
          optimizer=opt, lr_fn=lr_fn, total_steps=total, get_batch=get_batch, split=split,
          evaluate=evaluate, eval_every=max(spe // 2, 50) // 50 * 50 or 50, out=a.out, opt_slots=2,
          profile_batch=prof_batch, clip_norm=1.0,
          ctrl_kw=dict(window=a.window, frz_gv=a.frz_gv, frz_am=a.frz_am, sel_gv=a.sel_gv, sel_am=a.sel_am,
                       thaw_every=a.thaw_every, seed=a.seed), log_every=50)
print(json.dumps({"final": res["final"], "wall_s": res["wall_s"]}))
