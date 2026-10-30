"""Queue SST-2 downstream fine-tuning (same recipe for all) for every finished Track C run
that saved a checkpoint and has not been fine-tuned yet."""
import glob
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = []
for d in sorted(glob.glob(os.path.join(root, "results", "trackC", "C_*"))):
    name = os.path.basename(d)
    if not (os.path.exists(os.path.join(d, "final.json")) and os.path.exists(os.path.join(d, "model.pt"))):
        continue
    out = f"results/trackC_ft/FT_sst2_{name}"
    if os.path.exists(os.path.join(root, out)):
        continue
    L.append(f"FT_sst2_{name} | python finetune_glue.py --task sst2 --from_mlm results/trackC/{name} --epochs 3 --out {out}")
open(os.path.join(root, "jobs", "trackC_ft.txt"), "w").write("\n".join(L) + "\n")
print(len(L), "fine-tune jobs")
