"""Generate job files for the threshold-dependent runs (after calibration, see analysis.md).

Usage: python scripts/make_jobs.py trackB  --frz_gv X --frz_am Y --sel_gv Z
       python scripts/make_jobs.py trackC  --frz_gv X --frz_am Y --sel_gv Z --lrs mini=..,small=..,medium=..,base=..
       python scripts/make_jobs.py trackB_random   (needs B full seed-0 trace)
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("which")
ap.add_argument("--frz_gv", type=float)
ap.add_argument("--frz_am", type=float)
ap.add_argument("--sel_gv", type=float)
ap.add_argument("--sel_am", type=float, default=None)
ap.add_argument("--lrs", default="")
ap.add_argument("--steps", type=int, default=20000)
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
thr = lambda: f"--frz_gv {a.frz_gv} --frz_am {a.frz_am} --sel_gv {a.sel_gv}" + (f" --sel_am {a.sel_am}" if a.sel_am else "")
L = []

if a.which == "trackB":
    def j(n, args):
        L.append(f"{n} | python train_cnn.py {args} {thr()} --out results/trackB/{n}")
    for v in ("full", "frz", "2bw_frz", "sel", "full_freezeout", "full_adamreset"):
        j(f"B_c10_{v}_s0", f"--variant {v} --seed 0")
    j("B_c10_full_s1", "--variant full --seed 1")
    j("B_c10_frz_s1", "--variant frz --seed 1")
    L.append("B_c10_full_aggressive_s0 | python train_cnn.py --variant full --seed 0 --frz_gv 0.073 --frz_am 1e9 "
             "--sel_gv 0.0785 --out results/trackB/B_c10_full_aggressive_s0")
    for v in ("vanilla", "2bw", "frz", "full"):                 # CIFAR-100 core, last
        j(f"B_c100_{v}_s0", f"--data cifar100 --variant {v} --seed 0")

elif a.which == "trackB_random":
    for s in (0,):
        n = f"B_c10_full_random_s{s}"
        L.append(f"{n} | python train_cnn.py --variant full_random --seed {s} {thr()} "
                 f"--replay results/trackB/B_c10_full_s0/trace.jsonl --out results/trackB/{n}")

elif a.which == "trackC":
    lrs = dict(kv.split("=") for kv in a.lrs.split(","))
    def j(n, args, size):
        L.append(f"{n} | python train_bert.py --size {size} --lr {lrs[size]} --steps {a.steps} {args} {thr()} --save "
                 f"--out results/trackC/{n}")
    # ladder core: d=4, m=4, same token budget for every size
    for size in ("mini", "small", "medium", "base"):
        for v in ("vanilla", "2bw", "1bw", "full"):
            j(f"C_{size}_{v}_s0", f"--variant {v} --seed 0", size)
    # Medium ablations + second seed for the key rows
    for v in ("frz", "sel", "2bw_frz", "full_freezeout", "full_adamreset"):
        j(f"C_medium_{v}_s0", f"--variant {v} --seed 0", "medium")
    for v in ("vanilla", "2bw", "1bw", "full", "frz"):
        j(f"C_medium_{v}_s1", f"--variant {v} --seed 1", "medium")
    # 1BW depth / microbatch sweep on Medium (d, m)
    for d, m in ((2, 2), (2, 8), (4, 16), (8, 8), (8, 32)):
        j(f"C_medium_1bw_d{d}_m{m}_s0", f"--variant 1bw --d {d} --m {m} --seed 0", "medium")
    j("C_medium_2bw_d8_m8_s0", "--variant 2bw --d 8 --m 8 --seed 0", "medium")

elif a.which == "trackC_random":
    lrs = dict(kv.split("=") for kv in a.lrs.split(","))
    n = "C_medium_full_random_s0"
    L.append(f"{n} | python train_bert.py --size medium --lr {lrs['medium']} --steps {a.steps} --variant full_random "
             f"--seed 0 {thr()} --replay results/trackC/C_medium_full_s0/trace.jsonl --out results/trackC/{n}")

path = os.path.join(root, "jobs", f"{a.which}.txt")
open(path, "w").write("\n".join(L) + "\n")
print(path, len(L), "jobs")
