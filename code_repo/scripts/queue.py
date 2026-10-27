"""Tiny job queue: run commands from a jobs file with N concurrent jobs per GPU.

jobs file: one job per line, "<name> | <command>"; lines starting with # are ignored.
A job is skipped if results/<...>/final.json named by --out in its command exists.
Usage: python scripts/queue.py jobs.txt --gpus 0,1 --per-gpu 4
"""
import argparse
import os
import re
import shutil
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("jobs")
ap.add_argument("--gpus", default="0,1")
ap.add_argument("--per-gpu", type=int, default=4)
ap.add_argument("--settle", type=int, default=300, help="seconds to wait after a launch before the next")
ap.add_argument("--retries", type=int, default=3, help="re-run a job that died with CUDA OOM")
ap.add_argument("--min-free-gb", type=float, default=0.0,
                help="start a job only if the GPU has at least this much free memory")
args = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(root, "logs"), exist_ok=True)

jobs = []
for line in open(args.jobs):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, cmd = [x.strip() for x in line.split("|", 1)]
    m = re.search(r"--out\s+(\S+)", cmd)
    if m and os.path.exists(os.path.join(root, m.group(1), "final.json")):
        continue
    jobs.append((name, cmd))
tries = {}


def oom_died(name):
    try:
        with open(os.path.join(root, "logs", f"{name}.log"), errors="ignore") as f:
            return "OutOfMemoryError" in f.read()[-20000:]
    except OSError:
        return False

slots = {g: [None] * args.per_gpu for g in args.gpus.split(",")}
last_start = {g: 0.0 for g in slots}


def free_gb(g):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", g],
                         capture_output=True, text=True).stdout.strip()
    return float(out) / 1024 if out else 0.0
print(f"{len(jobs)} jobs to run on {list(slots)} x {args.per_gpu}", flush=True)
while jobs or any(p for ps in slots.values() for p in ps):
    for g, ps in slots.items():
        for i, pr in enumerate(ps):
            if pr is not None and pr[1].poll() is not None:
                print(f"[{time.strftime('%H:%M:%S')}] done gpu{g} {pr[0]} rc={pr[1].returncode}", flush=True)
                if pr[1].returncode != 0 and oom_died(pr[0]) and tries.get(pr[0], 0) < args.retries:
                    tries[pr[0]] = tries.get(pr[0], 0) + 1
                    m = re.search(r"--out\s+(\S+)", pr[2])
                    if m:
                        shutil.rmtree(os.path.join(root, m.group(1)), ignore_errors=True)
                    jobs.insert(0, (pr[0], pr[2]))
                    print(f"[{time.strftime('%H:%M:%S')}] OOM -> retry {tries[pr[0]]} {pr[0]}", flush=True)
                ps[i] = None
            # memory-aware start; wait 90 s after a launch so the new job's memory shows up
            if ps[i] is None and jobs and args.min_free_gb > 0 and (
                    time.time() - last_start[g] < args.settle or free_gb(g) < args.min_free_gb):
                continue
            if ps[i] is None and jobs:
                name, cmd = jobs.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
                log = open(os.path.join(root, "logs", f"{name}.log"), "a")
                proc = subprocess.Popen(cmd, shell=True, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                ps[i] = (name, proc, cmd)
                last_start[g] = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] start gpu{g} {name}", flush=True)
    time.sleep(10)
print("all jobs finished", flush=True)
