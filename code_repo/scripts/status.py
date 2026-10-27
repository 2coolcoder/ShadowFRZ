"""Compact progress report over all result directories + recent failures."""
import glob, json, os, re, time
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(time.strftime("%H:%M:%S"))
for track in ("trackA", "trackB", "trackC", "trackC_sweep", "trackC_ft"):
    done, run = [], []
    for d in sorted(glob.glob(os.path.join(root, "results", track, "*"))):
        n = os.path.basename(d)
        if os.path.exists(os.path.join(d, "final.json")):
            done.append(n)
        else:
            try:
                last = open(os.path.join(d, "log.jsonl")).read().strip().splitlines()[-1]
                r = json.loads(last)
                run.append(f"{n}@{r.get('step', r.get('epoch'))}")
            except Exception:
                run.append(f"{n}@start")
    print(f"[{track}] done {len(done)} | running: {' '.join(run)}")
fails = []
for q in glob.glob(os.path.join(root, "logs", "queue_*.log")):
    for line in open(q):
        if "rc=" in line and "rc=0" not in line and time.time() - os.path.getmtime(q) < 4 * 3600:
            fails.append(line.strip())
print("recent non-zero exits:", fails[-6:])
