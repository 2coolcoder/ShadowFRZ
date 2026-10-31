"""Aggregate results into Markdown tables and figures for analysis.md.

python analysis/report.py [trackA|trackB|trackC|calib|all]
Figures -> results/figs/*.png ; tables printed as Markdown.
"""
import glob
import json
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figs")
os.makedirs(FIG, exist_ok=True)

# reference categorical palette (light), fixed order; colour follows the entity
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ENTITY = {"vanilla": 0, "2bw": 1, "1bw": 2, "full": 3, "1bw_warm10": 4, "frz": 5, "sel": 6, "2bw_frz": 7,
          "K1": 0, "K2": 1, "K4": 2}
TEXT, TEXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
})


sys.path.insert(0, ROOT)
from pipeemu.runner import VARIANTS, summarize_trace  # noqa: E402

_CANON_PROF = {}


def canon_key(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    if "model" in cfg:
        return f"{cfg['model']}_{cfg.get('data', 'cifar10')}", cfg
    if "size" in cfg:
        return f"bert_{cfg['size']}", cfg
    return None, cfg


def canonical_profile(key, m):
    """Median per-unit timings over every run of the same model/microbatch; bytes identical."""
    if (key, m) in _CANON_PROF:
        return _CANON_PROF[(key, m)]
    profs = []
    for d in glob.glob(os.path.join(ROOT, "results", "track*", "*")):
        pf, cf = os.path.join(d, "profile.json"), os.path.join(d, "config.json")
        if not (os.path.exists(pf) and os.path.exists(cf)):
            continue
        k, cfg = canon_key(d)
        if k is None or "variant" not in cfg:
            continue
        if k.split("_cifar")[0] == key.split("_cifar")[0] and cfg.get("m", 4) == m and cfg.get("arch", "preln") == "preln":
            profs.append(json.load(open(pf)))
    base = dict(profs[0])
    for t in ("t_fwd", "t_bwd_train", "t_bwd_flbp"):
        base[t] = [st.median(p[t][u] for p in profs) for u in range(len(base[t]))]
    _CANON_PROF[(key, m)] = base
    return base


def resummarize(d, final):
    """Recompute the cost-model summary with the canonical profile and partition."""
    key, cfg = canon_key(d)
    dd, m = final.get("d", 4), final.get("m", 4)
    canon = os.path.join(ROOT, "results", "canonical", f"{key if key.startswith('bert') else key}_d{dd}.json")
    prof = canonical_profile(key, m)
    stages = json.load(open(canon)) if os.path.exists(canon) else final["stages"]
    pipeline = VARIANTS[final["variant"]][0]
    opt_slots = 1 if key.startswith("resnet") else 2
    final.update(summarize_trace(os.path.join(d, "trace.jsonl"), prof, stages, dd, m, opt_slots, pipeline))
    return final


def load(pattern, canonical=True):
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "results", pattern))):
        f = os.path.join(d, "final.json")
        if os.path.exists(f):
            fin = json.load(open(f))
            if canonical and "variant" in fin and os.path.exists(os.path.join(d, "trace.jsonl")):
                try:
                    fin = resummarize(d, fin)
                except Exception as e:      # keep the run's own summary if anything is missing
                    print(f"<!-- canonical summary failed for {os.path.basename(d)}: {e} -->")
            out[os.path.basename(d)] = {"final": fin, "dir": d}
    return out


def log_rows(d):
    p = os.path.join(d, "log.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def mean_rng(xs):
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{st.mean(xs):.2f} ± {(max(xs) - min(xs)) / 2:.2f}"


def smooth(ys, k=5):
    """Trailing moving average (noisy per-epoch test accuracy at high learning rates)."""
    out = []
    for i in range(len(ys)):
        w = ys[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(path, os.path.dirname(ROOT))


# ============================================================================ Track A
DDG_PAPER = {("resnet56", "cifar10"): (93.12, 93.11), ("resnet56", "cifar100"): (69.79, 70.17),
             ("resnet110", "cifar10"): (93.53, 93.41), ("resnet110", "cifar100"): (71.90, 71.39)}


def track_a():
    R = load("trackA/*")
    print("\n#### Track A results (final-epoch top-1 %; best-epoch in parentheses)\n")
    print("| Model | Data | Recipe | BP (K=1) | DDG K=2 | Δ K=2 | DDG K=4 | Δ K=4 | DDG paper BP / DDG |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in ("resnet56", "resnet110"):
        for data in ("cifar10", "cifar100"):
            for recipe in ("modern", "paper"):
                cell = {}
                for K in (1, 2, 4):
                    r = R.get(f"A_{model}_{data}_K{K}_{recipe}_s0")
                    if r:
                        cell[K] = (100 * r["final"]["final_acc"], 100 * r["final"]["best_acc"])
                if not cell:
                    continue
                f = lambda K: f"{cell[K][0]:.2f} ({cell[K][1]:.2f})" if K in cell else "–"
                dlt = lambda K: f"{cell[K][0] - cell[1][0]:+.2f}" if K in cell and 1 in cell else "–"
                pb, pd = DDG_PAPER[(model, data)]
                print(f"| {model} | {data} | {recipe} | {f(1)} | {f(2)} | {dlt(2)} | {f(4)} | {dlt(4)} | {pb} / {pd} |")
    # curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, (model, data) in zip(axes, (("resnet56", "cifar10"), ("resnet56", "cifar100"))):
        for K in (1, 2, 4):
            r = R.get(f"A_{model}_{data}_K{K}_modern_s0")
            d = os.path.join(ROOT, "results", "trackA", f"A_{model}_{data}_K{K}_modern_s0")
            rows = log_rows(d)
            if not rows:
                continue
            ax.plot([x["epoch"] for x in rows], smooth([100 * x["test_acc"] for x in rows]),
                    color=SLOTS[ENTITY[f"K{K}"]], label="BP" if K == 1 else f"DDG K={K}")
        ax.set_title(f"{model.replace('resnet', 'ResNet-')} / {data.replace('cifar', 'CIFAR-')}")
        ax.set_xlabel("epoch"); ax.set_ylabel("test top-1 (%), 5-epoch moving average")
        lo = 60 if data == "cifar10" else 30
        ax.set_ylim(lo, None)
        ax.legend(loc="lower right")
    print("\n![Track A curves](" + save(fig, "trackA_curves.png") + ")")


# ============================================================================ Track B / C shared
ROWS = [("vanilla", "0 Vanilla"), ("frz", "1 FRZ only"), ("2bw", "2 2BW"), ("1bw", "3 1BW"), ("1bw_warm10", "3b 1BW after 10% 2BW warm start"),
        ("sel", "4 2BW-Sel"), ("2bw_frz", "5 2BW + FRZ"), ("full", "6 Full (ours)"),
        ("full_random", "7 Full, random"), ("full_freezeout", "8 Full, FreezeOut sched."),
        ("full_adamreset", "9 Full, Adam reset")]


def variant_table(R, prefix, metric, metric_name, higher_better=True, scale=100.0):
    groups = {}
    for name, r in R.items():
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        v, seed = rest.rsplit("_s", 1)
        groups.setdefault(v, []).append(r["final"])
    if not groups:
        return groups
    base = groups.get("vanilla")
    b2 = groups.get("2bw")
    bm = st.mean([g["final"][metric] * scale for g in base]) if base else None
    b2mem = st.mean([g["mem_peak_maxstage_MB"] for g in b2]) if b2 else None
    b2s1 = st.mean([g["mem_avg_stage1_MB"] for g in b2]) if b2 else None
    b2t = st.mean([g["time_static_avg"] for g in b2]) if b2 else None
    b2r = st.mean([g["time_rebal_avg"] for g in b2]) if b2 else None
    print(f"| Row | {metric_name} | Δ vs vanilla | seeds | peak mem (max stage) vs 2BW | avg stage-1 mem vs 2BW | "
          f"time/batch static vs 2BW | rebalanced vs 2BW | frozen | no-shadow |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v, label in ROWS:
        g = groups.get(v)
        if not g:
            continue
        vals = [x["final"][metric] * scale for x in g]
        dv = f"{st.mean(vals) - bm:+.2f}" if bm is not None else "–"
        rel = lambda key, ref: f"{100 * (st.mean([x[key] for x in g]) / ref - 1):+.1f}%" if ref else "–"
        print(f"| {label} | {mean_rng(vals)} | {dv} | {len(g)} | {rel('mem_peak_maxstage_MB', b2mem)} | "
              f"{rel('mem_avg_stage1_MB', b2s1)} | {rel('time_static_avg', b2t)} | {rel('time_rebal_avg', b2r)} | "
              f"{100 * st.mean([x['frozen_frac_avg'] for x in g]):.0f}% | {100 * st.mean([x['noshadow_frac_avg'] for x in g]):.0f}% |")
    return groups


def attribution(groups, metric, scale, label):
    need = ("2bw", "sel", "2bw_frz", "full")
    if not all(k in groups for k in need):
        return
    m = {k: st.mean([x["final"][metric] * scale for x in groups[k]]) for k in need}
    mem = {k: st.mean([x["mem_avg_stage1_MB"] for x in groups[k]]) for k in need}
    t = {k: st.mean([x["time_rebal_avg"] for x in groups[k]]) for k in need}
    print(f"\n**Attribution ({label}), each relative to 2BW (row 2):**\n")
    print("| Quantity | total Δ6 | shadow Δ4 | freezing Δ5 | interaction |")
    print("|---|---|---|---|---|")
    for name, d, fmt in ((label, m, "{:+.2f}"), ("avg stage-1 memory (MB)", mem, "{:+.0f}"),
                         ("rebalanced time/batch (ms)", t, "{:+.2f}")):
        d6, d4, d5 = d["full"] - d["2bw"], d["sel"] - d["2bw"], d["2bw_frz"] - d["2bw"]
        print(f"| {name} | {fmt.format(d6)} | {fmt.format(d4)} | {fmt.format(d5)} | {fmt.format(d6 - d4 - d5)} |")


def curves(R, prefix, xkey, ykey, xs, title, fname, ylabel, ylim=None, yscale=1.0):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for v in ("vanilla", "2bw", "1bw", "full"):
        d = os.path.join(ROOT, "results", prefix.split("/")[0], f"{prefix.split('/')[1]}{v}_s0")
        rows = [x for x in log_rows(d) if ykey in x]
        if rows:
            ax.plot([x[xkey] / xs for x in rows], [x[ykey] * yscale for x in rows],
                    color=SLOTS[ENTITY[v]], label={"full": "Full (ours)", "2bw": "2BW (shadow)", "1bw": "1BW (no shadow)"}.get(v, v))
    ax.set_title(title); ax.set_xlabel("epoch" if xs > 1 else "step"); ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    return save(fig, fname)


def track_b():
    R = load("trackB/*")
    for data in ("c10", "c100"):
        print(f"\n#### Track B, ResNet-50, CIFAR-{data[1:]} (final test top-1 %)\n")
        g = variant_table(R, f"B_{data}_", "test_acc", "test acc", scale=100)
        if g:
            attribution(g, "test_acc", 100, "test acc (pts)")
    p = R.get("B_c10_plain_m1_s0")
    if p:
        print(f"\nPlain baseline (batch 128, no microbatching): {100 * p['final']['final']['test_acc']:.2f}%")
    d2 = R.get("B_c10_1bw_d2_s0")
    if d2:
        print(f"1BW at d=2 (m/d=2): {100 * d2['final']['final']['test_acc']:.2f}%")
    print("\n![Track B curves](" + curves(R, "trackB/B_c10_", "step", "test_acc", 390,
                                          "ResNet-50, CIFAR-10", "trackB_curves.png", "test top-1 (%)", (60, 100), 100) + ")")


def track_c():
    R = load("trackC/*")
    sizes = ("mini", "small", "medium", "base")
    for size in sizes:
        print(f"\n#### Track C, BERT-{size} (final MLM validation loss; lower is better)\n")
        g = variant_table(R, f"C_{size}_", "valid_loss", "valid loss", higher_better=False, scale=1.0)
        if size == "medium" and g:
            attribution(g, "valid_loss", 1.0, "valid MLM loss")
    # depth / microbatch sweep
    print("\n#### 1BW depth / microbatch sweep (BERT-Medium)\n")
    print("| d | m | m/d | valid loss | Δ vs vanilla | mismatched microbatch-units |")
    print("|---|---|---|---|---|---|")
    van = R.get("C_medium_vanilla_s0")
    vl = van["final"]["final"]["valid_loss"] if van else None
    for name, r in sorted(R.items()):
        f = r["final"]
        if name.startswith("C_medium_1bw") and f["variant"] == "1bw" and "warm10" not in name:
            mm = f["mismatch"]
            frac = mm["mismatched_microbatch_units"] / max(mm["microbatch_units"], 1)
            print(f"| {f['d']} | {f['m']} | {f['m'] // f['d']} | {f['final']['valid_loss']:.4f} | "
                  f"{(f['final']['valid_loss'] - vl) if vl else float('nan'):+.4f} | {100 * frac:.1f}% |")
    # scaling plot: gap to vanilla and memory saving vs params
    params = {"mini": 11.1, "small": 28.5, "medium": 41.1, "base": 108.9}
    lrs = {"mini": "1e-3", "small": "5e-4", "medium": "5e-4", "base": "1e-4"}
    names = {"2bw": "2BW", "1bw": "1BW (no shadow)", "full": "Full (ours)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for v in ("2bw", "1bw", "full"):
        pts = []
        for s_ in sizes:
            a, b, c = R.get(f"C_{s_}_{v}_s0"), R.get(f"C_{s_}_vanilla_s0"), R.get(f"C_{s_}_2bw_s0")
            if a and b and c:
                pts.append((s_, params[s_], a["final"]["final"]["valid_loss"] - b["final"]["final"]["valid_loss"],
                            100 * (a["final"]["mem_avg_stage1_MB"] / c["final"]["mem_avg_stage1_MB"] - 1)))
        if not pts:
            continue
        col = SLOTS[ENTITY[v]]
        main = [p_ for p_ in pts if p_[0] != "base"]
        for ax, k in ((axes[0], 2), (axes[1], 3)):
            ax.plot([p_[1] for p_ in main], [p_[k] for p_ in main], marker="o", ms=5, color=col, label=names[v])
            for p_ in pts:
                if p_[0] == "base":      # different (conservative) learning rate: hollow, unconnected
                    ax.plot([p_[1]], [p_[k]], marker="o", ms=6, mfc="none", mec=col, mew=1.5, ls="none")
    axes[0].axhline(0, color=TEXT2, lw=0.8)
    axes[0].set_title("Validation-loss gap to vanilla"); axes[0].set_ylabel("Δ MLM validation loss")
    axes[1].set_title("Average stage-1 memory vs 2BW"); axes[1].set_ylabel("change (%)")
    ticks = [params[s_] for s_ in sizes]
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(ticks)
        ax.set_xticklabels([f"{s_.capitalize()}\n{params[s_]:.0f}M\nlr {lrs[s_]}" for s_ in sizes], fontsize=7)
        ax.minorticks_off(); ax.legend(fontsize=7)
    fig.text(0.5, -0.08, "Hollow markers: Base, trained at a more conservative learning rate (not connected); no 1BW run at Base.",
             ha="center", fontsize=7, color=TEXT2)
    print("\n![Track C scaling](" + save(fig, "trackC_scaling.png") + ")")
    print("\n![Track C curves](" + curves(R, "trackC/C_medium_", "step", "valid_loss", 1,
                                          "BERT-Medium MLM", "trackC_curves.png", "valid MLM loss") + ")")




def shadow_fig():
    """Train-loss curves: vanilla vs 2BW vs 1BW (+ warm-start 1BW), one panel per BERT size."""
    sizes = [s for s in ("mini", "small", "medium") if os.path.exists(os.path.join(ROOT, "results", "trackC", f"C_{s}_1bw_s0", "final.json"))]
    if not sizes:
        return None
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 3.1), squeeze=False)
    labels = {"vanilla": "vanilla", "2bw": "2BW (shadow)", "1bw": "1BW (no shadow)", "1bw_warm10": "1BW after 2BW warm start"}
    for ax, size in zip(axes[0], sizes):
        for v in ("vanilla", "2bw", "1bw", "1bw_warm10"):
            rows = log_rows(os.path.join(ROOT, "results", "trackC", f"C_{size}_{v}_s0"))
            if not rows:
                continue
            ax.plot([r["step"] for r in rows], smooth([r["train_loss"] for r in rows], 3), color=SLOTS[ENTITY[v]],
                    lw=1.6 if v != "1bw_warm10" else 1.2, ls="-" if v != "1bw_warm10" else "--", label=labels[v])
        ax.set_xscale("log"); ax.set_ylim(2.7, 8.0)
        ax.set_title(f"BERT-{size.capitalize()}"); ax.set_xlabel("step (log)"); ax.set_ylabel("MLM train loss")
    axes[0][0].legend(fontsize=7, loc="lower left")
    return save(fig, "trackC_shadow.png")

# ============================================================================ calibration
def calib(trace, title, fname):
    rows = [json.loads(l) for l in open(trace)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    n = len(rows[0]["gv"])
    pick = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 2]))
    for i, u in enumerate(pick):
        xs = [r["frac"] for r in rows if r["gv"][u] is not None]
        axes[0].plot(xs, [r["gv"][u] for r in rows if r["gv"][u] is not None], color=SLOTS[i], label=f"unit {u}")
        xa = [r["frac"] for r in rows if r["am_rel"][u] is not None]
        axes[1].plot(xa, [r["am_rel"][u] for r in rows if r["am_rel"][u] is not None], color=SLOTS[i], label=f"unit {u}")
    axes[0].set_title("Gradient variation GV"); axes[0].set_ylabel("GV")
    axes[1].set_title("Relative gradient magnitude AM / AM(ref)"); axes[1].set_ylabel("AM_rel")
    for ax in axes:
        ax.set_xlabel("fraction of training"); ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return save(fig, fname)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("trackA", "all"):
        track_a()
    if which in ("trackB", "all"):
        track_b()
    if which in ("trackC", "all"):
        track_c()
        f = shadow_fig()
        if f:
            print("\n![shadow vs no shadow](" + f + ")")
