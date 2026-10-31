"""Figures made for the final presentation (written to final_slides/figs/).

python analysis/final_figs.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report  # noqa: E402  (palette, rcParams, helpers)
from report import SLOTS, TEXT2, ROOT  # noqa: E402

OUT = os.path.join(os.path.dirname(ROOT), "final_slides", "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def final(run):
    return json.load(open(os.path.join(ROOT, "results", run, "final.json")))


# ---------------------------------------------------------------- 1. calibration signals
def fig_calib():
    specs = [("trackB/B_c10_vanilla_s0", "ResNet-50 / CIFAR-10", {0: "stem", 6: "block 6", 12: "block 12", 16: "block 16"}),
             ("trackC/C_mini_vanilla_s0", "BERT-Mini (Pre-LN)", {0: "embedding", 1: "layer 1", 3: "layer 3", 4: "layer 4"})]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.6), sharex=True)
    for row, (run, title, units) in enumerate(specs):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", run, "trace.jsonl"))]
        for i, (u, lab) in enumerate(units.items()):
            xs = [r["frac"] for r in rows if r["gv"][u] is not None]
            axes[row][0].plot(xs, [r["gv"][u] for r in rows if r["gv"][u] is not None], color=SLOTS[i], lw=1.4, label=lab)
            xa = [r["frac"] for r in rows if r["am_rel"][u] is not None]
            axes[row][1].plot(xa, [r["am_rel"][u] for r in rows if r["am_rel"][u] is not None], color=SLOTS[i], lw=1.4, label=lab)
        axes[row][0].axhline(0.1, color=TEXT2, lw=0.8, ls="--")
        axes[row][0].text(0.5, 0.11, "pure-noise level (1/√window)", ha="center", va="bottom", fontsize=7, color=TEXT2,
                          transform=axes[row][0].get_yaxis_transform())
        axes[row][0].set_ylabel(f"{title}\nGV")
        axes[row][1].set_ylabel("AM / AM(ref)")
        axes[row][1].legend(fontsize=7, ncol=2, loc="upper left")
        axes[row][0].set_ylim(0, 0.35)
    axes[0][0].set_title("Gradient variation (GV)")
    axes[0][1].set_title("Relative gradient magnitude (AM)")
    for ax in axes[1]:
        ax.set_xlabel("fraction of training")
    fig.tight_layout()
    save(fig, "calib_signals.png")


# ---------------------------------------------------------------- 2. freezing trade-off
def fig_freeze_tradeoff():
    van = (final("trackB/B_c10_vanilla_s0")["final"]["test_acc"] + final("trackB/B_c10_vanilla_s1")["final"]["test_acc"]) / 2
    pts = [("FRZ only (PolicyFRZ rule)", 4.0, final("trackB/B_c10_frz_s0")["final"]["test_acc"]),
           ("Full (ours)", 4.1, (final("trackB/B_c10_full_s0")["final"]["test_acc"] + final("trackB/B_c10_full_s1")["final"]["test_acc"]) / 2),
           ("Full, random units", 3.9, final("trackB/B_c10_full_random_s0")["final"]["test_acc"]),
           ("FreezeOut schedule", 18.7, final("trackB/B_c10_full_freezeout_s0")["final"]["test_acc"]),
           ("Aggressive thresholds", 8.4, final("trackB/B_c10_full_aggressive_s0")["final"]["test_acc"])]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.axhspan(-0.2, 0.2, color="#e4e3df", alpha=0.6, lw=0)
    ax.text(12.5, -0.19, "seed noise (±0.2)", fontsize=7, color=TEXT2, ha="center", va="bottom")
    offs = {"FRZ only (PolicyFRZ rule)": (8, 6), "Full (ours)": (9, -3), "Full, random units": None,
            "FreezeOut schedule": (-8, -14), "Aggressive thresholds": (8, -2)}
    for lab, x, acc in pts:
        y = 100 * (acc - van)
        ax.plot([x], [y], "o", ms=8, color=SLOTS[0], mec="#fcfcfb", mew=1.5)
        if offs[lab] is None:
            continue
        dx, dy = offs[lab]
        if lab == "Full (ours)":
            lab = "Full (ours) and the same budget\nwith random units (overlap)"
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    ha="left" if dx > 0 else "right", color=report.TEXT)
    ax.plot([27.9], [0.1], "o", ms=8, mfc="none", mec=SLOTS[1], mew=1.8)
    ax.annotate("PolicyFRZ, reported\n(vs. its own 92.0% baseline)", (27.9, 0.1), xytext=(-8, -26),
                textcoords="offset points", fontsize=7.5, ha="right", color=report.TEXT)
    ax.set_xlim(0, 30); ax.set_ylim(-1.1, 0.45)
    ax.set_xlabel("training compute saved by freezing (%)")
    ax.set_ylabel("Δ test accuracy vs. vanilla (pts)")
    ax.set_title("ResNet-50 / CIFAR-10: what freezing buys, and what it costs")
    save(fig, "freeze_tradeoff.png")


# ---------------------------------------------------------------- 3. DDG gaps
def fig_ddg():
    def acc(n):
        return 100 * final(f"trackA/{n}")["final_acc"]
    cfgs = [("ResNet-56\nCIFAR-10", "resnet56", "cifar10", ("s0",)),
            ("ResNet-56\nCIFAR-100", "resnet56", "cifar100", ("s0", "s1")),
            ("ResNet-110\nCIFAR-10", "resnet110", "cifar10", ("s0",)),
            ("ResNet-110\nCIFAR-100", "resnet110", "cifar100", ("s0",))]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    w = 0.34
    for j, K in enumerate((2, 4)):
        xs, ys = [], []
        for i, (lab, mdl, data, seeds) in enumerate(cfgs):
            d = [acc(f"A_{mdl}_{data}_K{K}_modern_{s}") - acc(f"A_{mdl}_{data}_K1_modern_{s}") for s in seeds]
            y = sum(d) / len(d)
            xs.append(i + (j - 0.5) * w * 1.1); ys.append(y)
            ax.text(xs[-1], y + (0.04 if y >= 0 else -0.04), f"{y:+.2f}", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=7.5, color=report.TEXT)
        ax.bar(xs, ys, width=w, color=SLOTS[j], label=f"DDG, K = {K}", edgecolor="#fcfcfb", linewidth=2)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([c[0] for c in cfgs], fontsize=8)
    ax.set_ylabel("Δ top-1 vs. backprop (pts)")
    ax.set_ylim(-1.9, 0.8)
    ax.set_title("DDG vs. a modern backprop baseline")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "ddg_gaps.png")


# ---------------------------------------------------------------- 4. mismatch vs gap
def fig_mismatch():
    van = (final("trackC/C_medium_vanilla_s0")["final"]["valid_loss"] + final("trackC/C_medium_vanilla_s1")["final"]["valid_loss"]) / 2
    tbw = (final("trackC/C_medium_2bw_s0")["final"]["valid_loss"] + final("trackC/C_medium_2bw_s1")["final"]["valid_loss"]) / 2
    pts = []
    for run, lab in (("C_medium_1bw_d2_m2_s0", "d=2, m=2"), ("C_medium_1bw_s0", "d=4, m=4"),
                     ("C_medium_1bw_d8_m8_s0", "d=8, m=8"), ("C_medium_1bw_d4_m16_s0", "d=4, m=16")):
        f = final(f"trackC/{run}")
        mm = f["mismatch"]
        pts.append((lab, 100 * mm["mismatched_microbatch_units"] / mm["microbatch_units"], f["final"]["valid_loss"] - van))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(tbw - van, color=SLOTS[1], lw=1.4, ls="--")
    ax.text(1, tbw - van + 0.006, "2BW (every microbatch 1 step stale)", color=report.TEXT, fontsize=7.5, va="bottom")
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.text(1, 0.006, "vanilla", color=TEXT2, fontsize=7.5, va="bottom")
    for lab, x, y in pts:
        ax.plot([x], [y], "o", ms=8, color=SLOTS[2], mec="#fcfcfb", mew=1.5)
        dx, dy, ha = {"d=4, m=4": (-8, 6, "right")}.get(lab, (7, -3, "left"))
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=report.TEXT, ha=ha)
    ax.set_xlim(0, 60); ax.set_ylim(-0.03, 0.28)
    ax.set_xlabel("microbatch-units with mismatched weights (%)")
    ax.set_ylabel("Δ MLM validation loss")
    ax.set_title("1BW on BERT-Medium: cost tracks the mismatch fraction")
    save(fig, "mismatch_vs_gap.png")


if __name__ == "__main__":
    fig_calib(); fig_freeze_tradeoff(); fig_ddg(); fig_mismatch()
