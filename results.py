"""
FA-CRS Day 13-15: Results Figures + Final Paper Table
------------------------------------------------------
Reads the JSON outputs from all previous steps and produces:

1. results_table.txt       — formatted comparison table for paper
2. fut_curve_ndcg.png      — FUT curve: NDCG vs p (accuracy tradeoff)
3. fut_curve_spd.png       — FUT curve: SPD vs p (fairness improvement)
4. fut_curve_combined.png  — both axes on one figure (main paper figure)
5. bias_comparison.png     — bar chart: SPD across three systems

Requirements:
    pip install matplotlib
"""

import json
import os
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD RESULTS ─────────────────────────────────────────────────────────────

with open("outputs/baseline/baseline_results.json") as f:
    baseline = json.load(f)

with open("outputs/kg/kg_results.json") as f:
    kg = json.load(f)

with open("outputs/fair/fair_results.json") as f:
    fair = json.load(f)

fut = fair["fut_curve"]
p_values = [r["p"] for r in fut]

# ─── 1. FORMATTED RESULTS TABLE ───────────────────────────────────────────────

def make_table():
    primary = next(r for r in fut if r["p"] == 0.3)

    lines = []
    lines.append("=" * 72)
    lines.append("TABLE 1: System Comparison (FA-CRS on MovieLens-25M)")
    lines.append("=" * 72)
    lines.append(f"{'Metric':<22} {'Baseline':>12} {'KG':>12} {'KG+FA*IR':>12} {'Δ (Fair)':>12}")
    lines.append("-" * 72)

    rows = [
        ("NDCG@10",       "ndcg_at_10"),
        ("Precision@10",  "precision_at_10"),
        ("Recall@10",     "recall_at_10"),
        ("Gender SPD",    "gender_spd"),
        ("Gender EOD",    "gender_eod"),
        ("Region SPD",    "region_spd"),
        ("Region EOD",    "region_eod"),
    ]

    for label, key in rows:
        b = baseline.get(key, 0)
        k = kg.get(key, 0)
        f = primary.get(key, 0)
        d = f - k
        lines.append(f"{label:<22} {b:>12.4f} {k:>12.4f} {f:>12.4f} {d:>+12.4f}")

    lines.append("=" * 72)
    lines.append("FA*IR p=0.3, alpha=0.1. Δ = KG+FA*IR minus KG.")
    lines.append("")
    lines.append("Key findings:")
    lines.append(f"  Gender SPD improved by {primary['gender_spd'] - kg['gender_spd']:+.4f} ({abs((primary['gender_spd'] - kg['gender_spd']) / kg['gender_spd']) * 100:.1f}% reduction in bias)")
    lines.append(f"  Region SPD improved by {primary['region_spd'] - kg['region_spd']:+.4f} ({abs((primary['region_spd'] - kg['region_spd']) / kg['region_spd']) * 100:.1f}% reduction in bias)")
    lines.append(f"  NDCG@10 cost:          {primary['ndcg_at_10'] - kg['ndcg_at_10']:+.4f}")

    table_str = "\n".join(lines)
    print(table_str)

    with open(os.path.join(OUTPUT_DIR, "results_table.txt"), "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\nSaved: results_table.txt")


# ─── 2. FUT CURVE — COMBINED (main paper figure) ──────────────────────────────

def plot_fut_combined():
    ndcg_vals   = [r["ndcg_at_10"]  for r in fut]
    gender_spd  = [abs(r["gender_spd"]) for r in fut]  # abs so higher = more bias
    region_spd  = [abs(r["region_spd"]) for r in fut]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_ndcg   = "#2563EB"
    color_gender = "#DC2626"
    color_region = "#16A34A"

    ax1.plot(p_values, ndcg_vals, "o-", color=color_ndcg,
             linewidth=2, markersize=6, label="NDCG@10 (accuracy)")
    ax1.set_xlabel("Fairness Constraint Strength (p)", fontsize=12)
    ax1.set_ylabel("NDCG@10", fontsize=12, color=color_ndcg)
    ax1.tick_params(axis="y", labelcolor=color_ndcg)
    ax1.set_ylim(0, max(ndcg_vals) * 1.4)

    ax2 = ax1.twinx()
    ax2.plot(p_values, gender_spd, "s--", color=color_gender,
             linewidth=2, markersize=6, label="|Gender SPD|")
    ax2.plot(p_values, region_spd, "^--", color=color_region,
             linewidth=2, markersize=6, label="|Region SPD|")
    ax2.set_ylabel("|SPD| (lower = fairer)", fontsize=12)
    ax2.set_ylim(0, 1.1)

    # Reference lines for baseline bias
    ax2.axhline(abs(baseline["gender_spd"]), color=color_gender,
                linestyle=":", linewidth=1, alpha=0.5)
    ax2.axhline(abs(baseline["region_spd"]), color=color_region,
                linestyle=":", linewidth=1, alpha=0.5)

    # Annotate baseline reference lines
    ax2.text(p_values[-1] + 0.01, abs(baseline["gender_spd"]),
             "Baseline\nGender SPD", fontsize=7, color=color_gender, va="center")
    ax2.text(p_values[-1] + 0.01, abs(baseline["region_spd"]) - 0.04,
             "Baseline\nRegion SPD", fontsize=7, color=color_region, va="center")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, framealpha=0.9)

    plt.title("Fairness-Utility Tradeoff (FUT) Curve\nKG + FA*IR on MovieLens-25M",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fut_curve_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fut_curve_combined.png")


# ─── 3. BIAS COMPARISON BAR CHART ─────────────────────────────────────────────

def plot_bias_comparison():
    primary = next(r for r in fut if r["p"] == 0.3)

    systems = ["Baseline", "KG", "KG+FA*IR\n(p=0.3)"]
    gender_spd_vals = [
        abs(baseline["gender_spd"]),
        abs(kg["gender_spd"]),
        abs(primary["gender_spd"])
    ]
    region_spd_vals = [
        abs(baseline["region_spd"]),
        abs(kg["region_spd"]),
        abs(primary["region_spd"])
    ]

    x     = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width/2, gender_spd_vals, width,
                   label="|Gender SPD|", color="#DC2626", alpha=0.85)
    bars2 = ax.bar(x + width/2, region_spd_vals, width,
                   label="|Region SPD|", color="#16A34A", alpha=0.85)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("|SPD| (lower = fairer)", fontsize=12)
    ax.set_title("Bias Comparison Across Systems\n(Statistical Parity Difference)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: bias_comparison.png")


# ─── 4. ACCURACY ACROSS SYSTEMS ───────────────────────────────────────────────

def plot_accuracy_comparison():
    primary = next(r for r in fut if r["p"] == 0.3)

    systems = ["Baseline", "KG", "KG+FA*IR\n(p=0.3)"]
    ndcg_vals = [
        baseline["ndcg_at_10"],
        kg["ndcg_at_10"],
        primary["ndcg_at_10"]
    ]
    prec_vals = [
        baseline["precision_at_10"],
        kg["precision_at_10"],
        primary["precision_at_10"]
    ]
    rec_vals = [
        baseline["recall_at_10"],
        kg["recall_at_10"],
        primary["recall_at_10"]
    ]

    x     = np.arange(len(systems))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width, ndcg_vals,  width, label="NDCG@10",      color="#2563EB", alpha=0.85)
    ax.bar(x,         prec_vals,  width, label="Precision@10",  color="#7C3AED", alpha=0.85)
    ax.bar(x + width, rec_vals,   width, label="Recall@10",     color="#0891B2", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Accuracy Metrics Across Systems", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: accuracy_comparison.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...\n")
    make_table()
    print()
    plot_fut_combined()
    plot_bias_comparison()
    plot_accuracy_comparison()

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("\nFiles for your paper:")
    print("  - results_table.txt       -> paste into Table 1")
    print("  - fut_curve_combined.png  -> main contribution figure")
    print("  - bias_comparison.png     -> fairness results figure")
    print("  - accuracy_comparison.png -> accuracy results figure")


if __name__ == "__main__":
    main()
