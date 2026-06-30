"""Rebuild the cross-experiment comparison plots in results/ and results/new/.

Usage:
    python scripts/compare_plots.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from aggregate_results import PHASE1_RUNS, SYNTH_RUNS, ROOT


def line_comparison(runs, column, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, path in runs.items():
        df = pd.read_csv(ROOT / path).sort_values("gen")
        ax.plot(df["gen"], df[column], label=label, linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def final_best_bar(runs, out_path):
    labels, vals = [], []
    for label, path in runs.items():
        df = pd.read_csv(ROOT / path).sort_values("gen")
        labels.append(label)
        vals.append(float(df.iloc[-1]["best"]))
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, vals)
    ax.set_ylabel("Final best fitness")
    ax.set_title("Final Best Fitness by Configuration")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    line_comparison(PHASE1_RUNS, "best", "Best Fitness Across Phase 1+2 Configs", "Best fitness",
                     ROOT / "results/plot_best_fitness_all.png")
    line_comparison(PHASE1_RUNS, "mean", "Mean Fitness Across Phase 1+2 Configs", "Mean fitness",
                     ROOT / "results/plot_mean_fitness_all.png")
    line_comparison(PHASE1_RUNS, "diversity", "Population Diversity Across Phase 1+2 Configs", "Diversity",
                     ROOT / "results/plot_diversity_all.png")
    final_best_bar(PHASE1_RUNS, ROOT / "results/plot_final_best_bar.png")

    line_comparison(SYNTH_RUNS, "best", "Best Fitness Across Phase 3 Synth Configs", "Best fitness",
                     ROOT / "results/new/synth_best_fitness_curves.png")
    line_comparison(SYNTH_RUNS, "mean", "Mean Fitness Across Phase 3 Synth Configs", "Mean fitness",
                     ROOT / "results/new/synth_mean_fitness_curves.png")
    line_comparison(SYNTH_RUNS, "diversity", "Population Diversity Across Phase 3 Synth Configs", "Diversity",
                     ROOT / "results/new/synth_diversity_curves.png")

    print("wrote comparison plots to results/ and results/new/")
