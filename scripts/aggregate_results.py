"""Rebuild the cross-experiment summary tables in results/ from run CSVs.

Usage:
    python scripts/aggregate_results.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PHASE1_RUNS = {
    "sel random":         "runs/p1_sel_random/p1_sel_random.csv",
    "sel roulette":       "runs/p1_sel_roulette/p1_sel_roulette.csv",
    "xo onepoint":        "runs/p1_xo_onepoint/p1_xo_onepoint.csv",
    "xo uniform":         "runs/p1_xo_uniform/p1_xo_uniform.csv",
    "pop 200":            "runs/p1_pop_200/p1_pop_200.csv",
    "food regrow slow":   "runs/p1_food_regrow_slow/p1_food_regrow_slow.csv",
    "ticks 1500":         "runs/p1_ticks_1500/p1_ticks_1500.csv",
    "baseline":           "runs/p1_baseline/p1_baseline.csv",
    "phase2 prey predator": "runs/phase2_prey_predator/phase2_prey_predator.csv",
}

SYNTH_RUNS = {
    "synth_best":      "runs/synth_best/synth_best.csv",
    "synth_no_anneal": "runs/synth_no_anneal/synth_no_anneal.csv",
    "synth_no_regrow": "runs/synth_no_regrow/synth_no_regrow.csv",
}


def summarize(csv_path):
    df = pd.read_csv(csv_path).sort_values("gen")
    last = df.iloc[-1]
    final_best = float(last["best"])
    peak_row = df.loc[df["best"].idxmax()]
    threshold = 0.9 * final_best
    reached = df[df["best"] >= threshold]
    gen_to_90pct = int(reached["gen"].iloc[0]) if not reached.empty else int(last["gen"])
    return {
        "last_gen": int(last["gen"]),
        "final_best": final_best,
        "final_mean": float(last["mean"]),
        "final_diversity": float(last["diversity"]),
        "best_peak": float(peak_row["best"]),
        "gen_peak": int(peak_row["gen"]),
        "gen_to_90pct": gen_to_90pct,
    }


def build_phase1_tables():
    rows = {label: summarize(ROOT / path) for label, path in PHASE1_RUNS.items()}

    finals = pd.DataFrame([
        {
            "config": label,
            "last_gen": r["last_gen"],
            "best_mean": r["final_best"],
            "best_ci": "",
            "mean_mean": r["final_mean"],
            "mean_ci": "",
        }
        for label, r in rows.items()
    ]).sort_values("best_mean", ascending=False)
    finals.to_csv(ROOT / "results/finals_table.csv", index=False)

    convergence = pd.DataFrame([
        {"config": label, "final_best": r["final_best"], "gen_to_90pct": r["gen_to_90pct"]}
        for label, r in rows.items()
    ]).sort_values("final_best", ascending=False)
    convergence.to_csv(ROOT / "results/convergence_table.csv", index=False)


def build_synth_table():
    rows = []
    for label, path in SYNTH_RUNS.items():
        r = summarize(ROOT / path)
        rows.append({
            "config": label,
            "gens": r["last_gen"],
            "best_final": r["final_best"],
            "mean_final": r["final_mean"],
            "diversity_final": r["final_diversity"],
            "best_peak": r["best_peak"],
            "gen_peak": r["gen_peak"],
            "gen_to_90pct": r["gen_to_90pct"],
        })
    pd.DataFrame(rows).to_csv(ROOT / "results/new/synth_summary.csv", index=False)


if __name__ == "__main__":
    build_phase1_tables()
    build_synth_table()
    print("wrote results/finals_table.csv, results/convergence_table.csv, results/new/synth_summary.csv")
