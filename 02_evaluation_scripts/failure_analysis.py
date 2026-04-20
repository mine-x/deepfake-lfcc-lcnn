#!/usr/bin/env python
"""
Failure mode and score distribution analysis for augmented evaluation results.

Computes per-condition:
  - EER and delta-EER relative to clean baseline
  - FAR/FRR at a fixed reference threshold (clean baseline's EER threshold)
  - Dominant failure mode classification
  - Score statistics (mean, std) per group (bonafide / spoof)

Usage:
    python 02_evaluation_scripts/failure_analysis.py \
        --protocol ~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv \
        --clean-scores 04_completed_evals/clean_weighted/inference_20260226_092649_scores.txt \
        --score-dir 04_completed_evals/clean_weighted/augmented \
        --output 04_completed_evals/clean_weighted/augmented/robustness_summary.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from evaluator import compute_eer, load_protocol, load_scores


BALANCE_TOLERANCE = 0.03  # FAR and FRR within this → "balanced"


def compute_far_frr(bonafide_scores, spoof_scores, threshold):
    """Compute FAR and FRR at a fixed threshold.

    Convention (higher score = more likely bonafide):
      FAR = fraction of spoof samples with score > threshold  (accepted as bonafide)
      FRR = fraction of bonafide samples with score <= threshold  (rejected as bonafide)
    """
    far = np.mean(spoof_scores > threshold)
    frr = np.mean(bonafide_scores <= threshold)
    return far, frr


def classify_failure(far, frr):
    """Label dominant failure mode."""
    diff = abs(far - frr)
    if diff <= BALANCE_TOLERANCE:
        return "balanced"
    elif far > frr:
        return "spoof collapse"
    else:
        return "bonafide collapse"


def analyse_condition(protocol_df, score_path, ref_threshold):
    """Run full analysis for one condition file.

    Returns a dict with all metrics.
    """
    score_df = load_scores(score_path)
    merged = protocol_df.merge(score_df, on="trial", how="inner")

    bon_scores = merged[merged["label"] == "bonafide"]["score"].values
    spf_scores = merged[merged["label"] == "spoof"]["score"].values

    if len(bon_scores) == 0 or len(spf_scores) == 0:
        raise ValueError(f"No bonafide or spoof trials in {score_path}")

    # Step 1 — EER + its own threshold
    eer, _ = compute_eer(bon_scores, spf_scores)

    # Step 2 — FAR / FRR at the fixed reference threshold
    far, frr = compute_far_frr(bon_scores, spf_scores, ref_threshold)
    failure_mode = classify_failure(far, frr)

    # Step 3 — Score statistics per group
    return {
        "eer": eer,
        "far": far,
        "frr": frr,
        "failure_mode": failure_mode,
        "bon_mean": np.mean(bon_scores),
        "bon_std": np.std(bon_scores),
        "spf_mean": np.mean(spf_scores),
        "spf_std": np.std(spf_scores),
    }


def condition_name_from_path(path):
    """Extract condition label from filename.

    scores_noise_ambient_10dB.txt → noise_ambient_10dB
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"scores_(.+)", basename)
    return m.group(1) if m else basename


def discover_score_files(score_dir):
    """Glob scores_*.txt and return sorted list of paths."""
    pattern = os.path.join(score_dir, "scores_*.txt")
    paths = sorted(glob.glob(pattern))
    return paths


def build_table(rows):
    """Pretty-print the results table to stdout."""
    header = (
        f"{'Condition':<28} | {'EER':>7} | {'ΔEER':>7} | "
        f"{'FAR†':>7} | {'FRR†':>7} | {'Failure Mode':<19} | "
        f"{'Bonafide Mean':>13} | {'Bonafide Std':>12} | {'Spoof Mean':>11} | {'Spoof Std':>10}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    clean_eer = rows[0]["eer"] if rows else None

    for row in rows:
        delta = row["eer"] - clean_eer if clean_eer is not None else 0.0
        delta_str = "  —" if row["condition"] == "clean" else f"{delta * 100:+.2f}"

        print(
            f"{row['condition']:<28} | "
            f"{row['eer'] * 100:6.2f}% | "
            f"{delta_str:>7} | "
            f"{row['far'] * 100:6.2f}% | "
            f"{row['frr'] * 100:6.2f}% | "
            f"{row['failure_mode']:<19} | "
            f"{row['bon_mean']:13.4f} | "
            f"{row['bon_std']:12.4f} | "
            f"{row['spf_mean']:11.4f} | "
            f"{row['spf_std']:10.4f}"
        )

    print(sep)
    print("† FAR/FRR computed at clean baseline EER threshold (fixed reference)")


def save_csv(rows, output_path):
    """Write results to CSV."""
    clean_eer = rows[0]["eer"] if rows else None
    records = []
    for row in rows:
        delta = row["eer"] - clean_eer if clean_eer is not None else 0.0
        records.append({
            "condition": row["condition"],
            "eer": round(row["eer"] * 100, 4),
            "delta_eer": round(delta * 100, 4) if row["condition"] != "clean" else None,
            "far_at_ref": round(row["far"] * 100, 4),
            "frr_at_ref": round(row["frr"] * 100, 4),
            "failure_mode": row["failure_mode"],
            "bonafide_mean": round(row["bon_mean"], 6),
            "bonafide_std": round(row["bon_std"], 6),
            "spoof_mean": round(row["spf_mean"], 6),
            "spoof_std": round(row["spf_std"], 6),
        })
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Robustness analysis: EER, FAR/FRR, score stats per condition"
    )
    parser.add_argument("--protocol", required=True, help="ASVspoof5 protocol TSV")
    parser.add_argument("--clean-scores", required=True, help="Clean baseline score file")
    parser.add_argument("--score-dir", required=True, help="Directory with scores_*.txt files")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    # Load protocol once
    protocol_df = load_protocol(args.protocol)

    # Analyse clean baseline first to get the reference threshold
    print("Analysing clean baseline...")
    clean_score_df = load_scores(args.clean_scores)
    merged_clean = protocol_df.merge(clean_score_df, on="trial", how="inner")
    bon_clean = merged_clean[merged_clean["label"] == "bonafide"]["score"].values
    spf_clean = merged_clean[merged_clean["label"] == "spoof"]["score"].values
    clean_eer, ref_threshold = compute_eer(bon_clean, spf_clean)

    print(f"  Clean EER: {clean_eer * 100:.2f}%  |  Reference threshold: {ref_threshold:.6f}")

    # Build clean row
    clean_far, clean_frr = compute_far_frr(bon_clean, spf_clean, ref_threshold)
    rows = [{
        "condition": "clean",
        "eer": clean_eer,
        "far": clean_far,
        "frr": clean_frr,
        "failure_mode": classify_failure(clean_far, clean_frr),
        "bon_mean": np.mean(bon_clean),
        "bon_std": np.std(bon_clean),
        "spf_mean": np.mean(spf_clean),
        "spf_std": np.std(spf_clean),
    }]

    # Discover and analyse augmented conditions
    score_files = discover_score_files(args.score_dir)
    if not score_files:
        print(f"No scores_*.txt files found in {args.score_dir}")
    else:
        print(f"Found {len(score_files)} augmented score files\n")

    for sf in score_files:
        cond = condition_name_from_path(sf)
        print(f"  Analysing {cond}...")
        metrics = analyse_condition(protocol_df, sf, ref_threshold)
        metrics["condition"] = cond
        rows.append(metrics)

    print()
    build_table(rows)

    if args.output:
        save_csv(rows, args.output)


if __name__ == "__main__":
    main()
