#!/usr/bin/env python
"""
Score calibration via Z-normalization with condition-aware statistics.

Reads raw score files and per-condition statistics from robustness_summary.csv,
applies Z-normalization per condition, and evaluates calibrated scores.

The key insight: individual per-condition EER is unchanged by Z-norm (monotonic
transform), but a single global threshold on Z-normalized scores outperforms a
single threshold on raw scores when conditions are mixed — i.e., the realistic
deployment scenario where audio arrives with unknown degradation.

Usage:
    python calibrate_scores.py \
        --stats 04_completed_evals/noise_aug_v5/robustness_summary.csv \
        --score-dir 04_completed_evals/noise_aug_v5 \
        --protocol ~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv \
        --output-dir 04_completed_evals/noise_aug_v5/calibrated
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# Import evaluator functions from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import compute_eer


# =============================
# Lightweight I/O (avoid pandas for large score files)
# =============================

def load_scores_np(score_path):
    """Load score file into (trial_ids, scores) using numpy. Memory-efficient."""
    trial_ids = []
    scores = []
    with open(score_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                trial_ids.append(parts[0])
                scores.append(float(parts[1]))
            elif len(parts) == 4:
                # Raw format: Output trial_id label score
                trial_ids.append(parts[1].rstrip(","))
                scores.append(float(parts[3]))
    return trial_ids, np.array(scores, dtype=np.float64)


def write_scores(output_path, trial_ids, scores):
    """Write score file efficiently."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for tid, s in zip(trial_ids, scores):
            f.write(f"{tid} {s:.6f}\n")


# =============================
# Statistics & Z-normalization
# =============================

def compute_pooled_stats(bon_mean, bon_std, spf_mean, spf_std,
                         n_bon=138688, n_spf=542086):
    """
    Compute weighted pooled mean and std from bonafide/spoof group statistics.

    Uses the law of total variance:
        pooled_var = w_b*(s_b^2 + m_b^2) + w_s*(s_s^2 + m_s^2) - pooled_mean^2
    """
    n_total = n_bon + n_spf
    w_b = n_bon / n_total
    w_s = n_spf / n_total

    pooled_mean = w_b * bon_mean + w_s * spf_mean
    pooled_var = (w_b * (bon_std**2 + bon_mean**2)
                  + w_s * (spf_std**2 + spf_mean**2)
                  - pooled_mean**2)
    pooled_std = math.sqrt(max(pooled_var, 1e-12))

    return pooled_mean, pooled_std


def load_condition_stats(stats_path):
    """Load per-condition statistics from robustness_summary.csv."""
    df = pd.read_csv(stats_path)
    df = df.drop_duplicates(subset="condition")

    stats = {}
    for _, row in df.iterrows():
        cond = row["condition"]
        pooled_mean, pooled_std = compute_pooled_stats(
            row["bonafide_mean"], row["bonafide_std"],
            row["spoof_mean"], row["spoof_std"]
        )
        stats[cond] = {
            "pooled_mean": pooled_mean,
            "pooled_std": pooled_std,
            "eer": row["eer"],
        }
    return stats


# =============================
# Protocol → label lookup
# =============================

def build_label_map(protocol_path):
    """Build trial_id → label dict from protocol. Memory-efficient."""
    labels = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 9:
                trial_id = parts[1]
                label = parts[8]  # 'bonafide' or 'spoof'
                labels[trial_id] = label
    return labels


def split_scores_by_label(trial_ids, scores, label_map):
    """Split scores into bonafide/spoof arrays using label lookup."""
    bon = []
    spf = []
    for tid, s in zip(trial_ids, scores):
        lab = label_map.get(tid)
        if lab == "bonafide":
            bon.append(s)
        elif lab == "spoof":
            spf.append(s)
    return np.array(bon, dtype=np.float64), np.array(spf, dtype=np.float64)


# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser(
        description="Z-normalize scores per condition and evaluate calibration"
    )
    parser.add_argument("--stats", required=True,
                        help="Path to robustness_summary.csv")
    parser.add_argument("--score-dir", required=True,
                        help="Directory containing scores_<condition>.txt files")
    parser.add_argument("--protocol", required=True,
                        help="Path to ASVspoof5 eval protocol TSV")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for calibrated score files")
    args = parser.parse_args()

    # Build label lookup (dict is much lighter than full protocol DataFrame)
    label_map = build_label_map(args.protocol)

    # Load per-condition statistics
    cond_stats = load_condition_stats(args.stats)

    print("=" * 72)
    print("Score Calibration via Z-Normalization")
    print("=" * 72)

    # Print pooled stats
    print(f"\n{'Condition':<30} {'Pooled Mean':>12} {'Pooled Std':>12}")
    print("-" * 56)
    for cond, s in sorted(cond_stats.items()):
        print(f"{cond:<30} {s['pooled_mean']:>12.4f} {s['pooled_std']:>12.4f}")

    # --- Per-condition calibration ---
    print(f"\n{'Condition':<30} {'Raw EER%':>10} {'Cal EER%':>10} {'Delta':>8} {'Cal Thresh':>10}")
    print("-" * 72)

    condition_thresholds = {}
    # For mixed eval: collect bonafide/spoof score arrays per condition
    raw_bon_list, raw_spf_list = [], []
    cal_bon_list, cal_spf_list = [], []

    for cond in sorted(cond_stats):
        s = cond_stats[cond]
        score_path = os.path.join(args.score_dir, f"scores_{cond}.txt")
        if not os.path.exists(score_path):
            print(f"  SKIP {cond}: not found")
            continue

        # Load scores once
        trial_ids, raw_scores = load_scores_np(score_path)

        # Split by label
        raw_bon, raw_spf = split_scores_by_label(trial_ids, raw_scores, label_map)
        raw_eer, raw_thresh = compute_eer(raw_bon, raw_spf)

        # Z-normalize
        cal_scores = (raw_scores - s["pooled_mean"]) / s["pooled_std"]
        cal_bon, cal_spf = split_scores_by_label(trial_ids, cal_scores, label_map)
        cal_eer, cal_thresh = compute_eer(cal_bon, cal_spf)

        # Write calibrated file
        cal_path = os.path.join(args.output_dir, f"scores_{cond}_calibrated.txt")
        write_scores(cal_path, trial_ids, cal_scores)

        delta = (cal_eer - raw_eer) * 100
        condition_thresholds[cond] = {
            "raw_eer": raw_eer, "raw_threshold": raw_thresh,
            "cal_eer": cal_eer, "cal_threshold": cal_thresh,
        }
        print(f"{cond:<30} {raw_eer*100:>10.4f} {cal_eer*100:>10.4f} {delta:>+8.4f} {cal_thresh:>10.4f}")

        # Collect split scores for mixed eval (much smaller than full arrays)
        raw_bon_list.append(raw_bon)
        raw_spf_list.append(raw_spf)
        cal_bon_list.append(cal_bon)
        cal_spf_list.append(cal_spf)

        # Free large arrays
        del trial_ids, raw_scores, cal_scores

    # --- Sanity check: clean calibrated scores ---
    print("\n--- Sanity Check: Clean Calibrated Scores ---")
    clean_cal_path = os.path.join(args.output_dir, "scores_clean_calibrated.txt")
    if os.path.exists(clean_cal_path):
        _, clean_scores = load_scores_np(clean_cal_path)
        print(f"  Overall:  mean = {clean_scores.mean():.4f}, std = {clean_scores.std():.4f}  "
              f"(expect ~0, ~1)")
        del clean_scores

    # --- Mixed-condition evaluation ---
    if len(raw_bon_list) > 1:
        print("\n--- Mixed-Condition Evaluation (All Conditions Combined) ---\n")

        raw_bon_all = np.concatenate(raw_bon_list)
        raw_spf_all = np.concatenate(raw_spf_list)
        cal_bon_all = np.concatenate(cal_bon_list)
        cal_spf_all = np.concatenate(cal_spf_list)

        # Free the lists
        del raw_bon_list, raw_spf_list, cal_bon_list, cal_spf_list

        raw_mix_eer, raw_mix_thresh = compute_eer(raw_bon_all, raw_spf_all)
        cal_mix_eer, cal_mix_thresh = compute_eer(cal_bon_all, cal_spf_all)

        n_raw = len(raw_bon_all) + len(raw_spf_all)
        n_cal = len(cal_bon_all) + len(cal_spf_all)

        print(f"  {'Metric':<25} {'Raw':>12} {'Calibrated':>12}")
        print(f"  {'-'*51}")
        print(f"  {'Mixed EER%':<25} {raw_mix_eer*100:>12.4f} {cal_mix_eer*100:>12.4f}")
        print(f"  {'Mixed Threshold':<25} {raw_mix_thresh:>12.4f} {cal_mix_thresh:>12.4f}")
        print(f"  {'Total trials':<25} {n_raw:>12d} {n_cal:>12d}")
        improvement = (raw_mix_eer - cal_mix_eer) * 100
        print(f"\n  Mixed EER improvement from calibration: {improvement:+.4f} pp")

    # --- Condition-specific optimal thresholds ---
    print(f"\n{'Condition':<30} {'EER%':>8} {'Raw Thresh':>12} {'Cal Thresh':>12}")
    print("-" * 66)
    for cond in sorted(condition_thresholds):
        t = condition_thresholds[cond]
        print(f"{cond:<30} {t['raw_eer']*100:>8.4f} {t['raw_threshold']:>12.6f} "
              f"{t['cal_threshold']:>12.4f}")

    # Write threshold lookup table
    thresh_path = os.path.join(args.output_dir, "threshold_table.csv")
    rows = []
    for cond in sorted(condition_thresholds):
        t = condition_thresholds[cond]
        rows.append({
            "condition": cond,
            "eer_pct": round(t["raw_eer"] * 100, 4),
            "raw_threshold": round(t["raw_threshold"], 6),
            "calibrated_threshold": round(t["cal_threshold"], 6),
        })
    pd.DataFrame(rows).to_csv(thresh_path, index=False)
    print(f"\nThreshold table written to: {thresh_path}")

    print("\n" + "=" * 72)
    print("Calibration complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
