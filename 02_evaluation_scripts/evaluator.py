#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd


# =========================
# Metric functions
# =========================

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((
        np.ones(target_scores.size),
        np.zeros(nontarget_scores.size)
    ))

    # Sort scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute cumulative sums
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (
        nontarget_scores.size -
        (np.arange(1, n_scores + 1) - tar_trial_sums)
    )

    frr = np.concatenate(([0], tar_trial_sums / target_scores.size))
    far = np.concatenate(([1], nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((
        [all_scores[indices[0]] - 0.001],
        all_scores[indices]
    ))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


# =========================
# ASVspoof5 Evaluation
# =========================

def load_protocol(protocol_path):
    """
    Load ASVspoof5 protocol file (space-separated, no header).

    Column layout (0-indexed):
        0: speaker_id, 1: trial, 2: gender, 3: codec, 4: num,
        5: ref_id, 6: ac_cond, 7: attack_type, 8: label, 9: extra
    """
    col_names = ['speaker_id', 'trial', 'gender', 'codec', 'num',
                 'ref_id', 'ac_cond', 'attack_type', 'label', 'extra']
    df = pd.read_csv(protocol_path, sep=r'\s+', header=None, names=col_names)
    return df


def load_scores(score_path):
    """
    Accepts two formats:
        trial_id score
        Output trial_id label score   (raw inference.log format)
    """
    df = pd.read_csv(score_path, sep=r'\s+', header=None, engine="python")

    if df.shape[1] == 4:
        # Raw format: Output  trial_id  label  score
        df = df.iloc[:, [1, 3]]
    elif df.shape[1] == 2:
        df = df.iloc[:, [0, 1]]
    else:
        raise ValueError(f"Unexpected score file format: {df.shape[1]} columns")

    df.columns = ["trial", "score"]
    df["score"] = pd.to_numeric(df["score"])
    return df


def evaluate(protocol_path, score_path, subset=None):

    protocol_df = load_protocol(protocol_path)
    score_df = load_scores(score_path)

    # Merge
    df = protocol_df.merge(score_df, on="trial", how="inner")

    if subset is not None:
        if "subset" not in df.columns:
            raise ValueError("Subset column not found in protocol.")
        df = df[df["subset"] == subset]

    # Split scores
    bonafide_scores = df[df["label"] == "bonafide"]["score"].values
    spoof_scores = df[df["label"] == "spoof"]["score"].values

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        raise ValueError("No bonafide or spoof trials found.")

    eer, threshold = compute_eer(bonafide_scores, spoof_scores)

    print("===================================")
    print(f"Trials evaluated: {len(df)}")
    print(f"Bonafide trials: {len(bonafide_scores)}")
    print(f"Spoof trials: {len(spoof_scores)}")
    print("-----------------------------------")
    print(f"EER: {eer * 100:.4f} %")
    print(f"EER Threshold: {threshold:.6f}")
    print("===================================")

    return eer


# =========================
# CLI
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute EER for ASVspoof5"
    )

    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Path to ASVspoof5 TSV protocol file"
    )

    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to model score file"
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset filter (e.g., dev, eval)"
    )

    args = parser.parse_args()

    evaluate(
        protocol_path=args.protocol,
        score_path=args.scores,
        subset=args.subset
    )