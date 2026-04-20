#!/usr/bin/env python
"""
gradcam.py — GradCAM saliency maps for LFCC-LCNN deepfake detection model.

Generates heatmaps showing which frequency bands and time regions the model
attends to when making bonafide/spoof decisions. Output is a pair of stacked
subplots per sample: raw LFCC spectrogram on top, GradCAM overlay on bottom.

Requires PYTHONPATH to include the project root:
    export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN

Usage:
    python gradcam.py \\
        --model ../../_completed_runs/clean_weighted/trained_network.pt \\
        --audio-dir ~/asvspoof5_dataset/flac_E_eval \\
        --protocol ~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv \\
        --output-dir ../../saliency_maps/clean_baseline \\
        --n 10
"""

import os
import sys
import argparse
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as torch_nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end

# ---- Constants ----

# Target conv layer index in m_transform[0] Sequential.
# Index 25 = Conv2d(32, 64, 3x3) — last 3x3 conv before the final MaxPool.
# After 3 MaxPools, spatial dims are (~frames/8, ~60/8) = (~100, 7).
# Output shape: (batch, 64, ~100, 7) — good resolution for upsampling.
TARGET_LAYER_IDX = 25
SEED = 42
SAMPLE_RATE = 16000
FRAME_HOP = 160  # samples per LFCC frame shift


# ---- Lightweight model (mirrors model.py architecture) ----

class LCNNModel(torch_nn.Module):
    """LFCC-LCNN model matching the checkpoint key structure.

    Recreates the architecture from model.py without framework dependencies
    (core_scripts, prj_conf, etc.), enabling standalone GradCAM inference.
    """

    def __init__(self, emb_dim=1):
        super().__init__()
        self.emb_dim = emb_dim
        lfcc_dim = 60  # 20 LFCC * 3 (base + delta + delta-delta)

        self.m_frontend = torch_nn.ModuleList([
            nii_front_end.LFCC(320, 160, 1024, 16000, 20,
                               with_energy=True, max_freq=0.5)
        ])

        self.m_transform = torch_nn.ModuleList([
            torch_nn.Sequential(
                # Block 1
                torch_nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),   # [0]
                nii_nn.MaxFeatureMap2D(),                              # [1]
                torch_nn.MaxPool2d([2, 2], [2, 2]),                    # [2]
                # Block 2
                torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),   # [3]
                nii_nn.MaxFeatureMap2D(),                              # [4]
                torch_nn.BatchNorm2d(32, affine=False),                # [5]
                torch_nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),   # [6]
                nii_nn.MaxFeatureMap2D(),                              # [7]
                torch_nn.MaxPool2d([2, 2], [2, 2]),                    # [8]
                # Block 3
                torch_nn.BatchNorm2d(48, affine=False),                # [9]
                torch_nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),   # [10]
                nii_nn.MaxFeatureMap2D(),                              # [11]
                torch_nn.BatchNorm2d(48, affine=False),                # [12]
                torch_nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),  # [13]
                nii_nn.MaxFeatureMap2D(),                              # [14]
                torch_nn.MaxPool2d([2, 2], [2, 2]),                    # [15]
                # Block 4
                torch_nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),  # [16]
                nii_nn.MaxFeatureMap2D(),                              # [17]
                torch_nn.BatchNorm2d(64, affine=False),                # [18]
                torch_nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),   # [19]
                nii_nn.MaxFeatureMap2D(),                              # [20]
                torch_nn.BatchNorm2d(32, affine=False),                # [21]
                # Block 5
                torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),   # [22]
                nii_nn.MaxFeatureMap2D(),                              # [23]
                torch_nn.BatchNorm2d(32, affine=False),                # [24]
                torch_nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),   # [25] <-- target
                nii_nn.MaxFeatureMap2D(),                              # [26]
                torch_nn.MaxPool2d([2, 2], [2, 2]),                    # [27]
                torch_nn.Dropout(0.7)                                  # [28]
            )
        ])

        blstm_dim = (lfcc_dim // 16) * 32  # = 96
        self.m_before_pooling = torch_nn.ModuleList([
            torch_nn.Sequential(
                nii_nn.BLSTMLayer(blstm_dim, blstm_dim),
                nii_nn.BLSTMLayer(blstm_dim, blstm_dim)
            )
        ])

        self.m_output_act = torch_nn.ModuleList([
            torch_nn.Linear(blstm_dim, self.emb_dim)
        ])

        # OC-Softmax center (only used when emb_dim > 1)
        if self.emb_dim > 1:
            self.oc_loss = torch_nn.Module()
            self.oc_loss.center = torch_nn.Parameter(torch.randn(1, self.emb_dim))

        # Dummy parameters to match checkpoint keys
        self.input_mean = torch_nn.Parameter(torch.zeros(1), requires_grad=False)
        self.input_std = torch_nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_mean = torch_nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_std = torch_nn.Parameter(torch.ones(1), requires_grad=False)


# ---- Model loading ----

def load_model(checkpoint_path, device='cpu'):
    """Load trained LFCC-LCNN model from checkpoint.

    Auto-detects OC-Softmax models by checking for oc_loss.center in the
    state dict and adjusts emb_dim accordingly.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Detect OC-Softmax from output layer shape
    emb_dim = 1
    for key in state_dict:
        if 'm_output_act.0.weight' in key:
            emb_dim = state_dict[key].shape[0]
            break

    model = LCNNModel(emb_dim=emb_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if emb_dim > 1:
        print(f'[gradcam] OC-Softmax model detected (emb_dim={emb_dim})')
    return model


# ---- GradCAM ----

def compute_gradcam(model, wav_np, target_layer_idx=TARGET_LAYER_IDX):
    """Compute GradCAM saliency map for a single audio sample.

    Forward pass: wav -> LFCC -> CNN (m_transform) -> BLSTM -> Linear -> score
    Hooks capture activations and gradients at the target Conv2d layer.

    Args:
        model: LCNNModel in eval mode
        wav_np: numpy float32, shape (samples,)
        target_layer_idx: index of target Conv2d in m_transform[0]

    Returns:
        lfcc: numpy array (frames, 60)
        cam: numpy array (frames, 60), normalized to [0, 1]
        score: float, raw model output (higher = more likely bonafide)
    """
    activations = {}
    gradients = {}
    target_layer = model.m_transform[0][target_layer_idx]

    def fwd_hook(module, inp, out):
        activations['value'] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_backward_hook(bwd_hook)

    try:
        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)  # (1, samples)

        # LFCC extraction (no grad needed for frontend)
        with torch.no_grad():
            lfcc = model.m_frontend[0](wav_tensor)  # (1, frames, 60)
        lfcc_np = lfcc.squeeze(0).numpy().copy()

        # CNN forward — detach from no_grad context, enable grad for hooks
        x = lfcc.unsqueeze(1).detach().requires_grad_(True)  # (1, 1, frames, 60)
        hidden = model.m_transform[0](x)  # (1, 32, frames//16, 60//16)

        # Reshape: (batch, ch, H, W) -> (batch, H, ch*W)
        batch_size = hidden.shape[0]
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden.shape[1]
        hidden = hidden.view(batch_size, frame_num, -1)

        # BLSTM with residual connection + mean pooling + linear
        hidden_lstm = model.m_before_pooling[0](hidden)
        emb = model.m_output_act[0](
            (hidden_lstm + hidden).mean(1)
        )  # (1, emb_dim)

        if model.emb_dim > 1:
            # OC-Softmax: cosine similarity to bonafide center
            w = F.normalize(model.oc_loss.center, p=2, dim=1)
            x_norm = F.normalize(emb, p=2, dim=1)
            score_tensor = (x_norm @ w.transpose(0, 1)).squeeze(1)  # (1,)
        else:
            score_tensor = emb  # (1, 1)

        score = score_tensor.item()

        # Backward from score
        model.zero_grad()
        score_tensor.backward()

        # GradCAM: global-average-pool gradients -> channel weights -> weighted sum
        grads = gradients['value']   # (1, C, H, W)
        acts = activations['value']  # (1, C, H, W)

        weights = grads.mean(dim=[2, 3], keepdim=True)          # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)          # (1, 1, H, W)
        cam = torch.relu(cam)                                    # keep positive
        cam = cam.squeeze(0).squeeze(0)                          # (H, W)

        # Normalize to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Bilinear upsample from (H, W) to (frames, 60)
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(lfcc_np.shape[0], lfcc_np.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().numpy()

        return lfcc_np, cam_up, score

    finally:
        h_fwd.remove()
        h_bwd.remove()


# ---- Protocol parsing ----

def parse_protocol(protocol_path):
    """Parse ASVspoof5 protocol TSV.

    Returns:
        labels: dict mapping utterance ID -> 'bonafide' or 'spoof'
    """
    labels = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            labels[parts[1]] = parts[-2]  # second-to-last column
    return labels


def load_scores(scores_path):
    """Load score file (utterance_id score).

    Returns:
        dict mapping utterance ID -> float score
    """
    scores = {}
    with open(scores_path) as f:
        for line in f:
            uid, s = line.strip().split()
            scores[uid] = float(s)
    return scores


def stratified_select(labels, scores, n_per_type=2, seed=SEED, threshold=0.0):
    """Select representative samples stratified by correctness and confidence.

    Categories (using threshold as decision boundary — above = bonafide):
      - correct_highconf: correctly classified with high confidence
      - correct_lowconf:  correctly classified but near decision boundary
      - misclassified:    incorrectly classified

    Args:
        labels: dict uid -> 'bonafide'/'spoof'
        scores: dict uid -> float
        n_per_type: samples per (category, label) pair
        seed: random seed
        threshold: decision threshold (default 0.0, use EER threshold for consistency)

    Returns:
        list of (utt_id, label, category) tuples
    """
    rng = random.Random(seed)

    # Split by label, sorted by score
    bonafide = sorted(
        [(uid, scores[uid]) for uid in scores if labels.get(uid) == 'bonafide'],
        key=lambda x: x[1])
    spoof = sorted(
        [(uid, scores[uid]) for uid in scores if labels.get(uid) == 'spoof'],
        key=lambda x: x[1])

    def pick(pool, n):
        return rng.sample(pool, min(n, len(pool)))

    selected = []

    # Correct high-confidence: bonafide with highest scores, spoof with lowest
    selected += [(u, 'bonafide', 'correct_highconf') for u, _ in pick(bonafide[-100:], n_per_type)]
    selected += [(u, 'spoof', 'correct_highconf')    for u, _ in pick(spoof[:100], n_per_type)]

    # Correct low-confidence: bonafide just above threshold, spoof just below threshold
    margin = 2.0
    lc_bon = [x for x in bonafide if threshold < x[1] < threshold + margin]
    lc_spf = [x for x in spoof if threshold - margin < x[1] < threshold]
    selected += [(u, 'bonafide', 'correct_lowconf') for u, _ in pick(lc_bon, n_per_type)]
    selected += [(u, 'spoof', 'correct_lowconf')    for u, _ in pick(lc_spf, n_per_type)]

    # Misclassified: bonafide with lowest scores, spoof with highest
    selected += [(u, 'bonafide', 'misclassified') for u, _ in pick(bonafide[:100], n_per_type)]
    selected += [(u, 'spoof', 'misclassified')    for u, _ in pick(spoof[-100:], n_per_type)]

    return selected


# ---- Visualization ----

def plot_saliency(lfcc, cam, output_path, title):
    """Plot LFCC spectrogram and GradCAM overlay as stacked subplots.

    Top: Raw LFCC spectrogram (frequency x time)
    Bottom: LFCC with semi-transparent GradCAM heatmap overlay

    Args:
        lfcc: numpy array (frames, 60)
        cam: numpy array (frames, 60), values in [0, 1]
        output_path: path to save PNG
        title: plot title string
    """
    n_frames, n_feats = lfcc.shape
    time_axis = np.arange(n_frames) * FRAME_HOP / SAMPLE_RATE
    t_max = time_axis[-1] if len(time_axis) > 0 else 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: raw LFCC spectrogram
    im1 = ax1.imshow(lfcc.T, aspect='auto', origin='lower',
                      extent=[0, t_max, 0, n_feats], cmap='viridis')
    ax1.set_ylabel('LFCC index')
    ax1.set_title('LFCC spectrogram')
    plt.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)
    for y in [20, 40]:
        ax1.axhline(y=y, color='w', lw=0.5, ls='--', alpha=0.7)
    ax1.set_yticks([10, 30, 50])
    ax1.set_yticklabels(['LFCC\n(0-19)', 'Delta\n(20-39)', 'DD\n(40-59)'])

    # Bottom: LFCC + GradCAM overlay
    ax2.imshow(lfcc.T, aspect='auto', origin='lower',
               extent=[0, t_max, 0, n_feats], cmap='viridis')
    im_cam = ax2.imshow(cam.T, aspect='auto', origin='lower',
                         extent=[0, t_max, 0, n_feats],
                         cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax2.set_ylabel('LFCC index')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('GradCAM overlay')
    plt.colorbar(im_cam, ax=ax2, fraction=0.02, pad=0.02, label='Attention')
    for y in [20, 40]:
        ax2.axhline(y=y, color='w', lw=0.5, ls='--', alpha=0.7)
    ax2.set_yticks([10, 30, 50])
    ax2.set_yticklabels(['LFCC\n(0-19)', 'Delta\n(20-39)', 'DD\n(40-59)'])

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Generate GradCAM saliency maps for LFCC-LCNN model')
    parser.add_argument('--model', required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--audio-dir', required=True,
                        help='Directory containing eval audio files (FLAC)')
    parser.add_argument('--protocol', required=True,
                        help='Path to ASVspoof5 protocol TSV')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save saliency map PNGs')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of samples per class (random mode) '
                             'or per stratification type (stratified mode)')
    parser.add_argument('--scores', type=str, default=None,
                        help='Score file (uid score) for stratified selection. '
                             'Without this, falls back to random selection.')
    parser.add_argument('--augment', type=str, default=None,
                        help='Augmentation condition (e.g. noise_ambient_10dB)')
    parser.add_argument('--noise-subdir', type=str, default='train',
                        help='Subdirectory under noise_clips/ for noise clips')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducible sample selection')
    parser.add_argument('--target-layer', type=int, default=TARGET_LAYER_IDX,
                        help='Conv2d layer index in m_transform (default: 25)')
    parser.add_argument('--sample-ids', type=str, nargs='+', default=None,
                        help='Specific utterance IDs to process (bypasses selection)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Decision threshold for stratification (default: 0.0, use EER threshold for consistency)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f'[gradcam] Loading model from {args.model}')
    model = load_model(args.model)
    print('[gradcam] Model loaded')

    # Parse protocol
    labels = parse_protocol(args.protocol)
    n_bon = sum(1 for v in labels.values() if v == 'bonafide')
    n_spf = sum(1 for v in labels.values() if v == 'spoof')
    print(f'[gradcam] Protocol: {n_bon} bonafide, {n_spf} spoof')

    # Select samples
    if args.sample_ids:
        samples = [(uid, labels.get(uid, 'unknown'), 'selected') for uid in args.sample_ids]
        print(f'[gradcam] Using {len(samples)} explicitly specified samples')
    elif args.scores:
        # Stratified selection using pre-computed scores
        scores = load_scores(args.scores)
        print(f'[gradcam] Loaded {len(scores)} scores — using stratified selection')
        samples = stratified_select(labels, scores, n_per_type=args.n, seed=args.seed, threshold=args.threshold)
        print(f'[gradcam] Selected {len(samples)} samples across 3 categories')
    else:
        # Random selection fallback
        random.seed(args.seed)
        bon_ids = [u for u, l in labels.items() if l == 'bonafide']
        spf_ids = [u for u, l in labels.items() if l == 'spoof']
        sel_bon = random.sample(bon_ids, min(args.n, len(bon_ids)))
        sel_spf = random.sample(spf_ids, min(args.n, len(spf_ids)))
        samples = [(u, 'bonafide', 'random') for u in sel_bon] + \
                  [(u, 'spoof', 'random') for u in sel_spf]

    # Optional augmentation
    augmentor = None
    if args.augment:
        from augment import Augmentor
        augmentor = Augmentor(args.augment, noise_subdir=args.noise_subdir)

    condition_label = args.augment if args.augment else 'clean'

    # Load score lookup for title display (use official inference scores when available)
    score_lookup = {}
    if args.scores:
        score_lookup = load_scores(args.scores)

    # Process samples
    for i, (utt_id, label, category) in enumerate(samples):
        audio_path = os.path.join(args.audio_dir, f'{utt_id}.flac')
        if not os.path.isfile(audio_path):
            print(f'  [SKIP] {utt_id} — file not found')
            continue

        wav, sr = sf.read(audio_path, dtype='float32')
        if sr != SAMPLE_RATE:
            print(f'  [SKIP] {utt_id} — sr={sr}, expected {SAMPLE_RATE}')
            continue

        # Apply augmentation if specified
        if augmentor is not None:
            wav = augmentor(wav, utt_id)
            if wav is None:
                print(f'  [SKIP] {utt_id} — too short for augmentation')
                continue

        # Compute GradCAM
        lfcc, cam, score = compute_gradcam(model, wav, args.target_layer)

        # Save plot — use score file value if available, otherwise gradcam's re-computed score
        display_score = score_lookup.get(utt_id, score)
        title = (f'{utt_id}  |  {label}  |  score: {display_score:.4f}'
                 f'  |  {condition_label}  |  {category}')
        out_path = os.path.join(
            args.output_dir, f'{category}_{label}_{utt_id}.png')
        plot_saliency(lfcc, cam, out_path, title)

        print(f'  [{i+1}/{len(samples)}] {category} {label} {utt_id} '
              f'score={display_score:.4f} -> {out_path}')

    print(f'[gradcam] Done. Outputs in {args.output_dir}')


if __name__ == '__main__':
    main()

