#!/usr/bin/env python
"""
alert_demo.py — Real-time deepfake alert simulation.

Processes audio files through the LFCC-LCNN deepfake detector and produces
four-tier severity decisions (CRITICAL / HIGH / MONITOR / UNCONFIRMED)
anchored to precision operating points (99% / 95% / 90%). Simulates an
enterprise vishing detection pipeline.

Standalone — no training framework dependency.

Usage:
    # Single file
    python alert_demo.py --model ../../trained_network.pt --audio call.flac

    # Batch (directory)
    python alert_demo.py --model ../../trained_network.pt --audio-dir incoming/

    # OC-Softmax model (auto-detected)
    python alert_demo.py --model ../../ocsoftmax_model.pt --audio call.flac

    # Custom thresholds (override any tier)
    python alert_demo.py --model ../../trained_network.pt --audio call.flac \
        --critical-threshold -11.0 --high-threshold -7.0 --monitor-threshold -4.0

Requires PYTHONPATH to include the project root:
    export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN
"""

import argparse
import datetime
import json
import os
import subprocess
import glob
import tempfile

import soundfile as sf
import torch
import torch.nn as torch_nn
import torch.nn.functional as F

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end

SAMPLE_RATE = 16000
NATIVE_EXTENSIONS = {'.flac', '.wav'}
FFMPEG_EXTENSIONS = {'.mp3', '.ogg', '.opus', '.m4a', '.aac', '.wma', '.amr', '.webm'}
ALL_EXTENSIONS = NATIVE_EXTENSIONS | FFMPEG_EXTENSIONS

# Default thresholds anchored to precision operating points
# (from risk_threshold_analysis.md)
#
# Tier        Precision   BCE Thresh   OC Thresh   Action
# CRITICAL    99%         -11.48       -0.314      Auto-escalate, terminate/flag
# HIGH        95%          -6.35        0.096      Queue for analyst review
# MONITOR     90%          -3.78        0.690      Log for trend analysis
DEFAULT_THRESHOLDS_BCE = {'critical': -11.48, 'high': -6.35, 'monitor': -3.78}
DEFAULT_THRESHOLDS_OC = {'critical': -0.314, 'high': 0.096, 'monitor': 0.690}


# ---- Audio loading ----

def load_audio(filepath):
    """Load audio from any supported format, returning (samples, sample_rate).

    FLAC and WAV are loaded directly via soundfile. Other formats (MP3, Opus,
    OGG, M4A, AAC, WMA, AMR, WebM) are converted to WAV via ffmpeg first.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in NATIVE_EXTENSIONS:
        wav, sr = sf.read(filepath, dtype='float32')
        return wav, sr

    if ext in FFMPEG_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', filepath,
                 '-ar', str(SAMPLE_RATE), '-ac', '1',
                 '-f', 'wav', tmp_path],
                capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")
            wav, sr = sf.read(tmp_path, dtype='float32')
            return wav, sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    raise ValueError(f"Unsupported audio format: {ext}")


# ---- Model (same as gradcam.py) ----

class LCNNModel(torch_nn.Module):
    """Standalone LFCC-LCNN model for inference."""

    def __init__(self, emb_dim=1):
        super().__init__()
        self.emb_dim = emb_dim
        lfcc_dim = 60

        self.m_frontend = torch_nn.ModuleList([
            nii_front_end.LFCC(320, 160, 1024, 16000, 20,
                               with_energy=True, max_freq=0.5)
        ])

        self.m_transform = torch_nn.ModuleList([
            torch_nn.Sequential(
                torch_nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.MaxPool2d([2, 2], [2, 2]),
                torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(32, affine=False),
                torch_nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.MaxPool2d([2, 2], [2, 2]),
                torch_nn.BatchNorm2d(48, affine=False),
                torch_nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(48, affine=False),
                torch_nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.MaxPool2d([2, 2], [2, 2]),
                torch_nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(64, affine=False),
                torch_nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(32, affine=False),
                torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(32, affine=False),
                torch_nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.MaxPool2d([2, 2], [2, 2]),
                torch_nn.Dropout(0.7)
            )
        ])

        blstm_dim = (lfcc_dim // 16) * 32
        self.m_before_pooling = torch_nn.ModuleList([
            torch_nn.Sequential(
                nii_nn.BLSTMLayer(blstm_dim, blstm_dim),
                nii_nn.BLSTMLayer(blstm_dim, blstm_dim)
            )
        ])

        self.m_output_act = torch_nn.ModuleList([
            torch_nn.Linear(blstm_dim, self.emb_dim)
        ])

        if self.emb_dim > 1:
            self.oc_loss = torch_nn.Module()
            self.oc_loss.center = torch_nn.Parameter(torch.randn(1, self.emb_dim))

        self.input_mean = torch_nn.Parameter(torch.zeros(1), requires_grad=False)
        self.input_std = torch_nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_mean = torch_nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_std = torch_nn.Parameter(torch.ones(1), requires_grad=False)


def load_model(checkpoint_path, device='cpu'):
    """Load model, auto-detecting OC-Softmax from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    emb_dim = 1
    for key in state_dict:
        if 'm_output_act.0.weight' in key:
            emb_dim = state_dict[key].shape[0]
            break

    model = LCNNModel(emb_dim=emb_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


# ---- Inference ----

def score_audio(model, wav_np):
    """Run a single audio sample through the model and return a score.

    Returns:
        float: score (higher = more likely bonafide)
    """
    with torch.no_grad():
        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
        lfcc = model.m_frontend[0](wav_tensor)

        x = lfcc.unsqueeze(1)
        hidden = model.m_transform[0](x)

        batch_size = hidden.shape[0]
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden.shape[1]
        hidden = hidden.view(batch_size, frame_num, -1)

        hidden_lstm = model.m_before_pooling[0](hidden)
        emb = model.m_output_act[0]((hidden_lstm + hidden).mean(1))

        if model.emb_dim > 1:
            w = F.normalize(model.oc_loss.center, p=2, dim=1)
            x_norm = F.normalize(emb, p=2, dim=1)
            score = (x_norm @ w.transpose(0, 1)).squeeze().item()
        else:
            score = emb.squeeze().item()

    return score


# ---- Alert logic ----

def classify(score, thresholds):
    """Apply three-tier thresholds to produce CRITICAL / HIGH / MONITOR / UNCONFIRMED.

    Thresholds are precision-anchored (99% / 95% / 90%). Lower score = more
    likely spoof. Tiers are evaluated from most severe to least.
    """
    if score < thresholds['critical']:
        return "CRITICAL"
    elif score < thresholds['high']:
        return "HIGH"
    elif score < thresholds['monitor']:
        return "MONITOR"
    else:
        return "UNCONFIRMED"


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Deepfake alert simulation — processes audio and produces '
                    'CRITICAL / HIGH / MONITOR / UNCONFIRMED decisions')
    parser.add_argument('--model', required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to a single audio file (.flac/.wav)')
    parser.add_argument('--audio-dir', type=str, default=None,
                        help='Directory of audio files to process')
    parser.add_argument('--critical-threshold', type=float, default=None,
                        help='CRITICAL tier threshold (99%% precision)')
    parser.add_argument('--high-threshold', type=float, default=None,
                        help='HIGH tier threshold (95%% precision)')
    parser.add_argument('--monitor-threshold', type=float, default=None,
                        help='MONITOR tier threshold (90%% precision)')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to JSON log file (appended per run)')
    args = parser.parse_args()

    if args.audio is None and args.audio_dir is None:
        parser.error('Provide --audio or --audio-dir')

    # Load model
    model = load_model(args.model)
    is_oc = model.emb_dim > 1
    model_type = "OC-Softmax" if is_oc else "Noise-augmented"

    # Set thresholds (per-tier overrides or defaults)
    defaults = DEFAULT_THRESHOLDS_OC if is_oc else DEFAULT_THRESHOLDS_BCE
    thresholds = {
        'critical': args.critical_threshold if args.critical_threshold is not None else defaults['critical'],
        'high': args.high_threshold if args.high_threshold is not None else defaults['high'],
        'monitor': args.monitor_threshold if args.monitor_threshold is not None else defaults['monitor'],
    }

    print(f"Model:     {model_type} (emb_dim={model.emb_dim})")
    print(f"Thresholds (precision-anchored):")
    print(f"  CRITICAL (99%): score < {thresholds['critical']:.4f}  → analyst review / auto-escalate")
    print(f"  HIGH     (95%): score < {thresholds['high']:.4f}  → analyst review")
    print(f"  MONITOR  (90%): score < {thresholds['monitor']:.4f}  → log/trend")
    print(f"  UNCONFIRMED:           score >= {thresholds['monitor']:.4f}  → no action")
    print()

    # Collect audio files
    audio_files = []
    if args.audio:
        audio_files.append(args.audio)
    if args.audio_dir:
        for ext in ALL_EXTENSIONS:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, f'*{ext}')))
        audio_files = sorted(set(audio_files))

    if not audio_files:
        print("No audio files found.")
        return

    # Process
    results = []
    counts = {"CRITICAL": 0, "HIGH": 0, "MONITOR": 0, "UNCONFIRMED": 0}

    TIER_DISPLAY = {
        "CRITICAL":    "\033[91m[CRITICAL]\033[0m    ",
        "HIGH":        "\033[38;5;208m[HIGH]\033[0m        ",
        "MONITOR":     "\033[93m[MONITOR]\033[0m     ",
        "UNCONFIRMED": "\033[92m[UNCONFIRMED]\033[0m ",
    }

    log_initialized = False

    for filepath in audio_files:
        filename = os.path.splitext(os.path.basename(filepath))[0]

        try:
            wav, sr = load_audio(filepath)
        except Exception as e:
            print(f"  [SKIP] {filename} — {e}")
            continue
        if sr != SAMPLE_RATE:
            print(f"  [SKIP] {filename} — sr={sr}, expected {SAMPLE_RATE}")
            continue

        score = score_audio(model, wav)
        tier = classify(score, thresholds)
        counts[tier] += 1

        print(f"  {TIER_DISPLAY[tier]} {filename}  score: {score:+.4f}")

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "file": filename,
            "score": round(score, 6),
            "tier": tier,
            "model": model_type,
            "thresholds": thresholds,
        }
        results.append(entry)

        if args.log:
            if not log_initialized:
                os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
                log_initialized = True
            with open(args.log, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    # Summary
    total = sum(counts.values())
    print()
    print(f"--- Summary ---")
    print(f"Files processed: {total}")
    if total > 0:
        for tier in ["CRITICAL", "HIGH", "MONITOR", "UNCONFIRMED"]:
            n = counts[tier]
            print(f"  {tier:<13} {n:>6}  ({n/total*100:.1f}%)")

    if args.log and results:
        print(f"Log written: {args.log} ({len(results)} entries)")


if __name__ == '__main__':
    main()
