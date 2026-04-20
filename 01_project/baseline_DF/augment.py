"""
augment.py — On-the-fly audio augmentation for robustness evaluation.

Standalone module with no dependency on core_scripts.

Waveform convention:
    numpy float32, shape (num_samples,), range [-1, 1], 16 kHz
"""

import os
import sys
import glob
import hashlib
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

# Default location for noise clips: <repo root>/noise_clips/
# (augment.py lives at <repo>/01_project/baseline_DF/, noise_clips/ sits at the repo root)
DEFAULT_NOISE_CLIPS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "noise_clips")
)


def add_noise(signal, noise, snr_db):
    """Add noise to signal at a target SNR (dB).

    Args:
        signal: np.ndarray, shape (N,), float32, clean waveform
        noise:  np.ndarray, shape (M,), float32, noise waveform (M >= N)
        snr_db: float, target signal-to-noise ratio in dB

    Returns:
        np.ndarray, shape (N,), float32, noisy waveform clipped to [-1, 1]
    """
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0 or sig_power == 0:
        return signal.copy()

    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)

    mixed = signal + scale * noise[:len(signal)]
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def truncate(wav, sr, duration_sec):
    """Center-crop waveform to target duration.

    Args:
        wav:          np.ndarray, shape (N,), float32
        sr:           int, sampling rate
        duration_sec: float, target duration in seconds

    Returns:
        np.ndarray — cropped to target duration, or original if already shorter
    """
    target_samples = int(sr * duration_sec)
    if wav.shape[0] <= target_samples:
        return wav.copy()
    start = (wav.shape[0] - target_samples) // 2
    return wav[start : start + target_samples].copy()


class Augmentor:
    """Applies one or more augmentation conditions to waveforms.

    Supports single conditions and composite conditions joined by '+'.
    Composite conditions are applied left-to-right (e.g., truncate then noise).

    Usage:
        aug = Augmentor("noise_babble_10dB")
        aug = Augmentor("short_3s+noise_babble_20dB")  # truncate, then noise
        result = aug(wav_array, "E_0000000001")

    Condition names:
        noise_ambient_10dB,  noise_ambient_20dB,
        noise_babble_10dB,   noise_babble_20dB,
        short_3s, short_5s
    """

    NOISE_SOURCES = {
        "ambient": "ambient",
        "babble":  "babble",
    }

    def __init__(self, condition, musan_dir=DEFAULT_NOISE_CLIPS_DIR,
                 noise_subdir="train", sr=SAMPLE_RATE, max_noise_clips=None):
        self.condition = condition
        self.noise_subdir = noise_subdir
        self.sr = sr

        # Composite condition: split on '+' and create sub-augmentors
        if '+' in condition:
            self.aug_type = "composite"
            self._sub_augmentors = [
                Augmentor(sub.strip(), musan_dir=musan_dir,
                          noise_subdir=noise_subdir, sr=sr,
                          max_noise_clips=max_noise_clips)
                for sub in condition.split('+')
            ]
            print(f"[augment] Composite condition: "
                  f"{[a.condition for a in self._sub_augmentors]}",
                  file=sys.stderr)
        else:
            self.aug_type = None
            self.noise_clips = []
            self._sub_augmentors = []
            self._parse_condition(condition, musan_dir, noise_subdir, max_noise_clips)

    def _parse_condition(self, condition, musan_dir, noise_subdir, max_noise_clips):
        parts = condition.split("_")

        if parts[0] == "noise":
            noise_type = parts[1]
            self.snr_db = int(parts[2].replace("dB", ""))
            self.aug_type = "noise"
            self._load_noise_clips(musan_dir, noise_subdir, noise_type, max_noise_clips)

        elif parts[0] == "short":
            self.duration_sec = int(parts[1].replace("s", ""))
            self.aug_type = "truncate"

        else:
            raise ValueError(f"Unknown augmentation condition: {condition}")

    def _load_noise_clips(self, musan_dir, noise_subdir, noise_type, max_clips):
        folder = self.NOISE_SOURCES[noise_type]
        noise_dir = os.path.join(musan_dir, noise_subdir, folder)

        if not os.path.isdir(noise_dir):
            raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

        wav_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))
        if max_clips is not None:
            wav_files = wav_files[:max_clips]

        if len(wav_files) == 0:
            raise FileNotFoundError(f"No noise files found in {noise_dir}")

        for fpath in wav_files:
            data, file_sr = sf.read(fpath, dtype='float32')
            if file_sr != self.sr:
                raise ValueError(
                    f"Noise file {fpath} has sr={file_sr}, expected {self.sr}")
            if data.ndim > 1:
                data = data[:, 0]
            self.noise_clips.append(data)

        print(f"[augment] Loaded {len(self.noise_clips)} noise clips "
              f"for condition '{self.condition}'", file=sys.stderr)

    def _filename_seed(self, file_name):
        h = hashlib.md5(file_name.encode('utf-8')).hexdigest()
        return int(h[:8], 16)

    def __call__(self, wav, file_name):
        """Apply augmentation to a single waveform.

        Args:
            wav:       np.ndarray, shape (N,), float32, range [-1,1]
            file_name: str, utterance ID for deterministic seeding

        Returns:
            np.ndarray or None (if file should be skipped)
        """
        if self.aug_type == "composite":
            for sub in self._sub_augmentors:
                wav = sub(wav, file_name)
                if wav is None:
                    return None
            return wav
        elif self.aug_type == "noise":
            return self._apply_noise(wav, file_name)
        elif self.aug_type == "truncate":
            return self._apply_truncate(wav)

    def _apply_noise(self, wav, file_name):
        rng = np.random.RandomState(self._filename_seed(file_name))

        clip_idx = rng.randint(0, len(self.noise_clips))
        noise_clip = self.noise_clips[clip_idx]

        # tile if noise is shorter than signal
        if noise_clip.shape[0] < wav.shape[0]:
            reps = (wav.shape[0] // noise_clip.shape[0]) + 1
            noise_clip = np.tile(noise_clip, reps)

        max_start = noise_clip.shape[0] - wav.shape[0]
        start = rng.randint(0, max_start + 1)
        noise_segment = noise_clip[start : start + wav.shape[0]]

        return add_noise(wav, noise_segment, self.snr_db)

    def _apply_truncate(self, wav):
        return truncate(wav, self.sr, self.duration_sec)


class TrainingAugmentor:
    """Randomly applies noise augmentation during training.

    Each call has a 50% chance of returning the original waveform (clean)
    and a 50% chance of adding noise. When augmenting, the noise type
    (ambient/babble) and SNR (10-25dB) are chosen randomly.

    Uses filename-based seeding combined with an epoch counter so that:
    - The same file gets different augmentation each epoch
    - Results are reproducible given the same epoch number

    Usage:
        aug = TrainingAugmentor()
        result = aug(wav_array, "E_0000000001")
        aug.set_epoch(1)  # call at start of each epoch
    """

    SNR_RANGE = (10, 25)  # inclusive bounds in dB

    def __init__(self, musan_dir=DEFAULT_NOISE_CLIPS_DIR, noise_subdir="train",
                 sr=SAMPLE_RATE, augment_prob=0.5, max_noise_clips=None, curriculum=False):
        self.sr = sr
        self.augment_prob = augment_prob
        self.epoch = 0
        self.curriculum = curriculum

        # Load all noise clips (ambient + babble)
        self.noise_pools = {}
        for noise_type, folder in Augmentor.NOISE_SOURCES.items():
            noise_dir = os.path.join(musan_dir, noise_subdir, folder)
            if not os.path.isdir(noise_dir):
                raise FileNotFoundError(f"Noise directory not found: {noise_dir}")
            clips = []
            wav_files = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))
            if max_noise_clips is not None:
                wav_files = wav_files[:max_noise_clips]
            for fpath in wav_files:
                data, file_sr = sf.read(fpath, dtype='float32')
                if file_sr != sr:
                    raise ValueError(
                        f"Noise file {fpath} has sr={file_sr}, expected {sr}")
                if data.ndim > 1:
                    data = data[:, 0]
                clips.append(data)
            self.noise_pools[noise_type] = clips

        self.noise_types = list(self.noise_pools.keys())
        total_clips = sum(len(v) for v in self.noise_pools.values())
        print(f"[train-augment] Loaded {total_clips} noise clips "
              f"({', '.join(f'{k}: {len(v)}' for k, v in self.noise_pools.items())}), "
              f"augment_prob={augment_prob}, SNR={self.SNR_RANGE[0]}-{self.SNR_RANGE[1]}dB",
              file=sys.stderr)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _filename_seed(self, file_name):
        key = f"{file_name}_epoch{self.epoch}"
        h = hashlib.md5(key.encode('utf-8')).hexdigest()
        return int(h[:8], 16)

    def __call__(self, wav, file_name):
        """Apply random noise augmentation.

        Args:
            wav:       np.ndarray, shape (N,), float32, range [-1,1]
            file_name: str, utterance ID

        Returns:
            np.ndarray — original or noise-augmented waveform
        """
        rng = np.random.RandomState(self._filename_seed(file_name))

        # 50% chance: return clean
        if rng.random() > self.augment_prob:
            return wav

        # Pick random noise type and SNR
        noise_type = self.noise_types[rng.randint(0, len(self.noise_types))]

        # Curriculum: gradually lower minimum SNR over epochs
        if self.curriculum:
            if self.epoch < 2:
                snr_range = (20, 30)        # easy: mild noise only
            elif self.epoch < 4:
                snr_range = (10, 30)        # medium: moderate noise
            else:
                snr_range = self.SNR_RANGE  # full range (5-30dB)
        else:
            snr_range = self.SNR_RANGE

        snr_db = rng.randint(snr_range[0], snr_range[1] + 1)
        clips = self.noise_pools[noise_type]

        clip_idx = rng.randint(0, len(clips))
        noise_clip = clips[clip_idx]

        # Tile if noise is shorter than signal
        if noise_clip.shape[0] < wav.shape[0]:
            reps = (wav.shape[0] // noise_clip.shape[0]) + 1
            noise_clip = np.tile(noise_clip, reps)

        max_start = noise_clip.shape[0] - wav.shape[0]
        start = rng.randint(0, max_start + 1)
        noise_segment = noise_clip[start : start + wav.shape[0]]

        return add_noise(wav, noise_segment, snr_db)


def dump_samples(augment_fn, input_dir, output_dir, n=20,
                 ext=".flac", sr=SAMPLE_RATE):
    """Save augmented samples to disk for listening verification."""
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, f"*{ext}")))[:n]

    saved = 0
    skipped = 0
    for fpath in files:
        data, _ = sf.read(fpath, dtype='float32')
        file_name = os.path.splitext(os.path.basename(fpath))[0]

        result = augment_fn(data, file_name)

        if result is None:
            skipped += 1
            continue

        out_path = os.path.join(output_dir, f"{file_name}_aug.wav")
        sf.write(out_path, result, sr)
        saved += 1

    print(f"[augment] dump_samples: saved {saved}, skipped {skipped} "
          f"to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augmentation dump utility")
    parser.add_argument("--condition", type=str, required=True,
                        help="e.g. noise_office_10dB, short_3s")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--musan-dir", type=str,
                        default=DEFAULT_NOISE_CLIPS_DIR)
    parser.add_argument("--noise-subdir", type=str, default="selected",
                        help="Subdirectory under musan/ for noise clips")
    args = parser.parse_args()

    aug = Augmentor(args.condition, musan_dir=args.musan_dir,
                    noise_subdir=args.noise_subdir)
    dump_samples(aug, args.input_dir, args.output_dir, n=args.n)
