# Voice Deepfake Detection for Enterprise Security
**Georgia Tech CS 6727 Practicum — Mina Xu**

This project extends the LFCC-LCNN baseline (ASVspoof 2021 DF track) into a robustness-oriented deepfake detection framework, evaluated on the **ASVspoof5 (2024)** dataset. The goal is to produce actionable security signals for detecting vishing (voice-based social engineering) attacks in enterprise environments.

---

## ASVspoof5 Adaptations

The original baseline was designed for ASVspoof 2021, which has a different protocol format and data layout than ASVspoof5. The following changes were made for compatibility:

- **Protocol parsing** (`model.py`): Updated to correctly read ASVspoof5's TSV label format and extract trial IDs and ground truth labels
- **Batch inference** (`model.py`): Fixed a bug where only the first file in each batch was scored — all files are now scored individually
- **Dataset paths** (`main.py`): Updated throughout to point to the ASVspoof5 directory structure
- **EER evaluator** (`02_evaluation_scripts/evaluator.py`): Computes EER directly from ASVspoof5 protocol files and model score outputs
  - Loads headerless ASVspoof5 TSV protocols and model score files (handles both raw and clean formats)
  - Merges scores with ground truth labels and computes EER using the ASVspoof eval toolkit
- **Inference script** (`01_project/baseline_DF/02_eval_clean.sh`): End-to-end evaluation script that runs inference, extracts scores, and computes EER via `evaluator.py`
  - Usage: `bash 02_eval_clean.sh [trained_model]` (defaults to `./trained_network.pt`)
- **Augmentation module** (`01_project/baseline_DF/augment.py`): On-the-fly noise and truncation augmentations applied during inference data loading, integrated into the framework's `f_post_data_process()` hook
  - Noise conditions: adds ambient or babble noise at a target SNR (20dB or 10dB). All active clips (ambient and babble) originate from [Freesound](https://freesound.org/). Ambient clips come via MUSAN's `noise/free-sound/` subset (MUSAN packages Freesound content under CC BY 4.0; see the [MUSAN paper](https://arxiv.org/abs/1510.08484)); babble clips were pulled directly from Freesound (per-clip CC licenses). Individual Freesound clips are not redistributed in this repository; see `noise_clips/README.md` for the clip inventory and reconstruction instructions.
  - Short-clip conditions: center-crops waveforms to 3s or 5s duration
  - Reproducible via deterministic filename-based seeding: each utterance ID is MD5-hashed to produce a per-file seed (`_filename_seed`), which controls noise clip selection and offset. Same file always gets the same augmentation across runs, independent of worker count or batch order
  - Includes a `dump_samples` utility to save augmented audio for listening verification
- **Pre-trained training script** (`01_project/baseline_DF/01_train.sh`): Adapted training script that fine-tunes from ASVspoof 2019 LA pre-trained weights instead of training from scratch
  - Uses `--ignore-training-history-in-trained-model` to load weights but reset optimizer and epoch counter
  - Includes LR scheduling (ReduceLROnPlateau with 0.5 decay factor) starting at LR 0.0003
  - Pre-trained weights at `__pretrained/trained_network.pt`, downloaded from ASVspoof 2021 repository
- **Codec augmentation** (`01_project/baseline_DF/codec_augment.py`): Batch script to generate MP3/Opus compressed versions of the eval set via ffmpeg encode-decode roundtrip
  - Encodes each FLAC file to a lossy codec, then decodes back to 16kHz FLAC — spectral damage persists after decoding
  - Supports parallel processing and is resumable (skips already-generated files)
- **Codec evaluation script** (`01_project/baseline_DF/03_eval_codec.sh`): Adapted evaluation script for codec-augmented eval sets
  - Auto-generates `.lst` file from the codec directory and uses `config_auto.py` so `config.py` stays untouched
  - Runs inference, extracts scores, and cleans up intermediate files
  - Usage: `bash 03_eval_codec.sh <codec_dir> <condition_name> [trained_model]`
- **Failure analysis** (`02_evaluation_scripts/failure_analysis.py`): Computes per-condition FAR/FRR at a fixed reference threshold, classifies dominant failure modes (spoof collapse vs bonafide collapse), and reports score distribution statistics (mean, std) per group
  - Auto-discovers `scores_*.txt` files in a directory and compares all conditions against the clean baseline
  - Outputs a summary table to stdout and optionally saves to CSV
  - Usage: `python 02_evaluation_scripts/failure_analysis.py --protocol <TSV> --clean-scores <scores.txt> --score-dir <dir> [--output <csv>]`
- **GradCAM saliency maps** (`01_project/baseline_DF/gradcam.py`): Generates heatmaps showing which LFCC frequency bands and time regions the model attends to when making decisions
  - Standalone model loading (no core_scripts dependency) — recreates the LFCC-LCNN architecture and loads checkpoint weights
  - Supports stratified sample selection (correct high-confidence, correct low-confidence, misclassified) when a score file is provided, or random selection as fallback
  - Optional `--augment` flag to apply on-the-fly augmentations before GradCAM computation
  - Outputs stacked subplot PNGs: raw LFCC spectrogram on top, GradCAM overlay on bottom
  - Example maps stored under `04_completed_evals/clean_weighted/saliency_maps/` (clean-weighted baseline) and `04_completed_evals/ocsoftmax_v1/saliency_maps/` (final OC-Softmax model)
  - Usage: `python gradcam.py --model <.pt> --audio-dir <dir> --protocol <TSV> --output-dir <dir> [--scores <scores.txt>] [--augment <condition>] [--n <samples>]`

---

## Baseline Training

**From scratch** (plain BCELoss, no class weighting, no pre-trained weights):

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.0165 | 1.7752 | Yes |
| 1 | 0.0034 | 3.5691 | No |

Stopped after 2 epochs due to overfitting. Best model = epoch 0.

**With pre-trained 2019 weights** (fine-tuned from ASVspoof 2019 LA, LR 0.0003, ReduceLROnPlateau with 0.5 decay):

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.0394 | 1.7929 | Yes |
| 1 | 0.0045 | 6.3971 | No |

Early stopping after 4 epochs (3 consecutive non-improvements). Best model = epoch 0.

---

## Baseline Results

Reference point: In the ASVspoof 2021 challenge (deepfake track), LFCC-LCNN achieved **23.48%** EER as an organizer-provided baseline (B03) on the 2021 eval set ([Yamagishi et al., ASVspoof 2021 challenge summary](https://arxiv.org/abs/2210.02437)).

### Clean Evaluation

- Evaluated on full **ASVspoof5** eval set (680,774 trials: 138,688 bonafide, 542,086 spoof)
- Tested with and without pre-trained weights

| Condition | EER |
|-----------|-----|
| Baseline (plain BCELoss/no class weighting) | 43.84% |
| Baseline (pre-trained 2019 weights) | **38.72%** |

### Augmented Evaluation

- Pre-trained 2019 weights model evaluated on augmented ASVspoof5 eval set
- Augmentations applied to waveforms before LFCC extraction
- Noise conditions evaluated with all 12 clips combined (6 training + 6 unseen from `noise_clips/eval_train_combined/`) for the most representative real-world estimate
- Comparing against baseline EER = 38.72%

**SNR (Signal-to-Noise Ratio)** — ratio of speech power to noise power in decibels:
- Higher SNR = cleaner audio
- Clean eval files: ~37–56dB SNR (median ~46dB), estimated from frame energy analysis on 1,000 random samples (n=976 after excluding short files)
- **20dB** = speech is 100x louder than noise (barely noticeable)
- **10dB** = speech is 10x louder than noise (clearly audible)
- Typical VoIP quality: 15–25dB SNR (office/home), 10–15dB (speakerphone/noisy)
- The 20dB and 10dB test points bracket the realistic operating range for enterprise voice calls

FAR and FRR computed at the clean baseline's EER threshold (**1.2149**) as a fixed reference point. This reveals how each distortion shifts the score distributions relative to where the model was calibrated on clean data.

| Condition | EER | ΔEER | FAR† | FRR† | Failure Mode | Bonafide Mean | Bonafide Std | Spoof Mean | Spoof Std |
|-----------|-----|------|------|------|--------------|---------------|--------------|------------|-----------|
| Clean | 38.72% | — | 38.72% | 38.72% | balanced | 2.17 | 3.85 | -1.63 | 6.98 |
| Noise: ambient 20dB | 44.11% | +5.39 | 47.33% | 40.15% | spoof collapse | 2.36 | 4.50 | 0.15 | 6.14 |
| Noise: ambient 10dB | 47.98% | +9.26 | 55.73% | 39.37% | spoof collapse | 2.61 | 4.89 | 1.45 | 5.77 |
| Noise: babble 20dB | 43.21% | +4.49 | 57.34% | 26.63% | spoof collapse | 3.04 | 3.31 | 1.29 | 4.71 |
| Noise: babble 10dB | 46.60% | +7.88 | **78.73%** | 14.35% | spoof collapse | 3.92 | 2.83 | 3.29 | 3.44 |
| Short clip: 5s* | 38.36% | -0.36 | 33.01% | 46.51% | bonafide collapse | 1.35 | 5.49 | -4.06 | 9.43 |
| Short clip: 3s* | 39.09% | +0.37 | 32.86% | **47.88%** | bonafide collapse | 1.11 | **6.41** | -4.39 | **10.09** |
| Codec: MP3 64kbps | 38.88% | +0.16 | 36.18% | 43.36% | bonafide collapse | 1.73 | 4.03 | -2.19 | 7.27 |
| Codec: Opus 32kbps | 38.36% | -0.36 | 37.24% | 40.12% | balanced | 2.06 | 3.95 | -1.86 | 7.01 |

† FAR/FRR at the clean baseline's EER threshold (fixed reference).

\*Short clip conditions center-crop files longer than the target duration; files already shorter are scored at their original length. Max 5s: 570k truncated, 110k as-is. Max 3s: 679k truncated, 2k as-is.

#### Combined Augmentation Conditions

Individual distortions are tested in isolation above, but real telephony and VoIP channels present multiple distortions simultaneously. The following combinations simulate realistic deployment scenarios using 20dB SNR (representative of typical office/VoIP quality):

| Combined Conditions | EER | ΔEER | FAR† | FRR† | Failure Mode | Bonafide Mean | Bonafide Std | Spoof Mean | Spoof Std |
|-----------|-----|------|------|------|--------------|---------------|--------------|------------|-----------|
| Noise + Codec (babble 20dB + MP3) | 43.53% | +4.81 | 55.25% | 29.45% | spoof collapse | 2.80 | 3.44 | 1.04 | 4.89 |
| Noise + Short clip (babble 20dB + 3s) | 43.39% | +4.67 | 46.47% | 39.95% | spoof collapse | 2.02 | 5.45 | -0.48 | 6.98 |
| Codec + Short clip (MP3 + 3s) | 39.25% | +0.53 | 31.36% | 50.42% | bonafide collapse | 0.71 | 6.50 | -4.80 | 10.15 |

**Dominant failure modes:**

1. **Noise → spoof collapse (FAR >> FRR).** All four noise conditions cause spoof scores to shift upward past the decision threshold. The model increasingly classifies spoof samples as bonafide. Babble noise at 10dB is the most severe case: FAR reaches 78.73% while FRR drops to 14.35%. This indicates additive noise masks the spectral artifacts the model relies on to detect spoofing, making spoof audio indistinguishable from bonafide in the LFCC feature space.

2. **Short clips → bonafide collapse (FRR >> FAR).** Truncation has the opposite effect: bonafide scores shift slightly downward, causing the model to reject real audio. The effect is mild (FRR ~47% vs FAR ~33%) and EER barely changes, suggesting the model's discrimination ability is preserved but the score distribution shifts.

3. **Codec compression → negligible impact.** Both MP3 64kbps and Opus 32kbps produce score distributions nearly identical to clean (ΔEER < 0.4pp). Opus is classified as balanced; MP3 shows a very mild bonafide collapse (FRR 43% vs FAR 36%) but the shift is minor. Lossy compression at these bitrates does not destroy the spectral artifacts the model relies on for detection.

**Confidence instability:**

- **Short clips produce the widest score spread.** Spoof Std jumps from 6.98 (clean) to 10.09 (3s), and Bonafide Std from 3.85 to 6.41. The model becomes inconsistent when given less temporal context — individual scores vary widely within each class, making threshold calibration unreliable.
- **Noise compresses score distributions but shifts them.** Babble 10dB collapses Spoof Std from 6.98 to 3.44 and Bonafide Std from 3.85 to 2.83. The scores cluster tightly, but both clusters land above the threshold (Spoof Mean = 3.29, Bonafide Mean = 3.92), explaining the severe spoof collapse. The model is confident but wrong — it consistently scores noisy spoof audio as bonafide.

**GradCAM saliency observations:**

GradCAM heatmaps were generated for all conditions to visualize which parts of the audio the model focuses on. Cross-model analysis (baseline, noise-augmented, OC-Softmax) in `04_completed_evals/ocsoftmax_v1/saliency_maps/README.md`; baseline-only viewer guide in `04_completed_evals/clean_weighted/saliency_maps/README.md`.

- **The model uses two distinct attention strategies.** For bonafide classification, it looks broadly across the base voice features and their transitions (LFCC + Delta bands). For spoof detection, it focuses on sparse, localized hotspots in the fine-grained timing band (Delta-delta), where subtle synthesis artifacts are most visible.
- **Noise redirects the model's attention from the right features to the wrong ones.** Under babble 10dB, the focused hotspots used for spoof detection disappear entirely, replaced by the broad pattern the model associates with bonafide audio. The model doesn't just lose confidence; it actively reads noisy spoof audio as natural speech.
- **Flipped examples confirm this.** A spoof sample correctly detected on clean (score -33.5) was misclassified under babble 10dB (score +10.9), and a bonafide sample correctly classified on clean (score +12.7) was misclassified under babble 10dB (score -0.9). Saliency maps show noise redirecting the model's attention in both directions. The reverse also occurs: samples misclassified on clean can be accidentally corrected by noise. Maps in `04_completed_evals/clean_weighted/saliency_maps/flipped_examples/` (clean-weighted baseline) and `04_completed_evals/ocsoftmax_v1/saliency_maps/flipped_examples/` (OC-Softmax).
- **Short clips amplify existing patterns.** The model focuses on the same features as full-length audio but more intensely, explaining the wider score spread without a large EER change.
- **Codec compression is invisible.** MP3 and Opus saliency maps are indistinguishable from clean baseline maps.

**Combined condition observations:**

- **Codec compression is invisible in every combination.** Noise + Codec (42.75%) is near-identical to noise alone (42.35%), and Codec + Short clip (39.25%) is near-identical to short clip alone (39.09%). Adding MP3 64kbps compression on top of any other distortion does not measurably change performance. The audio features the model relies on survive lossy compression regardless of what other distortions are present.
- **Noise and truncation compound rather than cancel.** Noise + Short clip (43.44%) is worse than either alone (babble 20dB: 42.35%, short 3s: 39.09%). Despite opposing failure modes (noise → spoof collapse, truncation → bonafide collapse), noise dominates the combined result (FAR 60.56% vs FRR 25.73% → spoof collapse). Truncation partially counteracts by pulling FRR upward and widening score distributions, but the result is a degraded version of noise-only failure rather than a new failure pattern.

---

## Enhancement Strategy

Primary goal: address noise-induced failures where up to 91% of fake audio passes through undetected (babble noise at 10dB), the dominant and most operationally dangerous failure mode.

### Priority 1: Noise-Augmented Training

**Target:** Fake audio passing through under noisy conditions

Saliency analysis (GradCAM) shows that noise hides the subtle timing patterns the model uses to detect fakes, causing it to misread noisy fake audio as real. Training with noisy samples forces the model to find detection cues that survive background noise.

**Setup:**
- Fine-tuned from pre-trained 2019 LA weights; training script: `06_train_noise_aug.sh`
- **Per-epoch randomness:** seeded by `MD5(filename + epoch)` for reproducibility
- **Training noise:** `noise_clips/train/` — 3 ambient (MUSAN) + 3 babble (Freesound). Use `--noise-subdir train`.
- **Eval noise:** `noise_clips/eval_train_combined/` — all 12 clips combined for most representative estimate.

**Iterative approach:** Across v1–v5, we tested noise clip complexity, learning rate, regularization, warmup, curriculum, and label smoothing. Full comparison table and per-version details in `ENHANCEMENT_ITERATIONS.md`.


### Priority 2: Weighted Loss Function — Abandoned

**Target:** Training data imbalance (8.7x more fake samples than real)

Tested giving more weight to real-audio errors during training (pos_weight=2.0). Result: **worse than the noise-augmented model across all conditions** (+1.2–3.6pp). Noise augmentation had already shifted the model's bias from letting fakes through to occasionally rejecting real calls — extra weighting pushed this too far. Full results in `ENHANCEMENT_ITERATIONS.md`.

### Priority 3: Score Calibration / Condition-Aware Thresholds — Minimal Gain

**Target:** Adjusting alert thresholds based on audio quality conditions

- Different distortions shift scores in different directions — noise pushes scores up, short clips spread them out. A single fixed threshold can't account for all conditions.
- Applied statistical normalization per condition to align score ranges
- Post-processing only — no model changes, no retraining
- **Result:** Even with perfect knowledge of the condition, EER improved by only 0.12pp (39.09% → 38.97%). The noise-augmented model already produces consistent score ranges across conditions, leaving almost no room for post-processing improvement.
- Script: `02_evaluation_scripts/calibrate_scores.py` (calibrated score outputs not committed)

### Priority 4: Mixup Augmentation

**Target:** Overfitting — all noise-augmented iterations overfit within 1 epoch (best model always epoch 0 or 1)

Mixup blends pairs of training samples together (e.g., 70% of sample A + 30% of sample B), forcing the model to handle ambiguous inputs rather than memorizing individual examples. STC (Speech Technology Center) credited mixup as critical for their ASVspoof 2021 system, which reached 15.64% EER on the DF track using mixup plus codec simulation, a significant improvement over the 23.48% organizer-provided baseline ([Tomilov et al., ASVspoof 2021 Workshop](https://www.isca-archive.org/asvspoof_2021/tomilov21_asvspoof.pdf)).

- Applied at the LFCC feature level during training, no architecture change
- Controlled by `--mixup-alpha` (blending intensity; α=0.2 is standard)

### Priority 5: OC-Softmax Loss

**Target:** Generalization to unseen spoofing attacks

The current BCELoss treats spoofing detection as symmetric binary classification ("is this bonafide or spoof?"), which learns spoof-specific patterns from training data and fails on novel attacks. OC-Softmax (one-class softmax) reframes the problem: learn a compact boundary around bonafide speech, push everything else away. Novel attacks fail because they don't match bonafide — not because they match known spoof.

- Up to 33% relative EER reduction on ASVspoof 2019 LA ([Zhang et al., IEEE SPL 2021](https://arxiv.org/pdf/2010.13995))
- Outputs a -1 to +1 similarity score (how closely the audio matches the learned "real speech" template) instead of a raw number
- Same model architecture, only the loss function and output layer change
- Reference implementation: `yzyouzhang/AIR-ASVspoof` on GitHub

### Priority 6: FIR Codec Simulation in Training — Not Pursued

**Target:** Codec-induced distortions and channel variability

Applies random frequency filters during training to simulate the distortion patterns of various audio codecs (MP3, Opus, AMR, telephony). Not implemented because eval-time codec conditions showed negligible impact on the baseline (ΔEER < 0.4pp for MP3 64kbps and Opus 32kbps), so training-time codec simulation was deprioritized. Worth revisiting if lower-bitrate or telephony-specific codecs are introduced.

---

## Enhanced Results

### Noise-Augmented Model

- **Training:** 50% noise augmentation (ambient/babble, SNR 10–25dB), 50% clean
- **Settings:** LR 0.0003, warmup 1000 steps, no L2, augment_prob 0.5
- **Noise clips:** `noise_clips/train/` — 3 ambient (MUSAN) + 3 babble (Freesound)
- **Best model:** epoch 0 (train loss 0.1007, dev loss 0.9357)
- **Training script:** `06_train_noise_aug.sh`

| Condition | Baseline EER | Noise-Aug EER | ΔEER | FAR† | FRR† | Failure Mode | Bonafide Mean | Bonafide Std | Spoof Mean | Spoof Std |
|-----------|-------------|---------------|------|------|------|--------------|---------------|--------------|------------|-----------|
| Clean | 38.72% | **37.99%** | -0.73pp | 37.99% | 37.99% | balanced | -1.32 | 3.45 | -4.52 | 5.76 |
| Noise: ambient 20dB | 44.11% | **39.30%** | -4.81pp | 33.56% | 47.41% | bonafide collapse | -2.21 | 3.56 | -4.82 | 5.24 |
| Noise: ambient 10dB | 47.98% | **42.12%** | -5.86pp | 36.97% | 48.10% | bonafide collapse | -2.28 | 3.52 | -4.13 | 4.60 |
| Noise: babble 20dB | 43.21% | **38.65%** | -4.56pp | 32.19% | 47.15% | bonafide collapse | -2.17 | 3.17 | -4.61 | 4.50 |
| Noise: babble 10dB | 46.60% | **40.10%** | -6.50pp | 43.41% | 36.92% | spoof collapse | -1.58 | 2.63 | -3.11 | 3.14 |
| Short clip: 3s | 39.09% | **38.79%** | -0.30pp | 36.97% | 41.32% | bonafide collapse | -1.26 | 5.78 | -6.00 | 8.57 |
| Short clip: 5s | 38.36% | **38.03%** | -0.33pp | 37.06% | 39.48% | balanced | -1.03 | 5.19 | -5.76 | 8.10 |
| Codec: MP3 64kbps | 38.88% | **37.95%** | -0.93pp | 35.41% | 42.26% | bonafide collapse | -1.63 | 3.65 | -4.99 | 6.02 |
| Codec: Opus 32kbps | 38.36% | **38.26%** | -0.10pp | 36.64% | 40.95% | bonafide collapse | -1.54 | 3.51 | -4.70 | 5.73 |
| Combined: MP3+babble 20dB | 43.53% | **38.92%** | -4.61pp | 29.95% | 50.82% | bonafide collapse | -2.46 | 3.32 | -4.97 | 4.69 |
| Combined: MP3+short 3s | 39.25% | **38.68%** | -0.57pp | 34.96% | 44.00% | bonafide collapse | -1.65 | 5.87 | -6.44 | 8.60 |
| Combined: babble 20dB+short 3s | 43.39% | **39.75%** | -3.64pp | 30.89% | 50.19% | bonafide collapse | -2.70 | 5.38 | -6.22 | 6.88 |

† FAR/FRR at noise-augmented model's clean EER threshold (-2.3163, fixed reference).

#### Training Takeaways

- The LFCC-LCNN operates in a narrow fine-tuning regime — one epoch of augmented training is optimal; further epochs overfit regardless of regularization.
- LR warmup was the single most impactful technique, improving epoch 0 quality without requiring longer training.
- Across 5 iterations, the key factors were noise clip complexity (simple vs realistic), learning rate, and warmup. L2 regularization, curriculum scheduling, and label smoothing provided no benefit or actively hurt.

#### Observations

- **All conditions improved**, not just noise. Noise augmentation acted as a general regularizer, with the largest gains on noise conditions (4–7pp) and smaller improvements on clean, codec, and short clips.
- **Failure mode reversed under noise.** Baseline let fake audio through (spoof collapse). Now the model errs toward rejecting real audio instead (bonafide collapse). In a security context, this is preferable — false rejections are recoverable, false acceptances are not.
- **Babble 10dB still hardest** but FAR dropped from 79% to 43%. At this noise level speech and noise are nearly equal in power, a fundamental limit for frequency-based detection.
- **Score distributions shifted down.** Both bonafide and spoof means moved negative while maintaining similar separation, reflected in the EER threshold shifting from +1.21 to -2.32.
- **Short clips still have wide score spread.** Noise augmentation doesn't help here — this is an insufficient temporal context problem. Score calibration was tested separately (Priority 3) but did not meaningfully reduce the spread.
- **Combined conditions show no new failure modes.** Distortions compound additively, not synergistically.

### Weighted Loss — Abandoned

Tested weighted BCELoss (pos_weight=2.0) on top of the noise-augmented model to address 1:8.7 bonafide:spoof class imbalance. **Worse across all conditions (+1.2–3.6pp).** Noise augmentation already reversed spoof collapse to bonafide collapse; upweighting bonafide loss amplified this further. Full per-condition comparison in `ENHANCEMENT_ITERATIONS.md`.

### Score Calibration (Z-Normalization) — Minimal Gain

Per-condition Z-normalization (`z = (score - pooled_mean) / pooled_std`) applied as post-processing to align score distributions across conditions. No model changes. Script: `02_evaluation_scripts/calibrate_scores.py` (calibrated score outputs not committed).

| Metric | Raw | Calibrated |
|--------|-----|------------|
| Mixed EER (all 12 conditions pooled) | 39.09% | **38.97%** |
| Per-condition EER | unchanged | unchanged (Z-norm preserves score ordering) |

**Improvement: +0.12pp.** The noise-augmented model already produces well-aligned score distributions across conditions, leaving negligible headroom for post-processing calibration. Automatic condition detection (e.g., SNR estimation) was not implemented: the best-case ceiling was too low to justify. A per-condition threshold table can be regenerated via `calibrate_scores.py` for operational use.

### Mixup Augmentation — Abandoned

LFCC-level mixup (α=0.2) on top of noise-augmented model settings. **Worse across all conditions (+2–3pp).** Blending audio features together creates unrealistic training inputs that hurt detection accuracy on real data despite appearing to improve on the dev set. Full results in `ENHANCEMENT_ITERATIONS.md`.

### Enhanced Model (OC-Softmax + Noise Augmentation)

OC-Softmax loss (Priority 5) layered on top of the v5 noise-augmented training pipeline. See Priority 5 above for the loss function rationale.

- **Training:** From 2019 LA pre-trained weights, LR 0.0003, warmup 1000 steps, 50% noise augmentation (ambient/babble, SNR 10–25dB), emb_dim=64, r_real=0.9, r_fake=0.2, alpha=20.0
- **Best model:** epoch 4 — first model to improve past epoch 0
- **Training script:** `08_train_ocsoftmax.sh`

| Condition | Baseline EER | Noise-Aug EER | Enhanced EER | Δ vs Noise-Aug |
|-----------|-------------|-------------------|---------------|---------|
| Clean | 38.72% | 37.99% | 38.03% | +0.04pp |
| Noise: ambient 20dB | 44.11% | 39.30% | **38.30%** | **-1.00pp** |
| Noise: ambient 10dB | 47.98% | 42.12% | **40.98%** | **-1.14pp** |
| Noise: babble 20dB | 43.21% | 38.65% | **38.23%** | **-0.42pp** |
| Noise: babble 10dB | 46.60% | **40.10%** | 41.36% | +1.26pp |
| Short clip: 3s | 39.09% | 38.79% | **38.70%** | **-0.09pp** |
| Short clip: 5s | 38.36% | 38.03% | **37.93%** | **-0.10pp** |
| Codec: MP3 64kbps | 38.88% | 37.95% | **37.86%** | **-0.09pp** |
| Codec: Opus 32kbps | 38.36% | **38.26%** | 38.39% | +0.13pp |
| Combined: MP3+babble 20dB | 43.53% | 38.92% | **38.29%** | **-0.63pp** |
| Combined: babble 20dB+short 3s | 43.39% | 39.75% | **39.46%** | **-0.29pp** |
| Combined: MP3+short 3s | 39.25% | 38.68% | **38.53%** | **-0.15pp** |

**Improved 9 of 12 conditions vs noise-augmented model.** Best gains on moderate noise (ambient: -1.0 to -1.1pp). Only babble 10dB regressed (+1.26pp) — the extreme case where speech and noise have nearly equal power, a fundamental limit for spectral features.

FAR/FRR at OC-Softmax clean EER threshold (0.9095, cosine similarity scale):

| Condition | EER | FAR† | FRR† | Failure Mode | Bonafide Mean | Bonafide Std | Spoof Mean | Spoof Std |
|-----------|-----|------|------|--------------|---------------|--------------|------------|-----------|
| Clean | 38.03% | 38.03% | 38.03% | balanced | 0.786 | 0.334 | 0.411 | 0.569 |
| Noise: ambient 20dB | 38.30% | 31.20% | 50.84% | bonafide collapse | 0.665 | 0.414 | 0.332 | 0.562 |
| Noise: ambient 10dB | 40.98% | 29.20% | 57.64% | bonafide collapse | 0.607 | 0.425 | 0.366 | 0.526 |
| Noise: babble 20dB | 38.23% | 28.37% | 55.41% | bonafide collapse | 0.631 | 0.420 | 0.311 | 0.549 |
| Noise: babble 10dB | 41.36% | 29.60% | 56.97% | bonafide collapse | 0.648 | 0.385 | 0.443 | 0.475 |
| Short clip: 3s | 38.70% | 33.66% | 46.80% | bonafide collapse | 0.655 | 0.462 | 0.292 | 0.622 |
| Short clip: 5s | 37.93% | 34.49% | 44.42% | bonafide collapse | 0.699 | 0.429 | 0.304 | 0.619 |
| Codec: MP3 64kbps | 37.86% | 36.13% | 41.05% | bonafide collapse | 0.759 | 0.358 | 0.377 | 0.578 |
| Codec: Opus 32kbps | 38.39% | 37.26% | 40.51% | bonafide collapse | 0.765 | 0.352 | 0.397 | 0.572 |
| Combined: MP3+babble 20dB | 38.29% | 26.98% | 58.17% | bonafide collapse | 0.596 | 0.439 | 0.279 | 0.555 |
| Combined: babble 20dB+short 3s | 39.46% | 24.83% | 62.11% | bonafide collapse | 0.463 | 0.536 | 0.171 | 0.598 |
| Combined: MP3+short 3s | 38.53% | 32.20% | 48.89% | bonafide collapse | 0.633 | 0.474 | 0.270 | 0.623 |

† FAR/FRR at clean EER threshold (0.9095).

#### Observations

- **OC-Softmax enabled the longest useful training.** Best model at epoch 4, compared to epoch 0 for most BCE variants. OC-Softmax achieved multi-epoch improvement without sacrificing detection quality.
- **Starting from the noise-augmented weights was counterproductive.** A second run (v2) fine-tuning from noise-augmented model weights produced worse results (40.1% clean EER) — those features are optimized for BCE's single-score output and don't transfer well to the 64-dimensional similarity space. The general 2019 pre-trained weights gave OC-Softmax room to reshape features over multiple epochs.
- **Clean EER unchanged.** OC-Softmax's gains are concentrated on degraded conditions, consistent with its design goal of better generalization when audio quality varies.
- **Uniformly bonafide collapse.** Unlike the noise-augmented model (which had spoof collapse at babble 10dB), OC-Softmax produces bonafide collapse across all degraded conditions. The model consistently errs toward rejecting real audio rather than accepting fake audio — a safer failure mode for security applications.
- **Scores on cosine similarity scale (0–1).** Bonafide mean ~0.79, spoof mean ~0.41 on clean. Noise pushes both scores toward the middle: babble 10dB narrows the gap to 0.65 vs 0.44 (separation of 0.20 vs 0.38 on clean), explaining the EER regression at extreme noise.
- **Babble 10dB: spoof mean (0.443) nearly equals bonafide mean (0.648).** The r_fake=0.2 margin (how aggressively the model pushes fake audio away from the bonafide center) may be too lenient for this condition — fake audio under heavy noise lands close to the bonafide center. A tighter r_fake (0.5) was tested in v3 but did not improve on v1 overall.
- **OC-Softmax compressed both score distributions dramatically.** Bonafide std dropped ~11x (3.85 → 0.334) and spoof std dropped ~12x (6.98 → 0.569) compared to the baseline. The baseline's wide spoof std (6.98) was the root cause of spoof collapse under noise — scores were spread so widely that distortion easily pushed the tail past the threshold. OC-Softmax's bounded [-1, +1] cosine range and angular margin loss structurally prevent this, making alert tier thresholds stable across conditions.

---

## Metrics

- **EER** (Equal Error Rate) — the threshold where FAR equals FRR. Lower is better. Primary metric for detection performance.
- **ΔEER** — difference between a distorted condition's EER and clean baseline EER. Shows how much a distortion degrades performance.
- **FAR** (False Acceptance Rate) — fraction of spoof samples incorrectly accepted as bonafide (scored above the threshold). High FAR = the model is letting fake audio through.
- **FRR** (False Rejection Rate) — fraction of bonafide samples incorrectly rejected as spoof (scored below the threshold). High FRR = the model is blocking real audio.
- **Failure Mode** — when FAR and FRR are computed at a fixed threshold, the dominant direction reveals the type of failure: "spoof collapse" (FAR >> FRR, fake audio passes as real) or "bonafide collapse" (FRR >> FAR, real audio rejected as fake).
- **Bonafide Mean / Bonafide Std** — average and standard deviation of model scores for real (bonafide) audio. Shows where the bonafide score distribution is centered and how spread out it is.
- **Spoof Mean / Spoof Std** — average and standard deviation of model scores for fake (spoof) audio. A large Std means the model is inconsistent within that group, making threshold calibration difficult.
- **Risk scoring**: Raw scores mapped to three-tier severity levels (CRITICAL / HIGH / MONITOR) anchored to precision operating points (99% / 95% / 90%). See Risk Scoring section.

---

## Risk Scoring

Risk threshold trade-off analysis in `04_completed_evals/risk_threshold_analysis.md`. Alert simulation via `01_project/baseline_DF/alert_demo.py`.

### Operational Model

The system operates as a **deepfake tripwire** — it flags likely deepfakes for investigation but cannot positively confirm audio as safe. Three severity tiers are anchored to precision operating points, each mapping to a distinct enterprise SOC workflow.

| Tier | Precision | Meaning | Action |
|------|-----------|---------|--------|
| **CRITICAL** | 99% | Very high confidence deepfake | Auto-escalate to SOC, terminate/flag call |
| **HIGH** | 95% | Strong deepfake indicators | Queue for analyst review within SLA |
| **MONITOR** | 90% | Moderate deepfake indicators | Log for trend analysis, no immediate action |
| **UNCONFIRMED** | — | No detection triggered | No action — not verified safe, just not flagged |

### Three-Tier Thresholds (Precision-Anchored)

Thresholds are derived empirically from precision/detection rate analysis on the ASVspoof5 eval set (680,774 trials). Each tier boundary is set at the score where the corresponding precision target is met. The trade-off is remarkably linear — each 1% of precision costs about 3-4% of detection rate, with no natural elbow.

**Enhanced Model (recommended):**

| Tier | Threshold | Precision | Detection Rate | FPR | False Alarms / 1K calls |
|------|-----------|-----------|----------------|-----|------------------------|
| **CRITICAL** | score < -0.314 | 99% | 15.5% | 0.6% | ~6 |
| **HIGH** | score < 0.096 | 95% | 36.8% | 7.6% | ~76 |
| **MONITOR** | score < 0.690 | 90% | 52.9% | 23.0% | ~230 |

**Noise-Augmented Model:**

| Tier | Threshold | Precision | Detection Rate | FPR | False Alarms / 1K calls |
|------|-----------|-----------|--------|-----|------------------------|
| **CRITICAL** | score < -11.48 | 99% | 11.2% | 0.4% | ~4 |
| **HIGH** | score < -6.35 | 95% | 33.2% | 6.8% | ~68 |
| **MONITOR** | score < -3.78 | 90% | 51.1% | 22.2% | ~222 |

- **CRITICAL (99%)**: very few false alarms (~6/1K), safe for automated response. Catches ~1 in 6 deepfakes — only the most obvious ones
- **HIGH (95%)**: 19 out of 20 alerts are real deepfakes. Manageable analyst queue (~76/1K). Catches ~1 in 3 deepfakes
- **MONITOR (90%)**: informational only — no disruptive action. Higher FPR (~23%) is acceptable because this tier only logs for trend analysis. Catches ~1 in 2 deepfakes
- **UNCONFIRMED**: absence of alert means "not detected," not "confirmed safe." At 38% EER, the bonafide/spoof distributions overlap too heavily for a positive safety determination

### Why No "Safe/Pass" Tier

At ~38% EER with a 1:3.9 bonafide:spoof ratio in the eval set, the bonafide and spoof score distributions overlap heavily. Even in the most bonafide-leaning score region, 71% of trials are actually spoof. A "verified safe" label would be misleading — the absence of an alert means "not detected as deepfake," not "confirmed as real."

---

## Final Enhanced Model

The **enhanced model** (OC-Softmax loss layered on top of the v5 noise-augmented training pipeline) is selected for enterprise deployment over the noise-augmented model alone, based on robustness, score geometry, and operational simplicity.

### Head-to-Head: Enhanced vs Noise-Augmented

| Metric | Enhanced | Noise-Augmented |
|--------|---------------|--------------|
| Clean EER | 38.03% | 37.99% |
| Score range | [-1, +1] (bounded cosine) | ~[-15, +10] (unbounded logit) |
| Bonafide mean / std | 0.786 / 0.334 | -1.317 / 3.448 |
| Spoof mean / std | 0.411 / 0.569 | -4.518 / 5.763 |

### Robustness Comparison (ΔEER from clean baseline)

| Condition | Enhanced ΔEER | NA ΔEER | Winner |
|-----------|---------|---------|--------|
| noise_ambient_10dB | +2.95 | +4.13 | Enhanced |
| noise_ambient_20dB | +0.27 | +1.31 | Enhanced |
| noise_babble_10dB | +3.33 | +2.11 | NA |
| noise_babble_20dB | +0.20 | +0.66 | Enhanced |
| mp3+babble_20dB | +0.26 | +0.93 | Enhanced |
| babble_20dB+short_3s | +1.43 | +1.76 | Enhanced |
| Codecs / short clips | ~tied | ~tied | — |

The enhanced model wins 5 conditions (especially ambient noise — the dominant enterprise telephony degradation), noise-augmented alone wins 1 (babble 10dB), 6 tied.

### Why the Enhanced Model for Enterprise

| Factor | Assessment |
|--------|------------|
| **Threshold interpretability** | Output is a -1 to +1 similarity score — analysts can understand it at a glance |
| **Threshold stability** | No extreme outlier values, so thresholds behave consistently across conditions |
| **Calibration overhead** | Scores are usable directly — no normalization step required before applying thresholds |
| **Bonafide cluster tightness** | Real calls cluster tightly (std 0.334 vs 3.448), so fewer end up in the uncertain zone between tiers |
| **Ambient noise robustness** | 1.2pp less degradation under ambient noise — the most common distortion in enterprise calls |

The only scenario favoring noise-augmented alone is heavy multi-speaker background noise (call centers, trading floors), where its ΔEER advantage is 1.2pp. For general enterprise telephony, the enhanced model provides better robustness with simpler deployment.

---

## AI Assistance

This project used [Claude Code](https://claude.ai/code) (Anthropic) throughout the research, planning, and implementation lifecycle, with a human in the loop at every step: directing research priorities, reviewing implementation plans, validating results, and making all architectural and deployment decisions. Model assistance came from Claude Opus and Claude Sonnet (Anthropic's Claude 4 family).

The upstream LFCC-LCNN framework under `core_modules/` and `core_scripts/` is the ASVspoof 2021 baseline by Xin Wang (National Institute of Informatics, NII); AI involvement in upstream files was limited to the targeted patches called out under *Implementation* below.

**Research & Planning:**
- Literature review of LFCC-LCNN improvements, loss functions (OC-Softmax, P2SGrad, A-Softmax), augmentation techniques (RawBoost, mixup, FFM), and SSL front-ends
- Enhancement strategy development and prioritization based on published evidence and architecture constraints

**Implementation:**

*Patches to upstream code (from the NII / Xin Wang baseline):*
- `01_project/baseline_DF/model.py` — ASVspoof5 protocol parsing (column index), `forward()` signature update to accept `fileinfo`, batch inference fix (per-file scoring), OC-Softmax / SE attention / mixup / frequency masking flags
- `01_project/baseline_DF/main.py` — dataset path updates for ASVspoof5
- `01_project/baseline_DF/config.py` — portable dataset path resolution via env var / home expansion
- `core_scripts/nn_manager/nn_manager.py` — suppressed spurious "No output saved" warning; OC-Softmax partial weight-loading support

*AI-authored modules and scripts:*
- `01_project/baseline_DF/augment.py` — on-the-fly noise / short-clip augmentation with deterministic filename-based seeding
- `01_project/baseline_DF/codec_augment.py` — MP3 / Opus codec augmentation batch generator
- `01_project/baseline_DF/gradcam.py` — GradCAM saliency map generation with OC-Softmax support
- `01_project/baseline_DF/alert_demo.py` — three-tier severity alert demo with multi-codec support
- `01_project/baseline_DF/config_auto.py` — auto-generated `.lst` handling for codec eval
- `02_evaluation_scripts/evaluator.py` — EER computation against ASVspoof5 protocols
- `02_evaluation_scripts/failure_analysis.py` — per-condition FAR / FRR, failure mode classification, score distribution statistics
- `02_evaluation_scripts/calibrate_scores.py` — Z-normalization score calibration
- Training and evaluation shell scripts: `01_project/baseline_DF/01_train.sh` through `08_train_ocsoftmax.sh`

**Analysis & Documentation:**
- This `README.md` — project overview, ASVspoof5 adaptations, enhancement results, and operational framework
- `ENHANCEMENT_ITERATIONS.md` — per-iteration training experiment tracking
- `04_completed_evals/ocsoftmax_v1/saliency_maps/README.md` — cross-model GradCAM analysis (baseline, noise-augmented, OC-Softmax)
- `04_completed_evals/clean_weighted/saliency_maps/README.md` — baseline-only saliency map viewer guide
- `04_completed_evals/risk_threshold_analysis.md` — risk tier trade-off analysis
- `noise_clips/README.md` — noise clip sourcing and reconstruction instructions
