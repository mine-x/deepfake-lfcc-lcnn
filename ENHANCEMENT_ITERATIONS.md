# Enhancement Iterations

## Overview

The enhancement pipeline was developed in two stages:

1. **Noise-augmented training iterations (v1–v5)** — explored training configurations to address the baseline's spoof collapse under noise. v5 was adopted as the noise-augmented baseline.
2. **Strategy tests on top of v5** — evaluated additional enhancement strategies. OC-Softmax was adopted as the final enhancement; weighted loss and mixup were abandoned. Post-adoption experiments with SE attention blocks and frequency feature masking (FFM) did not yield further gains.

**Final enhanced model:** v5 noise-augmented training + OC-Softmax loss. Clean EER 38.03%, improved 9 of 12 evaluation conditions relative to v5 alone.

Score calibration (listed as one of five enhancement strategies in the report) was evaluated as a post-processing step on v5 outputs and produced only ~0.1 pp improvement, so it is not covered in detail in this document.

## Summary of All Iterations

| Iteration | Configuration | Best Epoch | Clean EER | Status |
|-----------|---------------|------------|-----------|--------|
| Baseline | LFCC-LCNN, 2019 pre-trained weights fine-tuned on ASVspoof5 | 0 | 38.72% | reference |
| v1 | 4 simple noise clips (2 MUSAN + 2 Librivox babble), LR 0.0003 | 1 | **35.75%** | noise-aug exploration |
| v2 | 6 realistic noise clips (3 MUSAN + 3 Freesound babble), LR 0.0003 | 0 | 39.16% | noise-aug exploration |
| v3 | v2 clips + lower LR (0.0001) + L2 regularization | 0 | 38.62% | noise-aug exploration |
| v4 | Warmup + curriculum + label smoothing + SNR 5–30dB | 2 | 41.05% | noise-aug exploration |
| v5 | v2 settings + warmup only | 0 | **37.99%** | **adopted (noise-aug)** |
| Weighted loss | pos_weight=2.0 on top of v5 | 0 | 39.75% | abandoned |
| Mixup | α=0.2 on top of v5 | 0 | 40.91% | abandoned |
| OC-Softmax (from 2019 weights) | v5 + OC-Softmax loss, r_fake=0.2 | 4 | **38.03%** | **adopted (final)** |
| OC-Softmax (from v5 backbone) | Alternative starting point | 0 | 40.12% | abandoned |
| OC-Softmax (r_fake=0.5) | Tighter spoof margin | 3 | 39.62% | abandoned |
| SE attention + OC-Softmax | Stacked on top of adopted OC-Softmax | 1 | 39.21% | abandoned |
| FFM + OC-Softmax | Stacked on top of adopted OC-Softmax | 1 | 38.45% | abandoned |

## EER Comparison Across Noise-Augmented Iterations (v1–v5)

| Condition | Baseline | v1* | v2 | v3 | v4 | v5 |
|-----------|---------|-----|-----|-----|-----|-----|
| Clean | 38.72% | **35.75%** | 39.16% | 38.62% | 41.05% | **37.99%** |
| Ambient 20dB | 44.11% | 40.29%* | 40.22% | 40.52% | 42.87% | **39.30%** |
| Ambient 10dB | 47.98% | 44.34%* | 44.00% | 44.21% | 45.53% | **42.12%** |
| Babble 20dB | 43.21% | 39.07%* | 39.91% | 40.04% | 43.96% | **38.65%** |
| Babble 10dB | 46.60% | 43.08%* | 43.53% | 43.59% | 45.62% | **40.10%** |
| Short 3s | 39.09% | **37.02%** | 40.47% | 39.29% | 41.39% | 38.79% |
| Short 5s | 38.36% | **35.87%** | 39.51% | 38.18% | 40.73% | 38.03% |
| MP3 64kbps | 38.88% | **35.64%** | 39.17% | 38.51% | 40.97% | 37.95% |
| Opus 32kbps | 38.36% | **35.52%** | 39.03% | 38.72% | 41.23% | 38.26% |
| MP3+babble 20dB | — | — | — | — | 44.02% | 38.92% |
| MP3+short 3s | — | — | — | — | 41.42% | 38.68% |
| Babble 20dB+short 3s | — | — | — | — | 44.51% | 39.75% |

*All versions evaluated with the same 12 noise clips. For v1 all 12 were unseen during training (v1 trained with 4 different simple clips). For v2–v5, 6 of the 12 eval clips overlap with training clips.

---

# Part 1: Noise-Augmented Training Iterations (v1–v5)

## v1 — 4 simple noise clips

4 noise clips (2 MUSAN ambient + 2 Librivox 2-voice babble). Best overall results, but Librivox babble not representative of real-world noise.

### Training

| Setting | Value |
|---------|-------|
| LR | 0.0003 |
| L2 | — |
| Augment prob | 0.5 |
| SNR range | 10–25dB |
| Warmup | — |
| Noise clips | 4 (simple) |
| Best epoch | 1 |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | — | — | — |
| 1 | 0.0196 | 1.1042 | Yes |
| 4 | — | — | Early stop |

### Failure Analysis

| Condition | EER | FAR | FRR | Failure Mode |
|-----------|-----|-----|-----|--------------|
| Clean | 35.75% | 35.75% | 35.75% | balanced |
| Ambient 20dB | 38.65% | 32.95% | 46.02% | bonafide collapse |
| Ambient 10dB | 42.56% | 36.35% | 50.05% | bonafide collapse |
| Babble 20dB | 38.86% | 32.76% | 46.99% | bonafide collapse |
| Babble 10dB | 41.68% | 46.26% | 36.71% | spoof collapse |
| Short 3s | 37.02% | 42.88% | 28.89% | spoof collapse |
| Short 5s | 35.87% | 44.76% | 23.38% | spoof collapse |
| MP3 64kbps | 35.64% | 35.53% | 35.83% | balanced |
| Opus 32kbps | 35.52% | 35.94% | 34.94% | balanced |

FAR/FRR at clean EER threshold (-1.6781). Baseline showed spoof collapse under all noise (FAR up to 91%). v1 reversed this — noise now causes bonafide collapse instead. Only babble 10dB retains spoof collapse (FAR 46% vs 91% baseline).

### Unseen Noise Generalization

| Condition | Training Noise EER | Unseen Noise EER | Delta |
|-----------|-------------------|-----------------|-------|
| Ambient 20dB | 38.65% | 40.29% | +1.64pp |
| Ambient 10dB | 42.56% | 44.34% | +1.78pp |
| Babble 20dB | 38.86% | 39.07% | +0.21pp |
| Babble 10dB | 41.68% | 43.08% | +1.40pp |

4 training clips generalized well (0.2–1.8pp degradation on unseen noise).

## v2 — Switch to realistic noise (immediate overfitting)

Switched to realistic Freesound babble clips. Overfit at epoch 0 — realistic noise caused destructive weight updates at LR 0.0003.

### Training

| Setting | Value | Change from v1 |
|---------|-------|----------------|
| LR | 0.0003 | — |
| L2 | — | — |
| Augment prob | 0.5 | — |
| SNR range | 10–25dB | — |
| Warmup | — | — |
| Noise clips | 6 (realistic) | +2 clips, Freesound babble |
| Best epoch | 0 | regressed |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.0669 | 0.9455 | Yes |
| 1 | 0.0153 | 1.0901 | No |
| 2 | 0.0106 | 1.2097 | No |
| 3 | 0.0082 | 1.9155 | No |

Clean regressed +0.44pp. Noise improved 3–4pp but less than v1. Short/codec regressed 0.3–1.4pp.

**Why v3:** Lower LR + L2 to slow convergence.

## v3 — Lower LR + L2 (partial fix)

Lower LR and L2 fixed clean regression but still overfit at epoch 0.

### Training

| Setting | Value | Change from v2 |
|---------|-------|----------------|
| LR | 0.0001 | 0.0003 → 0.0001 |
| L2 | 0.0001 | added |
| Augment prob | 0.5 | — |
| SNR range | 10–25dB | — |
| Warmup | — | — |
| Noise clips | 6 (realistic) | — |
| Best epoch | 0 | same |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.0759 | 0.8701 | Yes |
| 1 | 0.0167 | 0.9885 | No |
| 2 | 0.0110 | 1.3470 | No |
| 3 | 0.0078 | 1.5008 | No |

Clean EER 38.62% (-0.10pp vs baseline). Noise 3–4pp improvement. Short/codec regressions mostly fixed.

**Why v4:** Still overfits at epoch 0. Tried increasing augmentation diversity.

## v4 — Warmup + curriculum + label smoothing (too conservative)

Added warmup, curriculum, label smoothing. Trained to epoch 2 (first time past epoch 0), but EER regressed — techniques were too conservative.

### Training

| Setting | Value | Change from v3 |
|---------|-------|----------------|
| LR | 0.0001 | — |
| L2 | 0.0001 | — |
| Augment prob | 0.7 | 0.5 → 0.7 |
| SNR range | 5–30dB | 10–25 → 5–30 |
| Warmup | 1000 steps | added |
| Curriculum | yes (easy→hard) | added |
| Label smoothing | 0.1 | added |
| Noise clips | 6 (realistic) | — |
| Best epoch | 2 | improved |

| Epoch | Train Loss | Dev Loss | Best? | Curriculum Phase |
|-------|------------|----------|-------|-----------------|
| 0 | 0.3364 | 0.6523 | Yes | Easy (20-30dB) |
| 1 | 0.2120 | 0.6368 | Yes | Easy (20-30dB) |
| 2 | 0.2100 | 0.5834 | Yes | Medium (10-30dB) |
| 3 | 0.2070 | 0.6523 | No | Medium (10-30dB) |
| 4 | 0.2069 | 0.6085 | No | Full (5-30dB) |
| 5 | 0.2055 | 0.6760 | No | Full (5-30dB) |

Clean regressed +2.33pp. Noise improvement only 1–2pp (vs 3–4pp for v3). Label smoothing changed the loss landscape — lower dev loss didn't correlate with better EER. Train loss floor raised to ~0.21 (vs ~0.07 without smoothing).

**Why v5:** Label smoothing and curriculum hurt. Tested v2 settings + warmup only.

## v5 — v2 settings + warmup only (adopted)

v2 settings + warmup only. Best model with realistic noise clips. Warmup smoothed epoch 0's early batches, producing a better local minimum.

### Training

| Setting | Value | Change from v2 |
|---------|-------|----------------|
| LR | 0.0003 | — |
| L2 | — | — |
| Augment prob | 0.5 | — |
| SNR range | 10–25dB | — |
| Warmup | 1000 steps | added (only change) |
| Noise clips | 6 (realistic) | — |
| Best epoch | 0 | same |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.1007 | 0.9357 | Yes |
| 1 | 0.0187 | 1.1475 | No |
| 2 | 0.0117 | 1.1786 | No |

Epoch 0 dev loss better than v2 (0.9357 vs 0.9455). Still overfit at epoch 1, but the warmed-up epoch 0 produced significantly better features.

### Failure Analysis

| Condition | EER | FAR | FRR | Failure Mode |
|-----------|-----|-----|-----|--------------|
| Clean | 37.99% | 37.99% | 37.99% | balanced |
| Ambient 20dB | 39.30% | 33.56% | 47.41% | bonafide collapse |
| Ambient 10dB | 42.12% | 36.97% | 48.10% | bonafide collapse |
| Babble 20dB | 38.65% | 32.19% | 47.15% | bonafide collapse |
| Babble 10dB | 40.10% | 43.41% | 36.92% | spoof collapse |
| Short 3s | 38.79% | 36.97% | 41.32% | bonafide collapse |
| Short 5s | 38.03% | 37.06% | 39.48% | balanced |
| MP3 64kbps | 37.95% | 35.41% | 42.26% | bonafide collapse |
| Opus 32kbps | 38.26% | 36.64% | 40.95% | bonafide collapse |

FAR/FRR at clean EER threshold (-2.3163).

**Conclusion:** Warmup was the key missing ingredient for v2. The model still only trains for one useful epoch, but that epoch produces significantly better features than without warmup. Clean EER improved 0.73pp over baseline, noise improved 4.6–6.5pp. v5 adopted as the noise-augmented baseline for subsequent strategy tests.

---

# Part 2: Strategy Tests on Top of v5

The strategies below were tested on the v5 noise-augmented model to evaluate further gains. OC-Softmax was adopted as the final enhancement; the others were abandoned.

## Weighted Loss (abandoned)

Tested weighted BCELoss on top of v5 settings to address 1:8.7 bonafide:spoof class imbalance. Bonafide samples weighted 2x higher in loss computation. Result: worse than v5 across all conditions.

### Training

| Setting | Value | Change from v5 |
|---------|-------|----------------|
| LR | 0.0003 | — |
| Warmup | 1000 steps | — |
| Augment prob | 0.5 | — |
| SNR range | 10–25dB | — |
| pos_weight | 2.0 | added |
| Best epoch | 0 | same |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.1652 | 1.3711 | Yes |
| 1 | 0.0287 | 1.4350 | No |

### EER Comparison

| Condition | v5 EER | Weighted EER | Delta |
|-----------|--------|-------------|-------|
| Clean | **37.99%** | 39.75% | +1.76pp |
| Ambient 20dB | **39.30%** | 40.51% | +1.21pp |
| Ambient 10dB | **42.12%** | 43.88% | +1.76pp |
| Babble 20dB | **38.65%** | 40.30% | +1.65pp |
| Babble 10dB | **40.10%** | 43.65% | +3.55pp |
| Short 3s | **38.79%** | 40.22% | +1.43pp |
| Short 5s | **38.03%** | 39.60% | +1.57pp |
| MP3 64kbps | **37.95%** | 39.64% | +1.69pp |
| Opus 32kbps | **38.26%** | 39.79% | +1.53pp |

**Conclusion:** Weighted loss is counterproductive on top of noise augmentation. v5 already reversed the baseline's spoof collapse to bonafide collapse — upweighting bonafide loss further amplified this imbalance, worsening EER by 1.2–3.6pp across all conditions. The class imbalance in training data (1:8.7) is already effectively addressed by noise augmentation's regularization effect. Moved to mixup test next.

## Mixup Augmentation (abandoned)

Tested LFCC-level mixup on top of v5 noise-augmented settings. Mixup blends pairs of training samples and labels (`x_mix = λ*x1 + (1-λ)*x2`, `y_mix = λ*y1 + (1-λ)*y2`) to regularize against overfitting. Applied at the LFCC spectrogram level before the LCNN conv layers.

### Training

| Setting | Value | Change from v5 |
|---------|-------|----------------|
| LR | 0.0003 | — |
| Warmup | 1000 steps | — |
| Augment prob | 0.5 | — |
| SNR range | 10–25dB | — |
| Mixup alpha | 0.2 | added |
| Best epoch | 0 | same |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.1555 | 0.7113 | Yes |
| 1 | 0.0715 | 0.7842 | No |
| 2 | 0.0606 | 0.7588 | No |
| 3 | 0.0589 | 0.8085 | No |

Dev loss significantly lower than v5 (0.7113 vs 0.9357) but did not translate to better EER. Train loss higher than v5 (0.1555 vs 0.1007) — expected, mixup makes training harder.

### EER Comparison

| Condition | v5 EER | Mixup EER | Delta |
|-----------|--------|-----------|-------|
| Clean | **37.99%** | 40.91% | +2.92pp |
| Ambient 20dB | **39.30%** | 41.53% | +2.23pp |
| Ambient 10dB | **42.12%** | 43.77% | +1.65pp |
| Babble 20dB | **38.65%** | 41.15% | +2.50pp |
| Babble 10dB | **40.10%** | 42.64% | +2.54pp |
| Short 3s | **38.79%** | 41.29% | +2.50pp |
| Short 5s | **38.03%** | 40.72% | +2.69pp |
| MP3 64kbps | **37.95%** | 40.74% | +2.79pp |
| Opus 32kbps | **38.26%** | 41.18% | +2.92pp |
| MP3+babble 20dB | **38.92%** | 41.17% | +2.25pp |
| Babble 20dB+short 3s | **39.75%** | 41.81% | +2.06pp |
| MP3+short 3s | **38.68%** | 41.10% | +2.42pp |

**Conclusion:** Mixup at α=0.2 on LFCC features is consistently worse than v5 by 2–3pp across all conditions. Blending LFCC spectrograms creates unrealistic inputs — a blended bonafide+spoof spectrogram doesn't represent "partially fake" audio, it's an artifact that doesn't exist in nature. The model learns to handle these artificial blends at the expense of real-world discrimination. Lower dev loss (0.71 vs 0.94) reflects better fit to the mixup training objective, not better spoofing detection. STC's success with mixup ([Tomilov et al., 2021](https://www.isca-archive.org/asvspoof_2021/tomilov21_asvspoof.pdf)) may have used waveform-level mixup or different architecture/feature combinations. Moved to OC-Softmax test next.

## OC-Softmax Loss (adopted, final enhancement)

Replaced BCELoss with OC-Softmax ([Zhang et al., IEEE SPL 2021](https://arxiv.org/pdf/2010.13995)), which learns a compact bonafide boundary in embedding space with angular margins. Instead of "is this bonafide or spoof?" (BCE), the model learns "does this match bonafide?" — novel attacks fail because they don't match bonafide, not because they match known spoof.

Architecture change: output layer from `Linear(dim, 1)` → `Linear(dim, 64)`, no sigmoid. OCSoftmax module L2-normalizes embeddings and a learnable center, computes cosine similarity, and applies angular margins (`r_real`, `r_fake`) with a scaling factor (`alpha`).

Three variants were tested. OC-Softmax v1 (from 2019 weights, r_fake=0.2) was adopted as the final model.

### OC-Softmax v1 — from 2019 pre-trained weights (adopted)

| Setting | Value | Change from v5 |
|---------|-------|----------------|
| LR | 0.0003 | — |
| Warmup | 1000 steps | — |
| Augment prob | 0.5 | — |
| SNR range | 10–25dB | — |
| Loss | OC-Softmax | BCELoss → OC-Softmax |
| emb_dim | 64 | 1 → 64 |
| r_real | 0.9 | — |
| r_fake | 0.2 | — |
| alpha | 20.0 | — |
| Pre-trained from | 2019 LA weights | same as v5 |
| Best epoch | 4 | 0 → 4 |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.5096 | 4.2766 | Yes |
| 1 | 0.1459 | 3.9778 | Yes |
| 2 | 0.1080 | 4.2020 | No |
| 3 | 0.0911 | 4.2012 | No |
| 4 | 0.0794 | 3.7246 | Yes |
| 5 | 0.0714 | 4.5549 | No |
| 6 | 0.0637 | 4.9585 | No |
| 7 | — | — | CUDA crash (early stop would have triggered) |

First model to improve past epoch 0. Dev loss not directly comparable to BCE (different loss scale; OC-Softmax uses alpha=20 multiplier).

#### EER Comparison

| Condition | v5 EER | OC-Soft v1 EER | Delta |
|-----------|--------|---------------|-------|
| Clean | 37.99% | 38.03% | +0.04pp |
| Ambient 20dB | 39.30% | **38.30%** | **-1.00pp** |
| Ambient 10dB | 42.12% | **40.98%** | **-1.14pp** |
| Babble 20dB | 38.65% | **38.23%** | **-0.42pp** |
| Babble 10dB | **40.10%** | 41.36% | +1.26pp |
| Short 3s | 38.79% | **38.70%** | **-0.09pp** |
| Short 5s | 38.03% | **37.93%** | **-0.10pp** |
| MP3 64kbps | 37.95% | **37.86%** | **-0.09pp** |
| Opus 32kbps | **38.26%** | 38.39% | +0.13pp |
| MP3+babble 20dB | 38.92% | **38.29%** | **-0.63pp** |
| Babble 20dB+short 3s | 39.75% | **39.46%** | **-0.29pp** |
| MP3+short 3s | 38.68% | **38.53%** | **-0.15pp** |

**Improved 9 of 12 conditions.** Best gains on moderate noise (ambient 10dB: -1.14pp, ambient 20dB: -1.00pp). Only babble 10dB regressed (+1.26pp) — the extreme case where speech and noise are nearly equal power. Clean essentially unchanged.

### OC-Softmax v2 — from v5 backbone (abandoned)

Tested whether starting from v5's ASVspoof5-adapted backbone would give a better starting point than the generic 2019 weights.

| Setting | Value | Change from v1 |
|---------|-------|----------------|
| Pre-trained from | v5 noise-aug model | 2019 → v5 |
| All other settings | same | — |
| Best epoch | 0 | 4 → 0 |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.0684 | 3.9648 | Yes |
| 1 | 0.0613 | 3.9780 | No |
| 2 | 0.0556 | 4.5858 | No |
| 3 | 0.0495 | 4.0915 | No |

| Condition | v5 EER | OC-Soft v2 EER | Delta |
|-----------|--------|---------------|-------|
| Clean | **37.99%** | 40.12% | +2.13pp |
| Ambient 20dB | **39.30%** | 40.47% | +1.17pp |
| Ambient 10dB | **42.12%** | 43.00% | +0.88pp |
| Babble 20dB | **38.65%** | 40.82% | +2.17pp |
| Babble 10dB | **40.10%** | 44.06% | +3.96pp |
| Short 3s | **38.79%** | 40.30% | +1.51pp |
| Short 5s | **38.03%** | 39.85% | +1.82pp |
| MP3 64kbps | **37.95%** | 40.05% | +2.10pp |
| Opus 32kbps | **38.26%** | 40.22% | +1.96pp |
| MP3+babble 20dB | **38.92%** | 40.90% | +1.98pp |
| Babble 20dB+short 3s | **39.75%** | 41.15% | +1.40pp |
| MP3+short 3s | **38.68%** | 40.24% | +1.56pp |

**Worse than both v5 and v1 across all conditions.** The v5 backbone features are optimized for BCE's scalar output. Forcing them through a 64-dim angular margin space with only one useful epoch doesn't work. The 2019 backbone is more general and gave OC-Softmax room to reshape features over 5 epochs.

### OC-Softmax v3 — tighter spoof margin, r_fake=0.5 (abandoned)

Tested tighter spoof margin to address v1's babble 10dB regression. Hypothesis: r_fake=0.2 is too lenient — spoof samples under heavy noise land close to the bonafide center without sufficient penalty.

| Setting | Value | Change from v1 |
|---------|-------|----------------|
| r_fake | 0.5 | 0.2 → 0.5 |
| All other settings | same | — |
| Pre-trained from | 2019 LA weights | same |
| Best epoch | 3 | 4 → 3 |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.3272 | 2.6218 | Yes |
| 1 | 0.1002 | 2.5140 | Yes |
| 2 | 0.0793 | 2.8147 | No |
| 3 | 0.0622 | 2.3986 | Yes |
| 4 | 0.0549 | 2.4844 | No |
| 5 | 0.0519 | 2.7674 | No |
| 6 | 0.0438 | 2.8262 | No |

#### EER Comparison (partial — eval stopped early, pattern clear)

| Condition | v5 EER | OC-Soft v1 EER | OC-Soft v3 EER | Δ v3 vs v1 |
|-----------|--------|---------------|---------------|------------|
| Clean | **37.99%** | 38.03% | 39.62% | +1.59pp |
| Ambient 20dB | 39.30% | **38.30%** | 39.51% | +1.21pp |
| Ambient 10dB | 42.12% | **40.98%** | 42.44% | +1.46pp |
| Babble 20dB | 38.65% | **38.23%** | 40.22% | +1.99pp |
| Babble 10dB | **40.10%** | 41.36% | 43.15% | +1.79pp |
| Short 3s | 38.79% | **38.70%** | 40.22% | +1.52pp |
| Short 5s | 38.03% | **37.93%** | 39.62% | +1.69pp |
| MP3 64kbps | 37.95% | **37.86%** | 39.46% | +1.60pp |

**Worse than both v5 and v1 across all conditions (~1.5pp worse than v1).** The tighter margin did not fix babble 10dB (43.15% vs 41.36%) — it made it worse. r_fake=0.5 is too restrictive: it over-constrains the embedding space, pushing the model to create excessively tight boundaries that don't generalize. The lower dev loss (2.40 vs 3.72 for v1) reflects a more constrained optimization, not better detection.

**OC-Softmax conclusion:** v1 (from 2019 weights, r_fake=0.2) is the adopted final model. The loss function change enabled multi-epoch training for the first time and improved 9 of 12 conditions. Starting from a task-adapted backbone (v2) is counterproductive — OC-Softmax benefits from a more general starting point. Tightening the spoof margin (v3) over-constrains the embedding space.

## SE Attention Blocks + OC-Softmax (abandoned, experimental add-on)

Tested stacking SE attention on top of the adopted OC-Softmax v1 configuration. Added Squeeze-and-Excitation (SE) channel attention blocks ([Hu et al., CVPR 2018](https://arxiv.org/abs/1709.01507)) to the LCNN, inserted after each MaxPool layer (4 blocks at 32, 48, 64, 32 channels). SE blocks learn per-channel importance via global average pooling → FC → ReLU → FC → Sigmoid. Motivated by Ma & Liang (ICASSP 2021) who achieved 42% relative EER reduction on ASVspoof 2019 with attention-enhanced LCNN.

### Training

| Setting | Value | Change from OC-Soft v1 |
|---------|-------|------------------------|
| SE attention | enabled | added |
| All other settings | same as v1 | — |
| Additional params | +2,112 (SE blocks) | 275,904 → 283,540 total |
| Best epoch | 1 | 4 → 1 |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.7100 | 3.9129 | Yes |
| 1 | 0.1935 | 3.8146 | Yes |
| 2 | 0.1277 | 4.0667 | No |
| 3 | 0.1094 | 3.9415 | No |
| 4 | 0.0952 | 4.3865 | No |

### EER Comparison

| Condition | v5 EER | OC-Soft v1 EER | SE+OC-Soft EER | Δ vs v1 |
|-----------|--------|---------------|---------------|---------|
| Clean | **37.99%** | 38.03% | 39.21% | +1.18pp |
| Ambient 20dB | 39.30% | **38.30%** | 40.07% | +1.77pp |
| Ambient 10dB | 42.12% | **40.98%** | 43.40% | +2.42pp |
| Babble 20dB | 38.65% | **38.23%** | 39.57% | +1.34pp |
| Babble 10dB | **40.10%** | 41.36% | 43.14% | +1.78pp |
| Short 3s | 38.79% | **38.70%** | 40.63% | +1.93pp |
| Short 5s | 38.03% | **37.93%** | 39.90% | +1.97pp |
| MP3 64kbps | 37.95% | **37.86%** | 39.18% | +1.32pp |
| Opus 32kbps | 38.26% | **38.39%** | 39.68% | +1.29pp |
| MP3+babble 20dB | 38.92% | **38.29%** | 39.62% | +1.33pp |
| Babble 20dB+short 3s | 39.75% | **39.46%** | 40.44% | +0.98pp |
| MP3+short 3s | 38.68% | **38.53%** | — | — |

**Worse than both v5 and v1 across all conditions (~1–2pp worse than v1).** The SE blocks add fresh-initialized parameters that need multiple epochs to converge, but the model only trains usefully for 2 epochs (best at epoch 1). The combination of new SE weights + new OC-Softmax output layer + new OC-Softmax center is too many uninitialized components to learn simultaneously from 2019 pre-trained weights. The 42% relative improvement reported by Ma & Liang was on ASVspoof 2019 with BCE loss and longer training — different regime.

**Conclusion:** SE attention may help with BCE loss (fewer fresh parameters) or with a longer training curriculum, but does not stack well with OC-Softmax under the current training regime.

## Frequency Feature Masking + OC-Softmax (abandoned, experimental add-on)

Tested stacking frequency feature masking (FFM) ([Kwak et al., ADD 2022](https://ikwak2.github.io/publications/ddam004-kwak.pdf)) on top of the adopted OC-Softmax v1 configuration. Randomly masks frequency bands (low, high, or random contiguous band) during training to force the model to learn spoofing cues from all frequency regions instead of relying on specific bands. Motivated by GradCAM finding that noise masks the Delta-delta hotspots the model relies on. No new parameters — pure data augmentation.

### Training

| Setting | Value | Change from OC-Soft v1 |
|---------|-------|------------------------|
| freq_mask_width | 10 (out of 60 LFCC bins) | added |
| All other settings | same as v1 | — |
| Best epoch | 1 | 4 → 1 |

| Epoch | Train Loss | Dev Loss | Best? |
|-------|------------|----------|-------|
| 0 | 0.7380 | 4.0048 | Yes |
| 1 | 0.2272 | 3.8154 | Yes |
| 2 | 0.1548 | 4.9439 | No |
| 3 | — | — | OOM crash (early stop would have triggered) |

### EER Comparison

| Condition | v5 EER | OC-Soft v1 EER | FFM+OC-Soft EER | Δ vs v1 |
|-----------|--------|---------------|----------------|---------|
| Clean | **37.99%** | 38.03% | 38.45% | +0.42pp |
| Ambient 20dB | 39.30% | **38.30%** | 39.99% | +1.69pp |
| Ambient 10dB | 42.12% | **40.98%** | 43.35% | +2.37pp |
| Babble 20dB | 38.65% | **38.23%** | 40.64% | +2.41pp |
| Babble 10dB | **40.10%** | 41.36% | 44.53% | +3.17pp |
| Short 3s | 38.79% | **38.70%** | 38.87% | +0.17pp |
| Short 5s | 38.03% | **37.93%** | 37.87% | -0.06pp |
| MP3 64kbps | 37.95% | **37.86%** | 38.35% | +0.49pp |
| Opus 32kbps | **38.26%** | 38.39% | 38.36% | -0.03pp |
| MP3+babble 20dB | 38.92% | **38.29%** | 40.78% | +2.49pp |
| Babble 20dB+short 3s | 39.75% | **39.46%** | 41.35% | +1.89pp |
| MP3+short 3s | 38.68% | **38.53%** | 38.91% | +0.38pp |

**Worse than v1 across nearly all conditions, especially noise (+2–3pp).** Masking 10 of 60 LFCC bins (~17%) during training is too aggressive — it destroys critical spectral information that the model needs to learn spoofing cues, particularly in the Delta-delta bands that are already vulnerable to noise. The frequency masking compounds with noise augmentation to create excessively degraded training inputs. Short clip and codec conditions are nearly unchanged, confirming the damage is concentrated in the frequency domain.

**Conclusion:** FFM does not stack well with noise augmentation + OC-Softmax. The noise augmentation already forces robustness to frequency-domain corruption; additional frequency masking is redundant and destructive.
