# GradCAM Saliency Maps: Cross-Model Comparison

GradCAM analysis for the OC-Softmax LFCC-LCNN model, placed alongside the baseline (clean-weighted) and noise-augmented variants on identical conditions. The PNGs in this directory are the OC-Softmax model's output; the baseline maps live at `../../clean_weighted/saliency_maps/` and are referenced throughout the comparison tables below.

## Models Compared

| Model | Training | Loss | Clean EER | Failure Mode Under Noise |
|-------|----------|------|-----------|--------------------------|
| **Baseline** (clean_weighted) | Pre-trained 2019 weights, no augmentation | BCELoss | 38.72% | Spoof collapse (FAR >> FRR) |
| **Noise-augmented** | Pre-trained 2019 + noise augmentation | BCELoss | 37.99% | Bonafide collapse (FRR >> FAR) |
| **OC-Softmax** | Pre-trained 2019 + noise augmentation + OC-Softmax | Angular margin | 38.03% | Bonafide collapse (FRR >> FAR, uniform) |

See `PROJECT.md` §"Enhanced Results" for the full per-condition EER tables.

## Subfolder Index

Each per-condition folder contains 12 OC-Softmax maps (two bonafide plus two spoof per bucket: correct high-confidence, correct low-confidence, misclassified). EER and failure modes come from `../robustness_summary.csv`.

- `noise_ambient_20dB/`, `noise_ambient_10dB/` — ambient noise; OC-Softmax wins (1.0 to 1.1pp better than noise-augmented)
- `noise_babble_20dB/`, `noise_babble_10dB/` — babble noise; OC-Softmax wins at 20dB, loses at 10dB
- `mp3_64kbps/`, `opus_32kbps/` — codec compression; indistinguishable from clean
- `short_3s/`, `short_5s/` — center-cropped clips; bonafide collapse
- `corrected_examples/` — samples misclassified on clean that flipped correct under babble 10dB
- `flipped_examples/` — the E_0005089198 spoof and E_0000994221 bonafide samples used in the flipped-example comparison

## How to Read the Plots

**LFCC feature bands (y-axis).** The 60-dimensional feature splits into three bands. LFCC (indices 0 to 19) carries basic voice features: tone, pitch, vocal quality, and overall frequency content. Delta (20 to 39) captures how those features change over time, the speed of transitions between sounds. Delta-delta (40 to 59) captures how those rates of change themselves speed up or slow down, and is the most sensitive band for subtle synthesis artifacts.

**Scores.** Each model outputs a single score per utterance. The baseline uses raw BCELoss scores with magnitudes typically in the range of ±15 to ±50. The noise-augmented model uses the same BCELoss scale but with distributions shifted negative (roughly ±10 to ±40, EER threshold near -2.3). OC-Softmax uses a cosine-similarity scale bounded in [-1, +1] with the EER threshold near 0. When comparing across models, relative patterns (high-conf vs. low-conf, correct vs. misclassified) matter more than absolute score magnitudes.

**Sample stratification.** Each model's samples are bucketed using its own EER threshold as the decision boundary, not score sign. Seed 42 is fixed across all runs so that within a condition, the three models are compared on the same four-per-bucket utterances where possible.

**Heatmap intensity.** Red and yellow indicate high attention; blue indicates low. The heatmap shows which time-frequency regions most influenced the model's prediction, not where synthesis is located.

---

## Noise Conditions

### Ambient Noise 20dB (light)

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | Broad LFCC + Delta attention tracking speech-active regions. Classic bonafide pathway intact through light ambient noise. |
| **Noise-augmented** | Correct high-conf bonafide | Broader, more distributed LFCC + Delta attention than baseline, with more continuous coverage across the utterance. Attention extends into Delta-delta at speech-active regions; the noise-augmented model engages all three bands for bonafide detection, not just LFCC + Delta. |
| **OC-Softmax** | Correct high-conf bonafide | Sparser, more selective attention than both baseline and noise-augmented. Concentrated LFCC + Delta hotspots at specific speech-active time points rather than broad coverage. Delta-delta engagement is minimal. OC-Softmax uses fewer but more targeted cues for bonafide recognition. |
| **Baseline** | Correct high-conf spoof | Sparse, localized Delta-delta (40 to 59) hotspots at specific time points. Classic spoof detection pathway. |
| **Noise-augmented** | Correct high-conf spoof | Sparse Delta-delta + LFCC hotspots, but with notably more LFCC (0 to 19) engagement than baseline. The noise-augmented model's spoof detection pathway uses a wider frequency range: it has learned to detect artifacts visible in both basic features and acceleration, not just Delta-delta alone. |
| **OC-Softmax** | Correct high-conf spoof | Very sparse, highly localized hotspots focused in Delta (20 to 39) and LFCC (0 to 19) bands with pinpoint Delta-delta engagement. The most spatially precise of the three models: fewer hotspots but with higher peak intensity. OC-Softmax concentrates its spoof detection on narrow time-frequency regions. |
| **Baseline** | Misclassified bonafide | Strong Delta-delta activation at specific time points, triggering the spoof pathway. Noise-induced distortions compound channel artifacts. |
| **Noise-augmented** | Misclassified bonafide | Sparse, isolated LFCC + Delta hotspots with low overall activation. The model shows uncertainty; attention is fragmented with no clear bonafide or spoof signature. Score magnitudes are stronger negative than baseline misclassified bonafide (e.g., -19.3 vs. baseline's -17.8), suggesting the noise-augmented model's spoof-leaning bias under noise is more decisive. |
| **OC-Softmax** | Misclassified bonafide | Scattered Delta + LFCC hotspots with moderate activation. Marginally misclassified (scores near -0.5), consistent with OC-Softmax's tighter decision boundary. The attention pattern shows neither a clean bonafide nor a clean spoof signature; the model is genuinely uncertain. |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta attention spanning full utterance, indistinguishable from correct bonafide. Noise masks synthesis artifacts. |
| **Noise-augmented** | Misclassified spoof | Broad LFCC + Delta attention with moderate intensity, similar to the noise-augmented model's bonafide pattern. Slightly more scattered than baseline's misclassified spoof; the noise-augmented model has learned to look harder but still fails on these samples. |
| **OC-Softmax** | Misclassified spoof | Sparse but distributed LFCC + Delta + Delta-delta attention. Unlike the baseline's uniform bonafide-mimicking pattern, OC-Softmax's misclassified spoof maps show mixed signals; some Delta-delta activation is present but insufficient to push the score below threshold. The model is closer to correct than baseline or noise-augmented on these cases. |

### Ambient Noise 10dB (moderate)

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | Broad LFCC + Delta attention with notable Delta-delta engagement. Bonafide pathway remains robust. Scores +14.5 to +14.9. |
| **Noise-augmented** | Correct high-conf bonafide | Very broad, continuous LFCC + Delta coverage spanning nearly the entire utterance with multiple strong hotspot regions. More densely activated than baseline; the noise-augmented model gathers evidence from more time points. Scores +10.0. |
| **OC-Softmax** | Correct high-conf bonafide | Distributed Delta + LFCC attention with scattered hotspots across the full utterance. More fragmented than the noise-augmented model but covers all three bands. Scores near +1.0 (high-conf on OC-Softmax scale). |
| **Baseline** | Correct high-conf spoof | Sparse Delta-delta hotspots, sparser than at 20dB. Score magnitudes weaker than clean. |
| **Noise-augmented** | Correct high-conf spoof | Dense Delta-delta + LFCC hotspots concentrated in the second half of the utterance. The noise-augmented model finds artifact-rich regions even through 10dB ambient noise. Score magnitudes (-36.1) comparable to clean, a major improvement over baseline's weakened scores at this noise level. |
| **OC-Softmax** | Correct high-conf spoof | Extremely sparse, isolated Delta + LFCC hotspots at 2 to 3 specific time points. Very focal detection; OC-Softmax commits to a small number of high-confidence artifact detections rather than aggregating many weak ones. Score -0.66 (confident on OC-Softmax scale). |
| **Baseline** | Misclassified bonafide | Strong Delta-delta activation at specific time regions. Score -17.8. |
| **Noise-augmented** | Misclassified bonafide | Sparse, low-intensity hotspots scattered across LFCC + Delta with very little concentrated activation. Score -19.3. The noise-augmented model is more confident in its misclassification, but the attention map shows it is working harder (more distributed scanning) to reach this decision. |
| **OC-Softmax** | Misclassified bonafide | Sparse Delta + LFCC hotspots concentrated at 1 to 2 time regions. Score -0.53 (marginal misclassification). The model commits to very few regions, and those happen to produce spoof-leaning evidence. |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta + Delta-delta spanning full utterance. Score +17.2. Classic spoof collapse pattern. |
| **Noise-augmented** | Misclassified spoof | Broad LFCC + Delta coverage with distributed hotspots. Score +11.7. Similar to baseline's failure mode but with lower confidence. |
| **OC-Softmax** | Misclassified spoof | Concentrated LFCC + Delta hotspot at a single time region (~5s) with low activation elsewhere. Score +1.0. OC-Softmax's misclassification is driven by a single region of strong bonafide-like activation rather than uniform coverage, a qualitatively different failure mode. |

### Babble Noise 20dB (light)

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | Broad LFCC + Delta attention with Delta-delta engagement tracking speech. Bonafide pathway stable. |
| **Noise-augmented** | Correct high-conf bonafide | Dense, continuous LFCC + Delta attention spanning most of the utterance. More uniformly activated than baseline. Delta-delta shows moderate engagement at speech transitions. |
| **OC-Softmax** | Correct high-conf bonafide | Scattered LFCC + Delta hotspots at specific speech-active time points. More selective than both baseline and noise-augmented. Delta-delta is mostly quiet. |
| **Baseline** | Correct high-conf spoof | Sparse Delta-delta hotspots. Score magnitudes weaker than clean (-18/-24 vs -46). |
| **Noise-augmented** | Correct high-conf spoof | Sparse Delta-delta + LFCC hotspots with some Delta involvement. Scores moderately strong. The noise-augmented model maintains spoof detection through babble better than baseline. |
| **OC-Softmax** | Correct high-conf spoof | Very sparse, highly localized Delta + LFCC hotspots. Minimal spatial extent but high peak intensity. The most focused spoof attention of the three models. |
| **Baseline** | Misclassified bonafide | Scattered Delta-delta + LFCC hotspots at onset and intermittent regions. |
| **Noise-augmented** | Misclassified bonafide | Dense LFCC + Delta hotspots resembling the bonafide pattern but scored below threshold. The noise-augmented model's misclassified bonafide under babble shows a fundamentally different pattern from baseline: the model sees bonafide-like features but its shifted decision boundary rejects them. |
| **OC-Softmax** | Misclassified bonafide | Scattered Delta + LFCC hotspots with intermittent Delta-delta. Multiple discrete attention regions. Score near -0.5 (marginal misclassification, consistent with bonafide collapse pattern). |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta spanning full utterance. Indistinguishable from correct bonafide. |
| **Noise-augmented** | Misclassified spoof | Broad LFCC + Delta with moderate intensity. Similar failure pattern to baseline but slightly less confident. |
| **OC-Softmax** | Misclassified spoof | Moderate LFCC + Delta attention with some Delta-delta engagement. Less uniformly bonafide-like than baseline's or the noise-augmented model's failures; mixed signals present but insufficient for correct classification. |

### Babble Noise 10dB (moderate)

This is the most challenging noise condition and best reveals the differences between models.

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | Broad LFCC + Delta with Delta-delta engagement. Bonafide pathway stable even at 10dB. Additional Delta-delta activation may reflect overlapping babble voices. Scores +14.0, +13.1. |
| **Noise-augmented** | Correct high-conf bonafide | Very broad, continuous LFCC + Delta coverage. Multiple speech-active regions produce strong hotspots across the full utterance. The densest activation pattern of any model at any noise level. Scores moderate (+5 to +10 range on noise-augmented scale). |
| **OC-Softmax** | Correct high-conf bonafide | Selective, sparse Delta + LFCC hotspots at a few specific time points. Very different from baseline / noise-augmented; OC-Softmax extracts bonafide evidence from fewer regions but with higher precision. Delta-delta is minimally engaged. Score near +1.0. |
| **Baseline** | Correct high-conf spoof | Sparse, isolated Delta / Delta-delta hotspots. Score magnitudes much weaker than clean (-9.8/-13.7 vs -45.6). |
| **Noise-augmented** | Correct high-conf spoof | Concentrated Delta-delta + LFCC hotspots, fewer than at lighter noise but still present. Some residual Delta band involvement. Scores significantly stronger than baseline at this noise level; the noise-augmented model preserves spoof detection ability that baseline loses. |
| **OC-Softmax** | Correct high-conf spoof | Sparse, distributed LFCC + Delta + Delta-delta hotspots across the utterance with intermittent high-intensity peaks. The attention spans all three bands, suggesting OC-Softmax extracts spoof cues from multiple feature types simultaneously. Score -0.64. |
| **Baseline** | Correct low-conf bonafide | Broad LFCC + Delta attention with scattered hotspots. Score +2.17 (barely above threshold). |
| **Noise-augmented** | Correct low-conf bonafide | Moderate LFCC + Delta coverage with scattered, moderate-intensity hotspots. Score -1.48 (correct on noise-augmented scale but marginal). Less activation than high-conf bonafide. |
| **OC-Softmax** | Correct low-conf bonafide | Very sparse, isolated hotspots at 2 to 3 time points, predominantly in LFCC + Delta. Score +0.97 (marginal). The model finds minimal bonafide evidence. |
| **Baseline** | Correct low-conf spoof | Fragmented, sparse hotspots. Score -0.10 (barely below threshold). |
| **Noise-augmented** | Correct low-conf spoof | Scattered Delta-delta + LFCC hotspots appearing in the second half of the utterance. Score -3.73 (more confident than baseline). Noise-augmented training lets the model find enough artifact evidence for a marginal detection. |
| **OC-Softmax** | Correct low-conf spoof | Sparse Delta + LFCC hotspots distributed across the utterance. Score -0.17 (very marginal). |
| **Baseline** | Misclassified bonafide | Sparse, fragmented hotspots with low overall activation. Weakly spoof scores. Score -13.6. |
| **Noise-augmented** | Misclassified bonafide | Dense LFCC + Delta hotspots tracking speech-active regions, strong activation. Paradoxically, the attention map looks similar to correct bonafide, but the model scores it strongly negative (-15.4). This reveals the noise-augmented model's bonafide collapse mechanism: the model sees bonafide-like features but has learned to distrust them under noise, over-correcting from the baseline's spoof collapse. |
| **OC-Softmax** | Misclassified bonafide | Scattered Delta + LFCC hotspots at several time points with moderate intensity. Score -0.48 (marginal misclassification). Some Delta-delta engagement present that may be triggering the spoof pathway. |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta spanning entire utterance. Score +14.5 to +15.7. Core mechanism of 91.17% FAR. |
| **Noise-augmented** | Misclassified spoof | Broad LFCC + Delta coverage with distributed hotspots similar to the noise-augmented model's bonafide pattern. Score +9.3. The same bonafide-mimicking pattern as baseline but with lower magnitude. |
| **OC-Softmax** | Misclassified spoof | Sparse LFCC + Delta + Delta-delta with scattered hotspots. Score +1.0. Mixed attention with some spoof-like Delta-delta engagement visible; OC-Softmax is closer to the correct answer even when wrong, consistent with its more balanced misclassification pattern. |

---

## Short Clip Conditions

### Short 3s

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | Concentrated LFCC + Delta attention over a single central region. Model over-commits to the limited speech available. Score +19.8 (higher magnitude than full-length clean). |
| **Noise-augmented** | Correct high-conf bonafide | Dense, broad LFCC + Delta + Delta-delta attention spanning almost the entire 3s clip. Multiple overlapping hotspot regions. Score +13.5. The noise-augmented model distributes attention more evenly across the short clip rather than concentrating on one region. |
| **OC-Softmax** | Correct high-conf bonafide | Sparse, selective LFCC + Delta hotspots at 2 to 3 specific time points. Low overall activation area. Score +1.0. OC-Softmax maintains its selective attention strategy even with limited temporal context; it does not over-commit. |
| **Baseline** | Correct high-conf spoof | LFCC (0 to 19) + isolated Delta-delta hotspots. Score magnitudes stronger than clean (-51.1). Truncation amplifies confidence. |
| **Noise-augmented** | Correct high-conf spoof | Sparse Delta-delta + LFCC hotspots at utterance onset and isolated later points. Score -44.0. Similar amplification effect as baseline. The artifact-dense region captured by the 3s crop dominates the score. |
| **OC-Softmax** | Correct high-conf spoof | Distributed Delta + LFCC + Delta-delta hotspots across the 3s clip, with multiple discrete regions of activation. Score -0.69. OC-Softmax shows more multi-band engagement for spoof detection in short clips than at full length. |
| **Baseline** | Misclassified bonafide | Delta-delta + LFCC hotspots at start and end of clip with dead zones in the middle. Score -22.8. Center crop captured unfavorable regions. |
| **Noise-augmented** | Misclassified bonafide | Scattered LFCC + Delta hotspots concentrated at utterance onset with sparser activation later. Score -22.3. Similar failure mode to baseline; truncation captures artifact-like regions. The onset concentration is notable: the noise-augmented model may be over-weighting initial transients. |
| **OC-Softmax** | Misclassified bonafide | Focused Delta-delta hotspot at one specific time region (~0.7s) with sparse scattered attention elsewhere. Score -0.62. OC-Softmax's misclassification is driven by a single localized artifact detection rather than distributed evidence, consistent with its focal attention strategy. |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta spanning full 3s. Score +21.2. Center crop captured natural-sounding transitions. |
| **Noise-augmented** | Misclassified spoof | Very broad LFCC + Delta attention covering the entire 3s clip continuously, with strong hotspots in the Delta band. Score +15.3. Similar failure mode to baseline but the noise-augmented model's attention is even more distributed; it reads the entire clip as natural speech. |
| **OC-Softmax** | Misclassified spoof | Distributed LFCC + Delta + Delta-delta attention with multiple hotspot regions. Score +1.0. Like the noise conditions, OC-Softmax's misclassified spoof maps show mixed multi-band signals rather than a clean bonafide pattern; it is seeing conflicting evidence. |

### Short 5s

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | LFCC + Delta attention distributed across multiple speech regions. Score +18.4. Less concentrated than 3s; more temporal context allows the model to sample multiple regions. |
| **Noise-augmented** | Correct high-conf bonafide | Dense, continuous LFCC + Delta coverage spanning most of the 5s clip with strong overlap between hotspot regions. Score +12.9. Very broad activation, similar to the noise-augmented model's pattern at full length. |
| **OC-Softmax** | Correct high-conf bonafide | Sparse, isolated hotspots at a few specific time points. Minimal overall activation. Score +1.0. Maintains selectivity even with more available context. |
| **Baseline** | Correct high-conf spoof | LFCC + sparse Delta-delta hotspots. Score -46.5. Closer to clean-condition magnitudes than 3s. |
| **Noise-augmented** | Correct high-conf spoof | Sparse Delta-delta + LFCC hotspots at utterance onset and a few later time points. Score -40.7. Strong detection preserved. |
| **OC-Softmax** | Correct high-conf spoof | Extremely precise Delta + LFCC hotspots at 2 to 3 narrow time regions with very high peak intensity. Score -0.69. The most spatially concentrated spoof detection of any model at 5s. |
| **Baseline** | Misclassified bonafide | Delta-delta and LFCC hotspots at edges. Score -22.7. Same edge-heavy pattern as at 3s. |
| **Noise-augmented** | Misclassified bonafide | Dense Delta-delta + LFCC hotspots concentrated at utterance onset and scattered throughout. Score -22.0. Strong onset activation driving the misclassification. |
| **OC-Softmax** | Misclassified bonafide | Focused Delta-delta + LFCC hotspots at 1 to 2 specific time regions. Score -0.58. Marginal misclassification from localized artifact-like features. |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta spanning full 5s. Score +19.5. |
| **Noise-augmented** | Misclassified spoof | Very broad LFCC + Delta attention with strong, continuous coverage. Score +15.2. |
| **OC-Softmax** | Misclassified spoof | Single narrow LFCC hotspot at one time point with near-zero activation elsewhere. Score +1.0. OC-Softmax's misclassification is driven by one convincing bonafide-like region. |

---

## Codec Conditions

### MP3 64kbps and Opus 32kbps

| Model | Category | Attention Pattern |
|-------|----------|-------------------|
| **Baseline** | Correct high-conf bonafide | Broad LFCC + Delta attention, indistinguishable from clean baseline. |
| **Noise-augmented** | Correct high-conf bonafide | Dense, continuous LFCC + Delta attention spanning the full utterance with strong multi-region hotspots. Denser than baseline's codec pattern and visually identical to the noise-augmented model's clean bonafide pattern. |
| **OC-Softmax** | Correct high-conf bonafide | Sparse, selective LFCC + Delta hotspots at specific time points. Identical to OC-Softmax's clean pattern. |
| **Baseline** | Correct high-conf spoof | Sparse Delta-delta hotspots, identical to clean. Score magnitudes comparable to clean. |
| **Noise-augmented** | Correct high-conf spoof | Sparse Delta-delta + LFCC hotspots. Identical to the noise-augmented model's clean spoof detection pattern. |
| **OC-Softmax** | Correct high-conf spoof | Very sparse, highly localized Delta + LFCC hotspots with pinpoint precision. Identical to OC-Softmax's clean pattern. |
| **Baseline** | Misclassified bonafide | Delta-delta + LFCC hotspots at utterance edges. Same as clean baseline failures. |
| **Noise-augmented** | Misclassified bonafide | Sparse Delta-delta + LFCC hotspots at scattered time points. Same as the noise-augmented model's clean misclassification pattern. |
| **OC-Softmax** | Misclassified bonafide | Scattered LFCC + Delta + Delta-delta hotspots. Same as OC-Softmax's clean failures. |
| **Baseline** | Misclassified spoof | Broad LFCC + Delta mimicking bonafide. Same as clean. |
| **Noise-augmented** | Misclassified spoof | Broad LFCC + Delta coverage. Same as the noise-augmented model's clean failures. |
| **OC-Softmax** | Misclassified spoof | Distributed multi-band attention. Same as OC-Softmax's clean failures. |

**Codec compression is invisible to every model.** All three architectures show attention patterns indistinguishable from their respective clean-condition patterns under both MP3 64kbps and Opus 32kbps. The training differences (noise augmentation, loss function) do not change this finding, consistent with lossy compression at these bitrates preserving the spectral cues the LFCC front-end relies on.

---

## Flipped Examples: Same Sample, Clean vs. Babble 10dB

Samples E_0005089198 (spoof) and E_0000994221 (bonafide) were selected as cases where the baseline flipped its decision between clean and babble 10dB. All three models were run on both conditions of each sample.

### E_0005089198 (spoof)

| Model | Clean Score | Babble 10dB Score | Clean Attention | Noisy Attention |
|-------|------------|-------------------|-----------------|-----------------|
| **Baseline** | -33.5 (correct) | -2.8 (correct, marginal) | Sparse Delta-delta hotspots at ~1s, 2s, 3s, 5s, 6 to 7s. Strong artifact detection at multiple time points. | Shifted to broader LFCC + Delta coverage with some Delta-delta remnants. Noise reduced but did not fully eliminate artifact cues; score dropped from -33.5 to -2.8 but remained correct. |
| **Noise-augmented** | -28.6 (correct) | -8.5 (correct) | Sparse Delta-delta + LFCC hotspots similar to baseline but with more LFCC engagement. Artifact detection present at same time regions as baseline. | Delta-delta hotspots persist with broader LFCC + Delta coverage emerging. The noise-augmented model maintains more artifact evidence through noise; score drops less (-28.6 to -8.5) than baseline. |
| **OC-Softmax** | -0.64 (correct) | -0.27 (correct) | Sparse, focused Delta + LFCC hotspots at specific time points. Fewer regions activated than baseline or noise-augmented but confident. | Slightly reorganized attention with some hotspots shifting position, but overall pattern preserved. Score barely changes (-0.64 to -0.27); OC-Softmax is the most stable across conditions. |

**OC-Softmax's attention is the most noise-invariant.** All three models correctly classify this sample under both conditions, but with very different stability profiles. OC-Softmax shows the smallest score swing (0.37) compared to baseline (30.7) and noise-augmented (20.1). The attention maps reveal why: OC-Softmax's sparse, targeted attention identifies artifact cues that are robust to noise, while baseline's distributed Delta-delta scanning loses most of its evidence when noise masks the weaker hotspots.

### E_0000994221 (bonafide)

| Model | Clean Score | Babble 10dB Score | Clean Attention | Noisy Attention |
|-------|------------|-------------------|-----------------|-----------------|
| **Baseline** | +12.7 (correct) | +3.4 (correct, marginal) | Broad LFCC + Delta attention tracking speech regions throughout. Classic bonafide pattern. | Attention pattern largely preserved with some fragmentation at edges. Delta-delta begins to show scattered hotspots. Score drops from +12.7 to +3.4 but remains correct. |
| **Noise-augmented** | +7.0 (correct) | +4.0 (correct) | Broad LFCC + Delta attention similar to baseline but with more Delta-delta engagement throughout. Dense multi-band coverage. | Attention shifts to more concentrated LFCC + Delta hotspots with reduced Delta-delta. Pattern simplifies under noise but core bonafide evidence preserved. Score drops modestly (7.0 to 4.0). |
| **OC-Softmax** | +1.0 (correct) | +1.0 (correct) | Concentrated Delta + LFCC hotspot at ~3s with moderate activation at onset. Very focal bonafide recognition. | LFCC + Delta hotspots shift slightly but maintain similar sparsity and distribution. Pattern and score are essentially unchanged. |

**OC-Softmax's score swing is effectively zero on bonafide too.** All three models correctly handle this sample under both conditions. OC-Softmax's swing (~0.0) is consistent with its sparse, noise-resilient attention strategy seen on the spoof sample above.

---

## Corrected Examples: Misclassified on Clean, Correct Under Babble 10dB

### E_0009460641 (bonafide, misclassified on clean by all three models)

| Model | Clean Score | Babble 10dB Score | Clean Attention | Noisy Attention |
|-------|------------|-------------------|-----------------|-----------------|
| **Baseline** | -22.4 (misclassified) | -7.6 (misclassified) | Strong Delta-delta + LFCC hotspots at multiple time points; model detects channel / recording artifacts and triggers spoof pathway. Concentrated activity at speech transitions. | Attention shifts to sparser LFCC + Delta hotspots. Delta-delta activation reduced. Score shifts upward (-22.4 to -7.6) but remains misclassified. |
| **Noise-augmented** | -17.9 (misclassified) | -10.5 (misclassified) | Sparse Delta-delta + LFCC hotspots at fewer time points than baseline, concentrated around 6 to 7s. Less distributed artifact detection but still sufficient to trigger misclassification. | Even sparser activation with isolated LFCC hotspots. Score changes modestly. |
| **OC-Softmax** | +0.22 (correct, marginal) | -0.23 (misclassified, marginal) | Mixed LFCC + Delta + Delta-delta attention spanning the utterance. Moderate activation across multiple bands. Score is barely positive; model sees conflicting evidence. | Sparser attention with reduced activation. Score flips just below threshold. |

**Recording artifacts trigger spoof-like attention in every model.** This sample contains genuine channel / recording artifacts. Baseline and noise-augmented confidently misclassify under both conditions. OC-Softmax is borderline in both cases (score near 0), reflecting its tighter score distribution: the angular margin loss does not provide confident separation for this ambiguous sample, but the error magnitude stays minimal.

### E_0009263224 (spoof, misclassified on clean by all three models)

| Model | Clean Score | Babble 10dB Score | Clean Attention | Noisy Attention |
|-------|------------|-------------------|-----------------|-----------------|
| **Baseline** | +11.7 (misclassified) | +8.0 (misclassified) | Broad LFCC + Delta attention at speech transitions. No Delta-delta artifact detection. Model reads this spoof sample as natural speech. | Broader LFCC + Delta coverage with additional Delta-delta hotspots emerging at later time points. Score shifts down but still misclassified. |
| **Noise-augmented** | +1.1 (misclassified, marginal) | -0.9 (correct, marginal) | Sparse LFCC + Delta hotspots at speech-active time points. Less broad than baseline's attention on this sample. Score barely positive. | Scattered LFCC + Delta + Delta-delta hotspots. Some artifact evidence emerges. Score crosses to correct side by a thin margin. |
| **OC-Softmax** | +0.99 (misclassified, marginal) | +0.71 (misclassified, marginal) | Mixed LFCC + Delta + Delta-delta attention with strong activation at 5 to 8s, including Delta-delta engagement. Despite seeing some spoof-like cues, bonafide evidence outweighs. | Similar pattern with slightly reorganized hotspots. Score barely changes. |

**Noise-augmented is the only model that accidentally corrects under noise.** The noise-augmented model barely crosses to correct because its clean score was already marginal. OC-Softmax maintains a near-identical score regardless of condition; its noise-invariance works against it here since it cannot leverage noise to accidentally correct a wrong decision.

---

## Cross-Model Attention Architecture Comparison

### Fundamental Attention Strategies

Each model has developed a qualitatively distinct attention architecture:

| Feature | Baseline | Noise-augmented | OC-Softmax |
|---------|----------|-----------------|------------|
| **Bonafide attention** | Broad LFCC + Delta, tracking speech-active regions | Very dense, continuous LFCC + Delta with moderate Delta-delta, spanning most of the utterance | Sparse, selective LFCC + Delta hotspots at specific time points |
| **Spoof attention** | Sparse, localized Delta-delta hotspots | Sparse Delta-delta + LFCC hotspots (wider frequency range than baseline) | Very sparse, pinpoint Delta + LFCC hotspots with high peak intensity |
| **Bonafide / spoof distinguishability** | High: broad vs. sparse, clearly different bands | Moderate: both pathways use LFCC + Delta, differentiated by Delta-delta presence | Moderate-low: both pathways use similar sparse patterns, differentiated by band focus and peak distribution |
| **Spatial extent** | Moderate | Highest (densest coverage) | Lowest (most selective) |
| **Band usage** | Two-pathway: LFCC + Delta vs. Delta-delta | Multi-band: all three bands contribute to both decisions | Multi-band with Delta emphasis: Delta (20 to 39) more prominent than in other models |
| **Score sensitivity to perturbation** | High (large score swings) | Moderate (reduced swings) | Low (minimal score changes) |

### How Noise-Augmented Training Changes Attention (Baseline vs. Noise-Augmented)

**Broader frequency engagement for both classes.** The noise-augmented model activates LFCC (0 to 19) during spoof detection and Delta-delta (40 to 59) during bonafide detection, whereas the baseline keeps these pathways more separated. Noise augmentation forced the model to learn redundant features across bands.

**Denser temporal coverage.** The noise-augmented model consistently shows more hotspot regions per utterance than baseline, covering more of the available time axis. This is a "gather more evidence" strategy: by sampling more time-frequency regions, it reduces the chance that a few masked regions dominate the decision.

**Stronger spoof detection under noise but reversed failure mode.** The noise-augmented model's correct high-conf spoof samples under noise maintain stronger score magnitudes than baseline's. This came at the cost of bonafide recognition; the model became more suspicious overall, producing bonafide collapse (FRR >> FAR) rather than spoof collapse.

**Misclassified bonafide shows paradoxical patterns.** Under babble 10dB, the noise-augmented model's misclassified bonafide samples often show bonafide-like attention maps (broad LFCC + Delta) yet receive strongly negative scores. The model learned to distrust its own bonafide evidence under noise: it sees natural speech features but has been trained to expect that noisy bonafide-like signals might be spoofs.

### How OC-Softmax Changes Attention (Noise-Augmented vs. OC-Softmax)

**Dramatically sparser attention.** OC-Softmax consistently activates fewer spatial regions than the noise-augmented model (and baseline). The angular margin loss encourages the model to find a small number of highly discriminative features rather than aggregating many weak ones.

**Higher peak intensity, lower spatial extent.** When OC-Softmax activates a region, the peak attention intensity is often higher than the noise-augmented model's, but the activated area is much smaller. This is a quality-over-quantity strategy for evidence accumulation.

**Delta band (20 to 39) prominence.** OC-Softmax shows notably more Delta band activation than both baseline and noise-augmented. The angular margin loss may have shifted the model's preferred discriminative features from Delta-delta acceleration artifacts toward temporal dynamics (Delta), which are more robust to additive noise.

**Compressed score range with near-zero threshold.** OC-Softmax scores cluster tightly around 0, with high-confidence correct classifications at approximately ±1.0. This compressed range means misclassifications are always marginal (scores near the threshold), unlike baseline / noise-augmented where misclassified samples can have large wrong-direction magnitudes.

**Noise-invariant attention.** OC-Softmax's attention maps change less between clean and noisy conditions than either baseline or noise-augmented. The flipped / corrected examples demonstrate this: OC-Softmax shows the smallest score swings across conditions. The sparse, focal attention strategy means each hotspot either survives noise or doesn't; there is less gradual degradation of distributed evidence.

**Mixed-signal misclassifications.** Unlike baseline and noise-augmented, whose misclassified spoof maps cleanly mimic the bonafide pattern (broad LFCC + Delta), OC-Softmax's misclassified spoof maps typically show mixed multi-band signals with some Delta-delta engagement. The model sees conflicting evidence rather than being fully fooled.

---

## Misclassification Pattern Comparison

| Failure Type | Baseline | Noise-augmented | OC-Softmax |
|--------------|----------|-----------------|------------|
| **Misclassified spoof (noise)** | Broad LFCC + Delta spanning entire utterance, indistinguishable from bonafide. High-confidence wrong scores (+15 to +18). | Broad LFCC + Delta coverage, similar to baseline but slightly less uniform. Moderate wrong scores (+9 to +12). | Sparse, mixed multi-band attention with some Delta-delta present. Low wrong scores (+0.5 to +1.0). Qualitatively different from bonafide pattern. |
| **Misclassified bonafide (noise)** | Strong Delta-delta activation mimicking spoof pathway. Wrong scores -10 to -18. | Dense LFCC + Delta activation that looks bonafide-like but scored negative. Wrong scores -10 to -19. Model distrusts bonafide cues under noise. | Sparse, focal hotspots with mixed band activation. Marginal wrong scores (-0.2 to -0.6). |
| **Misclassified spoof (short clips)** | Broad LFCC + Delta spanning full clip. Score +21. | Very broad, dense LFCC + Delta coverage. Score +15. | Sparse attention at few regions. Score +1.0. |
| **Misclassified bonafide (short clips)** | Edge-heavy Delta-delta + LFCC. Score -23. | Onset-concentrated Delta-delta + LFCC. Score -22. | Single focused Delta-delta hotspot. Score -0.6. |

**OC-Softmax never produces high-confidence misclassifications.** Its compressed score range and sparse attention strategy mean that even when the model is wrong, it is wrong by a small margin and with visibly conflicting evidence in the attention map. This contrasts sharply with baseline and noise-augmented, where misclassifications can be highly confident and show clean, unambiguous wrong-pathway attention.

---

## Flipped / Corrected Examples: Cross-Model Stability

The flipped and corrected examples reveal how stable each model's decisions are across conditions:

| Stability Metric | Baseline | Noise-augmented | OC-Softmax |
|-----------------|----------|-----------------|------------|
| Score swing on flipped spoof (E_0005089198) | 30.7 | 20.1 | 0.37 |
| Score swing on flipped bonafide (E_0000994221) | 9.3 | 3.0 | ~0.0 |
| Corrected bonafide (E_0009460641) clean-to-noise swing | 14.8 | 7.4 | 0.45 |
| Corrected spoof (E_0009263224) clean-to-noise swing | 3.7 | 2.0 | 0.28 |
| **Attention map change (qualitative)** | Major reorganization (band switching) | Moderate reorganization (density change) | Minimal change (hotspot shifting) |

**OC-Softmax is decisively the most noise-invariant model** in both score stability and attention map consistency. Its decisions barely change between clean and noisy conditions. This is a double-edged sword: it maintains correct decisions robustly, but also maintains incorrect decisions robustly; noise cannot accidentally correct its errors the way it sometimes can for baseline.

---

## Key Findings

**Three distinct attention architectures emerged from different training regimes.** The baseline uses a two-pathway strategy (broad LFCC + Delta for bonafide, sparse Delta-delta for spoof). Noise augmentation produced a dense, multi-band strategy that engages all three frequency bands for both decisions. OC-Softmax produced an extremely sparse, focal strategy that commits to a small number of high-precision detections.

**Noise augmentation broadened the feature space but did not solve the fundamental problem.** The noise-augmented model engages more frequency bands and more temporal regions than baseline, making it more robust to isolated cue masking. This came at the cost of bonafide recognition; the model became generally more suspicious, reversing spoof collapse into bonafide collapse. The saliency maps show that the noise-augmented model's misclassified bonafide samples under noise often have bonafide-like attention patterns but receive strongly negative scores, revealing that the model has learned to distrust its own bonafide evidence.

**OC-Softmax fundamentally changed the decision geometry.** The angular margin loss produced scores compressed to a tight range around zero, attention patterns dramatically sparser than either baseline or noise-augmented, and decisions remarkably stable across conditions. Even when misclassifying, OC-Softmax shows mixed attention signals and marginal scores; it is never confidently wrong in the way baseline and noise-augmented can be.

**The Delta (20 to 39) band gained importance under OC-Softmax.** While baseline relies on Delta-delta (40 to 59) for spoof detection and LFCC (0 to 19) + Delta (20 to 39) for bonafide, OC-Softmax shows proportionally more Delta band activation in both pathways. This shift toward first-order temporal dynamics may be a more noise-robust feature basis, since Delta features are less susceptible to noise-induced acceleration artifacts that confuse Delta-delta-based detection.

**Codec compression remains invisible to every model.** All three models show attention patterns indistinguishable from their respective clean patterns under MP3 64kbps and Opus 32kbps. This is a shared architectural property of the LFCC-LCNN feature extraction pipeline, independent of training strategy.

**Short clips amplify each model's existing strategy.** Baseline's concentrated attention becomes more extreme at 3s. The noise-augmented model's dense coverage becomes even more saturated. OC-Softmax's sparse strategy becomes even sparser, sometimes relying on a single hotspot. The fundamental attention architecture is preserved but intensified under temporal constraint.

**OC-Softmax is the most noise-invariant model but not the most accurate.** The attention maps and score swings consistently show OC-Softmax as the most stable across conditions. This stability does not translate into the best EER because the model also stabilizes its errors. The optimal model would combine OC-Softmax's noise-invariant, focal attention with features that are more reliably discriminative under diverse conditions.

---

## Implications for Future Model Development

**Attention sparsity correlates with noise robustness.** OC-Softmax's focal strategy is the most noise-resilient because each attention hotspot is independently informative; losing one to noise masking has less impact when the model only relies on a few strong cues. Future loss functions or architectures that encourage even more targeted feature extraction may further improve noise robustness.

**Multi-band engagement is necessary but not sufficient.** The noise-augmented model's multi-band strategy provides more feature redundancy than baseline, but without the right decision geometry (loss function), the model cannot effectively leverage this redundancy. Combining noise augmentation with angular margin loss (as OC-Softmax does) shows the most promise.

**The bonafide collapse problem needs targeted attention.** Both the noise-augmented model and OC-Softmax suffer from bonafide collapse under noise; they correctly reject more spoofs than baseline but incorrectly reject too many bonafide samples. The saliency maps suggest this occurs because noise disrupts the sparse bonafide-recognition cues that these models rely on. A hybrid approach that maintains broad bonafide recognition (like baseline) while adding focused spoof detection (like OC-Softmax) could address this.

**Score calibration should account for model-specific dynamics.** Baseline scores swing widely, noise-augmented scores swing moderately, OC-Softmax scores barely move. Threshold calibration strategies need to be model-aware; a fixed-threshold approach works poorly for all three but for different reasons.

---

## Glossary

**Spoof collapse.** The model fails primarily by accepting fake audio as real. FAR >> FRR at a fixed threshold; most spoof samples are scored as bonafide while real speech is still correctly accepted.

**Bonafide collapse.** The model fails primarily by rejecting real audio as fake. FRR >> FAR at a fixed threshold; real speech is flagged as spoof while fake audio is still correctly rejected.

**Balanced.** FAR and FRR are roughly equal at the fixed threshold. The model's errors are distributed evenly across both classes.

**Score swing.** The absolute difference in a model's score on the same sample between two conditions (e.g., clean vs. babble 10dB). Larger swings indicate greater sensitivity to the perturbation.
