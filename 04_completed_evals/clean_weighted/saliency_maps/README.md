# Clean-Weighted Baseline Saliency Maps

GradCAM heatmaps for the clean-weighted LFCC-LCNN baseline (pre-trained 2019 LA weights, BCELoss, no noise augmentation). Each PNG stacks the raw LFCC spectrogram on top and the GradCAM attention overlay below. 108 maps total across 10 conditions and two narrative example sets.

## Model

**Training.** Pre-trained ASVspoof 2019 LA weights, fine-tuned on ASVspoof5 training set. Clean EER 38.72%. Under noise, the model collapses toward spoof acceptance (FAR >> FRR). See `PROJECT.md` §"Baseline Results" for the full results table.

**Decision threshold.** The EER threshold is 1.2149. Stratification into correct high-confidence, correct low-confidence, and misclassified buckets uses this threshold as the decision boundary, not score sign. Two bonafide and two spoof samples are drawn per bucket per condition (12 maps per condition). Sample seed: 42.

## How to Read the Plots

**LFCC feature bands (y-axis).** The 60-dimensional feature splits into three bands. LFCC (indices 0 to 19) captures basic voice features: tone, pitch, vocal quality, and overall frequency content. Delta (20 to 39) captures how those features change over time, the speed of transitions between sounds. Delta-delta (40 to 59) captures how those rates of change themselves speed up or slow down, and is the most sensitive band for subtle synthesis artifacts.

**Scores.** The model outputs a single raw score per utterance. Positive scores lean bonafide, negative lean spoof, and values near the EER threshold (1.2149) indicate uncertainty. Higher magnitude means higher confidence.

**Heatmap intensity.** Red and yellow regions indicate high attention; blue indicates low. The heatmap shows which time-frequency regions most influenced the model's prediction, not where synthesis is located. The same attention pattern can drive either a bonafide or a spoof decision depending on which bands activate.

## Subfolder Index

Per-condition folders each contain 12 maps (two bonafide plus two spoof per bucket, across three buckets). EER and failure modes come from `../robustness_summary.csv`.

- `noise_ambient_20dB/`, `noise_ambient_10dB/` — ambient noise; moderate-to-severe spoof collapse
- `noise_babble_20dB/`, `noise_babble_10dB/` — babble noise; worst case is babble 10dB (FAR 78.73%, the dominant operational failure)
- `mp3_64kbps/`, `opus_32kbps/` — codec compression; patterns indistinguishable from clean
- `short_3s/`, `short_5s/` — center-cropped clips; bonafide collapse
- `corrected_examples/` — samples misclassified on clean that were accidentally corrected under babble 10dB
- `flipped_examples/` — samples whose prediction flipped between clean and babble 10dB (E_0005089198 spoof, E_0000994221 bonafide)

## Model-Specific Attention Patterns

**Two-pathway attention.** Bonafide classification activates broad LFCC and Delta coverage tracking speech-active regions, reading the overall voice characteristics and how they evolve. Spoof detection activates sparse, localized Delta-delta hotspots at specific time points, picking up synthesis artifacts in the fine-grained acceleration of voice features. These two pathways are qualitatively distinct and are the baseline's entire decision mechanism.

**Low-confidence predictions are fragmented versions of the same pathways.** Rather than a different attention mode, uncertain scores arise from incomplete activation of either the bonafide or the spoof pathway. The model finds partial cues but not enough to commit.

**Misclassifications are pathway confusions, not attention failures.** Misclassified bonafide triggers the spoof pathway (Delta-delta activation from recording or transmission artifacts). Misclassified spoof triggers the bonafide pathway (broad LFCC and Delta coverage from high-quality synthesis that mimics natural speech). In both cases the attention is strong, just pointed at the wrong cues.

**Noise destroys the spoof-detection pathway.** Under babble 10dB, the sparse Delta-delta hotspots used to catch synthesis artifacts disappear, replaced by the broad LFCC and Delta coverage normally associated with bonafide. The model doesn't lose confidence; it confidently mislabels noisy spoof audio as natural speech. This is the mechanism behind the 78.73% FAR under babble 10dB.

## See Also

- `../../ocsoftmax_v1/saliency_maps/README.md` — cross-model comparison against the noise-augmented and OC-Softmax variants on the same conditions.
- `../../evaluation_results.xlsx` — combined EER / FAR / FRR tables for every condition and every model.
- `PROJECT.md` — full project context (augmentation setup, enhancement strategy, risk scoring).
