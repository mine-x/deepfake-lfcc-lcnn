#!/bin/bash
########################
# Generate GradCAM saliency maps for all augmented conditions.
#
# Usage:
#   conda activate pytorch-asvspoof2021
#   cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   bash 04_gen_saliency.sh
########################

set -e

export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN

MODEL=../../03_completed_training/clean_weighted/trained_network.pt
PROTOCOL=~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv
AUDIO_DIR=~/asvspoof5_dataset/flac_E_eval
OUT_BASE=../../04_completed_evals/clean_weighted/augmented/saliency_maps
SCORE_DIR=../../04_completed_evals/clean_weighted/augmented
N=2

# On-the-fly conditions (noise + short clips)
for condition in noise_ambient_20dB noise_ambient_10dB noise_babble_20dB noise_babble_10dB short_3s short_5s; do
    echo "=========================================="
    echo "Generating saliency maps: ${condition}"
    echo "=========================================="
    python gradcam.py \
        --model "${MODEL}" \
        --audio-dir "${AUDIO_DIR}" \
        --protocol "${PROTOCOL}" \
        --output-dir "${OUT_BASE}/${condition}" \
        --scores "${SCORE_DIR}/scores_${condition}.txt" \
        --augment "${condition}" \
        --n "${N}"
done

# Codec conditions (pre-generated audio, no --augment flag)
for condition in mp3_64kbps opus_32kbps; do
    echo "=========================================="
    echo "Generating saliency maps: ${condition}"
    echo "=========================================="
    python gradcam.py \
        --model "${MODEL}" \
        --audio-dir ~/asvspoof5_dataset/augmented/${condition} \
        --protocol "${PROTOCOL}" \
        --output-dir "${OUT_BASE}/${condition}" \
        --scores "${SCORE_DIR}/scores_${condition}.txt" \
        --n "${N}"
done

echo "=========================================="
echo "All saliency maps generated under ${OUT_BASE}/"
echo "=========================================="
