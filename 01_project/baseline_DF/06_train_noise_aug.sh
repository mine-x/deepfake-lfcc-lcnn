#!/bin/bash
########################
# Noise-augmented training (Mode B)
#
# Fine-tunes from pre-trained 2019 LA weights with on-the-fly noise
# augmentation: 50% of training samples receive random noise
# (ambient or babble, SNR 10-25dB), 50% remain clean.
#
# Usage:
#   1. Ensure config.py is properly configured
#   2. conda activate pytorch-asvspoof2021
#   3. cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   4. export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN
#   5. bash 06_train_noise_aug.sh
########################

pretrained_model=__pretrained/trained_network.pt
log_train="../../train_noise_aug_$(date +%Y%m%d_%H%M%S).log"

if [ ! -f "${pretrained_model}" ]; then
    echo "Error: Pre-trained model not found at ${pretrained_model}"
    echo "Download it first from the ASVspoof 2021 repository."
    exit 1
fi

echo "Noise-augmented training with pre-trained weights: ${pretrained_model}"
echo "Log: ${log_train}"

PYTHONUNBUFFERED=1 nohup python main.py \
    --model-forward-with-file-name \
    --num-workers 4 \
    --batch-size 32 \
    --epochs 25 \
    --no-best-epochs 3 \
    --sampler block_shuffle_by_length \
    --lr 0.0003 \
    --lr-decay-factor 0.5 \
    --lr-scheduler-type 1 \
    --trained-model ${pretrained_model} \
    --ignore-training-history-in-trained-model \
    --train-augment \
    --warmup-steps 1000 \
    --seed 1000 \
    --cudnn-benchmark-toggle \
    --save-model-dir ../../ \
    > "${log_train}" 2>&1 &

echo "Training started in background (PID: $!)"
echo "Monitor with: tail -f ${log_train}"
