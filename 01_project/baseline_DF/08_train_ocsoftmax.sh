#!/bin/bash
########################
# OC-Softmax training (Priority 5)
#
# Same as v5 noise-augmented training but replaces BCELoss with OC-Softmax.
# OC-Softmax learns a compact bonafide boundary with angular margin,
# improving generalization to unseen spoofing attacks.
#
# Note: Pre-trained weights are loaded with strict=False since the output
# layer changes shape (Linear(dim,1) -> Linear(dim,64)). The LFCC/LCNN/BiLSTM
# weights transfer; the output layer and OC-Softmax center are initialized fresh.
#
# Usage:
#   1. conda activate pytorch-asvspoof2021
#   2. cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   3. export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN
#   4. bash 08_train_ocsoftmax.sh
########################

pretrained_model=__pretrained/trained_network.pt
log_train="../../train_ocsoftmax_$(date +%Y%m%d_%H%M%S).log"

if [ ! -f "${pretrained_model}" ]; then
    echo "Error: Pre-trained model not found at ${pretrained_model}"
    echo "Download it first from the ASVspoof 2021 repository."
    exit 1
fi

echo "OC-Softmax + noise-augmented training: ${pretrained_model}"
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
    --oc-softmax \
    --emb-dim 64 \
    --oc-r-real 0.9 \
    --oc-r-fake 0.2 \
    --oc-alpha 20.0 \
    --seed 1000 \
    --cudnn-benchmark-toggle \
    --save-model-dir ../../ \
    > "${log_train}" 2>&1 &

echo "Training started in background (PID: $!)"
echo "Monitor with: tail -f ${log_train}"
