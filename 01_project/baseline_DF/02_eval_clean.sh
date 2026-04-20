#!/bin/bash
########################
# Script for evaluation on ASVspoof5 eval set
# Usage:
#   1. Ensure config.py is properly configured
#   2. conda activate pytorch-asvspoof2021
#   3. cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   4. bash 02_eval_clean.sh [trained_model]
########################

trained_model=${1:-../../trained_network.pt}
log_name="../../inference_$(date +%Y%m%d_%H%M%S)"

if [ ! -f "${trained_model}" ]; then
    echo "Error: Model not found: ${trained_model}"
    exit 1
fi

export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN

echo "Running evaluation"
echo "Model: ${trained_model}"
echo "Log: ${log_name}.log"

PYTHONUNBUFFERED=1 python main.py --inference --model-forward-with-file-name \
    --trained-model "${trained_model}" \
    --num-workers 8 \
    --batch-size 32 \
    --cudnn-benchmark-toggle \
    > "${log_name}.log" 2>&1

grep "Output," "${log_name}.log" | awk '{print $2" "$4}' | sed 's:,::g' > "${log_name}_scores.txt"
num_scores=$(wc -l < "${log_name}_scores.txt")
echo "Extracted ${num_scores} scores to ${log_name}_scores.txt"

# Compute EER
protocol=~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv
echo "Computing EER..."
python ../../02_evaluation_scripts/evaluator.py --protocol "${protocol}" --scores "${log_name}_scores.txt"
