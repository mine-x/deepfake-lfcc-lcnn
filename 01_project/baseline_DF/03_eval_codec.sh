#!/bin/bash
########################
# Evaluation script for codec-augmented eval sets
#
# Generates .lst file, runs inference, extracts scores, and cleans up.
# Uses config_auto.py so config.py is not modified.
#
# Usage:
#   conda activate pytorch-asvspoof2021
#   cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   bash 03_03_eval_codec.sh <codec_dir> <condition_name> [trained_model]
#
# Examples:
#   bash 03_03_eval_codec.sh ~/asvspoof5_dataset/augmented/mp3_64kbps mp3_64kbps
#   bash 03_03_eval_codec.sh ~/asvspoof5_dataset/augmented/opus_32kbps opus_32kbps
#   bash 03_03_eval_codec.sh ~/asvspoof5_dataset/augmented/mp3_64kbps mp3_64kbps ../../trained_network.pt
########################

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash 03_03_eval_codec.sh <codec_dir> <condition_name> [trained_model]"
    echo "  codec_dir:      absolute path to codec-augmented FLAC directory"
    echo "  condition_name:  e.g. mp3_64kbps, opus_32kbps"
    echo "  trained_model:   path to model (default: ../../trained_network.pt)"
    exit 1
fi

codec_dir=$1
condition_name=$2
trained_model=${3:-../../trained_network.pt}

if [ ! -d "${codec_dir}" ]; then
    echo "Error: Directory not found: ${codec_dir}"
    exit 1
fi

if [ ! -f "${trained_model}" ]; then
    echo "Error: Model not found: ${trained_model}"
    exit 1
fi

# Generate .lst file in scp directory (reuse if already exists)
lst_dir=~/asvspoof5_dataset/scp
lst_path="${lst_dir}/${condition_name}.lst"
if [ -f "${lst_path}" ]; then
    num_files=$(wc -l < "${lst_path}")
    echo "Reusing existing ${lst_path} (${num_files} files)"
else
    echo "Generating file list from ${codec_dir}..."
    ls "${codec_dir}" | sed 's/\.flac$//' > "${lst_path}"
    num_files=$(wc -l < "${lst_path}")
    echo "Saved ${lst_path} (${num_files} files)"
fi

# Set env vars for config_auto.py
export TEMP_DATA_NAME=${condition_name}
export TEMP_DATA_DIR=${codec_dir}
export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN

log_name="../../inference_${condition_name}"

echo "Running inference: ${condition_name}"
echo "Model: ${trained_model}"
echo "Log: ${log_name}.log"

PYTHONUNBUFFERED=1 python main.py \
    --inference \
    --model-forward-with-file-name \
    --trained-model "${trained_model}" \
    --module-config config_auto \
    --num-workers 2 \
    --batch-size 16 \
    --cudnn-benchmark-toggle \
    > "${log_name}.log" 2>&1

# Extract scores
grep "Output," "${log_name}.log" | awk '{print $2" "$4}' | sed 's:,::g' > "${log_name}_scores.txt"
num_scores=$(wc -l < "${log_name}_scores.txt")
echo "Extracted ${num_scores} scores to ${log_name}_scores.txt"

# Clean up intermediate files (lst kept in scp dir for reuse)
rm -f "${condition_name}_utt_length.dic"

# Compute EER
protocol=~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv
echo "Computing EER..."
python ../../02_evaluation_scripts/evaluator.py --protocol "${protocol}" --scores "${log_name}_scores.txt"
