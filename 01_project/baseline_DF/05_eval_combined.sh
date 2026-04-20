#!/bin/bash
########################
# Evaluation script for combined augmentation conditions
#
# Runs inference for 3 combined conditions:
#   1. Codec + Noise:      MP3 64kbps + babble 20dB
#   2. Noise + Short clip: babble 20dB + 3s
#   3. Codec + Short clip: MP3 64kbps + 3s
#
# Usage:
#   conda activate pytorch-asvspoof2021
#   cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   bash 05_eval_combined.sh [condition_number]
#
# Examples:
#   bash 05_eval_combined.sh      # run all 3
#   bash 05_eval_combined.sh 1    # codec + noise only
#   bash 05_eval_combined.sh 2    # noise + short only
#   bash 05_eval_combined.sh 3    # codec + short only
########################

set -e

MODEL=../../03_completed_training/clean_weighted/trained_network.pt
SCORE_DIR=../../04_completed_evals/clean_weighted/augmented
PROTOCOL=~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv
CODEC_DIR=~/asvspoof5_dataset/augmented/mp3_64kbps
EVAL_SCRIPTS=../../02_evaluation_scripts
export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN

run_condition=${1:-all}

run_codec_with_augment() {
    # Run inference on codec dir with an on-the-fly augmentation
    # Args: $1=condition_name, $2=codec_dir, $3=augment_flag
    local cond_name=$1
    local codec_dir=$2
    local augment=$3
    local log_path="${SCORE_DIR}/inference_${cond_name}"

    # Ensure .lst file exists for the codec dir
    local lst_name="mp3_64kbps"
    local lst_path=~/asvspoof5_dataset/scp/${lst_name}.lst
    if [ ! -f "${lst_path}" ]; then
        echo "Generating file list from ${codec_dir}..."
        ls "${codec_dir}" | sed 's/\.flac$//' > "${lst_path}"
    fi

    export TEMP_DATA_NAME=${lst_name}
    export TEMP_DATA_DIR=${codec_dir}

    echo "=== ${cond_name} ==="
    echo "  Audio dir: ${codec_dir}"
    echo "  Augment:   ${augment}"
    echo "  Log:       ${log_path}.log"

    PYTHONUNBUFFERED=1 python main.py \
        --inference \
        --model-forward-with-file-name \
        --trained-model "${MODEL}" \
        --module-config config_auto \
        --augment "${augment}" \
        --num-workers 2 \
        --batch-size 16 \
        --cudnn-benchmark-toggle \
        > "${log_path}.log" 2>&1

    # Extract scores
    grep "Output," "${log_path}.log" | awk '{print $2" "$4}' | sed 's:,::g' > "${SCORE_DIR}/scores_${cond_name}.txt"
    local num_scores=$(wc -l < "${SCORE_DIR}/scores_${cond_name}.txt")
    echo "  Scores: ${num_scores}"

    # Clean up
    rm -f "${lst_name}_utt_length.dic"

    # Compute EER
    python ${EVAL_SCRIPTS}/evaluator.py --protocol "${PROTOCOL}" --scores "${SCORE_DIR}/scores_${cond_name}.txt"
    echo ""
}

run_onthefly_combined() {
    # Run inference on original eval set with composite on-the-fly augmentation
    # Args: $1=condition_name, $2=augment_flag
    local cond_name=$1
    local augment=$2
    local log_path="${SCORE_DIR}/inference_${cond_name}"

    echo "=== ${cond_name} ==="
    echo "  Audio dir: original eval set"
    echo "  Augment:   ${augment}"
    echo "  Log:       ${log_path}.log"

    PYTHONUNBUFFERED=1 python main.py \
        --inference \
        --model-forward-with-file-name \
        --trained-model "${MODEL}" \
        --augment "${augment}" \
        --num-workers 2 \
        --batch-size 16 \
        --cudnn-benchmark-toggle \
        > "${log_path}.log" 2>&1

    # Extract scores
    grep "Output," "${log_path}.log" | awk '{print $2" "$4}' | sed 's:,::g' > "${SCORE_DIR}/scores_${cond_name}.txt"
    local num_scores=$(wc -l < "${SCORE_DIR}/scores_${cond_name}.txt")
    echo "  Scores: ${num_scores}"

    # Clean up
    rm -f "eval_utt_length.dic"

    # Compute EER
    python ${EVAL_SCRIPTS}/evaluator.py --protocol "${PROTOCOL}" --scores "${SCORE_DIR}/scores_${cond_name}.txt"
    echo ""
}

# Condition 1: Codec + Noise (MP3 64kbps + babble 20dB)
if [ "${run_condition}" = "all" ] || [ "${run_condition}" = "1" ]; then
    run_codec_with_augment "mp3_noise_babble_20dB" "${CODEC_DIR}" "noise_babble_20dB"
fi

# Condition 2: Noise + Short clip (babble 20dB + 3s)
if [ "${run_condition}" = "all" ] || [ "${run_condition}" = "2" ]; then
    run_onthefly_combined "noise_babble_20dB_short_3s" "short_3s+noise_babble_20dB"
fi

# Condition 3: Codec + Short clip (MP3 64kbps + 3s)
if [ "${run_condition}" = "all" ] || [ "${run_condition}" = "3" ]; then
    run_codec_with_augment "mp3_short_3s" "${CODEC_DIR}" "short_3s"
fi

echo "Done. Score files saved to ${SCORE_DIR}/"
