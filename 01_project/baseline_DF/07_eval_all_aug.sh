#!/bin/bash
########################
# Full augmented evaluation suite
#
# Runs clean + all augmented conditions, extracts scores, computes EER,
# and runs failure analysis. Works with any trained model.
#
# Usage:
#   conda activate pytorch-asvspoof2021
#   cd ~/2021/DF/Baseline-LFCC-LCNN/01_project/baseline_DF
#   export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN
#   bash 07_eval_all_aug.sh <output_dir> [trained_model] [noise_subdir]
#
# Examples:
#   bash 07_eval_all_aug.sh ../../04_completed_evals/noise_aug
#   bash 07_eval_all_aug.sh ../../04_completed_evals/noise_aug ../../trained_network.pt
#   bash 07_eval_all_aug.sh ../../04_completed_evals/noise_aug ../../trained_network.pt eval_noise_extended
########################

set -e

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash 07_eval_all_aug.sh <output_dir> [trained_model] [noise_subdir]"
    echo "  output_dir:    directory for logs and score files"
    echo "  trained_model: path to model (default: ../../trained_network.pt)"
    echo "  noise_subdir:  noise clip subdirectory under musan/ (default: selected)"
    exit 1
fi

OUTPUT_DIR=$1
trained_model=${2:-../../trained_network.pt}
NOISE_SUBDIR=${3:-train}
PROTOCOL=~/asvspoof5_dataset/asvspoof5_protocols/ASVspoof5.eval.track_1.tsv
EVAL_SCRIPTS=../../02_evaluation_scripts
CODEC_MP3=~/asvspoof5_dataset/augmented/mp3_64kbps
CODEC_OPUS=~/asvspoof5_dataset/augmented/opus_32kbps
export PYTHONPATH=~/2021/DF/Baseline-LFCC-LCNN

if [ ! -f "${trained_model}" ]; then
    echo "Error: Model not found: ${trained_model}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Full augmented evaluation suite"
echo "Model: ${trained_model}"
echo "Output: ${OUTPUT_DIR}"
echo "Noise subdir: ${NOISE_SUBDIR}"
echo "========================================"

NOISE_ARGS="--noise-subdir ${NOISE_SUBDIR}"
# Extra args passed via environment (e.g. EXTRA_ARGS="--oc-softmax --emb-dim 64")
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Helper: run inference on standard eval set (config.py) ---
run_standard() {
    local cond_name=$1
    local augment=$2  # empty string for clean
    local log_path="${OUTPUT_DIR}/inference_${cond_name}"

    if [ -f "${OUTPUT_DIR}/scores_${cond_name}.txt" ]; then
        echo "[SKIP] ${cond_name} -- scores already exist"
        return
    fi

    echo ""
    echo "=== ${cond_name} ==="
    local aug_args=""
    if [ -n "${augment}" ]; then
        aug_args="--augment ${augment}"
        echo "  Augment: ${augment}"
    fi

    PYTHONUNBUFFERED=1 python main.py \
        --inference \
        --model-forward-with-file-name \
        --trained-model "${trained_model}" \
        ${aug_args} \
        ${NOISE_ARGS} \
        ${EXTRA_ARGS} \
        --num-workers 8 \
        --batch-size 32 \
        --cudnn-benchmark-toggle \
        > "${log_path}.log" 2>&1

    grep "Output," "${log_path}.log" | awk '{print $2" "$4}' | sed 's:,::g' > "${OUTPUT_DIR}/scores_${cond_name}.txt"
    local n=$(wc -l < "${OUTPUT_DIR}/scores_${cond_name}.txt")
    echo "  Scores: ${n}"

    python ${EVAL_SCRIPTS}/evaluator.py --protocol "${PROTOCOL}" --scores "${OUTPUT_DIR}/scores_${cond_name}.txt"
}

# --- Helper: run inference on codec-augmented eval set (config_auto.py) ---
run_codec() {
    local cond_name=$1
    local codec_dir=$2
    local lst_name=$3
    local augment=$4  # optional on-the-fly augmentation on top of codec
    local log_path="${OUTPUT_DIR}/inference_${cond_name}"

    if [ -f "${OUTPUT_DIR}/scores_${cond_name}.txt" ]; then
        echo "[SKIP] ${cond_name} -- scores already exist"
        return
    fi

    # Ensure .lst file exists
    local lst_path=~/asvspoof5_dataset/scp/${lst_name}.lst
    if [ ! -f "${lst_path}" ]; then
        ls "${codec_dir}" | sed 's/\.flac$//' > "${lst_path}"
    fi

    export TEMP_DATA_NAME=${lst_name}
    export TEMP_DATA_DIR=${codec_dir}

    echo ""
    echo "=== ${cond_name} ==="
    echo "  Audio dir: ${codec_dir}"
    local aug_args=""
    if [ -n "${augment}" ]; then
        aug_args="--augment ${augment}"
        echo "  Augment: ${augment}"
    fi

    PYTHONUNBUFFERED=1 python main.py \
        --inference \
        --model-forward-with-file-name \
        --trained-model "${trained_model}" \
        --module-config config_auto \
        ${aug_args} \
        ${NOISE_ARGS} \
        ${EXTRA_ARGS} \
        --num-workers 2 \
        --batch-size 16 \
        --cudnn-benchmark-toggle \
        > "${log_path}.log" 2>&1

    grep "Output," "${log_path}.log" | awk '{print $2" "$4}' | sed 's:,::g' > "${OUTPUT_DIR}/scores_${cond_name}.txt"
    local n=$(wc -l < "${OUTPUT_DIR}/scores_${cond_name}.txt")
    echo "  Scores: ${n}"

    rm -f "${lst_name}_utt_length.dic"

    python ${EVAL_SCRIPTS}/evaluator.py --protocol "${PROTOCOL}" --scores "${OUTPUT_DIR}/scores_${cond_name}.txt"
}

# ========== Single conditions ==========

# Clean
run_standard "clean" ""

# Noise conditions
run_standard "noise_ambient_20dB" "noise_ambient_20dB"
run_standard "noise_ambient_10dB" "noise_ambient_10dB"
run_standard "noise_babble_20dB" "noise_babble_20dB"
run_standard "noise_babble_10dB" "noise_babble_10dB"

# Short clip conditions
run_standard "short_3s" "short_3s"
run_standard "short_5s" "short_5s"

# Codec conditions
run_codec "mp3_64kbps" "${CODEC_MP3}" "mp3_64kbps" ""
run_codec "opus_32kbps" "${CODEC_OPUS}" "opus_32kbps" ""

# ========== Combined conditions ==========

# Codec + Noise (MP3 64kbps + babble 20dB)
run_codec "mp3_noise_babble_20dB" "${CODEC_MP3}" "mp3_64kbps" "noise_babble_20dB"

# Noise + Short clip (babble 20dB + 3s)
run_standard "noise_babble_20dB_short_3s" "short_3s+noise_babble_20dB"

# Codec + Short clip (MP3 64kbps + 3s)
run_codec "mp3_short_3s" "${CODEC_MP3}" "mp3_64kbps" "short_3s"

# ========== Failure analysis ==========

echo ""
echo "========================================"
echo "Running failure analysis..."
echo "========================================"

python ${EVAL_SCRIPTS}/failure_analysis.py \
    --protocol "${PROTOCOL}" \
    --clean-scores "${OUTPUT_DIR}/scores_clean.txt" \
    --score-dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/robustness_summary.csv"

echo ""
echo "Done. All results in ${OUTPUT_DIR}/"
