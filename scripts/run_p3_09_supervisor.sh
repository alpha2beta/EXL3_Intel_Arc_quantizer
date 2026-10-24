#!/usr/bin/env bash
# run_p3_09_supervisor.sh — Relaunch-on-failure supervisor for full 32-layer conversion.
#
# Usage:
#   BOARD=b580 ./scripts/run_p3_09_supervisor.sh
#   BOARD=b60  ./scripts/run_p3_09_supervisor.sh

set -u

BOARD="${BOARD:-b580}"
K="${K:-4}"
VENV="/home/yanchun/.venvs/exl3"
REPO_DIR="/mnt/usb/EXL3_Intel_Arc_quantizer"

cd "$REPO_DIR" || exit 1

export PATH="/opt/intel/oneapi/compiler/latest/bin:/home/yanchun/.local/bin:$PATH"
export TORCH_XPU_ARCH_LIST=""
export PYTHONPATH="."

if [ "$BOARD" = "b580" ]; then
    export ZE_AFFINITY_MASK=1
    export EXL3_MEMORY_RESERVE_GIB=4.0
    export EXL3_MEMORY_REFLECT_OUTSTANDING=1
    export EXL3_MAX_LAYERS_PER_RUN="${EXL3_MAX_LAYERS_PER_RUN:-1}"
elif [ "$BOARD" = "b60" ]; then
    export ZE_AFFINITY_MASK=0
    unset EXL3_MEMORY_RESERVE_GIB 2>/dev/null || true
    unset EXL3_MEMORY_REFLECT_OUTSTANDING 2>/dev/null || true
    export EXL3_MAX_LAYERS_PER_RUN="${EXL3_MAX_LAYERS_PER_RUN:-1}"
fi

MODEL_DIR="${MODEL_DIR:-/mnt/usb/models/jev9b}"
OUT_DIR="${OUT_DIR:-/mnt/usb/models/jev9b_exl3_k${K}_${BOARD}}"
EVIDENCE_DIR="${EVIDENCE_DIR:-reports/evidence_jev9b_k${K}_${BOARD}}"
CORPUS="${CORPUS:-/mnt/usb/models/qwen38/calibration_tokens_wikitext2_test.json}"
mkdir -p "$EVIDENCE_DIR"
LOG_FILE="${EVIDENCE_DIR}/run.log"

echo "=================================================================" | tee -a "$LOG_FILE"
echo "Starting P3-09 Supervisor for Board: ${BOARD^^} (K=${K}) at $(date -u)" | tee -a "$LOG_FILE"
echo "Output Directory: $OUT_DIR" | tee -a "$LOG_FILE"
echo "Log File:         $LOG_FILE" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

ATTEMPT=1
MAX_ATTEMPTS=100

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    MAX_ATTEMPTS=100
    echo "" | tee -a "$LOG_FILE"
    echo "--- Supervisor Attempt $ATTEMPT / $MAX_ATTEMPTS at $(date -u) ---" | tee -a "$LOG_FILE"

    # Check if conversion is already complete
    if [ -f "${OUT_DIR}/quantization_config.json" ]; then
        if grep -q '"status": "completed"' "${OUT_DIR}/quantization_config.json"; then
            echo ">>> Quantization manifest records status: completed! Exiting supervisor successfully." | tee -a "$LOG_FILE"
            exit 0
        fi
    fi

    # Launch driver
    "$VENV/bin/python" scripts/run_p3_09.py \
        --board "$BOARD" \
        -K "$K" \
        --model-dir "$MODEL_DIR" \
        --output-dir "$OUT_DIR" \
        --corpus "$CORPUS" \
        --evidence "$EVIDENCE_DIR" \
        --resume 2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        if [ -f "${OUT_DIR}/quantization_config.json" ] && grep -q '"status": "completed"' "${OUT_DIR}/quantization_config.json"; then
            echo ">>> Conversion completed cleanly with exit code 0." | tee -a "$LOG_FILE"
            exit 0
        fi
    fi

    echo ">>> Driver exited with code $EXIT_CODE. Checking status in 5s before next attempt..." | tee -a "$LOG_FILE"
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))
done

echo ">>> ERROR: Exceeded maximum supervisor attempts ($MAX_ATTEMPTS)." | tee -a "$LOG_FILE"
exit 1
