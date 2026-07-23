#!/usr/bin/env bash
set -e

# Usage: ./scripts/run_all_experiments.sh [PROVIDER] [MODEL_NAME]
# Example: ./scripts/run_all_experiments.sh google gemini-2.0-flash
# Example: ./scripts/run_all_experiments.sh propensity gemini/gemini-3.1-pro-preview
# Example: ./scripts/run_all_experiments.sh vllm Qwen/Qwen2-VL-7B-Instruct

PROVIDER=${1:-"google"}
MODEL_NAME=${2:-"gemini-2.0-flash"}

echo "=========================================================="
echo " EventLapse: Master Pipeline for Model: $PROVIDER / $MODEL_NAME"
echo "=========================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project Root: $PROJECT_ROOT"

# Ensure dataset exists, generate smoke test dataset if missing
if [ ! -f "data/manifest.jsonl" ]; then
    echo "[Data Setup] Manifest not found. Generating dataset (2 seeds per setting)..."
    python3 scripts/generate_dataset.py --num-seeds 2 --tasks all
fi

echo "=========================================================="
echo " [Experiment 1] Full N x F Matrix Capability Boundary Sweep"
echo "=========================================================="
python3 scripts/run_matrix_sweep.py \
  --provider "$PROVIDER" \
  --model-name "$MODEL_NAME" \
  --input-mode native_video \
  --prompt-condition structured_trace

echo "=========================================================="
echo " [Experiment 2] Frame Sampling Density Interventions"
echo "=========================================================="
for mode in native_video frames_1fps frames_2fps frames_4fps frames_8fps frames_10fps frames_16fps; do
  echo "--- Running Frame Density Mode: $mode ---"
  python3 scripts/run_matrix_sweep.py \
    --provider "$PROVIDER" \
    --model-name "$MODEL_NAME" \
    --input-mode "$mode" \
    --prompt-condition structured_trace
done

echo "=========================================================="
echo " [Experiment 3] Oracle Keyframe Evidence Interventions"
echo "=========================================================="
python3 scripts/run_matrix_sweep.py \
  --provider "$PROVIDER" \
  --model-name "$MODEL_NAME" \
  --input-mode oracle_evidence \
  --prompt-condition structured_trace

echo "=========================================================="
echo " [Experiment 4] Prompting & Reasoning Mode Interventions"
echo "=========================================================="
for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
  echo "--- Running Prompting Strategy: $cond ---"
  python3 scripts/run_matrix_sweep.py \
    --provider "$PROVIDER" \
    --model-name "$MODEL_NAME" \
    --input-mode native_video \
    --prompt-condition "$cond"
done

echo "=========================================================="
echo " [Experiment 5] Result Aggregation & 2D Matrix Heatmaps"
echo "=========================================================="
python3 scripts/aggregate_results.py
python3 scripts/make_matrix_heatmaps.py

echo "=========================================================="
echo " All 5 Experiments Completed for Model: $PROVIDER / $MODEL_NAME"
echo "=========================================================="
