#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " EventLapse: Master Experiment Execution Pipeline"
echo "=========================================================="

# Ensure script is run from project root or scripts directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project Root: $PROJECT_ROOT"

# Step 1: Check available models and provider capabilities
echo "[Step 1] Checking available models and provider capabilities..."
python3 scripts/check_available_models.py --provider google

# Step 2: Validate dataset (or generate smoke test dataset if missing)
if [ ! -f "data/manifest.jsonl" ]; then
    echo "[Step 2] Generating smoke test dataset (2 seeds per task)..."
    python3 scripts/generate_dataset.py --num-seeds 2 --tasks all
fi

echo "[Step 2] Validating dataset integrity..."
python3 scripts/validate_dataset.py

# Step 3: Run Group 1 Capability Boundaries Experiment Sweep (Dry Run default)
echo "[Step 3] Running capability boundary experiment sweep (Dry Run)..."
python3 scripts/run_experiment.py --provider google --model-name gemini-3.5-flash --input-mode native_video --prompt-condition structured_trace --dry-run

# Step 4: Aggregate results and generate paper figures
echo "[Step 4] Aggregating results and generating summary plots..."
python3 scripts/aggregate_results.py || true
python3 scripts/make_figures.py || true

echo "=========================================================="
echo " Pipeline execution completed successfully!"
echo "=========================================================="
