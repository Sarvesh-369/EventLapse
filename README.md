# EventLapse: Trace-Grounded Parametric Profiling of Temporal Capabilities in Frontier Video LLMs

> **Paper Title:** *EventLapse: Where Do Frontier Video LLMs Lose Events? Trace-Grounded Parametric Profiling of Temporal Capabilities*

EventLapse is a research framework for profiling and diagnosing the temporal reasoning capabilities of frontier Video-Language Models (VLMs) across 7 synthetic video tasks using Manim Community Edition and FFmpeg, aligned with the **MORSE** executable trace evaluation methodology.

---

## 🚀 Repository Quickstart & Installation

### 1. Clone & Install Dependencies
```bash
cd EventLapse
pip install -e .
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```
Ensure `GEMINI_API_KEY` is configured for Google Gemini experiments.

---

## 📋 Comprehensive Experiment Instructions

### 1. Check Available Models & Discovery
Inspect provider capabilities and discover available Gemini model IDs:
```bash
python3 scripts/check_available_models.py --provider google
```

### 2. Generate Synthetic Dataset (Smoke Test or Full 20 Seeds)
To render 2 smoke-test sample videos per task:
```bash
python3 scripts/generate_dataset.py --num-seeds 2 --tasks all
```
To generate the full paper dataset (20 seeds per control value across all tasks):
```bash
python3 scripts/generate_dataset.py --num-seeds 20 --tasks all
```
Videos are saved in `data/videos/{task_name}/`, traces in `data/traces/{task_name}/`, and ground-truth in `data/gt/{task_name}/`.

### 3. Validate Generated Dataset
Validate dataset samples against duration, event count, and contrast rules:
```bash
python3 scripts/validate_dataset.py
```

### 4. Run Model Smoke Test
Run a quick smoke test on model loading and optional video query:
```bash
python3 scripts/smoke_test_model.py --provider google --model-name gemini-3.5-flash
```

### 5. Run Capability Boundaries Experiment Sweep (Group 1)
Run Gemini 3.5 Flash on native video:
```bash
python3 scripts/run_experiment.py --provider google --model-name gemini-3.5-flash --input-mode native_video --prompt-condition structured_trace
```

### 6. Run Frame-Density Interventions (Group 3A)
Compare native video vs 2 FPS vs 10 FPS frame sampling:
```bash
python3 scripts/run_experiment.py --provider google --model-name gemini-3.5-flash --input-mode frames_2fps --prompt-condition structured_trace
python3 scripts/run_experiment.py --provider google --model-name gemini-3.5-flash --input-mode frames_10fps --prompt-condition structured_trace
```

### 7. Run Prompting & Thinking Interventions (Group 3C)
```bash
python3 scripts/run_experiment.py --provider google --model-name gemini-3.5-flash --input-mode native_video --prompt-condition direct
python3 scripts/run_experiment.py --provider google --model-name gemini-3.5-flash --input-mode native_video --prompt-condition thinking
```

### 8. Master Script to Run All Experiments
To run the master automated pipeline script:
```bash
./scripts/run_all_experiments.sh
```

### 9. Aggregate Results & Generate Paper Figures
Consolidate results into CSV and render 7-panel paper figures:
```bash
python3 scripts/aggregate_results.py
python3 scripts/make_figures.py
```

### 10. Run Unit & Integration Tests
```bash
pytest tests/
```

---

## 📁 Repository Structure
- `configs/`: Model, candidate, generation, and task YAML configs.
- `reference_code/`: Standalone reference scripts for synthetic Manim video generators.
- `data/`: Output dataset directory (`videos/`, `traces/`, `gt/`, `manifest.jsonl`). *Excluded from git tracking.*
- `src/eventlapse/`: Core python package:
  - `generation/`: Manim synthetic task video & trace generators.
  - `traces/`: Schemas, serialization, and validation.
  - `models/`: Abstract interface, `load_model.py` mandatory entry point, and Gemini / provider adapters.
  - `inference/`: Prompts, runner, and response parser.
  - `interventions/`: 2 FPS / 10 FPS frame extraction, oracle evidence, prompting, and thinking controls.
  - `evaluation/`: Exact match, Wilson 95% CIs, operational capability boundaries ($\tau=0.80$), and MORSE trace evaluator.
  - `datasets/`: RepCount-A natural transfer dataset parsers.
  - `experiments/`: Experiment group routines.
  - `utils/`: Logging, caching, seeds, paths.
- `scripts/`: Master runner scripts, dataset generation, model discovery, aggregation, and plotting.
- `tests/`: Pytest unit and integration test suite.
