# EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries

> **Paper Title:** *EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries*

EventLapse is a research framework for profiling and diagnosing the temporal reasoning capabilities of frontier Video-Language Models (VLMs) on **Event Counting** across **3 synthetic video domains** rendered with Manim Community Edition, using the **MORSE** executable trace evaluation methodology with Trace Precision / Recall / F1 scoring and operational capability boundary estimation.

> 📖 **Detailed Experiments Guide:** See [EXPERIMENTS.md](file:///Users/sarvesh/Documents/VLM_failures/EventLapse/EXPERIMENTS.md) for full technical details on each of the 5 paper experiments.

---

## 🚀 Quickstart & Installation

### 1. Clone & Install

```bash
git clone https://github.com/Sarvesh-369/EventLapse.git
cd EventLapse
pip install -e .
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Set API keys for the providers you intend to use:

| Provider | Environment Variable / Spec | Notes |
| :--- | :--- | :--- |
| Google Gemini | `GEMINI_API_KEY` | Native video supported |
| OpenAI | `OPENAI_API_KEY` | Frame sequences only |
| Anthropic | `ANTHROPIC_API_KEY` | Frame sequences only |
| **PropensityBench Gateway** | `PROPENSITY_GATEWAY_URL`, `RATE_PM=60` | Model spec: `<provider>/<original_model_name>` |
| **vLLM (open-source)** | `VLLM_BASE_URL` (no API key needed) | See §Open-Source Models via vLLM |

---

## 📊 Event Counting Visual Domains ($N \times F$ Matrix)

| Visual Domain | Event Description | Target Count Axis ($N$) | Frequency Axis ($F$) | Fixed Video Duration |
| :--- | :--- | :---: | :---: | :---: |
| **`bounce_ball`** | Ball contacting walls | $N \in \{0, 1, 2, 3, 4, 5, 6, 8, 10, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$ | 24.0 s |
| **`blinking`** | Object light pulses/blinks | $N \in \{0, 1, 2, 3, 4, 5, 6, 8, 10, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$ | 24.0 s |
| **`state_machine`** | Visual state transitions $\{A, B, C, D\}$ | $N \in \{0, 1, 2, 3, 4, 5, 6, 8, 10, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$ | 24.0 s |

Videos are constant-duration per task ($24.0\text{s}$) to prevent total duration from being a confounding variable.

---

## ⚡ Master Script: Run All 5 Experiments for Any Model

Run all 5 experiments sequentially for a model with a single command:

```bash
# Run all 5 experiments for Google Gemini 2.0 Flash
./scripts/run_all_experiments.sh google gemini-2.0-flash

# Run all 5 experiments via PropensityBench Gateway
./scripts/run_all_experiments.sh propensity gemini/gemini-3.1-pro-preview

# Run all 5 experiments for local vLLM open-source model
./scripts/run_all_experiments.sh vllm Qwen/Qwen2-VL-7B-Instruct
```

---

## 🧪 Step-by-Step Experiment Workflows

### 1. Generate Synthetic Dataset ($N \times F$)

```bash
# Quick test dataset (2 seeds per setting)
python3 scripts/generate_dataset.py --num-seeds 2 --tasks all

# Full benchmark dataset (20 seeds per setting)
python3 scripts/generate_dataset.py --num-seeds 20 --tasks all
```

Outputs are saved to:
- `data/videos/{domain}/` — Rendered `.mp4` video files
- `data/traces/{domain}/` — Ground-truth executable JSON traces
- `data/gt/{domain}/` — Ground-truth answer files
- `data/manifest.jsonl` — Dataset index manifest

### 2. Individual Experiment Commands

```bash
# Experiment 1: Full N x F Matrix Sweep
python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode native_video --prompt-condition structured_trace

# Experiment 2: Frame Sampling Density Sweep (Native, 1, 2, 4, 8, 10, 16 FPS)
for mode in native_video frames_1fps frames_2fps frames_4fps frames_8fps frames_10fps frames_16fps; do
  python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode ${mode} --prompt-condition structured_trace
done

# Experiment 3: Oracle Keyframe Evidence Sweep
python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode oracle_evidence --prompt-condition structured_trace

# Experiment 4: Prompting Strategy Sweep (5 conditions)
for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
  python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode native_video --prompt-condition ${cond}
done

# Experiment 5: Aggregate Results & Generate 2D Heatmaps
python3 scripts/aggregate_results.py
python3 scripts/make_matrix_heatmaps.py
```

---

## 🖥️ Open-Source Models via vLLM

Host any open-source Vision-Language Model using [vLLM](https://github.com/vllm-project/vllm)'s OpenAI-compatible API server:

```bash
# 1. Launch vLLM server
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

# 2. Run all 5 experiments against local vLLM
./scripts/run_all_experiments.sh vllm Qwen/Qwen2-VL-7B-Instruct
```

---

## 📏 Evaluation Metrics

| Metric | Description |
| :--- | :--- |
| **Exact Match (EM)** | Binary exact string/numeric match of final predicted answer vs. ground truth |
| **Trace Precision ($P$)** | Fraction of model-reported event steps matching ground-truth steps |
| **Trace Recall ($R$)** | Fraction of true ground-truth events successfully detected by model |
| **Trace F1 ($F_1$)** | Harmonic mean: $F_1 = \frac{2 \cdot P \cdot R}{P + R}$ |
| **Accidental Correctness Rate (ACR)** | Correct final answer but corrupted/hallucinated trace ($F_1 < 1.0$) |
| **Reasoning Failure Rate (RFR)** | Perfect trace ($F_1 = 1.0$) but wrong final answer |
| **Operational Boundary ($x^*$)** | Max difficulty where lower 95% Wilson CI bound ≥ $\tau = 0.80$ |

---

## 📁 Repository Structure

```
EventLapse/
├── configs/               # Model, generation, task, and experiment YAML configs
├── data/                  # Dataset outputs: videos/, traces/, gt/, manifest.jsonl
├── sample_data/           # Single-seed sample dataset for evaluation testing
├── scripts/
│   ├── run_all_experiments.sh  # Master script to run all 5 experiments for a model
│   ├── run_matrix_sweep.py     # N x F matrix sweep execution engine
│   ├── generate_dataset.py     # Event Counting dataset generator
│   ├── aggregate_results.py    # Consolidate results, token usage, and costs ($USD)
│   └── make_matrix_heatmaps.py # 2D N x F matrix heatmap plotter
├── src/eventlapse/
│   ├── generation/        # Event Counting Manim generators (bounce_ball, blinking, state_machine)
│   ├── models/            # Model dispatch and adapters (gemini, openai, anthropic, vllm, propensity)
│   ├── inference/         # Prompts (5 strategies), runner, and response parser
│   ├── interventions/     # Frame extraction (1–16 fps), oracle keyframe evidence, prompting controls
│   ├── evaluation/        # Exact match, Trace F1, Wilson 95% CIs, operational boundaries, MORSE evaluator
│   └── utils/             # Logging, caching, seeds, paths, cost calculator
└── tests/                 # Pytest unit test suite (14 tests)
```

---

## 🧪 Unit Tests

```bash
PYTHONPATH=src pytest tests/
```

All 14 tests pass cleanly.

---

## 📄 Citation

```bibtex
@article{eventlapse2026,
  title={EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries},
  author={},
  year={2026}
}
```
