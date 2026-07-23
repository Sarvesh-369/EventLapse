# EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries

> **Paper Title:** *EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries*

EventLapse is a research framework for profiling and diagnosing the temporal reasoning capabilities of frontier Video-Language Models (VLMs) on **Event Counting** across **3 synthetic video domains** rendered with Manim Community Edition, using the **MORSE** executable trace evaluation methodology with Trace Precision / Recall / F1 scoring and operational capability boundary estimation.

> 📖 **Detailed Experiments Guide:** See [EXPERIMENTS.md](file:///Users/sarvesh/Documents/VLM_failures/EventLapse/EXPERIMENTS.md) for full technical documentation on the $N \times F$ matrix sweep framework, PropensityBench gateway usage, vLLM hosting, interventions, error taxonomy, and metrics.

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

## 🧪 Running Experiments

### 1. Generate Synthetic Dataset ($N \times F$)

```bash
# Quick test dataset (2 seeds per setting — fast)
python3 scripts/generate_dataset.py --num-seeds 2 --tasks all

# Full benchmark dataset (20 seeds per setting)
python3 scripts/generate_dataset.py --num-seeds 20 --tasks all
```

Outputs are saved to:
- `data/videos/{domain}/` — Rendered `.mp4` video files
- `data/traces/{domain}/` — Ground-truth executable JSON traces
- `data/gt/{domain}/` — Ground-truth answer files
- `data/manifest.jsonl` — Dataset index manifest

### 2. Run $N \times F$ Matrix Evaluation Sweeps

```bash
# Google Gemini 2.0 Flash — native video
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode native_video \
  --prompt-condition structured_trace

# PropensityBench Gateway Call (Shayan's spec format: <provider>/<original_model_name>)
python3 scripts/run_matrix_sweep.py \
  --provider propensity \
  --model-name gemini/gemini-3.1-pro-preview \
  --input-mode native_video \
  --prompt-condition structured_trace
```

### 3. Frame-Density Intervention Sweeps

Run the model across 7 frame sampling rates (native video, 1 FPS, 2 FPS, 4 FPS, 8 FPS, 10 FPS, 16 FPS):

```bash
for mode in native_video frames_1fps frames_2fps frames_4fps frames_8fps frames_10fps frames_16fps; do
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode ${mode} \
    --prompt-condition structured_trace
done
```

### 4. Prompting & Reasoning Mode Interventions

```bash
# Evaluate all 5 prompting conditions (direct, structured_trace, multi_turn_verification, thinking, role_prompting)
for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode native_video \
    --prompt-condition ${cond}
done
```

### 5. Aggregate Results & Generate 2D Heatmap Figures

```bash
# Aggregate token usage, costs ($USD), latencies, and frame counts
python3 scripts/aggregate_results.py

# Generate 2D N x F matrix accuracy heatmaps
python3 scripts/make_matrix_heatmaps.py
```

---

## 🖥️ Open-Source Models via vLLM

Host any open-source Vision-Language Model using [vLLM](https://github.com/vllm-project/vllm)'s OpenAI-compatible API server:

```bash
# 1. Launch vLLM server
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

# 2. Run evaluation sweep
python3 scripts/run_matrix_sweep.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
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
├── scripts/               # Master runner scripts, dataset generation, matrix sweeps, and heatmaps
├── src/eventlapse/
│   ├── generation/        # Event Counting Manim generators (bounce_ball, blinking, state_machine)
│   ├── models/
│   │   ├── base.py        # BaseVideoModel abstract interface
│   │   ├── load_model.py  # Provider dispatch & model spec parser
│   │   └── adapters/      # Adapters: gemini, openai, anthropic, bedrock, fireworks, vllm, propensity_client
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
