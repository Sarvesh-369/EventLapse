# EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries

> **Paper Title:** *EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries*

EventLapse is a research framework for profiling and diagnosing the temporal reasoning capabilities of frontier Video-Language Models (VLMs) across **7 synthetic video tasks** rendered with Manim Community Edition, using the **MORSE** executable trace evaluation methodology with Trace Precision / Recall / F1 scoring and operational capability boundary estimation.

> 📖 **Detailed Experiments Guide:** See [EXPERIMENTS.md](file:///Users/sarvesh/Documents/VLM_failures/EventLapse/EXPERIMENTS.md) for full technical documentation on the $N \times F$ matrix sweep framework, vLLM hosting, interventions, error taxonomy, and metrics.


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

| Provider | Environment Variable | Notes |
| :--- | :--- | :--- |
| Google Gemini | `GEMINI_API_KEY` | Native video supported |
| OpenAI | `OPENAI_API_KEY` | Frame sequences only |
| Anthropic | `ANTHROPIC_API_KEY` | Frame sequences only |
| AWS Bedrock | AWS credentials | Frame sequences only |
| Fireworks AI | `FIREWORKS_API_KEY` | Frame sequences only |
| **vLLM (open-source)** | `VLLM_BASE_URL` (no API key needed) | See §Open-Source Models via vLLM |

---

## 📊 7 Synthetic Tasks

| Task | Control Axis | Parameter Range | Fixed Video Duration |
| :--- | :--- | :---: | :---: |
| **Event Counting** | Bounce events $N$ | $N \in \{1, 2, 4, 8, 10, 12, 16\}$ | 20.0 s |
| **Event Frequency** | Frequency $f$ (Hz) | $f \in \{0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$ | 7.9 s |
| **Temporal Ordering** | Sequence length $L$ | $L \in \{2, 3, 4, 8, 12, 16\}$ | 18.0 s |
| **Duration Comparison** | Duration ratio $r$ | $r \in \{1.05, 1.10, 1.25, 1.50, 2.00, 3.00\}$ | 13.0 s |
| **Causal Attribution** | Causal depth $C$ | $C \in \{1, 2, 3, 4, 5, 6\}$ | 18.0 s |
| **Future Prediction** | Prediction horizon $H$ | $H \in \{1, 2, 3, 4, 5, 6\}$ | 10.0 s |
| **Long-Term Dependency** | Intervening swaps $D$ | $D \in \{0, 2, 4, 8, 12, 16\}$ | 17.0 s |

Videos are constant-duration per task (durations are fixed regardless of control parameter value) to prevent duration being a confound.

---

## 🧪 Running Experiments

### 1. Generate Synthetic Dataset

```bash
# Smoke test (2 seeds per task — fast)
python3 scripts/generate_dataset.py --num-seeds 2 --tasks all

# Full paper dataset (30 seeds per task)
python3 scripts/generate_dataset.py --num-seeds 30 --tasks all
```

Outputs are saved to:
- `data/videos/{task_name}/` — Rendered `.mp4` files
- `data/traces/{task_name}/` — Ground-truth executable JSON traces
- `data/gt/{task_name}/` — Ground-truth answer files
- `data/manifest.jsonl` — Dataset index

### 2. Validate Dataset

```bash
python3 scripts/validate_dataset.py
```

### 3. Run Capability Boundary Sweep (Group 1)

```bash
# Google Gemini 2.0 Flash — native video
python3 scripts/run_experiment.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode native_video \
  --prompt-condition structured_trace

# GPT-4o — frame sequence at 2 fps
python3 scripts/run_experiment.py \
  --provider openai \
  --model-name gpt-4o \
  --input-mode frames_2fps \
  --prompt-condition structured_trace

# Claude 3.5 Sonnet — frame sequence at 2 fps
python3 scripts/run_experiment.py \
  --provider anthropic \
  --model-name claude-3-5-sonnet-20241022 \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

### 4. Frame-Density Intervention Sweep (Group 3A)

Run the same model at different frame sampling rates $\{1, 2, 4, 8, 16\}$ fps:

```bash
for fps in 1fps 2fps 4fps 8fps 16fps; do
  python3 scripts/run_experiment.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode frames_${fps} \
    --prompt-condition structured_trace
done
```

### 5. Prompting & Thinking Interventions (Group 3C)

```bash
# Direct answer (no structured trace CoT)
python3 scripts/run_experiment.py \
  --provider google --model-name gemini-2.0-flash \
  --input-mode native_video --prompt-condition direct

# Structured trace CoT
python3 scripts/run_experiment.py \
  --provider google --model-name gemini-2.0-flash \
  --input-mode native_video --prompt-condition structured_trace

# Extended thinking (Gemini thinking model or o3-mini)
python3 scripts/run_experiment.py \
  --provider google --model-name gemini-2.0-flash-thinking \
  --input-mode native_video --prompt-condition thinking
```

### 6. Master Pipeline Script

```bash
./scripts/run_all_experiments.sh
```

### 7. Aggregate Results & Generate Paper Figures

```bash
python3 scripts/aggregate_results.py
python3 scripts/make_figures.py
```

---

## 🖥️ Open-Source Models via vLLM

You can host any open-source Vision-Language Model using [vLLM](https://github.com/vllm-project/vllm)'s OpenAI-compatible API server and run EventLapse evaluations against it.

### Step 1: Launch vLLM Server

```bash
# Example: Qwen2-VL-7B-Instruct
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

# Example: LLaVA-NeXT-Video-7B
vllm serve lmms-lab/LLaVA-NeXT-Video-7B --port 8000

# Example: InternVL2-8B
vllm serve OpenGVLab/InternVL2-8B --port 8000
```

### Step 2: Set Environment Variables

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"   # Default: localhost:8000
```

> **Note:** vLLM does not require an API key. The adapter uses no authentication by default.

### Step 3: Run EventLapse Evaluation

```bash
python3 scripts/run_experiment.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

> **Tip:** Use `--input-mode frames_2fps` or `--input-mode frames_8fps` for models that do not natively support video file input. Use `--input-mode native_video` for models with native video support (e.g. Qwen2-VL).

---

## 📏 Evaluation Metrics

EventLapse computes dual-tier evaluation aligned with the MORSE methodology:

| Metric | Description |
| :--- | :--- |
| **Exact Match (EM)** | Binary exact string match of final predicted answer vs. ground truth |
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
├── data/                  # Dataset outputs (git-ignored): videos/, traces/, gt/, manifest.jsonl
├── scripts/               # Master runner scripts, dataset generation, aggregation, and plotting
├── src/eventlapse/
│   ├── generation/        # Manim synthetic task generators (7 tasks)
│   ├── models/
│   │   ├── base.py        # BaseVideoModel abstract interface
│   │   ├── load_model.py  # Provider dispatch (google, openai, anthropic, bedrock, fireworks, vllm)
│   │   └── adapters/      # Provider adapters: gemini, openai, anthropic, bedrock, fireworks, vllm
│   ├── inference/         # Prompts, runner, and response parser
│   ├── interventions/     # Frame extraction (1–16 fps), oracle evidence, prompting controls
│   ├── evaluation/        # Exact match, Trace F1, Wilson 95% CIs, operational boundaries, MORSE evaluator
│   ├── datasets/          # RepCount-A natural transfer dataset parsers
│   ├── experiments/       # Experiment group routines
│   └── utils/             # Logging, caching, seeds, paths
└── tests/                 # Pytest unit and integration test suite (15 tests)
```

---

## 🧪 Unit Tests

```bash
PYTHONPATH=src pytest tests/
```

All 15 tests should pass across: task generation, causal attribution, trace evaluation, model dispatch (including vLLM), and MORSE evaluator.

---

## 📄 Citation

If you use EventLapse in your research, please cite:

```bibtex
@article{eventlapse2026,
  title={EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries},
  author={},
  year={2026}
}
```
