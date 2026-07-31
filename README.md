# EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries

> **Paper Title:** *EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries*
> **AAAI 2027 Submission Code Package**

EventLapse is a research framework for profiling and diagnosing the temporal reasoning capabilities of frontier Video-Language Models (VLMs) on **Event Counting** across **3 synthetic video domains** rendered with Manim Community Edition, using the **MORSE** executable trace evaluation methodology with Trace Precision / Recall / F1 scoring and operational capability boundary estimation.

> 📖 **Detailed Experiments Guide:** See [EXPERIMENTS.md](EXPERIMENTS.md) for full technical details on each of the 5 paper experiments.

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

Set API keys for the providers you intend to evaluate:

| Provider | Environment Variable | Notes |
| :--- | :--- | :--- |
| Google Gemini | `GEMINI_API_KEY` | Native video supported |
| OpenAI | `OPENAI_API_KEY` | Frame sequences |
| Anthropic | `ANTHROPIC_API_KEY` | Frame sequences |
| PropensityBench Gateway | `PROPENSITY_GATEWAY_URL` | Multi-model evaluation gateway |
| vLLM (open-source) | `VLLM_BASE_URL` | Local open-weight models |

---

## 📊 Event Counting Visual Domains ($N \times F$ Matrix)

| Visual Domain | Event Description | Target Count Axis ($N$) | Frequency Axis ($F$) | Fixed Video Duration |
| :--- | :--- | :---: | :---: | :---: |
| **`bounce_ball`** | Ball contacting walls | $N \in \{0, 1, 2, 3, 4, 5, 6, 8, 10, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$ | 24.0 s |
| **`blinking`** | Object light pulses/blinks | $N \in \{0, 1, 2, 3, 4, 5, 6, 8, 10, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$ | 24.0 s |
| **`state_machine`** | Visual state transitions $\{A, B, C, D\}$ | $N \in \{0, 1, 2, 3, 4, 5, 6, 8, 10, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$ | 24.0 s |

---

## 🛠️ Step-by-Step Data Generation & Evaluation Workflow

### Step 1: Generate Synthetic Video Dataset & Ground-Truth Traces

To generate synthetic MP4 video files and exact executable ground-truth traces:

```bash
# Generate complete benchmark dataset (20 seeds per cell across all 3 domains)
python3 scripts/generate_dataset.py --num-seeds 20 --tasks all

# Quick sample generation for testing (2 seeds per cell)
python3 scripts/generate_dataset.py --num-seeds 2 --tasks all
```

This populates:
- `data/videos/{domain}/` — Rendered `.mp4` video files
- `data/traces/{domain}/` — Ground-truth executable JSON traces
- `data/gt/{domain}/` — Ground-truth integer answer files
- `data/manifest.jsonl` — Dataset index manifest

---

### Step 2: Run Benchmark Experiments

Run all 5 paper experiments for any model using the master script or individual experiment execution commands:

```bash
# Option A: Master Script (Runs Exp 1–5 sequentially for Gemini 2.0 Flash)
./scripts/run_all_experiments.sh google gemini-2.0-flash

# Option B: Run Individual Experiments
# Exp 1: Baseline N x F Matrix Sweep
python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode native_video --prompt-condition structured_trace

# Exp 2: Frame Sampling Density Interventions (Native, 1, 2, 4, 8, 10, 16 FPS)
for mode in native_video frames_1fps frames_2fps frames_4fps frames_8fps frames_10fps frames_16fps; do
  python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode ${mode} --prompt-condition structured_trace
done

# Exp 3: Oracle Keyframe Evidence Interventions
python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode oracle_evidence --prompt-condition structured_trace

# Exp 4: Prompting Strategy Interventions (5 conditions)
for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
  python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode native_video --prompt-condition ${cond}
done
```

Model prediction JSONL logs are stored in `outputs/`.

---

### Step 3: Compute Evaluation Metrics & Generate Paper Artifacts

Compute Exact Match, Trace Precision, Recall, Trace $F_1$, Accidental Correctness Rate (ACR), and Reasoning Failure Rate (RFR):

```bash
# Evaluate metrics across raw outputs
python3 scripts/evaluate_paper_metrics.py

# Aggregate results and build paper heatmap figures & LaTeX tables
python3 scripts/build_paper_figures_and_tables.py
```

Paper figures and tables are output to `figures/` and `outputs/paper/`.

---

### Step 4: Extract Qualitative Samples & Render Figure Panels

To extract qualitative model traces, build the interactive HTML dashboard, and render 3–2 split PDF figure panels for appendix inclusion:

```bash
# 1. Sample 30 qualitative cases across the 6 taxonomy categories
python3 scripts/qualitative/build_qualitative_dataset.py

# 2. Build interactive HTML dashboard
python3 scripts/qualitative/generate_qualitative_html.py

# 3. Render page-optimized 3-2 split PDF figure panels
python3 scripts/qualitative/render_split_qualitative_figures.py
```

---

## 🖥️ Evaluating Open-Source Models via vLLM

To evaluate local open-weight vision-language models:

```bash
# 1. Launch local vLLM server
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

# 2. Execute matrix sweep against local vLLM endpoint
./scripts/run_all_experiments.sh vllm Qwen/Qwen2-VL-7B-Instruct
```

---

## 📏 Evaluation Metrics Reference

| Metric | Definition & Formula |
| :--- | :--- |
| **Exact Match (EM)** | Binary exact integer prediction vs. ground truth ($\hat{y} = N$) |
| **Trace Precision ($P$)** | Fraction of model-reported timestamped steps matching ground-truth steps ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) |
| **Trace Recall ($R$)** | Fraction of true ground-truth event timestamps detected by model ($\frac{\text{TP}}{\text{TP} + \text{FN}}$) |
| **Trace F1 ($F_1$)** | Harmonic mean of trace precision and recall ($\frac{2 \cdot P \cdot R}{P + R}$) |
| **Accidental Correctness Rate (ACR)** | Correct integer answer ($\hat{y} = N$) with an ungrounded or corrupted trace ($F_1 < 1.0$) |
| **Reasoning Failure Rate (RFR)** | Perfect intermediate trace ($F_1 = 1.0$) but incorrect final integer count |
| **Operational Boundary ($x^*$)** | Maximum difficulty ($N$ or $F$) where 95% Wilson CI lower bound $\ge \tau = 0.80$ |

---

## 📁 Repository Structure

```
EventLapse/
├── configs/               # Model, generation, task, and experiment YAML configs
├── scripts/               # Master execution & evaluation scripts
│   ├── run_all_experiments.sh # Master script to run all 5 experiments
│   ├── run_matrix_sweep.py    # N x F matrix sweep execution engine
│   ├── generate_dataset.py    # Manim Event Counting video dataset generator
│   ├── evaluate_paper_metrics.py
│   ├── build_paper_figures_and_tables.py
│   └── qualitative/           # Qualitative dataset extraction & 3-2 split PDF renderer
├── src/eventlapse/        # Core Python package
│   ├── generation/        # Manim video generators (bounce_ball, blinking, state_machine)
│   ├── models/            # VLM adapters (gemini, openai, anthropic, vllm, propensity)
│   ├── inference/         # Prompts (5 strategies), runner, and response parsers
│   ├── interventions/     # Temporal frame density, keyframe evidence, prompt controls
│   ├── evaluation/        # Exact match, Trace F1, Wilson 95% CIs, MORSE evaluator
│   └── utils/             # Logging, caching, seeds, paths, cost calculator
├── reference_code/        # Reference evaluation routines
└── tests/                 # Pytest unit test suite (15/15 tests passing)
```

---

## 🧪 Unit Tests

Run the unit test suite to verify pipeline integrity:

```bash
PYTHONPATH=src pytest tests/
```

All 15 unit tests pass cleanly.

---

## 📄 Citation

```bibtex
@article{eventlapse2026,
  title={EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries},
  author={},
  year={2026}
}
```
