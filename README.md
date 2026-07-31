# The Low-Frequency Trap: Video–Language Models Fail at Simple Event Bookkeeping

> **Paper Title:** *The Low-Frequency Trap: Video–Language Models Fail at Simple Event Bookkeeping*

A research framework for profiling and diagnosing the temporal reasoning capabilities of frontier Video-Language Models (VLMs) on **Event Counting** across **3 synthetic video domains** rendered with Manim Community Edition, using the **MORSE** executable trace evaluation methodology with Trace Precision / Recall / F1 scoring and operational capability boundary estimation.

> 📖 **Detailed Experiments Guide:** See [EXPERIMENTS.md](EXPERIMENTS.md) for full technical details on each of the 5 paper experiments.

---

## 🚀 Quickstart & Installation

### 1. Clone & Install

```bash
git clone <repository_url>
cd repository
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
.
├── configs/               # Model, generation, task, and experiment YAML configs
├── pyproject.toml         # Python package setup
├── EXPERIMENTS.md         # Full specification of 5 paper experiments
├── README.md              # Main repository documentation
└── src/eventlapse/        # Core Python package
    ├── generation/        # Manim video generators (bounce_ball, blinking, state_machine)
    ├── models/            # VLM adapters (gemini, openai, anthropic, vllm, propensity)
    ├── inference/         # Prompts (5 strategies), runner, and response parsers
    ├── interventions/     # Temporal frame density, keyframe evidence, prompt controls
    ├── evaluation/        # Exact match, Trace F1, Wilson 95% CIs, MORSE evaluator
    └── utils/             # Logging, caching, seeds, paths, cost calculator
```
