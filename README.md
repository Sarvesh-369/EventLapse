<div align="center">

# The Low-Frequency Trap: Video–Language Models Fail at Simple Event Bookkeeping

**Official PyTorch Implementation of The Low-Frequency Trap**

[Sarvesh Baskar](https://sarvesh-369.github.io/)*<sup>1</sup>, [Zikui Cai](https://zikuicai.github.io/)*<sup>1</sup>, [Shayan Shabihi](https://shayanshabihi.github.io/)*<sup>1</sup>, [Anirudh Satheesh](https://anirudhsatheesh.github.io/)<sup>1</sup>,  
[Muhammad R. Islam](https://mrislam.github.io/)<sup>1</sup>, [Udari Madhushani Sewwog](https://udarim.github.io/)<sup>2</sup>, [Tom Goldstein](https://www.cs.umd.edu/~tomg/)<sup>1</sup>, [Furong Huang](https://furong-huang.com/)<sup>1</sup>

<sup>1</sup>*University of Maryland, College Park* &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup>*Scale AI*  
*\* Equal contribution*

<br>

[![Project Page](https://img.shields.io/badge/project-page-38BDF8?style=for-the-badge&logo=googlechrome&logoColor=white)](https://low-frequency-trap.github.io)
[![arXiv Paper](https://img.shields.io/badge/arxiv-paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.06361)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/Sarvesh-369/Low-Frequency-Trap)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

<br>

**[Quickstart](#-quickstart--installation) &nbsp;•&nbsp; [Visual Domains](#-event-counting-visual-domains) &nbsp;•&nbsp; [Evaluation Metrics](#-evaluation-metrics-reference) &nbsp;•&nbsp; [Experiments Guide](EXPERIMENTS.md)**

<br>

<img src="https://raw.githubusercontent.com/Low-Frequency-Trap/Low-Frequency-Trap.github.io/main/assets/images/hero_teaser.png" width="92%" alt="The Low-Frequency Trap Teaser Paradigm">

</div>

<br>

> **The Low-Frequency Trap** evaluates Video-Language Models (VLMs) on event bookkeeping by controlling event count ($N$) and frequency ($F$). Rather than scoring final answers alone, our benchmark audits timestamped model traces against executable ground truth.

---

## 🚀 Quickstart & Installation

### 1. Clone & Install

```bash
git clone https://github.com/Low-Frequency-Trap/The-Low-Frequency-Trap.git
cd The-Low-Frequency-Trap
pip install -e .
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Set API keys for the providers you intend to evaluate:

| Provider | Environment Variable | Notes |
| :--- | :--- | :--- |
| **Google Gemini** | `GEMINI_API_KEY` | Native video supported |
| **OpenAI** | `OPENAI_API_KEY` | Frame sequences |
| **Anthropic** | `ANTHROPIC_API_KEY` | Frame sequences |
| **PropensityBench Gateway** | `PROPENSITY_GATEWAY_URL` | Multi-model evaluation gateway |
| **vLLM (open-source)** | `VLLM_BASE_URL` | Local open-weight models (e.g. Qwen 3 VL) |

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

---

## 📄 Citation

If you find our work useful in your research, please cite:

```bibtex
@misc{baskar2026lowfrequencytrapvideo,
      title={The Low Frequency Trap: Video Language Models Fail at Simple Event Bookkeeping}, 
      author={Sarvesh Baskar and Zikui Cai and Shayan Shabihi and Anirudh Satheesh and Muhammad R. Islam and Udari Madhushani Sehwag and Tom Goldstein and Furong Huang},
      year={2026},
      eprint={2608.06361},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.06361}, 
}
```

---

## 📜 License

This repository is licensed under the [MIT License](LICENSE).
