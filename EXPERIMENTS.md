# EventLapse: Comprehensive Experiments & Evaluation Guide

This document provides a detailed, technical guide to the experimental protocol, the $N \times F$ parametric matrix evaluation framework, supported model providers (including vLLM open-source hosting), interventions, evaluation metrics, and step-by-step commands to reproduce all paper evaluations.

---

## 📌 Executive Summary of Experimental Architecture

EventLapse profiles the temporal reasoning boundaries of Video-Language Models (VLMs) by evaluating them across controlled **Parametric Axes** ($N \times F$ grids) and **Intervention Controls**. 

Evaluation is performed using **Trace-Grounded Evaluation (MORSE)**, assessing both final answer correctness (Exact Match) and intermediate reasoning trace fidelity (Trace Precision, Recall, F1, Accidental Correctness, and Reasoning Failures).

---

## 📐 1. The $N \times F$ Parametric Matrix Framework

Instead of evaluating VLMs on static or uncontrolled videos, EventLapse evaluates models across a systematically controlled 2D matrix of **Event Count ($N$)** vs **Event Frequency ($F$, Hz)**:

| Task Name | Control Parameter Axis | Parameter Grid / Range | Video Duration |
| :--- | :--- | :--- | :---: |
| **Event Counting** | Bounce events $N$ | $N \in \{1, 2, 3, 4, 6, 8, 10, 12, 16\}$ | 20.0s |
| **Event Frequency** | Frequency $f$ (Hz) | $f \in \{0.5, 1.0, 1.5, 2.0, 3.0, 4.0\}$ | 7.9s |
| **Temporal Ordering** | Sequence length $L$ | $L \in \{3, 4, 6, 8, 10, 12, 16\}$ | 18.0s |
| **Duration Comparison** | Duration ratio $r$ | $r \in \{1.05, 1.10, 1.25, 1.50, 2.00, 3.00\}$ | 13.0s |
| **Causal Attribution** | Causal depth $C$ | $C \in \{1, 2, 3, 4, 5, 6\}$ | 18.0s |
| **Future Prediction** | Prediction horizon $H$ | $H \in \{1, 2, 3, 4, 5\}$ | 10.0s |
| **Long-Term Dependency** | Intervening swaps $D$ | $D \in \{0, 2, 4, 8, 12, 16\}$ | 17.0s |

> **Note on Video Durations:** Each task enforces a **fixed constant video duration** across all control parameter values (padded with resting state prior to the question card). This guarantees that total video duration is never a confounder when measuring difficulty degradation.

---

## 🤖 2. Model Providers & API Gateways

EventLapse provides a unified model loading and execution layer (`load_model.py` and `BaseVideoModel` interface) supporting 6 primary model gateways:

| Provider Key | Supported Models | Input Mode Capabilities | Env Variables Required |
| :--- | :--- | :--- | :--- |
| **`google`** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-3.5-flash`, `gemini-2.0-flash-thinking` | Native Video + Frame Sequences | `GEMINI_API_KEY` |
| **`openai`** | `gpt-4o`, `gpt-4o-mini`, `o3-mini` | Frame Sequences | `OPENAI_API_KEY` |
| **`anthropic`** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` | Frame Sequences | `ANTHROPIC_API_KEY` |
| **`bedrock`** | AWS Bedrock hosted VLMs | Frame Sequences | AWS Credentials |
| **`fireworks`** | Fireworks AI hosted VLMs | Frame Sequences | `FIREWORKS_API_KEY` |
| **`vllm`** | Open-source VLMs (`Qwen2-VL`, `LLaVA-NeXT-Video`, `InternVL2`, etc.) | Native Video / Frame Sequences | `VLLM_BASE_URL` (No API key required) |

### Serving Open-Source Models via vLLM
To run evaluations against any open-source model using vLLM:
```bash
# 1. Launch vLLM OpenAI-compatible server on port 8000
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

# 2. Execute EventLapse evaluation sweep targeting localhost:8000
python3 scripts/run_matrix_sweep.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

---

## 🎛️ 3. Experimental Groups & Interventions

The evaluation pipeline is divided into 4 structured experiment groups:

### Group 1: Base $N \times F$ Capability Boundaries
Profiles raw model performance across task parameter axes using native video input and structured trace prompting. Operational capability boundaries ($x^*$) are estimated by identifying the maximum parameter difficulty where the lower 95% Wilson confidence interval bound meets or exceeds $\tau = 0.80$.

### Group 2: MORSE Trace Diagnosis & Error Taxonomy
Evaluates intermediate step-by-step reasoning outputs to classify failure modes into an 18-category error taxonomy (e.g. `missed_event`, `hallucinated_event`, `wrong_timestamp`, `wrong_root_cause`, `off_by_one_prediction`).

### Group 3: Systematic Interventions
1. **Group 3A — Frame Density Sweeps (`input_mode`)**:
   Evaluates performance as frame sampling rate varies:
   $$\text{Input Modes} \in \{\text{native\_video}, \text{frames\_1fps}, \text{frames\_2fps}, \text{frames\_4fps}, \text{frames\_8fps}, \text{frames\_16fps}\}$$
2. **Group 3B — Oracle Evidence (`oracle_evidence`)**:
   Provides models with oracle visual frames extracted specifically around ground-truth event windows ($\pm 0.1\text{s}$) to test whether perceptual resolution or reasoning capacity is the bottleneck.
3. **Group 3C — Prompting & Thinking Controls (`prompt_condition`)**:
   Tests model sensitivity to reasoning formats:
   - `direct`: Direct answer response with `\boxed{}` formatting only.
   - `structured_trace`: Full MORSE step-by-step CoT reasoning trace.
   - `thinking`: Extended reasoning models (e.g. Gemini Thinking, o3-mini).

### Group 4: Real-World Natural Transfer
Tests whether synthetic capability breakdown boundaries ($x^*$) predict performance degradation on real-world video datasets (e.g., RepCount-A repetitive movement counting).

---

## 📊 4. Measured Resource & Performance Metrics

Every single evaluation call executed by `scripts/run_experiment.py` or `scripts/run_matrix_sweep.py` automatically measures and logs the following metrics:

| Metric Name | Field in Output JSONL | Description |
| :--- | :--- | :--- |
| **Exact Match (EM)** | `exact_match_result` | Binary string/numeric match between model answer and ground truth |
| **Supplied Frames** | `num_frames` | Total number of frames supplied to the model (native video frame count or extracted JPEG count) |
| **Input Tokens** | `prompt_tokens` | Prompt token count sent to the API |
| **Output Tokens** | `completion_tokens` | Candidate token count generated by the model |
| **Total Tokens** | `total_tokens` | Combined token count (`prompt_tokens + completion_tokens`) |
| **Estimated Cost ($USD)** | `estimated_cost_usd` | Estimated financial cost of the API call in USD |
| **Latency (sec)** | `latency_sec` | Total end-to-end wall-clock latency of the inference call in seconds |
| **Trace Precision ($P$)** | `trace_precision` | Ratio of model-reported steps that match true ground-truth events |
| **Trace Recall ($R$)** | `trace_recall` | Ratio of ground-truth events correctly identified by the model |
| **Trace F1 ($F_1$)** | `trace_f1` | Harmonic mean of Trace Precision and Trace Recall |
| **Accidental Correctness** | `is_accidental_correct` | Correct final answer ($\text{EM}=1$) despite corrupted trace ($F_1 < 1.0$) |
| **Reasoning Failure** | `is_reasoning_failure` | Perfect trace ($F_1 = 1.0$) but wrong final answer ($\text{EM}=0$) |

---

## 🚀 5. Step-by-Step Execution Command Guide

### Step 1: Run $N \times F$ Matrix Evaluation Sweeps
```bash
# Evaluate Gemini 2.0 Flash on native video across all tasks
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode native_video \
  --prompt-condition structured_trace

# Evaluate GPT-4o on 2 FPS frame sequences
python3 scripts/run_matrix_sweep.py \
  --provider openai \
  --model-name gpt-4o \
  --input-mode frames_2fps \
  --prompt-condition structured_trace

# Evaluate local open-source Qwen2-VL via vLLM
python3 scripts/run_matrix_sweep.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

### Step 2: Run Frame-Density Intervention Sweeps
```bash
for fps in 1fps 2fps 4fps 8fps 16fps; do
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode frames_${fps} \
    --prompt-condition structured_trace
done
```

### Step 3: Run Oracle Evidence Intervention Sweeps
```bash
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode oracle_evidence \
  --prompt-condition structured_trace
```

### Step 4: Run Master Automated Pipeline Script
To run the full automated evaluation pipeline dry-run / integration check:
```bash
./scripts/run_all_experiments.sh
```

### Step 5: Aggregate Results & Compute Summaries
Consolidate all evaluation JSONL files into CSV format and compute resource & cost summaries:
```bash
python3 scripts/aggregate_results.py
```
This produces:
- `outputs/aggregated_results.csv` — Full sample-level results table.
- `outputs/mode_resource_summary.csv` — Summary table of input/output tokens, USD cost, latency, frame count, and accuracy per mode.

### Step 6: Generate $N \times F$ Matrix Heatmaps & Figures
```bash
# Render N x F matrix accuracy heatmaps (Event Count N vs Event Frequency F)
python3 scripts/make_matrix_heatmaps.py

# Render 7-panel paper figures
python3 scripts/make_figures.py
```
Outputs are saved as high-resolution PNGs in `outputs/`.
