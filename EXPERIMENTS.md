# EventLapse: Detailed Experiments & Evaluation Guide ($N \times F$ Matrix)

> **Paper Title:** *EventLapse: Where Do Frontier Video LLMs Break? Trace-Grounded Profiling of Temporal Capability Boundaries*

This document provides a comprehensive, publication-grade technical guide to the experimental protocol, the $N \times F$ parametric matrix evaluation framework, supported model providers (including PropensityBench rate-limited gateway and vLLM open-source hosting), systematic interventions, error taxonomy, and step-by-step commands to reproduce all evaluations.

---

## 📋 Table of Contents
1. [Executive Summary & Motivation](#-1-executive-summary--motivation)
2. [Benchmark Specifications & Visual Domains](#-2-benchmark-specifications--visual-domains)
3. [Detailed Breakdown of the 5 Paper Experiments](#-3-detailed-breakdown-of-the-5-paper-experiments)
   - [Experiment 1: Full $N \times F$ Matrix Capability Boundary Sweep](#experiment-1-full-n-times-f-matrix-capability-boundary-sweep)
   - [Experiment 2: Frame Sampling Density Interventions](#experiment-2-frame-sampling-density-interventions)
   - [Experiment 3: Oracle Keyframe Evidence Interventions](#experiment-3-oracle-keyframe-evidence-interventions)
   - [Experiment 4: Prompting & Reasoning Mode Interventions](#experiment-4-prompting--reasoning-mode-interventions)
   - [Experiment 5: MORSE Trace Diagnosis & Error Taxonomy](#experiment-5-morse-trace-diagnosis--error-taxonomy)
4. [Model Provider Gateways & vLLM Setup](#-4-model-provider-gateways--vllm-setup)
5. [Resource, Token & Financial Cost Tracking](#-5-resource-token--financial-cost-tracking)
6. [Reproducibility & Command Cheat-Sheet](#-6-reproducibility--command-cheat-sheet)

---

## 📌 1. Executive Summary & Motivation

Vision-Language Models (VLMs) frequently fail on fine-grained video reasoning tasks. However, standard benchmarks evaluate models on heterogeneous, uncalibrated videos where perceptual ambiguity and duration confound performance.

**EventLapse** addresses this by constructing a synthetic, systematically controlled benchmark centered on **Event Counting** across 3 visual domains rendered with Manim Community Edition. By holding global video duration constant ($24.0\text{s}$) and sweeping Event Count ($N$) against Event Frequency ($F$, Hz), EventLapse pinpoints exact **Operational Capability Boundaries ($x^*$)** where model reliability drops below $\tau = 0.80$ (lower 95% Wilson confidence interval bound).

---

## 📊 2. Benchmark Specifications & Visual Domains

The benchmark consists of 3 distinct visual domains designed to test different perceptual and temporal reasoning modalities:

| Visual Domain | Event Description | Control Axis | Count Grid ($N$) | Frequency Grid ($F$, Hz) | Fixed Video Duration |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`bounce_ball`** | Ball contacting walls in oscillating physics simulation | Wall contacts ($N$) | $[0, 1, 2, 3, 4, 5, 6, 8, 10, 12]$ | $[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]$ | 20.0s – 24.0s |
| **`blinking`** | Object pulsing ON/OFF periodically | Light blinks ($N$) | $[0, 1, 2, 3, 4, 5, 6, 8, 10, 12]$ | $[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]$ | 20.0s – 24.0s |
| **`state_machine`** | Visual transitions between states $\{A, B, C, D\}$ | Transitions ($N$) | $[0, 1, 2, 3, 4, 5, 6, 8, 10, 12]$ | $[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]$ | 20.0s – 24.0s |

> **Key Design Guarantee:** Global video duration is fixed across all parameter configurations ($N, F$). This ensures video length is never a confounder when measuring performance breakdown.

---

## 🧪 3. Detailed Breakdown of the 5 Paper Experiments

---

### Experiment 1: Full $N \times F$ Matrix Capability Boundary Sweep

#### 🎯 Motivation & Research Goal
Evaluate baseline VLM performance across the 2D matrix of **Event Count ($N$)** vs **Event Frequency ($F$)** to establish the primary **Operational Capability Boundary ($x^*$)**.

#### 🔬 Experimental Protocol
- **Domains Evaluated**: `bounce_ball`, `blinking`, `state_machine`.
- **Parameter Matrix**: 10 Count values $\times$ 8 Frequency values = 80 distinct parameter configurations per domain.
- **Input Mode**: `native_video` (for native video models like Gemini, Qwen2-VL) or `frames_2fps`.
- **Prompt Condition**: `structured_trace`.
- **Primary Metric**: Exact Match (EM) accuracy per $(N, F)$ cell, 2D heatmap matrix, and Wilson 95% CI boundary estimate $x^*$.

#### 💻 Execution Command
```bash
# Evaluate Google Gemini 2.0 Flash on full N x F grid
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode native_video \
  --prompt-condition structured_trace

# Evaluate local open-source Qwen2-VL via vLLM
python3 scripts/run_matrix_sweep.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

---

### Experiment 2: Frame Sampling Density Interventions

#### 🎯 Motivation & Research Goal
Investigate whether VLM performance degradation at high event frequencies ($F \ge 2.0\text{ Hz}$) is caused by **temporal frame sampling resolution limits** (missing events between sampled frames) or **internal reasoning capacity limits**.

#### 🔬 Experimental Protocol
- **Target Sampling Rates (7 Input Modes)**:
  $$\text{Input Modes} \in \{\text{native\_video}, \text{frames\_1fps}, \text{frames\_2fps}, \text{frames\_4fps}, \text{frames\_8fps}, \text{frames\_10fps}, \text{frames\_16fps}\}$$
- **Hypothesis**: If increasing frame density from 2 FPS $\rightarrow$ 10 FPS $\rightarrow$ 16 FPS $\rightarrow$ Native Video restores accuracy at higher frequencies $F$, the failure is perceptual sampling. If accuracy remains degraded despite high frame rates, the failure is temporal reasoning.

#### 💻 Execution Command
```bash
for mode in native_video frames_1fps frames_2fps frames_4fps frames_8fps frames_10fps frames_16fps; do
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode ${mode} \
    --prompt-condition structured_trace
done
```

---

### Experiment 3: Oracle Keyframe Evidence Interventions

#### 🎯 Motivation & Research Goal
Isolate perceptual video parsing from temporal reasoning logic by providing the model with **oracle event-centered keyframes**.

#### 🔬 Experimental Protocol
- **Methodology**: For each ground-truth event timestamp $t_i$, keyframes are automatically extracted in a narrow window around the event ($t_i \pm 0.1\text{s}$) and supplied to the model.
- **Input Mode**: `oracle_evidence`.
- **Hypothesis**: If supplying oracle keyframes restores counting accuracy for large sequence counts ($N \ge 8$), the model can count when visual perception is guaranteed. If counting fails even with oracle keyframes, the bottleneck is sequence tallying and memory accumulation.

#### 💻 Execution Command
```bash
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode oracle_evidence \
  --prompt-condition structured_trace
```

---

### Experiment 4: Prompting & Reasoning Mode Interventions

#### 🎯 Motivation & Research Goal
Evaluate the effect of prompt engineering, chain-of-thought trace structure, self-correction, extended reasoning budgets, and system instructions on operational capability boundaries.

#### 🔬 5 Prompting Strategies (`src/eventlapse/inference/prompts.py`)
1. **`direct` (Baseline Direct Prompting)**:
   - Direct zero-shot prompt requesting the final count inside `\boxed{N}` without step-by-step trace logging.
2. **`structured_trace` (Event-by-Event MORSE CoT)**:
   - Requires explicit step-by-step event logging (timestamp, event type, running count) before boxing `\boxed{N}`.
3. **`multi_turn_verification` (Audit & Self-Correction)**:
   - Asks the model to draft events, audit for missed or double-counted events, and state the verified count in `\boxed{N}`.
4. **`thinking` (Extended Reasoning Budget Mode)**:
   - Triggers deep reasoning mode for thinking models (e.g. Gemini 2.0 Flash Thinking, o3-mini) using `types.ThinkingConfig(include_thoughts=True)`.
5. **`role_prompting` (System Instruction Persona)**:
   - Sets official API `system_instruction`: `"You are an expert video analytics systems auditor specializing in fine-grained temporal event verification."`

#### 💻 Execution Command
```bash
for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode native_video \
    --prompt-condition ${cond}
done
```

---

### Experiment 5: MORSE Trace Diagnosis & Error Taxonomy

#### 🎯 Motivation & Research Goal
Evaluate intermediate step-by-step reasoning fidelity against executable ground-truth traces to measure *why* models fail and identify accidental correctness.

#### 🔬 Measured Trace Metrics & Failure Modes
- **Trace Precision ($P$)**: Ratio of model-reported event steps matching true ground-truth events.
- **Trace Recall ($R$)**: Ratio of ground-truth events correctly detected by the model.
- **Trace F1 ($F_1$)**: Harmonic mean: $F_1 = \frac{2 \cdot P \cdot R}{P + R}$.
- **Accidental Correctness Rate (ACR)**: Correct final answer ($\text{EM}=1$) despite corrupted trace ($F_1 < 1.0$).
- **Reasoning Failure Rate (RFR)**: Wrong final answer ($\text{EM}=0$) despite perfect trace ($F_1 = 1.0$).
- **18 Error Categories**: `missed_event`, `hallucinated_event`, `merged_events`, `duplicated_event`, `misordered_event`, `wrong_timestamp`, `incorrectly_accumulated_event`.

#### 💻 Execution Command
```bash
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode native_video \
  --prompt-condition structured_trace
```

---

## 🤖 4. Model Provider Gateways & vLLM Setup

EventLapse provides a unified model loading interface (`load_model.py`) supporting direct provider APIs, local vLLM hosting, and Shayan's **PropensityBench Gateway**.

### Supported Gateways

| Provider Key | Supported Models | Input Capabilities | Configuration Env Variables |
| :--- | :--- | :--- | :--- |
| **`google`** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-thinking` | Native Video + Frames | `GEMINI_API_KEY` |
| **`openai`** | `gpt-4o`, `gpt-4o-mini`, `o3-mini` | Frame Sequences | `OPENAI_API_KEY` |
| **`anthropic`** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` | Frame Sequences | `ANTHROPIC_API_KEY` |
| **`propensity`** | PropensityBench gateway calls | Native Video + Frames | `PROPENSITY_GATEWAY_URL`, `RATE_PM=60`, `RATE_LIMIT=true` |
| **`vllm`** | `Qwen/Qwen2-VL-7B-Instruct`, `LLaVA-NeXT-Video`, `InternVL2` | Native Video / Frames | `VLLM_BASE_URL="http://localhost:8000/v1"` (No API key needed) |

### PropensityBench Model Specification Format
Models can be specified directly using Shayan's PropensityBench format: `<provider>/<original_model_name>`
```bash
# Example: PropensityBench gateway call for Gemini 3.1 Pro Preview
python3 scripts/run_matrix_sweep.py \
  --provider propensity \
  --model-name gemini/gemini-3.1-pro-preview \
  --input-mode native_video \
  --prompt-condition structured_trace
```

---

## 💰 5. Resource, Token & Financial Cost Tracking

Every evaluation sample logged to `outputs/results_*.jsonl` automatically records complete execution metadata:

| JSON Field | Metric Description |
| :--- | :--- |
| `raw_model_response` | Full un-truncated raw text output returned by the model |
| `prompt_tokens` | Total input prompt tokens billed |
| `completion_tokens` | Total output candidate tokens generated |
| `total_tokens` | Combined token usage |
| `estimated_cost_usd` | Financial API cost estimated in USD ($) |
| `latency_sec` | Total wall-clock latency in seconds |
| `num_frames` | Total video frames / extracted JPEGs supplied to the model |
| `exact_match_result` | Exact match correctness boolean |

---

## 🚀 6. Reproducibility & Command Cheat-Sheet

### Step 1: Generate Full Dataset ($N \times F$)
```bash
# Generate full paper dataset across all 3 domains (20 seeds per setting)
python3 scripts/generate_dataset.py --num-seeds 20 --tasks all
```

### Step 2: Execute Matrix Sweeps & Interventions
```bash
# Experiment 1: N x F Matrix Sweep
python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode native_video --prompt-condition structured_trace

# Experiment 2: Frame Sampling Density Sweep
for mode in native_video frames_1fps frames_2fps frames_4fps frames_8fps frames_10fps frames_16fps; do
  python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode ${mode} --prompt-condition structured_trace
done

# Experiment 3: Oracle Evidence Sweep
python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode oracle_evidence --prompt-condition structured_trace

# Experiment 4: Prompting Strategy Sweep
for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
  python3 scripts/run_matrix_sweep.py --provider google --model-name gemini-2.0-flash --input-mode native_video --prompt-condition ${cond}
done
```

### Step 3: Aggregate Results & Generate 2D Heatmaps
```bash
# Consolidate results & compute token/cost summaries
python3 scripts/aggregate_results.py

# Render 2D N x F matrix accuracy heatmaps
python3 scripts/make_matrix_heatmaps.py
```
Output files saved to `outputs/`.
