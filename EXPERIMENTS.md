# EventLapse: Event Counting Experiments & Evaluation Guide ($N \times F$ Matrix)

This document provides a comprehensive technical guide to the experimental protocol for **Event Counting** evaluated across the 3 core video domains (**Bounce Ball**, **Blinking**, and **State Machine**), systematically sweeping the 2D **$N \times F$ Matrix** (Event Count $N$ vs Event Frequency $F$, Hz).

---

## 📌 Executive Summary of Research Design

EventLapse assesses Vision-Language Models (VLMs) on **Event Counting** across 3 controlled visual domains:

1. **`bounce_ball`**: Physics simulation of a ball bouncing between walls; models count the total number of wall contacts ($N$) across oscillation frequencies ($F$).
2. **`blinking`**: Periodic visual pulse simulation; models count the total number of light blinks/pulses ($N$) across blinking frequencies ($F$).
3. **`state_machine`**: Visual state transition simulation between states $\{A, B, C, D\}$; models count the total number of state transitions ($N$) across transition rates ($F$).

---

## 📐 1. The $N \times F$ Parametric Matrix Framework

Each domain is evaluated across a 2D matrix of **Event Count ($N$)** vs **Event Frequency ($F$, Hz)**:

| Visual Domain | Event Type | Event Count Grid ($N$) | Frequency Grid ($F$, Hz) | Video Duration |
| :--- | :--- | :--- | :--- | :---: |
| **`bounce_ball`** | Wall contact bounces | $N \in \{2, 4, 8, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}$ | 24.0s |
| **`blinking`** | Light blinks / pulses | $N \in \{2, 4, 8, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}$ | 24.0s |
| **`state_machine`** | State transitions | $N \in \{2, 4, 8, 12\}$ | $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}$ | 24.0s |

- **Event Count ($N$)**: The target quantity the VLM must count.
- **Event Frequency ($F$)**: The rate of occurrence (events per second). Higher frequencies reduce the temporal margin between events, testing temporal resolution limits.
- **Fixed Duration**: All videos use a fixed global duration ($24.0\text{s}$) so duration is never a confounder.

---

## 🤖 2. Supported Model Providers & Gateways

All evaluations execute via the unified `load_model.py` dispatch supporting 6 gateways:

| Provider | Model Identifier | Input Modes Supported | Env Variables |
| :--- | :--- | :--- | :--- |
| **`google`** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-3.5-flash` | Native Video + Frame Sequences | `GEMINI_API_KEY` |
| **`openai`** | `gpt-4o`, `gpt-4o-mini`, `o3-mini` | Frame Sequences | `OPENAI_API_KEY` |
| **`anthropic`** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` | Frame Sequences | `ANTHROPIC_API_KEY` |
| **`vllm`** | `Qwen/Qwen2-VL-7B-Instruct`, `LLaVA-NeXT-Video`, `InternVL2` | Native Video / Frame Sequences | `VLLM_BASE_URL` (No API key needed) |

### Hosting Open-Source Models via vLLM
```bash
# 1. Launch vLLM OpenAI-compatible server on port 8000
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

# 2. Run EventLapse N x F evaluation against local vLLM
python3 scripts/run_matrix_sweep.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

---

## 🎛️ 3. Input Modes & Prompt Conditions

### Input Modes (`--input-mode`)
- **`native_video`**: Pass the raw `.mp4` video file directly to models supporting native video (e.g. Gemini, Qwen2-VL).
- **`frames_Xfps`**: Sample frames at $X$ FPS (`frames_1fps`, `frames_2fps`, `frames_4fps`, `frames_8fps`, `frames_16fps`).
- **`oracle_evidence`**: Supply key visual frames extracted around ground-truth event timestamps ($\pm 0.1\text{s}$).

### Prompt Conditions (`--prompt-condition`)
- **`direct`**: Asks for direct final answer only in `\boxed{}`.
- **`structured_trace`**: Requires step-by-step MORSE CoT event tracking before boxing the final count.
- **`thinking`**: Enables extended reasoning mode on supported models (e.g., Gemini Thinking, o3-mini).

---

## 📊 4. Recorded Evaluation Metrics

Every evaluation sample logged to `outputs/results_*.jsonl` explicitly records:

| Metric | Output Field | Description |
| :--- | :--- | :--- |
| **Exact Match (EM)** | `exact_match_result` | Binary match between model predicted count and true count $N$ |
| **Supplied Frames** | `num_frames` | Total number of video frames / extracted JPEGs supplied |
| **Input Tokens** | `prompt_tokens` | Prompt token count |
| **Output Tokens** | `completion_tokens` | Candidate token count |
| **Total Tokens** | `total_tokens` | Total token count |
| **Estimated Cost ($USD)** | `estimated_cost_usd` | Estimated financial cost of the call in USD |
| **Latency (sec)** | `latency_sec` | Total end-to-end wall-clock inference time in seconds |
| **Trace Precision ($P$)** | `trace_precision` | Ratio of model-reported events matching ground-truth events |
| **Trace Recall ($R$)** | `trace_recall` | Ratio of true ground-truth events detected by model |
| **Trace F1 ($F_1$)** | `trace_f1` | Harmonic mean: $F_1 = \frac{2 P R}{P + R}$ |

---

## 🚀 5. Command Reference & Workflow

### 1. Generate Synthetic Dataset Sweeps ($N \times F$)
Generate videos for the 3 event counting domains across $N \in \{2, 4, 8, 12\}$ and $F \in \{0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4\}$:
```bash
# Generate 5 seeds per configuration across all 3 domains
python3 scripts/generate_dataset.py --num-seeds 5 --tasks all
```

### 2. Run $N \times F$ Evaluation Sweeps
```bash
# Run Gemini 2.0 Flash on native video
python3 scripts/run_matrix_sweep.py \
  --provider google \
  --model-name gemini-2.0-flash \
  --input-mode native_video \
  --prompt-condition structured_trace

# Run GPT-4o on 2 FPS frame sequences
python3 scripts/run_matrix_sweep.py \
  --provider openai \
  --model-name gpt-4o \
  --input-mode frames_2fps \
  --prompt-condition structured_trace

# Run local vLLM model
python3 scripts/run_matrix_sweep.py \
  --provider vllm \
  --model-name Qwen/Qwen2-VL-7B-Instruct \
  --input-mode frames_2fps \
  --prompt-condition structured_trace
```

### 3. Run Frame-Density Intervention Sweeps
```bash
for fps in 1fps 2fps 4fps 8fps 16fps; do
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode frames_${fps} \
    --prompt-condition structured_trace
done
```

### 4. Aggregate Results & Summarize Costs
```bash
python3 scripts/aggregate_results.py
```
Produces:
- `outputs/aggregated_results.csv` — Full sample-level results.
- `outputs/mode_resource_summary.csv` — Summary table of tokens, cost ($USD), latency, frame counts, and accuracy.

### 5. Generate $N \times F$ Heatmaps
```bash
python3 scripts/make_matrix_heatmaps.py
```
Produces 2D accuracy heatmap matrices ($N$ vs $F$) saved in `outputs/`.
