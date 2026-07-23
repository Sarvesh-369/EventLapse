# EventLapse: Synthetic Event-Counting Experiment Suite ($N \times F$)

This document provides a technical guide to the 5 core synthetic event-counting experiments executed in EventLapse across **Bouncing Ball** (`bounce_ball`), **Blinking** (`blinking`), and **State Transition** (`state_machine`).

---

## 🧪 The 5 Paper Experiments

### 1. Full Event-Count $\times$ Event-Frequency Grid ($N \times F$)
- **Objective**: Profile the exact operational capability boundary $x^*$ across the 2D matrix of **Event Count ($N$)** vs **Event Frequency ($F$, Hz)** on frontier anchor models (e.g. Gemini 2.0 Flash / 1.5 Pro, GPT-4o, Claude 3.5 Sonnet, vLLM open-source VLMs).
- **Domain Targets**: `bounce_ball`, `blinking`, `state_machine`
- **Parameter Grids**:
  - Event Count $N \in \{2, 4, 8, 12\}$
  - Event Frequency $F \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ Hz}$
  - Global Video Duration: $24.0\text{s}$ constant
- **Command**:
  ```bash
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode native_video \
    --prompt-condition structured_trace
  ```

---

### 2. Frame Sampling Density Interventions (Native Video, 1 FPS to 16 FPS)
- **Objective**: Vary frame sampling density across 7 sampling rates to measure whether denser frame sampling shifts or expands the temporal failure boundary:
  $$\text{Input Modes} \in \{\text{native\_video}, \text{frames\_1fps}, \text{frames\_2fps}, \text{frames\_4fps}, \text{frames\_8fps}, \text{frames\_10fps}, \text{frames\_16fps}\}$$
- **Command**:
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

### 3. Oracle Keyframe Evidence Interventions
- **Objective**: Supply oracle event-centered keyframes extracted around ground-truth event timestamps ($\pm 0.1\text{s}$) to evaluate whether perceptual visibility eliminates the counting breakdown.
- **Input Mode**: `oracle_evidence`
- **Command**:
  ```bash
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode oracle_evidence \
    --prompt-condition structured_trace
  ```

---

### 4. Prompting & Reasoning Mode Interventions
- **Objective**: Compare performance across 5 distinct prompting strategies:
  1. `direct`: Direct zero-shot prompt requiring boxed final count (`\boxed{N}`).
  2. `structured_trace`: Event-by-event MORSE CoT reasoning trace + `\boxed{N}`.
  3. `multi_turn_verification`: Audit turn verifying detected timestamps for missed/duplicate events before boxing `\boxed{N}`.
  4. `thinking`: Extended reasoning/thinking mode (e.g. Gemini 2.0 Flash Thinking, OpenAI o3-mini).
  5. `role_prompting`: System instruction setting expert video analytics auditor persona.
- **Commands**:
  ```bash
  # Evaluate all prompting conditions
  for cond in direct structured_trace multi_turn_verification thinking role_prompting; do
    python3 scripts/run_matrix_sweep.py \
      --provider google \
      --model-name gemini-2.0-flash \
      --input-mode native_video \
      --prompt-condition ${cond}
  done
  ```

---

### 5. MORSE Trace Diagnosis & Error Taxonomy
- **Objective**: Compare model-generated reasoning traces against ground-truth executable traces to classify intermediate errors:
  - **Missed Events**: True event omitted from trace.
  - **Hallucinated Events**: Non-existent event reported.
  - **Merged Events**: Multiple events combined into a single entry.
  - **Misordered Events**: Correct events listed out of temporal order.
  - **Incorrect Accumulation**: Correct event detections but arithmetic tallying error.
  - **Accidental Correctness Rate (ACR)**: EM=1 with corrupted trace ($F_1 < 1.0$).
  - **Reasoning Failure Rate (RFR)**: EM=0 despite perfect trace ($F_1 = 1.0$).

---

## 🤖 Model Provider & PropensityBench Gateway Setup

Supports direct querying via OpenAI, Gemini, Anthropic, Bedrock, Fireworks, vLLM, or Shayan's **PropensityBench Gateway** (`<provider>/<original_model_name>` spec):

```bash
# PropensityBench Litellm Rate-Limited Gateway Call
python3 scripts/run_matrix_sweep.py \
  --provider propensity \
  --model-name gemini/gemini-3.1-pro-preview \
  --input-mode native_video \
  --prompt-condition structured_trace
```

---

## 📊 Result Aggregation & Matrix Heatmaps

```bash
# 1. Aggregate results, tokens, USD costs, latencies, and frame counts
python3 scripts/aggregate_results.py

# 2. Generate 2D N x F matrix accuracy heatmaps (Count N vs Frequency F)
python3 scripts/make_matrix_heatmaps.py
```
Output files saved in `outputs/`.
