# EventLapse: Synthetic Event-Counting Experiments ($N \times F$)

This document details the 5 core synthetic event-counting experiments executed in EventLapse across **Bouncing Ball** (`bounce_ball`), **Blinking** (`blinking`), and **State Transition** (`state_machine`).

---

## Experiment 1: Full $N \times F$ Matrix Capability Boundary Sweep

- **Research Motivation**: Establish the baseline operational capability boundary ($x^*$) across Event Count ($N \in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]$) and Event Frequency ($F \in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]\text{ Hz}$) on frontier models.
- **Protocol**: Evaluates 80 parameter combinations per domain per seed, calculating Wilson 95% CI lower bounds to find where accuracy drops below $\tau = 0.80$.
- **Visual Domains**: `bounce_ball`, `blinking`, `state_machine`
- **Global Video Duration**: $24.0\text{s}$ fixed duration across all samples.
- **Evaluated Metrics**: Exact Match (EM) accuracy per $(N, F)$ matrix cell, Valid Output Rate, and Operational Boundary Estimate ($x^*$).
- **Execution Command**:
  ```bash
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode native_video \
    --prompt-condition structured_trace
  ```

---

## Experiment 2: Frame Sampling Density Interventions

- **Research Motivation**: Test whether performance breakdown at higher frequencies ($F \ge 2.0\text{ Hz}$) is caused by temporal perceptual sampling limits or internal reasoning limits.
- **Protocol**: Sweeps across 7 input sampling rates (`native_video`, `frames_1fps`, `frames_2fps`, `frames_4fps`, `frames_8fps`, `frames_10fps`, `frames_16fps`).
- **Visual Domains**: `bounce_ball`, `blinking`, `state_machine`
- **Evaluated Metrics**: Exact Match accuracy across frame sampling density rates, shift in operational boundary $x^*$.
- **Execution Command**:
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

## Experiment 3: Oracle Keyframe Evidence Interventions

- **Research Motivation**: Disambiguate video frame parsing from temporal tallying by supplying perfect oracle visual keyframes.
- **Protocol**: Automatically extracts keyframes in event windows ($t_i \pm 0.1\text{s}$) around ground-truth event timestamps and feeds them to the model (`--input-mode oracle_evidence`).
- **Visual Domains**: `bounce_ball`, `blinking`, `state_machine`
- **Evaluated Metrics**: Exact Match accuracy under oracle keyframes vs full video, recovery rate of sequence tallying.
- **Execution Command**:
  ```bash
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode oracle_evidence \
    --prompt-condition structured_trace
  ```

---

## Experiment 4: Prompting & Reasoning Mode Interventions

- **Research Motivation**: Measure the impact of prompt structures, reasoning formats, self-correction, and system instructions on capability breakdown.
- **Protocol**: Compares 5 distinct strategies (`direct`, `structured_trace`, `multi_turn_verification`, `thinking`, `role_prompting`).
  1. **`direct`**: Direct zero-shot prompt requiring boxed final count (`\boxed{N}`).
  2. **`structured_trace`**: Event-by-event MORSE CoT reasoning trace + `\boxed{N}`.
  3. **`multi_turn_verification`**: Audit turn verifying detected timestamps for missed/duplicate events before boxing `\boxed{N}`.
  4. **`thinking`**: Extended reasoning/thinking mode (e.g. Gemini 2.0 Flash Thinking, OpenAI o3-mini).
  5. **`role_prompting`**: System instruction setting expert video analytics auditor persona (`system_instruction`).
- **Visual Domains**: `bounce_ball`, `blinking`, `state_machine`
- **Execution Command**:
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

## Experiment 5: MORSE Trace Diagnosis & Error Taxonomy

- **Research Motivation**: Audit intermediate model reasoning traces against executable ground-truth traces to pinpoint exact failure modes and measure accidental correctness.
- **Protocol**: Evaluates Trace Precision ($P$), Trace Recall ($R$), Trace F1 ($F_1$), Accidental Correctness Rate (ACR), Reasoning Failure Rate (RFR), and 18 failure categories (`missed_event`, `hallucinated_event`, `merged_events`, `duplicated_event`, `misordered_event`, `wrong_timestamp`, `incorrectly_accumulated_event`).
- **Visual Domains**: `bounce_ball`, `blinking`, `state_machine`
- **Execution Command**:
  ```bash
  python3 scripts/run_matrix_sweep.py \
    --provider google \
    --model-name gemini-2.0-flash \
    --input-mode native_video \
    --prompt-condition structured_trace
  ```
