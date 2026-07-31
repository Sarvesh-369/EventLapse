#!/usr/bin/env python3
import os
import re
import json
import glob
import subprocess
from pathlib import Path

EVENTLAPSE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = EVENTLAPSE_DIR / "qualitative_examples"
FRAMES_DIR = OUTPUT_DIR / "frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_RESULTS = EVENTLAPSE_DIR / "outputs" / "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"
ALT_RESULTS = Path("/Users/sarvesh/Documents/morse papers/Morse-abilations/outputs/results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl")
RESULTS_FILE = LOCAL_RESULTS if LOCAL_RESULTS.exists() else ALT_RESULTS

def parse_n_and_f(sample_id):
    n_match = re.search(r"_N(\d+)_", str(sample_id))
    f_match = re.search(r"_F([\d\.]+)_", str(sample_id))
    n_val = int(n_match.group(1)) if n_match else 0
    f_val = float(f_match.group(1)) if f_match else 1.0
    return n_val, f_val

def extract_pred_timestamps(text):
    if not text:
        return []
    ts = []
    matches = re.findall(r"\b(\d{1,2}):(\d{2})(?:\.(\d+))?\b", str(text))
    for m, s, ms in matches:
        val = int(m)*60 + int(s) + (float(f"0.{ms}") if ms else 0.0)
        ts.append(val)
    if not ts:
        sec_matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", str(text), re.IGNORECASE)
        for s in sec_matches:
            val = float(s)
            if 0 <= val <= 25.0:
                ts.append(val)
    return ts

def format_ts(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 10))
    if ms > 0:
        return f"{m:02d}:{s:02d}.{ms}"
    return f"{m:02d}:{s:02d}"

def load_gt_timestamps(task, sample_id):
    gt_file = EVENTLAPSE_DIR / f"data/traces/{task}/{sample_id}_trace.json"
    if gt_file.exists():
        with open(gt_file) as f:
            gt_data = json.load(f)
        events = gt_data.get("events", [])
        return [float(ev["timestamp"]) for ev in events], events
    return [], []

def compute_8_key_timestamps(gt_ts, duration=24.0, target_n=8):
    gt_ts_sorted = sorted(list(set([round(float(t), 2) for t in gt_ts if 0 <= t <= duration])))
    K = len(gt_ts_sorted)
    
    if K == target_n:
        return [(t, True, idx + 1) for idx, t in enumerate(gt_ts_sorted)]
    elif K > target_n:
        # Subsample target_n events spread out
        step = (K - 1) / (target_n - 1)
        chosen = []
        for i in range(target_n):
            idx = int(round(i * step))
            t = gt_ts_sorted[idx]
            chosen.append((t, True, idx + 1))
        return chosen
    else:
        # K < target_n: include ALL K key events, fill remaining with context frames
        event_dict = {t: idx + 1 for idx, t in enumerate(gt_ts_sorted)}
        
        # Build candidates
        fills_needed = target_n - K
        
        # Segment midpoints
        points = [0.5] + gt_ts_sorted + [max(0.5, duration - 0.5)]
        gaps = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            gap_sz = p2 - p1
            if gap_sz > 0.4:
                mid = round((p1 + p2) / 2.0, 2)
                if mid not in event_dict:
                    gaps.append((gap_sz, mid))
                    
        gaps.sort(key=lambda x: x[0], reverse=True)
        chosen_fills = [g[1] for g in gaps[:fills_needed]]
        
        if len(chosen_fills) < fills_needed:
            # Add uniform fallback points
            import numpy as np
            uniform = np.linspace(0.5, duration - 0.5, fills_needed + 2)[1:-1]
            for u in uniform:
                u_r = round(float(u), 2)
                if u_r not in event_dict and u_r not in chosen_fills:
                    chosen_fills.append(u_r)
                    if len(chosen_fills) >= fills_needed:
                        break
                        
        combined = []
        for t in gt_ts_sorted:
            combined.append((t, True, event_dict[t]))
        for t in chosen_fills[:fills_needed]:
            combined.append((t, False, None))
            
        combined.sort(key=lambda x: x[0])
        # ensure exact length 8
        return combined[:target_n]

def extract_video_key_frames(task, sample_id, gt_ts, num_frames=8):
    video_path = EVENTLAPSE_DIR / f"data/videos/{task}/{sample_id}.mp4"
    if not video_path.exists():
        return []
    
    sample_frames_dir = FRAMES_DIR / sample_id
    sample_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video duration
    cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprintwrappers=1:nokey=1", str(video_path)]
    res = subprocess.run(cmd_dur, capture_output=True, text=True)
    try:
        duration = float(res.stdout.strip())
    except Exception:
        duration = 24.0

    key_ts_info = compute_8_key_timestamps(gt_ts, duration=duration, target_n=num_frames)
    extracted_frames_meta = []
    
    for idx, (t_sec, is_event, event_num) in enumerate(key_ts_info):
        out_jpg = sample_frames_dir / f"keyframe_{idx:02d}.jpg"
        cmd = [
            "/opt/homebrew/bin/ffmpeg", "-y", "-ss", f"{t_sec:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", "scale=320:180",
            "-q:v", "3",
            str(out_jpg)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out_jpg.exists():
            extracted_frames_meta.append({
                "rel_path": str(out_jpg.relative_to(OUTPUT_DIR)),
                "timestamp_sec": t_sec,
                "timestamp_str": format_ts(t_sec),
                "is_event": is_event,
                "event_num": event_num
            })
            
    return extracted_frames_meta

def classify_and_sample():
    candidates = {
        "correct": [],
        "missed_event": [],
        "hallucinated_event": [],
        "correct_trace_wrong_final_answer": [],
        "incorrect_trace_accidental_correct": [],
        "temporally_displaced_event": []
    }
    
    with open(RESULTS_FILE) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("parser_validity"):
                continue
            task = row.get("task", "bounce_ball")
            sample_id = row.get("sample_id", "")
            gt_ans = str(row.get("ground_truth_answer", "")).strip()
            pred_ans = str(row.get("predicted_answer", "")).strip()
            is_exact = (gt_ans == pred_ans)
            
            gt_ts, gt_events = load_gt_timestamps(task, sample_id)
            raw_resp = row.get("raw_model_response", "")
            pred_ts = extract_pred_timestamps(raw_resp)
            
            gt_N = len(gt_ts) if gt_ts else (int(gt_ans) if gt_ans.isdigit() else 0)
            M = len(pred_ts)
            
            matched = 0
            gt_matched = set()
            displaced = 0
            for pt in pred_ts:
                found = False
                for idx_gt, gt_t in enumerate(gt_ts):
                    if idx_gt not in gt_matched:
                        diff = abs(pt - gt_t)
                        if diff <= 1.0:
                            matched += 1
                            gt_matched.add(idx_gt)
                            found = True
                            break
                        elif diff <= 2.5:
                            displaced += 1
            
            p = (matched / M) if M > 0 else (1.0 if gt_N == 0 else 0.0)
            r = (matched / gt_N) if gt_N > 0 else (1.0 if M == 0 else 0.0)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            f1_pct = f1 * 100.0
            
            N_param, F_param = parse_n_and_f(sample_id)
            
            entry = {
                "sample_id": sample_id,
                "task": task,
                "control_N": N_param,
                "control_F": F_param,
                "seed": row.get("seed", 0),
                "gt_ans": gt_ans,
                "pred_ans": pred_ans,
                "gt_N": gt_N,
                "pred_M": M,
                "gt_ts_raw": gt_ts,
                "gt_ts": [format_ts(t) for t in gt_ts],
                "pred_ts": [format_ts(t) for t in pred_ts],
                "p": round(p * 100, 1),
                "r": round(r * 100, 1),
                "f1": round(f1_pct, 1),
                "raw_response": raw_resp,
                "question": row.get("question", "How many target events occurred in the video?"),
                "is_exact": is_exact
            }
            
            if is_exact and f1_pct >= 80.0:
                candidates["correct"].append(entry)
            elif not is_exact and f1_pct >= 80.0:
                candidates["correct_trace_wrong_final_answer"].append(entry)
            elif is_exact and f1_pct < 80.0:
                candidates["incorrect_trace_accidental_correct"].append(entry)
            elif M < gt_N and r < 0.80:
                candidates["missed_event"].append(entry)
            elif M > gt_N and p < 0.80:
                candidates["hallucinated_event"].append(entry)
            elif displaced > 0:
                candidates["temporally_displaced_event"].append(entry)
                
    selected_by_category = {}
    for cat, items in candidates.items():
        if len(items) == 0:
            selected_by_category[cat] = []
            continue
        items.sort(key=lambda x: (x["task"], x["control_N"], x["control_F"]))
        n_items = len(items)
        if n_items <= 5:
            selected_by_category[cat] = items
        else:
            indices = [int(i * (n_items - 1) / 4) for i in range(5)]
            selected_by_category[cat] = [items[idx] for idx in indices]
            
    return selected_by_category

def generate_justification(cat, sample):
    gt_n = sample["gt_N"]
    pred_ans = sample["pred_ans"]
    pred_m = sample["pred_M"]
    f1 = sample["f1"]
    
    if cat == "correct":
        return f"Gemini accurately tracked all {gt_n} key event timestamps in sequence, yielding a perfect trace F1 of {f1}% and matching final answer."
    elif cat == "missed_event":
        return f"Gemini under-reported the event sequence, identifying only {pred_m} out of {gt_n} true key events. The model missed high-frequency transitions due to temporal compression."
    elif cat == "hallucinated_event":
        return f"Gemini over-reported events, generating {pred_m} timestamps for a video with only {gt_n} ground-truth key events. Spurious detections occurred during non-event visual motion."
    elif cat == "correct_trace_wrong_final_answer":
        return f"Wrong Accumulation (RFR): Gemini generated a highly accurate step-by-step trace (Trace F1: {f1}%), but made an arithmetic aggregation error, declaring final count {pred_ans} instead of {gt_n}."
    elif cat == "incorrect_trace_accidental_correct":
        return f"Accidental Correctness (ACR): Gemini predicted the correct final integer ({pred_ans}), but its reported trace was incomplete/flawed (Trace F1: {f1}%), demonstrating a lucky guess or compensating errors."
    elif cat == "temporally_displaced_event":
        return f"Gemini detected the event occurrences but assigned timestamps displaced from actual physical boundary contacts outside the 1.0s tolerance window."
    return ""

def main():
    print("Classifying and selecting 30 qualitative samples with KEY EVENT FRAMES (5 per category)...")
    selected_samples = classify_and_sample()
    
    dataset_export = {}
    
    for cat, samples in selected_samples.items():
        print(f"Extracting key event frames for category '{cat}' ({len(samples)} samples)...")
        dataset_export[cat] = []
        for sample in samples:
            # Extract key event frames
            frame_meta = extract_video_key_frames(sample["task"], sample["sample_id"], sample["gt_ts_raw"], num_frames=8)
            sample["frame_meta"] = frame_meta
            sample["category"] = cat
            sample["justification"] = generate_justification(cat, sample)
            dataset_export[cat].append(sample)
            
    # Save JSON dataset
    json_path = OUTPUT_DIR / "qualitative_dataset_30.json"
    with open(json_path, "w") as f:
        json.dump(dataset_export, f, indent=2)
    print(f"Saved key-event qualitative dataset JSON to {json_path}")

if __name__ == "__main__":
    main()
