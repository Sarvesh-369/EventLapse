from pathlib import Path
from typing import List, Dict, Any

def construct_repcount_frequency_pairs(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for i in range(0, len(clips) - 1, 2):
        clip_a = clips[i]
        clip_b = clips[i+1]
        freq_a = clip_a.get("repetition_count", 1) / 10.0
        freq_b = clip_b.get("repetition_count", 1) / 10.0
        faster_side = "left" if freq_a >= freq_b else "right"

        pairs.append({
            "pair_id": f"pair_{i}",
            "clip_left": clip_a,
            "clip_right": clip_b,
            "faster_side": faster_side,
            "rate_ratio": max(freq_a, freq_b) / max(0.01, min(freq_a, freq_b))
        })

    return pairs
