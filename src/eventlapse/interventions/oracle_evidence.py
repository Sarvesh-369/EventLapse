import subprocess
from pathlib import Path
from typing import List, Dict, Any

def extract_oracle_frames(video_path: Path, trace_events: List[Dict[str, Any]], epsilon: float = 0.1, output_dir: Path = None) -> List[Path]:
    if output_dir is None:
        output_dir = video_path.parent / f"oracle_frames_{video_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    for idx, event in enumerate(trace_events):
        t_center = event.get("timestamp", 0.0)
        timestamps = [max(0.0, t_center - epsilon), t_center, t_center + epsilon]

        for sub_i, t in enumerate(timestamps):
            out_file = output_dir / f"event_{idx+1}_t{sub_i}_{t:.2f}s.jpg"
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{t:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(out_file)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if out_file.exists():
                extracted_files.append(out_file)

    return sorted(extracted_files)
