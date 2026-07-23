import subprocess
import shutil
from pathlib import Path
from typing import List

def extract_frames_at_fps(video_path: Path, target_fps: int, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = output_dir / "frame_%04d.jpg"

    # FFmpeg exact rate frame extraction
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={target_fps}",
        "-q:v", "2",
        str(out_pattern)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr.decode('utf-8')}")

    extracted_frames = sorted(list(output_dir.glob("frame_*.jpg")))
    return extracted_frames
