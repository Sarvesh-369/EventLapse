import subprocess
import shutil
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("eventlapse.interventions.frame_extraction")

def get_ffmpeg_binary() -> str:
    # 1. System PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # 2. Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return "ffmpeg"

def extract_frames_via_cv2(video_path: Path, target_fps: int, output_dir: Path) -> List[Path]:
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / target_fps)))

    frame_count = 0
    saved_count = 0
    extracted_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_file = output_dir / f"frame_{saved_count+1:04d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            extracted_frames.append(out_file)
            saved_count += 1

        frame_count += 1

    cap.release()
    return extracted_frames

def extract_frames_at_fps(video_path: Path, target_fps: int, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing extracted frames
    existing = sorted(list(output_dir.glob("frame_*.jpg")))
    if existing:
        return existing

    ffmpeg_bin = get_ffmpeg_binary()

    if shutil.which(ffmpeg_bin) or (Path(ffmpeg_bin).exists() and Path(ffmpeg_bin).is_file()):
        out_pattern = output_dir / "frame_%04d.jpg"
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-vf", f"fps={target_fps}",
            "-q:v", "2",
            str(out_pattern)
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                extracted = sorted(list(output_dir.glob("frame_*.jpg")))
                if extracted:
                    return extracted
        except FileNotFoundError:
            pass

    # OpenCV Fallback if FFmpeg is not found
    try:
        logger.info(f"FFmpeg binary not found. Falling back to OpenCV for frame extraction at {target_fps} FPS...")
        return extract_frames_via_cv2(video_path, target_fps, output_dir)
    except Exception as e:
        raise RuntimeError(
            f"Frame extraction failed: system FFmpeg binary not found and OpenCV fallback error ({e}). "
            f"Please run 'pip install imageio-ffmpeg' or install ffmpeg on your server system."
        )
