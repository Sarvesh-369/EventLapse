import os
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent

def get_data_dir() -> Path:
    env_dir = os.getenv("EVENTLAPSE_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = get_project_root() / p
        return p
    return get_project_root() / "data"

def get_outputs_dir() -> Path:
    env_dir = os.getenv("EVENTLAPSE_OUTPUT_DIR")
    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = get_project_root() / p
        return p
    return get_project_root() / "outputs"

def ensure_directories():
    data_dir = get_data_dir()
    (data_dir / "videos").mkdir(parents=True, exist_ok=True)
    (data_dir / "traces").mkdir(parents=True, exist_ok=True)
    (data_dir / "gt").mkdir(parents=True, exist_ok=True)
    get_outputs_dir().mkdir(parents=True, exist_ok=True)
