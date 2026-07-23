import json
from pathlib import Path
from typing import Dict, Any
from eventlapse.traces.schemas import ExecutableTrace

def load_trace(trace_path: Path) -> Dict[str, Any]:
    with open(trace_path, "r") as f:
        return json.load(f)

def save_trace(trace_data: Dict[str, Any], trace_path: Path):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2)
