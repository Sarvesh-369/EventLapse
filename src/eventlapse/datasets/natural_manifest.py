import json
from pathlib import Path
from typing import List, Dict, Any

def load_natural_video_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        return []

    items = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items
