import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

def compute_file_checksum(file_path: Path) -> str:
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

class SimpleCache:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save()
