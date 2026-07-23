import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class RepCountAParser:
    """
    Parser for RepCount-A natural dataset annotations.
    Extracts repetition cycle start and end locations into executable traces.
    """
    def __init__(self, annotation_file: Optional[Path] = None):
        self.annotation_file = annotation_file

    def parse_annotations(self) -> List[Dict[str, Any]]:
        if not self.annotation_file or not self.annotation_file.exists():
            return []

        parsed_clips = []
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        for video_id, ann in data.items():
            count = len(ann.get("count", []))
            cycles = ann.get("count", [])
            parsed_clips.append({
                "source_video_id": video_id,
                "repetition_count": count,
                "cycles": cycles
            })

        return parsed_clips
