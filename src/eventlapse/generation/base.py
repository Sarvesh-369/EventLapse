import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

@dataclass
class SyntheticSample:
    sample_id: str
    task_name: str
    control_parameter_name: str
    control_parameter_value: float
    seed: int
    video_path: Path
    question: str
    exact_answer: str
    executable_trace: Dict[str, Any]
    cot_text: str
    generation_config: Dict[str, Any]
    duration: float
    fps: int
    resolution: List[int]
    checksum: str
    git_commit: Optional[str] = None

class BaseTaskGenerator(abc.ABC):
    @property
    @abc.abstractmethod
    def task_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def control_parameter_name(self) -> str:
        pass

    @abc.abstractmethod
    def generate_sample(
        self,
        control_value: float,
        seed: int,
        output_dir: Path,
        resolution: List[int] = (1920, 1080),
        fps: int = 30
    ) -> SyntheticSample:
        pass
