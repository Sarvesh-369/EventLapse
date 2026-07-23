from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class TraceStep(BaseModel):
    state: Dict[str, Any] = Field(description="Structured state at this step s_i")
    event: Dict[str, Any] = Field(description="Visual/observable event e_i")
    operation: Dict[str, Any] = Field(description="Reasoning/computational operation o_i")

class ExecutableTrace(BaseModel):
    steps: List[TraceStep]
    final_answer: Optional[Union[str, int, float, bool]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
