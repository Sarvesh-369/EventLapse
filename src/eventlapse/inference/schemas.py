from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Structured JSON schemas for model output requests per task

class EventCountingEvidence(BaseModel):
    detected_contacts: List[Dict[str, Any]] = Field(default_factory=list, description="List of detected contacts with timestamps and wall identities")
    running_count: int = Field(description="Running count of detected wall contacts")
    final_answer: str = Field(description="Final integer count of wall contacts")

class EventFrequencyEvidence(BaseModel):
    detected_cycles: List[Dict[str, Any]] = Field(default_factory=list, description="Detected oscillation cycles or turning points per side")
    faster_side: str = Field(description="'left' or 'right'")
    final_answer: str = Field(description="Final answer: 'left' or 'right'")

class TemporalOrderingEvidence(BaseModel):
    ordered_crossings: List[Dict[str, Any]] = Field(default_factory=list, description="Chronological list of crossing objects")
    queried_position: int = Field(description="The position k queried")
    final_answer: str = Field(description="Color/name of the object crossing in position k")

class DurationComparisonEvidence(BaseModel):
    entry_and_exit_times: List[Dict[str, Any]] = Field(default_factory=list, description="Estimated entry and exit timestamps per object")
    estimated_durations: Dict[str, float] = Field(default_factory=dict, description="Estimated dwell duration per object")
    final_answer: str = Field(description="Which object remained stopped longer ('top object' or 'bottom object')")

class CausalAttributionEvidence(BaseModel):
    detected_events: List[Dict[str, Any]] = Field(default_factory=list, description="Detected state changes and activations")
    causal_edges: List[List[str]] = Field(default_factory=list, description="Directed causal links [parent, child]")
    root_cause_object: str = Field(description="Object color that initiated the causal chain")
    final_answer: str = Field(description="Final answer: initiating object color")

class FuturePredictionEvidence(BaseModel):
    observed_sequence: List[str] = Field(default_factory=list, description="Observed flash color sequence")
    inferred_rule: str = Field(description="Inferred cyclic sequence pattern")
    predicted_future_sequence: List[str] = Field(default_factory=list, description="Predicted future flashes")
    final_answer: str = Field(description="Square color that will flash h steps after video ends")

class LongTermDependencyEvidence(BaseModel):
    marked_parcel_color: str = Field(description="Color of parcel marked at beginning")
    observed_swaps: List[Dict[str, Any]] = Field(default_factory=list, description="Observed position swaps")
    final_parcel_in_bin: str = Field(description="Parcel color that entered delivery bin")
    final_answer: str = Field(description="Final answer: 'yes' or 'no'")
