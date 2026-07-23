import pytest
import tempfile
from pathlib import Path

from eventlapse.generation.event_counting import EventCountingGenerator
from eventlapse.generation.event_frequency import EventFrequencyGenerator
from eventlapse.generation.temporal_ordering import TemporalOrderingGenerator
from eventlapse.generation.duration_comparison import DurationComparisonGenerator
from eventlapse.generation.causal_attribution import CausalAttributionGenerator
from eventlapse.generation.future_prediction import FuturePredictionGenerator
from eventlapse.generation.long_term_dependency import LongTermDependencyGenerator

def test_event_counting_generator():
    gen = EventCountingGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = gen.generate_sample(control_value=3, seed=7, output_dir=Path(tmp_dir))
        assert sample.control_parameter_value == 3
        assert sample.exact_answer == "3"
        assert sample.video_path.exists()
        assert len(sample.executable_trace["events"]) == 3

def test_seed_nuisance_independence():
    gen = EventCountingGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample0 = gen.generate_sample(control_value=4, seed=0, output_dir=Path(tmp_dir))
        sample1 = gen.generate_sample(control_value=4, seed=1, output_dir=Path(tmp_dir))
        assert sample0.control_parameter_value == sample1.control_parameter_value == 4
        assert sample0.exact_answer == sample1.exact_answer == "4"

def test_long_term_dependency_balanced():
    gen = LongTermDependencyGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        s0 = gen.generate_sample(control_value=2, seed=0, output_dir=Path(tmp_dir))
        s1 = gen.generate_sample(control_value=2, seed=1, output_dir=Path(tmp_dir))
        assert {s0.exact_answer, s1.exact_answer} == {"yes", "no"}
