import pytest
import tempfile
from pathlib import Path

from eventlapse.generation.bounce_ball import BounceBallGenerator
from eventlapse.generation.blinking import BlinkingGenerator
from eventlapse.generation.state_machine import StateMachineGenerator

def test_bounce_ball_generator():
    gen = BounceBallGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample0 = gen.generate_sample(control_value=0, frequency=1.0, seed=5, output_dir=Path(tmp_dir))
        assert sample0.control_parameter_value == 0
        assert sample0.exact_answer == "0"
        assert len(sample0.executable_trace["events"]) == 0

        sample3 = gen.generate_sample(control_value=3, frequency=2.0, seed=7, output_dir=Path(tmp_dir))
        assert sample3.control_parameter_value == 3
        assert sample3.exact_answer == "3"
        assert len(sample3.executable_trace["events"]) == 3
        assert sample3.executable_trace["frequency_hz"] == 2.0

def test_blinking_generator():
    gen = BlinkingGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample0 = gen.generate_sample(control_value=0, frequency=0.5, seed=5, output_dir=Path(tmp_dir))
        assert sample0.control_parameter_value == 0
        assert sample0.exact_answer == "0"
        assert len(sample0.executable_trace["events"]) == 0

        sample4 = gen.generate_sample(control_value=4, frequency=3.0, seed=2, output_dir=Path(tmp_dir))
        assert sample4.control_parameter_value == 4
        assert sample4.exact_answer == "4"
        assert len(sample4.executable_trace["events"]) == 4
        assert sample4.executable_trace["frequency_hz"] == 3.0

def test_state_machine_generator():
    gen = StateMachineGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample0 = gen.generate_sample(control_value=0, frequency=1.5, seed=5, output_dir=Path(tmp_dir))
        assert sample0.control_parameter_value == 0
        assert sample0.exact_answer == "0"
        assert len(sample0.executable_trace["events"]) == 0

        sample2 = gen.generate_sample(control_value=2, frequency=4.0, seed=1, output_dir=Path(tmp_dir))
        assert sample2.control_parameter_value == 2
        assert sample2.exact_answer == "2"
        assert len(sample2.executable_trace["events"]) == 2
        assert sample2.executable_trace["frequency_hz"] == 4.0
