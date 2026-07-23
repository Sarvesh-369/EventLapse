import pytest
import tempfile
from pathlib import Path

from eventlapse.generation.bounce_ball import BounceBallGenerator
from eventlapse.generation.blinking import BlinkingGenerator
from eventlapse.generation.state_machine import StateMachineGenerator

def test_bounce_ball_generator():
    gen = BounceBallGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = gen.generate_sample(control_value=3, seed=7, output_dir=Path(tmp_dir))
        assert sample.control_parameter_value == 3
        assert sample.exact_answer == "3"
        assert sample.video_path.exists()
        assert len(sample.executable_trace["events"]) == 3

def test_blinking_generator():
    gen = BlinkingGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = gen.generate_sample(control_value=4, seed=2, output_dir=Path(tmp_dir))
        assert sample.control_parameter_value == 4
        assert sample.exact_answer == "4"
        assert sample.video_path.exists()
        assert len(sample.executable_trace["events"]) == 4

def test_state_machine_generator():
    gen = StateMachineGenerator()
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = gen.generate_sample(control_value=2, seed=1, output_dir=Path(tmp_dir))
        assert sample.control_parameter_value == 2
        assert sample.exact_answer == "2"
        assert sample.video_path.exists()
        assert len(sample.executable_trace["events"]) == 2
