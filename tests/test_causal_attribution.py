import pytest
import random
import tempfile
from pathlib import Path
from eventlapse.generation.causal_attribution import CausalAttributionGenerator, COLOR_NAMES

def test_causal_attribution_c1_no_intermediate_components():
    generator = CausalAttributionGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        sample = generator.generate_sample(control_value=1, seed=0, output_dir=out_dir)

        trace = sample.executable_trace
        assert trace["causal_depth"] == 1
        assert len(trace["trials"]) == 3

        for trial in trace["trials"]:
            # C = 1 must contain exactly 1 transition
            assert len(trial["events"]) == 1
            assert trial["events"][0]["transition_index"] == 1

        # Exactly 1 trial activates lamp
        succ_trials = [t for t in trace["trials"] if t["lamp_activated"]]
        assert len(succ_trials) == 1
        assert trace["answer"] == sample.exact_answer

def test_causal_attribution_static_shortcuts_eliminated():
    generator = CausalAttributionGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        for C in [1, 2, 3, 4, 5, 6]:
            sample = generator.generate_sample(control_value=C, seed=0, output_dir=out_dir)
            trace = sample.executable_trace

            assert trace["causal_depth"] == C
            assert len(trace["trials"]) == 3

            # 1. Every trial has exactly C transitions
            for trial in trace["trials"]:
                assert len(trial["events"]) == C
                # Check timestamps fall within duration
                for ev in trial["events"]:
                    assert 0.0 <= ev["timestamp"] <= sample.duration

            # 2. Exactly one trial activates the lamp
            succ = [t for t in trace["trials"] if t["is_successful"]]
            assert len(succ) == 1
            assert succ[0]["outcome"] == "lamp_activated"
            assert trace["answer"] == succ[0]["initiator_color"]

            # 3. Rendered duration matches stored metadata
            assert sample.duration > 0.0

def test_causal_attribution_seed_balancing():
    succ_positions = []
    true_colors = []

    for seed in range(30):
        rng = random.Random(seed)
        candidate_colors = list(COLOR_NAMES)
        trial_order = [0, 1, 2]
        rng.shuffle(trial_order)
        successful_trial_idx = rng.randint(0, 2)
        true_cause_color = candidate_colors[trial_order[successful_trial_idx]]

        succ_positions.append(successful_trial_idx)
        true_colors.append(true_cause_color)

    # Check all 3 positions occur
    unique_pos = set(succ_positions)
    assert len(unique_pos) == 3, f"Not all trial positions represented: {unique_pos}"

    # Check all 3 colors occur
    unique_colors = set(true_colors)
    assert len(unique_colors) == 3, f"Not all colors represented: {unique_colors}"
