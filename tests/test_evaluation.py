from eventlapse.evaluation.exact_match import compute_exact_match, compute_wilson_score_interval
from eventlapse.evaluation.boundaries import estimate_operational_boundary
from eventlapse.evaluation.morse_evaluator import MorseTraceEvaluator
from eventlapse.inference.parser import parse_model_response, extract_boxed_answer

def test_exact_match():
    assert compute_exact_match(" 4 ", "4") is True
    assert compute_exact_match("16", "16") is True
    assert compute_exact_match("left", "Right") is False

def test_boxed_parser_extraction():
    raw_text = "Final Answer inside \\boxed{} (e.g. \\boxed{4})\n\n- 0.0s: start count 0\n- 1.0s: bounce 1\n... 16 bounces total ...\n\n\\boxed{16}"
    pred_ans, evidence, valid = parse_model_response(raw_text)
    assert pred_ans == "16"
    assert valid is True
    assert evidence is not None
    assert evidence["boxed_answer"] == "16"

def test_wilson_score_interval():
    acc, lower, upper = compute_wilson_score_interval(18, 20)
    assert acc == 0.90
    assert lower > 0.65
    assert upper <= 1.0

def test_estimate_operational_boundary():
    accuracy_data = {
        1: (20, 20),
        2: (19, 20),
        3: (12, 20)
    }
    boundary = estimate_operational_boundary([1, 2, 3], accuracy_data, tau=0.70)
    assert boundary == 2

def test_morse_evaluator():
    evaluator = MorseTraceEvaluator()
    gt_trace = {"steps": [{"state": {}, "event": {}, "operation": {}}]}
    res = evaluator.evaluate_sample_trace("4", {"steps": []}, "4", gt_trace)
    assert res["exact_match"] is True
    assert res["step_alignment_score"] == 0.0
