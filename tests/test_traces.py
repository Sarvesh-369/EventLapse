from eventlapse.traces.validation import validate_trace_structure

def test_trace_validation():
    valid_trace = {
        "steps": [
            {
                "state": {"count": 1},
                "event": {"type": "bounce", "timestamp": 1.0},
                "operation": {"action": "increment"}
            }
        ]
    }
    is_valid, errors = validate_trace_structure(valid_trace)
    assert is_valid is True
    assert len(errors) == 0

def test_invalid_trace_structure():
    invalid_trace = {"steps": ["invalid_step_str"]}
    is_valid, errors = validate_trace_structure(invalid_trace)
    assert is_valid is False
    assert len(errors) > 0
