import json

import pytest

from check_kimi_access import build_batch_request, validate_batch_response


def _response(verdicts):
    return {"choices": [{"message": {"content": json.dumps(verdicts)}}]}


def test_build_batch_request_packs_controls_into_one_training_shaped_prompt():
    payload, expected = build_batch_request(3, 4096)

    assert payload["max_tokens"] == 4096
    assert expected == {
        "batch-check-0": True,
        "batch-check-1": False,
        "batch-check-2": True,
    }
    assert len(payload["messages"]) == 2
    user_message = payload["messages"][1]["content"]
    assert "batch-check-0" in user_message
    assert "batch-check-1" in user_message
    assert "batch-check-2" in user_message
    assert "Return exactly one JSON array" in user_message


def test_validate_batch_response_requires_all_ids_and_expected_control_verdicts():
    expected = {"batch-check-0": True, "batch-check-1": False}
    payload = _response(
        [
            {"id": "batch-check-0", "correct": True, "reason": "The answer is four."},
            {"id": "batch-check-1", "correct": False, "reason": "The answer is five."},
        ]
    )

    verdicts = validate_batch_response(payload, expected)

    assert verdicts["batch-check-0"][0] is True
    with pytest.raises(ValueError, match="incorrect control verdicts"):
        validate_batch_response(
            _response(
                [
                    {"id": "batch-check-0", "correct": False, "reason": "Wrong verdict."},
                    {"id": "batch-check-1", "correct": False, "reason": "The answer is five."},
                ]
            ),
            expected,
        )


def test_build_batch_request_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_batch_request(0, 4096)
