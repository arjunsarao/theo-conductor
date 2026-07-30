from datasets import Dataset
import pytest

from theo_conductor import data


def _rows(prefix: str, count: int) -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": f"{prefix}-{index}",
                "question": f"Question {prefix}-{index}",
                "answer": str(index),
                "answer_type": "freeForm",
                "reference_answer": f"Explanation {index}",
                "subject": "Physics",
            }
            for index in range(count)
        ]
    )


def test_combined_hle_gpqa_split_is_deterministic(monkeypatch):
    monkeypatch.setattr(data, "load_hle_physics_dataset", lambda: _rows("hle", 4))
    monkeypatch.setattr(data, "load_gpqa_physics_dataset", lambda seed=42: _rows("gpqa", 4))

    first = data.build_conductor_splits(
        "hle-gpqa",
        seed=7,
        total_samples=6,
        validation_samples=2,
    )
    second = data.build_conductor_splits(
        "hle-gpqa",
        seed=7,
        total_samples=6,
        validation_samples=2,
    )

    assert len(first["train"]) == 4
    assert len(first["test"]) == 2
    assert first["train"]["id"] == second["train"]["id"]
    assert first["test"]["id"] == second["test"]["id"]
    all_ids = list(first["train"]["id"]) + list(first["test"]["id"])
    assert {row_id.split("-", 1)[0] for row_id in all_ids} == {
        "hle",
        "gpqa",
    }


def test_dataset_split_rejects_invalid_name_and_validation_size(monkeypatch):
    with pytest.raises(ValueError, match="Unknown dataset"):
        data.load_conductor_dataset("unknown")

    monkeypatch.setattr(data, "load_hle_physics_dataset", lambda: _rows("hle", 3))
    with pytest.raises(ValueError, match="validation_samples"):
        data.build_conductor_splits("hle", validation_samples=3)


def test_hle_loader_filters_and_normalizes_physics_adjacent_rows(monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "id": "1",
                "question": "Physics question",
                "answer": "42",
                "answer_type": "exactMatch",
                "rationale": "Because physics.",
                "category": "Physics",
            },
            {
                "id": "2",
                "question": "History question",
                "answer": "No",
                "answer_type": "exactMatch",
                "rationale": "Because history.",
                "category": "Humanities/Social Science",
            },
        ]
    )
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: raw)

    loaded = data.load_hle_physics_dataset()

    assert len(loaded) == 1
    assert loaded[0] == {
        "id": "hle-1",
        "question": "Physics question",
        "answer": "42",
        "answer_type": "exactMatch",
        "reference_answer": "Because physics.",
        "subject": "Physics",
    }


def test_gpqa_loader_formats_choices_and_preserves_explanation(monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "wrong 1",
                "Incorrect Answer 2": "wrong 2",
                "Incorrect Answer 3": "wrong 3",
                "Explanation": "The correct option follows from the premise.",
                "Record ID": "record-1",
                "High-level domain": "Math",
            }
        ]
    )
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: raw)

    loaded = data.load_gpqa_physics_dataset(seed=9)

    assert len(loaded) == 1
    assert loaded[0]["id"] == "gpqa-record-1"
    assert loaded[0]["answer"] in {"A", "B", "C", "D"}
    assert loaded[0]["answer_type"] == "multipleChoice"
    assert loaded[0]["reference_answer"] == "The correct option follows from the premise."
    assert loaded[0]["subject"] == "Math"
    assert "A." in loaded[0]["question"]
