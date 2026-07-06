import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from hle import HLEDataset


def test_hle_dataset_loads_default_parquet():
    dataset = HLEDataset(columns=["id", "question", "answer", "answer_type"])

    assert len(dataset) == 2500
    assert dataset.column_names == ["id", "question", "answer", "answer_type"]

    row = dataset[0]
    assert set(row) == {"id", "question", "answer", "answer_type"}
    assert row["id"]
    assert row["question"]
    assert row["answer"]


def test_hle_dataset_supports_transform_and_negative_indexing():
    dataset = HLEDataset(
        columns=["question", "answer"],
        transform=lambda row: (row["question"], row["answer"]),
    )

    question, answer = dataset[-1]

    assert isinstance(question, str)
    assert isinstance(answer, str)


def test_hle_dataset_rejects_out_of_range_index():
    dataset = HLEDataset(columns=["id"])

    with pytest.raises(IndexError, match="out of range"):
        dataset[len(dataset)]
