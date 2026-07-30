import os
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv

load_dotenv()

GPQA_OPTION_LABELS = ["A", "B", "C", "D"]
PHYSICS_ADJACENT_DOMAINS = {
    "Physics",
    "Math",
    "Engineering",
    "Computer Science/AI",
}

MEGASCIENCE_DATASET_ID = "MegaScience/MegaScience"
DEFAULT_MEGASCIENCE_SAMPLES = 2_000
TRAINING_DATASETS = ("megascience", "hle", "gpqa", "hle-gpqa")


def format_mcq_batch(examples, seed=42):
    """
    examples: dict of lists when batched=True.
    Expected columns: Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3
    """
    formatted_questions = []
    answer_letters = []

    rng = random.Random(seed)

    for i in range(len(examples["Question"])):
        question = examples["Question"][i]

        options = [
            (0, examples["Correct Answer"][i]),
            (1, examples["Incorrect Answer 1"][i]),
            (2, examples["Incorrect Answer 2"][i]),
            (3, examples["Incorrect Answer 3"][i]),
        ]

        rng.shuffle(options)

        correct_pos = next(pos for pos, (orig_idx, _) in enumerate(options) if orig_idx == 0)
        answer_letter = GPQA_OPTION_LABELS[correct_pos]

        lines = [question, ""]
        for option_label, (_, text) in zip(GPQA_OPTION_LABELS, options):
            lines.append(f"{option_label}. {text}")

        formatted_questions.append("\n".join(lines))
        answer_letters.append(answer_letter)

    return {
        "question": formatted_questions,
        "answer": answer_letters,
    }


def is_physics_adjacent_domain(value: str | None) -> bool:
    return value in PHYSICS_ADJACENT_DOMAINS


def load_hle_physics_dataset():
    hle = load_dataset("cais/hle", split="test", token=os.getenv("HF_TOKEN"))
    return hle.filter(lambda ex: is_physics_adjacent_domain(ex.get("category"))).map(
        lambda ex: {
            "id": f"hle-{ex['id']}",
            "reference_answer": ex.get("rationale"),
            "subject": ex.get("category"),
        },
        remove_columns=["category", "rationale"],
    ).select_columns(
        ["id", "question", "answer", "answer_type", "reference_answer", "subject"]
    )


def load_gpqa_physics_dataset(seed=42):
    gpqa = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train", token=os.getenv("HF_TOKEN"))
    gpqa_physics = (
        gpqa.filter(lambda ex: is_physics_adjacent_domain(ex.get("High-level domain")))
        .select_columns(
            [
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
                "Explanation",
                "Record ID",
                "High-level domain",
            ]
        )
        .rename_columns(
            {
                "Explanation": "reference_answer",
                "Record ID": "id",
                "High-level domain": "subject",
            }
        )
    )

    gpqa_physics = gpqa_physics.map(
        format_mcq_batch,
        batched=True,
        remove_columns=[
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        ],
        fn_kwargs={"seed": seed},
    ).add_column("answer_type", ["multipleChoice"] * len(gpqa_physics))

    return gpqa_physics.map(lambda ex: {"id": f"gpqa-{ex['id']}"}).select_columns(
        ["id", "question", "answer", "answer_type", "reference_answer", "subject"]
    )


def load_megascience_dataset(
    seed: int = 42,
    max_samples: int | None = DEFAULT_MEGASCIENCE_SAMPLES,
) -> Dataset:
    """Load and normalize the MegaScience training data.

    MegaScience's finalized records expose ``question``, ``answer``,
    ``subject``, and ``reference_answer`` fields.  The conductor training
    pipeline needs the first two plus the metadata fields used by its reward
    function, so normalize them here and assign stable local IDs.

    Sampling happens after a seeded shuffle so ``max_samples`` is a genuine
    deterministic subset rather than simply the first rows in dataset order.
    Pass ``max_samples=None`` to load the complete split.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")

    dataset = load_dataset(
        MEGASCIENCE_DATASET_ID,
        split="train",
        token=os.getenv("HF_TOKEN"),
    ).select_columns(["question", "answer", "subject", "reference_answer"])

    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    dataset = dataset.add_column("id", [f"megascience-{index}" for index in range(len(dataset))])
    return dataset.add_column("answer_type", ["freeForm"] * len(dataset))


def build_training_dataset(
    seed: int = 42,
    max_samples: int | None = DEFAULT_MEGASCIENCE_SAMPLES,
) -> Dataset:
    """Build the default conductor training dataset from MegaScience."""
    return load_megascience_dataset(seed=seed, max_samples=max_samples)


def build_megascience_splits(
    seed: int = 42,
    total_samples: int = DEFAULT_MEGASCIENCE_SAMPLES,
    validation_samples: int = 200,
) -> DatasetDict:
    """Return deterministic train/validation splits from one MegaScience subset."""
    if validation_samples <= 0 or validation_samples >= total_samples:
        raise ValueError("validation_samples must be between 1 and total_samples - 1")

    dataset = load_megascience_dataset(seed=seed, max_samples=total_samples)
    return dataset.train_test_split(test_size=validation_samples, seed=seed, shuffle=True)


def load_conductor_dataset(
    dataset_name: str,
    *,
    seed: int = 42,
    max_samples: int | None = None,
) -> Dataset:
    """Load a normalized dataset used by conductor training and evaluation."""
    if dataset_name not in TRAINING_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; choose one of: {', '.join(TRAINING_DATASETS)}"
        )
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")

    if dataset_name == "megascience":
        # Preserve the established 2,000-row default for existing runs.
        limit = DEFAULT_MEGASCIENCE_SAMPLES if max_samples is None else max_samples
        return load_megascience_dataset(seed=seed, max_samples=limit)
    if dataset_name == "hle":
        dataset = load_hle_physics_dataset()
    elif dataset_name == "gpqa":
        dataset = load_gpqa_physics_dataset(seed=seed)
    else:
        dataset = concatenate_datasets(
            [load_hle_physics_dataset(), load_gpqa_physics_dataset(seed=seed)]
        )

    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def build_conductor_splits(
    dataset_name: str = "megascience",
    *,
    seed: int = 42,
    total_samples: int | None = None,
    validation_samples: int = 200,
) -> DatasetDict:
    """Return deterministic train/evaluation splits for a selected dataset."""
    dataset = load_conductor_dataset(dataset_name, seed=seed, max_samples=total_samples)
    if validation_samples <= 0 or validation_samples >= len(dataset):
        raise ValueError(
            f"validation_samples must be between 1 and {len(dataset) - 1} "
            f"for dataset {dataset_name!r}"
        )
    return dataset.train_test_split(test_size=validation_samples, seed=seed, shuffle=True)
