import os
import random

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

load_dotenv()

GPQA_OPTION_LABELS = ["A", "B", "C", "D"]
PHYSICS_ADJACENT_DOMAINS = {
    "Physics",
    "Math",
    "Engineering",
    "Computer Science/AI",
}


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
    return hle.filter(lambda ex: is_physics_adjacent_domain(ex.get("category"))).select_columns(
        ["id", "question", "answer", "answer_type", "rationale"]
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
            ]
        )
        .rename_columns({"Explanation": "rationale", "Record ID": "id"})
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

    return gpqa_physics


def build_training_dataset(seed=42):
    gpqa_physics = load_gpqa_physics_dataset(seed=seed)
    hle_physics = load_hle_physics_dataset()
    return concatenate_datasets([gpqa_physics, hle_physics])
