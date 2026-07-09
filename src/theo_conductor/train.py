from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig

from theo_conductor.data import build_training_dataset
from theo_conductor.grpo import build_grpo_trainer
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.prompt import build_prompt

load_dotenv()


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/grpo-conductor"


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = DEFAULT_OUTPUT_DIR
    config_dir: str = "configs"
    seed: int = 42
    max_train_samples: int | None = None
    max_steps: int = -1
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    num_generations: int = 4
    max_completion_length: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    use_vllm: bool = False
    bf16: bool | None = None
    fp16: bool = False
    report_to: str = "none"
    dry_run: bool = False


def resolve_chat_template(processor: Any) -> str:
    processor_chat_template = getattr(processor, "chat_template", None)
    if processor_chat_template:
        return processor_chat_template

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_chat_template = getattr(tokenizer, "chat_template", None)
    if tokenizer_chat_template:
        return tokenizer_chat_template

    raise ValueError("This processor does not expose a chat template.")


def build_example_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": "Create a valid conductor workflow for the provided physics problem.",
        },
    ]


def build_worker_model_lines(model_registry: ModelRegistry) -> list[str]:
    lines: list[str] = []

    for model_id, spec in model_registry._models.items():
        details = []
        if spec.display_name:
            details.append(f"name={spec.display_name}")
        if spec.provider:
            details.append(f"provider={spec.provider}")
        if spec.tags:
            details.append(f"tags={','.join(sorted(spec.tags))}")
        if spec.supports_json:
            details.append("supports_json=true")
        if spec.supports_tools:
            details.append("supports_tools=true")

        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- model_id={json.dumps(model_id)}{suffix}")

    return lines


def build_default_examples(model_registry: ModelRegistry) -> list[str]:
    model_ids = list(model_registry._models)
    if not model_ids:
        return []

    first_model_id = model_ids[0]
    example = {
        "task_type": "physics",
        "difficulty": "medium",
        "workflow": [
            {
                "step_id": "final",
                "model_id": first_model_id,
                "instruction": "Solve the problem and return only the final answer.",
                "access_list": ["question"],
            }
        ],
    }
    return [json.dumps(example, indent=2)]


def build_conductor_prompt(
    question: str,
    model_registry: ModelRegistry,
    *,
    tools: list[str] | None = None,
    examples: list[str] | None = None,
) -> str:
    return build_prompt(
        build_worker_model_lines(model_registry),
        tools or ["No external tools are available."],
        examples or build_default_examples(model_registry),
        question,
    )


def prepare_grpo_dataset(
    dataset: Dataset,
    model_registry: ModelRegistry,
    *,
    max_samples: int | None = None,
) -> Dataset:
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_row(row: dict[str, Any]) -> dict[str, Any]:
        question = row["question"]
        return {
            "prompt": build_conductor_prompt(question, model_registry),
            "question": question,
            "answer": row["answer"],
            "answer_type": row.get("answer_type"),
            "id": row.get("id"),
        }

    return dataset.map(format_row)


def build_training_args(config: TrainConfig) -> GRPOConfig:
    return GRPOConfig(
        output_dir=config.output_dir,
        seed=config.seed,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        use_vllm=config.use_vllm,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to=config.report_to,
        remove_unused_columns=False,
    )


def load_processing_class(model_name: str):
    try:
        return AutoProcessor.from_pretrained(model_name)
    except ValueError:
        return AutoTokenizer.from_pretrained(model_name)


def build_trainer(config: TrainConfig):
    model_registry = ModelRegistry.from_config_dir(config.config_dir)
    train_dataset = prepare_grpo_dataset(
        build_training_dataset(seed=config.seed),
        model_registry,
        max_samples=config.max_train_samples,
    )
    processor = load_processing_class(config.model_name)

    return build_grpo_trainer(
        model=config.model_name,
        train_dataset=train_dataset,
        processing_class=processor,
        args=build_training_args(config),
        model_registry=model_registry,
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the conductor with TRL GRPO.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    model_registry = ModelRegistry.from_config_dir(config.config_dir)
    train_dataset = prepare_grpo_dataset(
        build_training_dataset(seed=config.seed),
        model_registry,
        max_samples=config.max_train_samples,
    )

    if config.dry_run:
        print(f"Prepared {len(train_dataset)} training examples.")
        print(train_dataset[0]["prompt"])
        return

    processor = load_processing_class(config.model_name)
    trainer = build_grpo_trainer(
        model=config.model_name,
        train_dataset=train_dataset,
        processing_class=processor,
        args=build_training_args(config),
        model_registry=model_registry,
    )
    trainer.train()


if __name__ == "__main__":
    main()
