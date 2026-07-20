from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig

from theo_conductor.data import build_megascience_splits, build_training_dataset
from theo_conductor.grpo import (
    CORRECT_REWARD,
    INVALID_WORKFLOW_REWARD,
    MALFORMED_REWARD,
    VALID_WORKFLOW_REWARD,
    RewardTrace,
    build_grpo_trainer,
    compute_reward,
    parse_conductor_json,
)
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.prompt import build_conductor_prompt
from theo_conductor.runner import Runner
from theo_conductor.traces import TrainingTraceLogger

load_dotenv()


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B"
DEFAULT_OUTPUT_DIR = "outputs/grpo-conductor"


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = DEFAULT_OUTPUT_DIR
    config_path: str = "configs/local_small_models.yaml"
    seed: int = 42
    max_train_samples: int | None = None
    max_steps: int = 200
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 256
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_generations: int = 64
    generation_batch_size: int = 256
    max_completion_length: int = 1024
    max_worker_tokens: int = 4096
    worker_temperature: float = 0.2
    max_context_length: int | None = None
    validation_samples: int = 200
    eval_steps: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    use_vllm: bool = False
    execute_workflows: bool = False
    bf16: bool | None = None
    fp16: bool = False
    report_to: str = "wandb"
    wandb_project: str = "theo-conductor"
    wandb_run_name: str | None = None
    dry_run: bool = False
    preflight: bool = False
    skip_preflight: bool = False


def log_cuda_memory(stage: str) -> None:
    """Emit concise allocator and device totals for memory debugging."""

    if not torch.cuda.is_available():
        print(f"CUDA_MEMORY stage={stage} cuda_available=false", flush=True)
        return

    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>").split(",")
    gib = 1024**3
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        physical = visible_devices[device] if device < len(visible_devices) else "unknown"
        print(
            "CUDA_MEMORY "
            f"stage={stage} logical_gpu={device} physical_gpu={physical} "
            f"device_used_gib={(total - free) / gib:.2f} device_free_gib={free / gib:.2f} "
            f"pytorch_allocated_gib={allocated / gib:.2f} "
            f"pytorch_reserved_gib={reserved / gib:.2f} "
            f"pytorch_peak_allocated_gib={peak_allocated / gib:.2f}",
            flush=True,
        )


def run_isolated_preflight() -> None:
    """Run preflight in a child process so all of its CUDA state dies with it."""

    command = [sys.executable, "-m", "theo_conductor.train", *sys.argv[1:], "--preflight"]
    print("Starting isolated GRPO preflight process", flush=True)
    subprocess.run(command, check=True)
    print("Isolated GRPO preflight process exited cleanly", flush=True)


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
    training_kwargs = dict(
        output_dir=config.output_dir,
        seed=config.seed,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.999,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        beta=0.0,
        epsilon=0.2,
        sync_ref_model=False,
        use_vllm=config.use_vllm,
        report_to=config.report_to,
        run_name=config.wandb_run_name,
        eval_strategy="no" if config.preflight else "steps",
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=64,
        save_strategy="steps",
        save_steps=1 if config.preflight else 500,
        remove_unused_columns=False,
    )
    # Leave precision at the TRL/Transformers default when the operator did
    # not explicitly select one; passing None can be interpreted as True by
    # some installed Transformers versions.
    training_kwargs["bf16"] = config.bf16 if config.bf16 is not None else torch.cuda.is_available()
    if config.fp16:
        training_kwargs["fp16"] = True
    return GRPOConfig(**training_kwargs)


def load_processing_class(model_name: str):
    try:
        return AutoProcessor.from_pretrained(model_name)
    except ValueError:
        return AutoTokenizer.from_pretrained(model_name)


def build_trainer(config: TrainConfig):
    if config.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    model_registry = ModelRegistry.from_yaml_file(config.config_path)
    splits = build_megascience_splits(
        seed=config.seed,
        total_samples=2_000,
        validation_samples=config.validation_samples,
    )
    train_dataset = prepare_grpo_dataset(
        splits["train"],
        model_registry,
        max_samples=config.max_train_samples,
    )
    eval_dataset = prepare_grpo_dataset(splits["test"], model_registry)
    processor = load_processing_class(config.model_name)
    trace_logger = TrainingTraceLogger(
        config.output_dir,
        log_to_wandb=config.report_to == "wandb",
    )

    runner = (
        Runner(
            model_registry,
            max_worker_tokens=config.max_worker_tokens,
            worker_temperature=config.worker_temperature,
        )
        if config.execute_workflows
        else None
    )
    return build_grpo_trainer(
        model=config.model_name,
        train_dataset=train_dataset,
        processing_class=processor,
        args=build_training_args(config),
        model_registry=model_registry,
        runner=runner,
        execute_workflows=config.execute_workflows,
        eval_dataset=eval_dataset,
        trace_observer=trace_logger,
    )


def _tokenizer_for(processing_class: Any) -> Any:
    return getattr(processing_class, "tokenizer", processing_class)


def _prompt_token_count(processing_class: Any, prompt: str) -> int:
    encoded = _tokenizer_for(processing_class)(prompt, add_special_tokens=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    shape = getattr(input_ids, "shape", None)
    if shape:
        return int(shape[-1])
    return len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)


def _context_length(processing_class: Any, configured_length: int | None) -> int:
    if configured_length is not None:
        return configured_length

    length = getattr(_tokenizer_for(processing_class), "model_max_length", None)
    # Transformers uses very large sentinel values when a tokenizer does not
    # advertise a real limit; require an explicit operator choice in that case.
    if not isinstance(length, int) or length <= 0 or length >= 10**12:
        raise ValueError("Tokenizer has no finite context length; pass --max-context-length.")
    return length


def _reward_tier_probe(registry: ModelRegistry) -> None:
    valid_workflow = {
        "task_type": "physics",
        "difficulty": "medium",
        "question": "What is 2 + 2?",
        "workflow": [
            {
                "step_id": "final",
                "model_id": registry.model_ids()[0],
                "instruction": "Return the answer. End with FINAL: <answer>.",
                "access_list": ["question"],
            }
        ],
        "final_answer": "4",
    }
    invalid_workflow = {**valid_workflow, "workflow": [{**valid_workflow["workflow"][0], "step_id": "solve"}]}
    valid_incorrect = {**valid_workflow, "final_answer": "5"}
    rewards = compute_reward(
        ["not json", invalid_workflow, valid_incorrect, valid_workflow],
        ground_truth=["4"] * 4,
        model_registry=registry,
    )
    expected = [MALFORMED_REWARD, INVALID_WORKFLOW_REWARD, VALID_WORKFLOW_REWARD, CORRECT_REWARD]
    if rewards != expected:
        raise RuntimeError(f"Reward tier probe failed: expected {expected}, got {rewards}.")


def run_preflight(config: TrainConfig) -> None:
    """Run one real, checkpointed GRPO update before starting a full training job."""
    if not config.use_vllm:
        raise ValueError("Preflight requires --use-vllm so conductor generation uses vLLM.")

    registry = ModelRegistry.from_yaml_file(config.config_path)
    raw_dataset = build_training_dataset(seed=config.seed, max_samples=2_000)
    if len(raw_dataset) != 2_000:
        raise RuntimeError(f"MegaScience preflight requires 2,000 rows, but loaded {len(raw_dataset)}.")
    train_dataset = prepare_grpo_dataset(raw_dataset, registry)
    processing_class = load_processing_class(config.model_name)
    context_length = _context_length(processing_class, config.max_context_length)
    longest_prompt = max(_prompt_token_count(processing_class, row["prompt"]) for row in train_dataset)
    if longest_prompt + config.max_completion_length > context_length:
        raise RuntimeError(
            "MegaScience prompts exceed the selected context window: "
            f"{longest_prompt} prompt tokens + {config.max_completion_length} completion tokens > {context_length}."
        )

    _reward_tier_probe(registry)

    preflight_dir = Path(config.output_dir) / "preflight"
    traces: list[RewardTrace] = []
    trace_logger = TrainingTraceLogger(preflight_dir, log_to_wandb=False)

    def observe_preflight_traces(batch: list[RewardTrace]) -> None:
        traces.extend(batch)
        trace_logger(batch)

    preflight_config = replace(
        config,
        output_dir=str(preflight_dir),
        max_train_samples=2,
        max_steps=1,
        num_train_epochs=1.0,
        # Keep the generation batch divisible by num_generations when the
        # preflight is launched as a single process (the Slurm default).
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        # GRPO computes an advantage relative to the other completions for a
        # prompt, so TRL requires at least two generations even for preflight.
        num_generations=2,
        generation_batch_size=2,
        report_to="none",
        preflight=True,
    )
    args = build_training_args(preflight_config)
    runner = (
        Runner(
            registry,
            max_worker_tokens=config.max_worker_tokens,
            worker_temperature=config.worker_temperature,
        )
        if config.execute_workflows
        else None
    )
    trainer = build_grpo_trainer(
        model=config.model_name,
        train_dataset=train_dataset.select(range(2)),
        processing_class=processing_class,
        args=args,
        model_registry=registry,
        runner=runner,
        execute_workflows=config.execute_workflows,
        trace_observer=observe_preflight_traces,
    )
    trainer.train()

    if not traces:
        raise RuntimeError("GRPO preflight completed without a conductor generation.")
    generated = next((trace for trace in traces if trace.task is not None), None)
    if generated is None:
        raise RuntimeError(f"Conductor generation did not parse: {traces[0].error or 'unknown parse error'}")
    # Parsing separately makes this guarantee explicit even if reward behavior changes.
    parse_conductor_json(generated.completion, question=train_dataset[0]["question"], model_registry=registry)
    if config.execute_workflows:
        if generated.run_result is None:
            raise RuntimeError(
                "Parsed workflow did not execute through a configured vLLM worker: "
                f"{generated.error or 'no run result'}"
            )
        if any(registry.get(step.model_id).provider != "vllm" for step in generated.task.workflow):
            raise RuntimeError("Parsed workflow used a non-vLLM worker; select a vLLM-only model config for preflight.")
    if not any(preflight_dir.glob("checkpoint-*")):
        raise RuntimeError(f"GRPO preflight did not save a checkpoint in {preflight_dir}.")

    execution_check = "vLLM worker execution, " if config.execute_workflows else "format-only reward, "
    print(
        "Preflight passed: 2,000 MegaScience rows, "
        f"{longest_prompt}/{context_length} prompt+completion token budget, "
        f"parsed conductor workflow, {execution_check}reward tiers, and checkpoint."
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the conductor with TRL GRPO.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path", default="configs/local_small_models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--num-generations", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-conductor-completion-length",
        "--max-completion-length",
        dest="max_completion_length",
        type=int,
        default=1024,
        help="Maximum generated tokens for the trainable conductor model (default: 1024).",
    )
    parser.add_argument(
        "--max-worker-tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens for each worker-model workflow step (default: 4096).",
    )
    parser.add_argument(
        "--worker-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for worker-model workflow steps (default: 0.2).",
    )
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument(
        "--execute-workflows",
        action="store_true",
        help="Execute generated workflows through worker models while scoring; disabled by default.",
    )
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="theo-conductor")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--dry-run", action="store_true")
    preflight_mode = parser.add_mutually_exclusive_group()
    preflight_mode.add_argument("--preflight", action="store_true", help="Run only the end-to-end preflight.")
    preflight_mode.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Start full training immediately; use only after a separate preflight passed.",
    )

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    if config.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    if config.dry_run or config.preflight:
        log_cuda_memory("before_preflight")
        run_preflight(config)
        log_cuda_memory("after_preflight")
        return

    # A child process gives CUDA/vLLM a hard lifetime boundary. Constructing
    # both trainers in this process retained tens of GiB from preflight.
    if not config.skip_preflight:
        run_isolated_preflight()
        log_cuda_memory("after_isolated_preflight")
    trainer = build_trainer(config)
    log_cuda_memory("after_full_trainer_build")
    try:
        trainer.train()
    except torch.OutOfMemoryError:
        log_cuda_memory("full_training_oom")
        raise


if __name__ == "__main__":
    main()
