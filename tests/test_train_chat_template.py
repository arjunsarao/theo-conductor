from datasets import Dataset

from theo_conductor.models.fake import FakeModelClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelSpec
from theo_conductor.train import (
    TrainConfig,
    build_conductor_prompt,
    build_training_args,
    parse_args,
    prepare_grpo_dataset,
    resolve_chat_template,
    run_isolated_preflight,
)


class DummyTokenizer:
    def __init__(self, template: str | None = None):
        self.chat_template = template


class DummyProcessor:
    def __init__(self, template: str | None = None, tokenizer_template: str | None = None):
        self.chat_template = template
        self.tokenizer = DummyTokenizer(tokenizer_template)


def test_resolve_chat_template_falls_back_to_tokenizer_template():
    processor = DummyProcessor(tokenizer_template="tokenizer-template")

    assert resolve_chat_template(processor) == "tokenizer-template"


def test_resolve_chat_template_prefers_processor_template():
    processor = DummyProcessor(template="processor-template", tokenizer_template="tokenizer-template")

    assert resolve_chat_template(processor) == "processor-template"


def test_build_conductor_prompt_uses_registry_model_ids():
    registry = ModelRegistry(
        [
            ModelSpec(
                model_idx="solver",
                display_name="Physics Solver",
                provider="local",
                client=FakeModelClient("solver"),
            )
        ]
    )

    prompt = build_conductor_prompt("What is 2 + 2?", registry)

    assert 'model_id="solver"' in prompt
    assert "Physics Solver" in prompt
    assert "What is 2 + 2?" in prompt


def test_prepare_grpo_dataset_adds_prompt_and_reward_columns():
    registry = ModelRegistry([ModelSpec(model_idx="solver", client=FakeModelClient("solver"))])
    dataset = Dataset.from_list(
        [
            {
                "id": "example-1",
                "question": "What is 2 + 2?",
                "answer": "A",
                "answer_type": "multipleChoice",
            }
        ]
    )

    prepared = prepare_grpo_dataset(dataset, registry)

    assert prepared[0]["question"] == "What is 2 + 2?"
    assert prepared[0]["answer"] == "A"
    assert prepared[0]["answer_type"] == "multipleChoice"
    assert "Output JSON matching this schema" in prepared[0]["prompt"]


def test_build_training_args_maps_train_config_to_grpo_config():
    args = build_training_args(
        TrainConfig(
            output_dir="tmp-output",
            max_steps=3,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=128,
        )
    )

    assert args.output_dir == "tmp-output"
    assert args.max_steps == 3
    assert args.per_device_train_batch_size == 2
    assert args.num_generations == 2
    assert args.max_completion_length == 128


def test_paper_training_defaults_map_to_one_iteration_batch():
    config = TrainConfig()
    args = build_training_args(config)

    assert config.model_name == "Qwen/Qwen2.5-7B"
    assert args.max_steps == 200
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 256
    assert args.generation_batch_size == 256
    assert args.num_generations == 64
    assert args.generation_batch_size // args.num_generations == 4
    assert args.max_completion_length == 1024
    assert args.temperature == 1.0
    assert args.learning_rate == 1e-6
    assert str(args.lr_scheduler_type) == "SchedulerType.COSINE"
    assert args.warmup_ratio == 0.03
    assert args.adam_beta1 == 0.9
    assert args.adam_beta2 == 0.999
    assert args.epsilon == 0.2
    assert args.beta == 0.0
    assert args.sync_ref_model is False
    assert config.max_worker_tokens == 4096
    assert config.worker_temperature == 0.2
    assert config.execute_workflows is False


def test_worker_execution_is_cli_opt_in(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--execute-workflows"])

    assert parse_args().execute_workflows is True


def test_isolated_preflight_relaunches_training_in_a_child_process(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["train", "--max-steps", "10"])
    monkeypatch.setattr("subprocess.run", lambda command, check: calls.append((command, check)))

    run_isolated_preflight()

    [(command, check)] = calls
    assert command[1:3] == ["-m", "theo_conductor.train"]
    assert command[3:] == ["--max-steps", "10", "--preflight"]
    assert check is True
