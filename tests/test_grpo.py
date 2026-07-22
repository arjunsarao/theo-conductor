import asyncio
import json

import pytest

from theo_conductor.grpo import (
    ConductorParseError,
    JudgeBatchError,
    answers_match,
    build_grpo_trainer,
    compute_reward_traces,
    compute_reward,
    judge_reward_traces,
    parse_conductor_json,
)
from theo_conductor.models.fake import FakeModelClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelResponse, ModelSpec


VALID_COMPLETION = """
```json
{
  "task_type": "physics",
  "difficulty": "medium",
  "question": "What is 2 + 2?",
  "workflow": [
    {
      "step_id": "final",
      "model_id": "solver",
      "instruction": "Return the answer. End with FINAL: <answer>.",
      "access_list": ["question"]
    }
  ],
  "final_answer": "A"
}
```
"""


def test_parse_conductor_json_accepts_fenced_json():
    task = parse_conductor_json(VALID_COMPLETION)

    assert task.task_type == "physics"
    assert task.workflow[0].step_id == "final"


def test_parse_conductor_json_recovers_workflow_after_other_json_fragment():
    completion = "Model output:\n{}\nAssistant:\n" + VALID_COMPLETION.replace("```json", "").replace("```", "")

    task = parse_conductor_json(completion)

    assert task.task_type == "physics"
    assert task.workflow[0].step_id == "final"


def test_parse_conductor_json_treats_last_entry_as_final_regardless_of_name():
    completion = VALID_COMPLETION.replace('"step_id": "final"', '"step_id": "answer"')

    task = parse_conductor_json(completion)

    assert task.workflow[-1].step_id == "answer"


def test_parse_conductor_json_rejects_invalid_workflow():
    with pytest.raises(ConductorParseError):
        parse_conductor_json('{"workflow": []}')


def test_parse_conductor_json_allows_runner_to_add_final_output_protocol():
    completion = VALID_COMPLETION.replace("End with FINAL: <answer>.", "Return only the answer.")

    task = parse_conductor_json(completion)

    assert task.workflow[-1].instruction == "Return the answer. Return only the answer."


def test_compute_reward_scores_malformed_invalid_valid_and_correct():
    invalid_workflow = """
    {
      "task_type": "physics",
      "difficulty": "medium",
      "question": "What is 2 + 2?",
      "workflow": [
        {
          "step_id": "solve",
          "model_id": "solver",
          "instruction": "Solve.",
          "access_list": ["question", "missing"]
        }
      ]
    }
    """
    valid_without_answer = VALID_COMPLETION.replace('"final_answer": "A"', '"final_answer": "B"')

    assert compute_reward(
        ["not json", invalid_workflow, valid_without_answer, VALID_COMPLETION],
        ground_truth=["A", "A", "A", "A"],
        answer_type=["multipleChoice"] * 4,
    ) == [0.0, 0.2, 0.5, 1.0]


def test_compute_reward_validates_model_ids_when_registry_is_supplied():
    registry = ModelRegistry([ModelSpec(model_idx="other", client=FakeModelClient("other"))])

    assert compute_reward([VALID_COMPLETION], ground_truth=["A"], model_registry=registry) == [0.2]


def test_answers_match_accepts_close_scientific_numbers_and_equivalent_symbols():
    assert answers_match("6.022e23", "6.02214076 × 10^23")
    assert answers_match("0.3333334", "1/3")
    assert answers_match(r"\frac{1}{2}", "0.5")
    assert answers_match("(x + 1) * (x - 1)", "x^2 - 1")
    assert not answers_match("1.2", "1.0")


def test_reward_does_not_run_workers_unless_explicitly_enabled():
    calls = 0

    class CountingClient(FakeModelClient):
        async def generate(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return await super().generate(*args, **kwargs)

    registry = ModelRegistry([ModelSpec(model_idx="solver", client=CountingClient("FINAL: A"))])

    assert compute_reward([VALID_COMPLETION], ground_truth=["A"], model_registry=registry) == [1.0]
    assert calls == 0


def test_optional_execution_works_inside_an_active_event_loop():
    class AnswerClient:
        async def generate(self, **kwargs):
            return ModelResponse(text="working\nFINAL: A")

    registry = ModelRegistry([ModelSpec(model_idx="solver", client=AnswerClient())])

    async def reward_from_loop():
        return compute_reward(
            [VALID_COMPLETION],
            ground_truth=["A"],
            model_registry=registry,
            execute_workflows=True,
        )

    assert asyncio.run(reward_from_loop()) == [1.0]


def test_reward_traces_include_the_executed_vllm_workflow_result():
    class AnswerClient:
        async def generate(self, **kwargs):
            return ModelResponse(text="FINAL: A")

    registry = ModelRegistry([ModelSpec(model_idx="solver", provider="vllm", client=AnswerClient())])

    [trace] = compute_reward_traces(
        [VALID_COMPLETION],
        ground_truth=["A"],
        model_registry=registry,
        execute_workflows=True,
    )

    assert trace.reward == 1.0
    assert trace.task is not None
    assert trace.run_result is not None


def test_build_grpo_trainer_keeps_trace_observer_out_of_trl_kwargs(monkeypatch):
    captured: dict = {}

    class StubTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("theo_conductor.grpo.GRPOTrainer", StubTrainer)
    traces = []
    trainer = build_grpo_trainer(
        model="unused",
        train_dataset=[],
        args=object(),
        trace_observer=traces.extend,
    )

    assert "trace_observer" not in captured
    rewards = captured["reward_funcs"]([VALID_COMPLETION], ground_truth=["A"])

    assert isinstance(trainer, StubTrainer)
    assert rewards == [1.0]
    assert len(traces) == 1
    assert traces[0].reward == 1.0


class BatchJudgeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(text=response)


def test_judge_reward_traces_sends_the_whole_rollout_batch_once():
    traces = compute_reward_traces(
        [VALID_COMPLETION, VALID_COMPLETION.replace('"final_answer": "A"', '"final_answer": "B"')],
        ground_truth=["A", "A"],
        question=["First?", "Second?"],
        reference_answer=["A reference", "Another reference"],
    )
    client = BatchJudgeClient(
        [
            json.dumps(
                [
                    {"id": "rollout-0", "correct": True, "reason": "Matches."},
                    {"id": "rollout-1", "correct": False, "reason": "Does not match."},
                ]
            )
        ]
    )

    judged = judge_reward_traces(traces, client=client, retry_delay_seconds=0)

    assert len(client.calls) == 1
    request_items = json.loads(client.calls[0]["question"].split("\n", 1)[1])
    assert [item["id"] for item in request_items] == ["rollout-0", "rollout-1"]
    assert [trace.reward for trace in judged] == [1.0, 0.5]
    assert [trace.judge_correct for trace in judged] == [True, False]
    assert all(trace.judge_attempts == 1 for trace in judged)


def test_judge_reward_traces_retries_the_batch_then_succeeds():
    traces = compute_reward_traces([VALID_COMPLETION], ground_truth=["A"])
    client = BatchJudgeClient(
        [
            "not json",
            '[{"id":"rollout-0","correct":true,"reason":"Matches."}]',
        ]
    )

    [judged] = judge_reward_traces(traces, client=client, attempts=2, retry_delay_seconds=0)

    assert len(client.calls) == 2
    assert judged.reward == 1.0
    assert judged.judge_attempts == 2


def test_judge_reward_traces_raises_after_retries_without_heuristic_fallback():
    traces = compute_reward_traces([VALID_COMPLETION], ground_truth=["A"])
    client = BatchJudgeClient([RuntimeError("unavailable"), "still not json"])

    with pytest.raises(JudgeBatchError, match="failed after 2 attempts"):
        judge_reward_traces(traces, client=client, attempts=2, retry_delay_seconds=0)

    assert len(client.calls) == 2


def test_trainer_uses_kimi_verdict_instead_of_local_answer_match(monkeypatch):
    captured = {}

    class StubTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("theo_conductor.grpo.GRPOTrainer", StubTrainer)
    client = BatchJudgeClient(
        ['[{"id":"rollout-0","correct":false,"reason":"Substantively wrong."}]']
    )
    build_grpo_trainer(
        model="unused",
        train_dataset=[],
        args=object(),
        judge_client=client,
        judge_retry_delay_seconds=0,
    )

    rewards = captured["reward_funcs"]([VALID_COMPLETION], ground_truth=["A"])

    assert rewards == [0.5]
    assert len(client.calls) == 1


def test_executed_workflow_trainer_requires_an_explicit_judge_client(monkeypatch):
    monkeypatch.setattr("theo_conductor.grpo.GRPOTrainer", lambda **kwargs: kwargs)

    with pytest.raises(ValueError, match="requires a Kimi judge client"):
        build_grpo_trainer(model="unused", train_dataset=[], execute_workflows=True)
