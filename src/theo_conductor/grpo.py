from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from theo_conductor.models.registry import ModelRegistry
from theo_conductor.runner import Runner
from theo_conductor.schema import RunResult, Task
from theo_conductor.validate import validate_task


MALFORMED_REWARD = 0.0
INVALID_WORKFLOW_REWARD = 0.2
VALID_WORKFLOW_REWARD = 0.5
CORRECT_REWARD = 1.0


class ConductorParseError(ValueError):
    """Raised when a completion does not contain a usable conductor workflow."""


@dataclass(frozen=True)
class RewardTrace:
    completion: str
    reward: float
    task: Task | None = None
    run_result: RunResult | None = None
    final_answer: str | None = None
    error: str | None = None


def parse_conductor_json(
    completion: str | dict[str, Any] | list[Any],
    *,
    question: str | None = None,
    model_registry: ModelRegistry | None = None,
) -> Task:
    """Parse a conductor completion into a validated ``Task``.

    The conductor is expected to emit JSON. For practical model outputs this
    also accepts fenced JSON blocks and prose surrounding the JSON object.
    """

    data = _load_conductor_payload(_completion_to_text(completion))

    if isinstance(data, list):
        if question is None:
            raise ConductorParseError("Workflow-list JSON requires a question.")
        data = {
            "task_type": "physics",
            "difficulty": "medium",
            "question": question,
            "workflow": data,
        }
    elif isinstance(data, dict):
        data = dict(data)
        if question is not None:
            data.setdefault("question", question)
    else:
        raise ConductorParseError("Conductor JSON must be an object or workflow list.")

    try:
        task = Task.from_dict(data)
        validate_task(task, model_registry)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ConductorParseError(str(exc)) from exc

    return task


async def run_conductor_runner(
    completion: str | dict[str, Any] | list[Any],
    *,
    question: str | None = None,
    model_registry: ModelRegistry,
    runner: Runner | None = None,
) -> RunResult:
    """Parse a conductor completion and execute it through the worker runner."""

    task = parse_conductor_json(completion, question=question, model_registry=model_registry)
    return await (runner or Runner(model_registry)).run(task)


def compute_reward(
    completions: Sequence[Any],
    ground_truth: Sequence[str] | str | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible reward function for conductor JSON workflows.

    Scoring is intentionally simple and strict:
    - ``0.0`` malformed / no JSON
    - ``0.2`` parseable JSON but invalid workflow, model, or access graph
    - ``0.5`` valid workflow but no correct final answer
    - ``1.0`` final answer matches the gold answer

    Pass ``model_registry`` or ``runner`` in ``kwargs`` to execute workflows
    during reward computation. Without a runner, valid workflows receive the
    structural reward unless the completion itself contains ``final_answer``.
    """

    traces = compute_reward_traces(completions, ground_truth, **kwargs)
    return [trace.reward for trace in traces]


def compute_reward_traces(
    completions: Sequence[Any],
    ground_truth: Sequence[str] | str | None = None,
    **kwargs: Any,
) -> list[RewardTrace]:
    model_registry: ModelRegistry | None = kwargs.get("model_registry")
    runner: Runner | None = kwargs.get("runner")
    question_source = kwargs.get("question") if kwargs.get("question") is not None else kwargs.get("prompts")
    gold_source = ground_truth if ground_truth is not None else kwargs.get("answer", kwargs.get("gold_answer"))
    questions = _as_list(question_source, len(completions))
    gold_answers = _as_list(gold_source, len(completions))
    answer_types = _as_list(kwargs.get("answer_type"), len(completions))

    traces: list[RewardTrace] = []

    for index, completion in enumerate(completions):
        completion_text = _completion_to_text(completion)
        question = questions[index]
        gold_answer = gold_answers[index]
        answer_type = answer_types[index]

        try:
            raw_payload = _load_conductor_payload(completion_text)
        except ConductorParseError as exc:
            traces.append(RewardTrace(completion=completion_text, reward=MALFORMED_REWARD, error=str(exc)))
            continue

        try:
            task = parse_conductor_json(raw_payload, question=question, model_registry=model_registry)
        except ConductorParseError as exc:
            traces.append(RewardTrace(completion=completion_text, reward=INVALID_WORKFLOW_REWARD, error=str(exc)))
            continue

        run_result: RunResult | None = None
        final_answer = _extract_embedded_final_answer(raw_payload)

        if runner is not None or model_registry is not None:
            try:
                run_result = _run_runner_sync(runner or Runner(model_registry), task)  # type: ignore[arg-type]
                final_answer = _extract_final_answer(run_result)
            except Exception as exc:
                traces.append(
                    RewardTrace(
                        completion=completion_text,
                        reward=INVALID_WORKFLOW_REWARD,
                        task=task,
                        error=str(exc),
                    )
                )
                continue

        reward = (
            CORRECT_REWARD
            if gold_answer is not None
            and final_answer is not None
            and answers_match(final_answer, gold_answer, answer_type=answer_type)
            else VALID_WORKFLOW_REWARD
        )

        traces.append(
            RewardTrace(
                completion=completion_text,
                reward=reward,
                task=task,
                run_result=run_result,
                final_answer=final_answer,
            )
        )

    return traces


def build_grpo_trainer(
    *,
    model: str | Any,
    train_dataset: Any,
    processing_class: Any | None = None,
    args: GRPOConfig | None = None,
    model_registry: ModelRegistry | None = None,
    runner: Runner | None = None,
    reward_kwargs: dict[str, Any] | None = None,
    **trainer_kwargs: Any,
) -> GRPOTrainer:
    """Create a ``GRPOTrainer`` configured for conductor reward training."""

    reward_kwargs = {
        "model_registry": model_registry,
        "runner": runner,
        **(reward_kwargs or {}),
    }

    def reward_func(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return compute_reward(completions, **reward_kwargs, **kwargs)

    return GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args or GRPOConfig(),
        train_dataset=train_dataset,
        processing_class=processing_class,
        **trainer_kwargs,
    )


def answers_match(predicted: str, gold: str, *, answer_type: str | None = None) -> bool:
    if answer_type == "multipleChoice":
        return _extract_choice(predicted) == _extract_choice(gold)

    return _normalize_answer(predicted) == _normalize_answer(gold)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content", completion.get("text", completion))
        if content is completion:
            return json.dumps(completion)
        return _completion_to_text(content)

    if isinstance(completion, list):
        if completion and all(isinstance(item, dict) and "content" in item for item in completion):
            return "\n".join(_completion_to_text(item["content"]) for item in completion)
        return "\n".join(_completion_to_text(item) for item in completion)

    return str(completion)


def _load_conductor_payload(text: str) -> Any:
    candidates = _json_candidates(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ConductorParseError("Completion does not contain valid JSON.")


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]

    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])

    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _extract_embedded_final_answer(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("final_answer", "answer", "final"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _extract_final_answer(run_result: RunResult) -> str | None:
    final = run_result.outputs.get("final")
    if final is None or not final.text.strip():
        return None
    return final.text


def _normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"\b(final\s+answer|answer)\s*[:\-]\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .")


def _extract_choice(answer: str | None) -> str | None:
    if answer is None:
        return None

    text = answer.strip().upper()
    match = re.search(r"(?:^|\b)([A-D])(?:[\).:\s]|$)", text)
    return match.group(1) if match else None


def _as_list(value: Any, length: int) -> list[Any]:
    if value is None:
        return [None] * length

    if isinstance(value, str):
        return [value] * length

    if isinstance(value, Sequence):
        values = list(value)
        if len(values) == length:
            return values

    return [value] * length


def _run_runner_sync(runner: Runner, task: Task) -> RunResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(runner.run(task))

    raise RuntimeError("Synchronous reward execution cannot run inside an active event loop.")
