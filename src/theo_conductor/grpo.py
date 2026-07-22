from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from theo_conductor.benchmark import JUDGE_INSTRUCTION, build_judge_batch_question, parse_judge_batch
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.runner import Runner
from theo_conductor.schema import RunResult, Task
from theo_conductor.validate import validate_task


MALFORMED_REWARD = 0.0
INVALID_WORKFLOW_REWARD = 0.2
VALID_WORKFLOW_REWARD = 0.5
CORRECT_REWARD = 1.0

# This small, explicit protocol lets reward/evaluation code pull an answer out
# of a verbose worker response without guessing which sentence is the answer.
FINAL_ANSWER_MARKER = "FINAL:"
_FINAL_ANSWER_MARKER_RE = re.compile(r"(?im)^\s*final\s*(?:answer\s*)?:\s*(.+?)\s*$")
_NUMBER_RE = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")
_SAFE_SYMBOLIC_RE = re.compile(r"^[0-9A-Za-z_+\-*/^().,{}\\\s]+$")


class ConductorParseError(ValueError):
    """Raised when a completion does not contain a usable conductor workflow."""


@dataclass(frozen=True)
class RewardTrace:
    completion: str
    reward: float
    question: str | None = None
    gold_answer: str | None = None
    reference_answer: str | None = None
    answer_type: str | None = None
    task: Task | None = None
    run_result: RunResult | None = None
    final_answer: str | None = None
    error: str | None = None
    judge_correct: bool | None = None
    judge_model: str | None = None
    judge_reason: str | None = None
    judge_response: str | None = None
    judge_error: str | None = None
    judge_attempts: int = 0


class JudgeBatchError(RuntimeError):
    """Raised when a rollout batch cannot be judged after all attempts."""

    def __init__(self, attempts: int, cause: BaseException):
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"Kimi judge batch failed after {attempts} attempt{'s' if attempts != 1 else ''}: "
            f"{type(cause).__name__}: {cause}"
        )


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

    Reward calculation is structural by default and never calls worker models.
    Set ``execute_workflows=True`` (and supply ``model_registry`` or ``runner``)
    only for a small, explicit evaluation run. Valid workflows receive the
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
    reference_answers = _as_list(kwargs.get("reference_answer"), len(completions))
    answer_types = _as_list(kwargs.get("answer_type"), len(completions))
    execute_workflows = bool(kwargs.get("execute_workflows", False))
    use_heuristic_answer_matching = bool(kwargs.get("use_heuristic_answer_matching", True))

    traces: list[RewardTrace] = []

    for index, completion in enumerate(completions):
        completion_text = _completion_to_text(completion)
        question = questions[index]
        gold_answer = gold_answers[index]
        reference_answer = reference_answers[index]
        answer_type = answer_types[index]

        try:
            raw_payload = _load_conductor_payload(completion_text)
        except ConductorParseError as exc:
            traces.append(
                RewardTrace(
                    completion=completion_text,
                    reward=MALFORMED_REWARD,
                    question=question,
                    gold_answer=gold_answer,
                    reference_answer=reference_answer,
                    answer_type=answer_type,
                    error=str(exc),
                )
            )
            continue

        try:
            task = parse_conductor_json(raw_payload, question=question, model_registry=model_registry)
        except ConductorParseError as exc:
            traces.append(
                RewardTrace(
                    completion=completion_text,
                    reward=INVALID_WORKFLOW_REWARD,
                    question=question,
                    gold_answer=gold_answer,
                    reference_answer=reference_answer,
                    answer_type=answer_type,
                    error=str(exc),
                )
            )
            continue

        run_result: RunResult | None = None
        final_answer = _extract_embedded_final_answer(raw_payload)

        if execute_workflows and (runner is not None or model_registry is not None):
            try:
                run_result = _run_runner_sync(runner or Runner(model_registry), task)  # type: ignore[arg-type]
                final_answer = _extract_final_answer(run_result)
            except Exception as exc:
                traces.append(
                    RewardTrace(
                        completion=completion_text,
                        reward=INVALID_WORKFLOW_REWARD,
                        question=question,
                        gold_answer=gold_answer,
                        reference_answer=reference_answer,
                        answer_type=answer_type,
                        task=task,
                        error=str(exc),
                    )
                )
                continue

        reward = (
            CORRECT_REWARD
            if use_heuristic_answer_matching
            and gold_answer is not None
            and final_answer is not None
            and answers_match(final_answer, gold_answer, answer_type=answer_type)
            else VALID_WORKFLOW_REWARD
        )

        traces.append(
            RewardTrace(
                completion=completion_text,
                reward=reward,
                question=question,
                gold_answer=gold_answer,
                reference_answer=reference_answer,
                answer_type=answer_type,
                task=task,
                run_result=run_result,
                final_answer=final_answer,
            )
        )

    return traces


def judge_reward_traces(
    traces: Sequence[RewardTrace],
    *,
    client: Any,
    max_tokens: int = 8192,
    attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> list[RewardTrace]:
    """Judge all valid rollouts in one strict Kimi request.

    Structural and execution failures retain their 0.0/0.2 rewards and are not
    answer-judged. Every otherwise valid rollout is included in one request.
    A malformed response or request error retries the entire batch; exhausting
    retries raises ``JudgeBatchError`` instead of substituting a heuristic.
    """
    if max_tokens <= 0:
        raise ValueError("judge max tokens must be positive")
    if attempts <= 0:
        raise ValueError("judge attempts must be positive")
    if retry_delay_seconds < 0:
        raise ValueError("judge retry delay must be non-negative")

    judgeable = [
        (index, trace)
        for index, trace in enumerate(traces)
        if trace.task is not None and trace.error is None
    ]
    if not judgeable:
        return list(traces)

    records: list[tuple[str, dict[str, Any]]] = []
    for index, trace in judgeable:
        response_text = _final_worker_response(trace)
        records.append(
            (
                f"rollout-{index}",
                {
                    "question": trace.question,
                    "reference_answer": trace.reference_answer,
                    "gold_answer": trace.gold_answer,
                    "response": response_text,
                    "extracted_answer": trace.final_answer,
                },
            )
        )

    expected_ids = [item_id for item_id, _ in records]

    async def request_verdicts() -> tuple[dict[str, tuple[bool, str]], int]:
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = await client.generate(
                    instruction=JUDGE_INSTRUCTION,
                    question=build_judge_batch_question(records),
                    context={},
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return parse_judge_batch(response.text, expected_ids), attempt
            except Exception as exc:
                last_error = exc
                if attempt < attempts and retry_delay_seconds:
                    await asyncio.sleep(retry_delay_seconds * (2 ** (attempt - 1)))
        assert last_error is not None
        raise JudgeBatchError(attempts, last_error) from last_error

    verdicts, attempts_used = _run_async_sync(request_verdicts())
    judged = list(traces)
    for (index, trace), (item_id, _) in zip(judgeable, records, strict=True):
        correct, reason = verdicts[item_id]
        judged[index] = replace(
            trace,
            reward=CORRECT_REWARD if correct else VALID_WORKFLOW_REWARD,
            judge_correct=correct,
            judge_model=getattr(client, "model", None),
            judge_reason=reason,
            judge_response=json.dumps(
                {"id": item_id, "correct": correct, "reason": reason},
                ensure_ascii=False,
            ),
            judge_attempts=attempts_used,
        )
    return judged


def build_grpo_trainer(
    *,
    model: str | Any,
    train_dataset: Any,
    processing_class: Any | None = None,
    args: GRPOConfig | None = None,
    model_registry: ModelRegistry | None = None,
    runner: Runner | None = None,
    execute_workflows: bool = False,
    judge_client: Any | None = None,
    judge_max_tokens: int = 8192,
    judge_attempts: int = 3,
    judge_retry_delay_seconds: float = 1.0,
    reward_kwargs: dict[str, Any] | None = None,
    trace_observer: Callable[[Sequence[RewardTrace]], Any] | None = None,
    **trainer_kwargs: Any,
) -> GRPOTrainer:
    """Create a ``GRPOTrainer`` configured for conductor reward training."""

    if execute_workflows and judge_client is None:
        raise ValueError("Executed-workflow training requires a Kimi judge client.")
    if judge_client is not None:
        if judge_max_tokens <= 0:
            raise ValueError("judge max tokens must be positive")
        if judge_attempts <= 0:
            raise ValueError("judge attempts must be positive")
        if judge_retry_delay_seconds < 0:
            raise ValueError("judge retry delay must be non-negative")

    reward_kwargs = {"execute_workflows": execute_workflows, **(reward_kwargs or {})}
    if judge_client is not None:
        # Kimi is the sole source of semantic correctness during training.
        # The local matcher remains available to standalone probes/tests only.
        reward_kwargs["use_heuristic_answer_matching"] = False
    # Keep the registry for structural model-id validation, but do not give the
    # reward permission to make network worker calls unless explicitly asked.
    if model_registry is not None:
        reward_kwargs.setdefault("model_registry", model_registry)
    if runner is not None:
        reward_kwargs.setdefault("runner", runner)

    def reward_func(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        traces = compute_reward_traces(completions, **reward_kwargs, **kwargs)
        if judge_client is not None:
            try:
                traces = judge_reward_traces(
                    traces,
                    client=judge_client,
                    max_tokens=judge_max_tokens,
                    attempts=judge_attempts,
                    retry_delay_seconds=judge_retry_delay_seconds,
                )
            except JudgeBatchError as exc:
                if trace_observer is not None:
                    trace_observer(
                        [
                            replace(
                                trace,
                                judge_model=getattr(judge_client, "model", None),
                                judge_error=str(exc),
                                judge_attempts=exc.attempts,
                            )
                            if trace.task is not None and trace.error is None
                            else trace
                            for trace in traces
                        ]
                    )
                raise
        if trace_observer is not None:
            trace_observer(traces)
        return [trace.reward for trace in traces]

    return GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args or GRPOConfig(),
        train_dataset=train_dataset,
        processing_class=processing_class,
        **trainer_kwargs,
    )


def answers_match(
    predicted: str,
    gold: str,
    *,
    answer_type: str | None = None,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-4,
) -> bool:
    if answer_type == "multipleChoice":
        return _extract_choice(predicted) == _extract_choice(gold)

    if _normalize_answer(predicted) == _normalize_answer(gold):
        return True

    predicted_number = _extract_single_number(predicted)
    gold_number = _extract_single_number(gold)
    if predicted_number is not None and gold_number is not None:
        return abs(predicted_number - gold_number) <= absolute_tolerance + relative_tolerance * max(
            abs(predicted_number), abs(gold_number)
        )

    return _symbolically_equivalent(predicted, gold)


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
    if not candidates:
        raise ConductorParseError("Completion does not contain valid JSON.")
    try:
        return json.loads(candidates[0])
    except json.JSONDecodeError:
        pass

    parsed_candidates: list[Any] = []
    for candidate in candidates[1:]:
        try:
            parsed_candidates.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    # A completion can contain more than one JSON fragment (for example, an
    # empty example object followed by the actual response).  The broad
    # first-"{"/last-"}" candidate above cannot represent that situation, so
    # scan for independently decodable conductor-shaped values.  Do not accept
    # arbitrary embedded JSON: prose often contains incidental access lists.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            payload, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if _looks_like_conductor_payload(payload):
            parsed_candidates.append(payload)

    # Prefer an enclosing task object over a nested workflow array. This is
    # important when prose or another JSON fragment makes the broad object
    # candidate invalid but its inner workflow array remains valid JSON.
    for payload in parsed_candidates:
        if isinstance(payload, dict) and "workflow" in payload:
            return payload
    for payload in parsed_candidates:
        if _looks_like_conductor_payload(payload):
            return payload
    if parsed_candidates:
        return parsed_candidates[0]
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


def _looks_like_conductor_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        return "workflow" in payload
    return (
        isinstance(payload, list)
        and bool(payload)
        and all(isinstance(step, dict) and "step_id" in step for step in payload)
    )


def _extract_embedded_final_answer(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("final_answer", "answer", "final"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _extract_final_answer(run_result: RunResult) -> str | None:
    final_step_id = run_result.task.workflow[-1].step_id
    final = run_result.outputs.get(final_step_id)
    if final is None or not final.text.strip():
        return None
    matches = _FINAL_ANSWER_MARKER_RE.findall(final.text)
    return matches[-1].strip() if matches else None


def _final_worker_response(trace: RewardTrace) -> str | None:
    if trace.run_result is None or trace.task is None:
        return trace.final_answer
    final_step_id = trace.task.workflow[-1].step_id
    output = trace.run_result.outputs.get(final_step_id)
    return output.text if output is not None and output.text.strip() else trace.final_answer


def _extract_single_number(answer: str) -> float | None:
    """Extract one finite decimal/scientific value from a short answer."""

    text = _strip_answer_wrapper(answer)
    fraction = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*", text)
    if fraction:
        numerator, denominator = map(float, fraction.groups())
        if denominator:
            return numerator / denominator
    # Common scientific notation emitted by models, e.g. 6.02 × 10^23.
    text = re.sub(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:×|x|\\\\times)\s*10\s*(?:\^|\*\*)\s*([+-]?\d+)",
        r"\1e\2",
        text,
        flags=re.IGNORECASE,
    )
    numbers = _NUMBER_RE.findall(text.replace(",", ""))
    if len(numbers) != 1:
        return None
    try:
        value = float(numbers[0])
    except ValueError:
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _symbolically_equivalent(predicted: str, gold: str) -> bool:
    """Compare safe elementary expressions with SymPy when it is available."""

    try:
        from sympy import E, pi, simplify, sympify
    except ImportError:
        return False

    predicted_expression = _prepare_symbolic_expression(predicted)
    gold_expression = _prepare_symbolic_expression(gold)
    if predicted_expression is None or gold_expression is None:
        return False

    try:
        locals_map = {"pi": pi, "e": E}
        predicted_value = sympify(predicted_expression, locals=locals_map)
        gold_value = sympify(gold_expression, locals=locals_map)
        return bool(simplify(predicted_value - gold_value) == 0)
    except (ArithmeticError, NameError, SyntaxError, TypeError, ValueError):
        return False


def _prepare_symbolic_expression(answer: str) -> str | None:
    expression = _strip_answer_wrapper(answer)
    expression = expression.strip("$ ")
    expression = expression.replace("\\left", "").replace("\\right", "")
    expression = expression.replace("\\pi", "pi").replace("\\cdot", "*").replace("×", "*")
    expression = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expression)
    expression = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expression)
    expression = expression.replace("{", "(").replace("}", ")").replace("^", "**")
    if not expression or not _SAFE_SYMBOLIC_RE.fullmatch(expression) or "__" in expression:
        return None
    return expression


def _strip_answer_wrapper(answer: str) -> str:
    text = answer.strip()
    marker_matches = _FINAL_ANSWER_MARKER_RE.findall(text)
    if marker_matches:
        text = marker_matches[-1].strip()
    return re.sub(r"^\s*(?:final\s+answer|answer)\s*[:=\-]\s*", "", text, flags=re.IGNORECASE)


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
    result = _run_async_sync(runner.run(task))
    if result is None:
        raise RuntimeError("Runner completed without a result.")
    return result


def _run_async_sync(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # TRL can invoke a reward from notebook/server contexts that already own an
    # event loop. Run the optional evaluation in a separate thread in that case.
    result: Any = None
    error: BaseException | None = None

    def run_in_thread() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coroutine)
        except BaseException as exc:  # Re-raise the worker failure in the caller.
            error = exc

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result
