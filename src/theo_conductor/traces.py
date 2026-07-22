from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .grpo import RewardTrace


class TrainingTraceLogger:
    """Persist conductor plans and worker outputs produced by GRPO rewards.

    A JSON object is appended for every sampled completion. In distributed
    jobs each rank owns a separate file, avoiding locks across processes.
    Rank zero also publishes each reward batch as a W&B table when W&B
    reporting is enabled.
    """

    def __init__(self, output_dir: str | Path, *, log_to_wandb: bool = False) -> None:
        self.rank = int(os.getenv("RANK", "0"))
        trace_dir = Path(output_dir) / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"plans-and-worker-outputs-rank-{self.rank}.jsonl"
        self.log_to_wandb = log_to_wandb and self.rank == 0
        self._batch = 0
        self._lock = threading.Lock()

    def __call__(self, traces: Sequence[RewardTrace]) -> None:
        with self._lock:
            batch = self._batch
            self._batch += 1
            records = [reward_trace_to_dict(trace, batch=batch, index=index, rank=self.rank) for index, trace in enumerate(traces)]
            with self.path.open("a", encoding="utf-8") as trace_file:
                for record in records:
                    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if self.log_to_wandb:
                self._log_wandb(records)

    @staticmethod
    def _log_wandb(records: list[dict[str, Any]]) -> None:
        # Transformers initializes the W&B run before the first reward call.
        # If W&B was disabled or failed to initialize, local JSONL collection
        # remains authoritative and training continues.
        try:
            import wandb

            if wandb.run is None:
                return
            columns = [
                "batch",
                "sample",
                "reward",
                "question",
                "gold_answer",
                "final_answer",
                "plan_json",
                "worker_outputs_json",
                "error",
                "judge_correct",
                "judge_model",
                "judge_reason",
                "judge_attempts",
                "judge_error",
            ]
            rows = [
                [
                    record["batch"],
                    record["sample"],
                    record["reward"],
                    record["question"],
                    record["gold_answer"],
                    record["final_answer"],
                    json.dumps(record["plan"], ensure_ascii=False),
                    json.dumps(record["worker_outputs"], ensure_ascii=False),
                    record["error"],
                    record["judge_correct"],
                    record["judge_model"],
                    record["judge_reason"],
                    record["judge_attempts"],
                    record["judge_error"],
                ]
                for record in records
            ]
            wandb.log({"conductor/plans_and_worker_outputs": wandb.Table(columns=columns, data=rows)})
        except Exception as exc:  # logging must never abort an expensive run
            print(f"Warning: failed to log conductor traces to W&B: {exc}", flush=True)


def reward_trace_to_dict(trace: RewardTrace, *, batch: int, index: int, rank: int) -> dict[str, Any]:
    plan = trace.task.model_dump(mode="json") if trace.task is not None else None
    worker_outputs = (
        {step_id: output.model_dump(mode="json") for step_id, output in trace.run_result.outputs.items()}
        if trace.run_result is not None
        else {}
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "batch": batch,
        "sample": index,
        "reward": trace.reward,
        "question": trace.question,
        "gold_answer": trace.gold_answer,
        "reference_answer": trace.reference_answer,
        "answer_type": trace.answer_type,
        "final_answer": trace.final_answer,
        "judge_correct": trace.judge_correct,
        "judge_model": trace.judge_model,
        "judge_reason": trace.judge_reason,
        "judge_response": trace.judge_response,
        "judge_error": trace.judge_error,
        "judge_attempts": trace.judge_attempts,
        "conductor_completion": trace.completion,
        "plan": plan,
        "worker_outputs": worker_outputs,
        "error": trace.error,
    }
