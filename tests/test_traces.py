import json
import sys
from types import SimpleNamespace

from theo_conductor.grpo import RewardTrace
from theo_conductor.schema import Difficulty, RunResult, Step, StepOutput, Task
from theo_conductor.traces import TrainingTraceLogger, reward_trace_to_dict


def _trace() -> RewardTrace:
    task = Task(
        task_type="physics",
        difficulty=Difficulty.EASY,
        question="What is 2 + 2?",
        workflow=[Step(step_id="final", model_id="solver", instruction="Solve", access_list=["question"])],
    )
    result = RunResult(
        task=task,
        outputs={"final": StepOutput(step_id="final", model_id="solver", text="FINAL: 4")},
    )
    return RewardTrace(
        completion='{"workflow": []}',
        reward=1.0,
        question=task.question,
        gold_answer="4",
        answer_type="exactMatch",
        task=task,
        run_result=result,
        final_answer="4",
    )


def test_reward_trace_record_contains_plan_and_worker_outputs():
    record = reward_trace_to_dict(_trace(), batch=3, index=1, rank=0)

    assert record["plan"]["workflow"][0]["step_id"] == "final"
    assert record["worker_outputs"]["final"]["text"] == "FINAL: 4"
    assert record["gold_answer"] == "4"


def test_training_trace_logger_appends_jsonl(tmp_path):
    logger = TrainingTraceLogger(tmp_path)

    logger([_trace()])
    logger([_trace()])

    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [record["batch"] for record in records] == [0, 1]
    assert records[0]["conductor_completion"] == '{"workflow": []}'
    assert records[0]["worker_outputs"]["final"]["model_id"] == "solver"


def test_training_trace_logger_publishes_wandb_table(tmp_path, monkeypatch):
    logged = []
    fake_wandb = SimpleNamespace(
        run=object(),
        Table=lambda *, columns, data: {"columns": columns, "data": data},
        log=logged.append,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = TrainingTraceLogger(tmp_path, log_to_wandb=True)

    logger([_trace()])

    [payload] = logged
    table = payload["conductor/plans_and_worker_outputs"]
    assert "plan_json" in table["columns"]
    assert "FINAL: 4" in table["data"][0][7]
