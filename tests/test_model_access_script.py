import json

import pytest

from test_model_access import Worker, completion_parts, load_workers


def test_load_workers_uses_client_endpoints_and_models_from_yaml(tmp_path):
    config = tmp_path / "workers.yaml"
    config.write_text(
        """
models:
  - model_idx: worker-a
    display_name: Worker A
    client:
      base_url: http://model-a.example/v1/
      model: served-a
      api_key: secret-a
  - name: worker-b
    client:
      base_url: http://model-b.example/v1
      model: served-b
""",
        encoding="utf-8",
    )

    assert load_workers(config) == [
        Worker("worker-a", "Worker A", "http://model-a.example/v1", "served-a", "secret-a"),
        Worker("worker-b", "worker-b", "http://model-b.example/v1", "served-b", "EMPTY"),
    ]


def test_large_pool_defines_the_three_expected_workers():
    workers = load_workers("configs/worker_pool_large.yaml")

    assert [worker.worker_id for worker in workers] == [
        "kimi-k2.6",
        "glm-5.2-fp8",
        "deepseek-v4-flash",
    ]
    assert all(worker.base_url and worker.model for worker in workers)


def test_completion_parts_reads_reasoning_and_answer():
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"reasoning_content": "thinking", "content": "MODEL_ACCESS_OK"},
            }
        ]
    }

    assert completion_parts(payload) == ("thinking", "MODEL_ACCESS_OK", "stop")


def test_completion_parts_rejects_missing_choice():
    with pytest.raises(ValueError, match="did not contain a choice"):
        completion_parts(json.loads("{}"))
