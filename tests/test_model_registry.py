import pytest

from theo_conductor.models.openai_compat import OpenAICompatibleClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelSpec


def test_model_registry_loads_models_from_yaml_file(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  - model_idx: solver
    provider: vllm
    display_name: Solver
    context_length: 4096
    output_budget_observed_tokens: 3000
    max_output_tokens: 4096
    supports_tools: true
    supports_json: true
    tags:
      - local
      - physics
    client:
      type: openai_compatible
      base_url: http://localhost:8001/v1
      model: solver-model
""",
    )

    registry = ModelRegistry.from_yaml_file(config_file)
    spec = registry.get("solver")

    assert spec.display_name == "Solver"
    assert spec.context_length == 4096
    assert spec.output_budget_observed_tokens == 3000
    assert spec.max_output_tokens == 4096
    assert spec.supports_tools is True
    assert spec.tags == {"local", "physics"}
    assert isinstance(spec.client, OpenAICompatibleClient)
    assert spec.client.model == "solver-model"


def test_model_registry_loads_pricing_from_yaml_file(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  - model_idx: priced
    cost_per_1m_input_tokens: 1.25
    cost_per_1m_output_tokens: 5.5
    client:
      base_url: https://example.test/v1
      model: priced-model
""",
    )

    spec = ModelRegistry.from_yaml_file(config_file).get("priced")

    assert spec.cost_per_1m_input_tokens == 1.25
    assert spec.cost_per_1m_output_tokens == 5.5


def test_model_spec_rejects_negative_pricing():
    with pytest.raises(ValueError, match="must be non-negative"):
        ModelSpec(
            model_idx="invalid",
            client=object(),
            cost_per_1m_input_tokens=-1,
        )


def test_model_spec_rejects_nonpositive_output_budget():
    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        ModelSpec(model_idx="invalid", client=object(), max_output_tokens=0)


def test_frontier_config_contains_requested_openrouter_models_and_pricing():
    registry = ModelRegistry.from_yaml_file("configs/worker_pool_frontier.yaml")

    assert registry.conductor_model == "Qwen/Qwen3.8-27B"
    assert {"glm-5.3", "kimi-k3", "deepseek-v4-pro", "grok-4.6"}.issubset(
        registry.model_ids()
    )
    assert all(
        registry.get(model_id).cost_per_1m_input_tokens is not None
        and registry.get(model_id).cost_per_1m_output_tokens is not None
        for model_id in registry.model_ids()
    )
    assert {
        model_id: registry.get(model_id).max_output_tokens
        for model_id in registry.model_ids()
    } == {
        "gpt-5.5": 16_384,
        "claude-opus-4.8": 40_960,
        "gemini-3.7-flash": 12_288,
        "glm-5.3": 65_536,
        "kimi-k3": 32_768,
        "deepseek-v4-pro": 40_960,
        "grok-4.6": 24_576,
    }
    assert all(
        registry.get(model_id).role
        and registry.get(model_id).best_for
        and registry.get(model_id).useful_for
        and registry.get(model_id).routing_note
        for model_id in registry.model_ids()
    )


def test_model_registry_accepts_routing_node_alias(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  - model_idx: solver
    routing_node: Prefer for final synthesis.
    client:
      base_url: http://localhost:8001/v1
      model: solver-model
""",
    )

    spec = ModelRegistry.from_yaml_file(config_file).get("solver")

    assert spec.routing_note == "Prefer for final synthesis."


def test_model_registry_loads_conductor_model_from_yaml_file(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
conductor_model: Qwen/Qwen3.5-27B
models:
  - model_idx: solver
    client:
      base_url: http://localhost:8001/v1
      model: solver-model
""",
    )

    registry = ModelRegistry.from_yaml_file(config_file)

    assert registry.conductor_model == "Qwen/Qwen3.5-27B"


def test_model_registry_ignores_launcher_deployment_metadata(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  - model_idx: worker
    client:
      base_url: http://localhost:8001/v1
      model: served-model
    deployment:
      mode: local
      source_model: source/model
      gpu_set: "0,1"
      tensor_parallel_size: 2
      max_model_len: 32768
""",
    )

    registry = ModelRegistry.from_yaml_file(config_file)

    assert registry.model_ids() == ["worker"]
    assert registry.get("worker").client.model == "served-model"


def test_model_registry_loads_all_yaml_files_from_config_dir(tmp_path):
    (tmp_path / "first.yaml").write_text(
        """
models:
  - model_idx: solver
    client:
      base_url: http://localhost:8001/v1
      model: solver-model
""",
    )
    (tmp_path / "second.yml").write_text(
        """
models:
  - model_idx: final
    client:
      type: openai
      base_url: http://localhost:8002/v1
      model: final-model
""",
    )

    registry = ModelRegistry.from_config_dir(tmp_path)

    assert registry.get("solver").client.model == "solver-model"
    assert registry.get("final").client.model == "final-model"
