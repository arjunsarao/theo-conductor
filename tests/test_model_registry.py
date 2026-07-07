from theo_conductor.models.openai_compat import OpenAICompatibleClient
from theo_conductor.models.registry import ModelRegistry


def test_model_registry_loads_models_from_yaml_file(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  - model_idx: solver
    provider: vllm
    display_name: Solver
    context_length: 4096
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
    assert spec.supports_tools is True
    assert spec.tags == {"local", "physics"}
    assert isinstance(spec.client, OpenAICompatibleClient)
    assert spec.client.model == "solver-model"


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
