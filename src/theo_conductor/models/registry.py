from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml

from theo_conductor.schema import ModelSpec, ModelClient
from .openai_compat import OpenAICompatibleClient


class ModelRegistry:
    def __init__(self, specs: List[ModelSpec]):
        self._models = {self._key_for(spec): spec for spec in specs}

        if len(self._models) != len(specs):
            raise ValueError("Duplicate model_id found in specs")

    def get(self, model_idx: int | str) -> ModelSpec:
        try:
            return self._models[model_idx]
        except KeyError:
            valid_models = ", ".join(str(key) for key in self._models.keys())
            raise ValueError(f"Model '{model_idx}' not found. Valid models: {valid_models}")

    def client(self, model_id: str) -> ModelClient:
        return self.get(model_id).client

    def get_models(self) -> list[str]:
        return [spec.display_name for spec in self._models.values()]

    def model_ids(self) -> list[int | str]:
        """Return the stable IDs accepted by conductor workflow steps."""
        return list(self._models)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ModelRegistry:
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}

        specs = data.get("models", data) if isinstance(data, dict) else data
        if not isinstance(specs, list):
            raise ValueError(f"{path} must contain a list of models or a 'models' list")

        return cls([cls._spec_from_dict(spec, source=path) for spec in specs])

    @classmethod
    def from_config_dir(cls, path: str | Path = "configs") -> ModelRegistry:
        """Load all YAML model specs in a directory (backward-compatible API)."""
        path = Path(path)
        yaml_files = sorted([*path.glob("*.yaml"), *path.glob("*.yml")])
        if not yaml_files:
            raise ValueError(f"No YAML model config files found in {path}")

        specs: list[ModelSpec] = []
        for yaml_file in yaml_files:
            with yaml_file.open() as f:
                data = yaml.safe_load(f) or {}
            file_specs = data.get("models", data) if isinstance(data, dict) else data
            if not isinstance(file_specs, list):
                raise ValueError(f"{yaml_file} must contain a list of models or a 'models' list")
            specs.extend(cls._spec_from_dict(spec, source=yaml_file) for spec in file_specs)
        return cls(specs)


    @staticmethod
    def _key_for(spec: ModelSpec) -> int | str:
        if spec.model_idx is not None:
            return spec.model_idx
        if spec.name is not None:
            return spec.name
        if spec.display_name is not None:
            return spec.display_name
        raise ValueError("ModelSpec must define model_idx, name, or display_name")

    @staticmethod
    def _spec_from_dict(data: dict[str, Any], *, source: Path) -> ModelSpec:
        if not isinstance(data, dict):
            raise ValueError(f"Model entries in {source} must be mappings")

        raw = dict(data)
        client_config = raw.pop("client", None)
        if client_config is None:
            raise ValueError(f"Model entry in {source} is missing required 'client' config")

        if "tags" in raw:
            raw["tags"] = set(raw["tags"])

        return ModelSpec(client=ModelRegistry._client_from_dict(client_config, source=source), **raw)

    @staticmethod
    def _client_from_dict(data: dict[str, Any], *, source: Path) -> ModelClient:
        if not isinstance(data, dict):
            raise ValueError(f"Client config in {source} must be a mapping")

        raw = dict(data)
        client_type = raw.pop("type", "openai_compatible")

        if client_type in {"openai_compatible", "openai-compatible", "openai"}:
            return OpenAICompatibleClient(**raw)

        raise ValueError(f"Unsupported client type {client_type!r} in {source}")
