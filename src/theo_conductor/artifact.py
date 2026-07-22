from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
import hashlib
import json
import os
import re
import shutil
import tempfile


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Artifact:
    """A durable file published by a worker or tool during a run."""

    artifact_id: str
    path: Path
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum_sha256: str = ""
    size_bytes: int = 0

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        path = self.path
        if relative_to is not None:
            path = path.relative_to(relative_to)
        return {
            "artifact_id": self.artifact_id,
            "path": str(path),
            "kind": self.kind,
            "metadata": self.metadata,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
        }


class ArtifactStore:
    """A local, append-only artifact repository scoped to one run.

    Files and their JSON manifest are written atomically. Artifact IDs are
    immutable: publishing the same ID twice is rejected instead of silently
    changing an input that another worker may already have consumed.
    """

    def __init__(self, run_id: str | int, root: str | Path = "runs") -> None:
        run_name = str(run_id)
        if not _SAFE_ID.fullmatch(run_name):
            raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-'")

        self.run_id = run_name
        self.root = Path(root).resolve() / run_name / "artifacts"
        self.files_dir = self.root / "files"
        self.manifest_path = self.root / "manifest.json"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._artifacts = self._load_manifest()

    def publish(
        self,
        source: str | Path,
        *,
        artifact_id: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Copy a file into the store and atomically add it to the manifest."""
        self._validate_artifact_id(artifact_id)
        if not kind.strip():
            raise ValueError("kind must be a non-empty string")

        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact source is not a file: {source_path}")

        artifact_metadata = dict(metadata or {})
        # Fail before copying if the manifest could not represent this value.
        json.dumps(artifact_metadata)

        with self._lock:
            if artifact_id in self._artifacts:
                raise ValueError(f"Artifact {artifact_id!r} already exists")

            suffix = "".join(source_path.suffixes)
            destination = self.files_dir / f"{artifact_id}{suffix}"
            if destination.exists():
                raise ValueError(f"Artifact destination already exists: {destination}")

            temporary_path: Path | None = None
            try:
                with source_path.open("rb") as source_handle:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=self.files_dir, prefix=f".{artifact_id}-", delete=False
                    ) as temporary_handle:
                        temporary_path = Path(temporary_handle.name)
                        shutil.copyfileobj(source_handle, temporary_handle)
                        temporary_handle.flush()
                        os.fsync(temporary_handle.fileno())

                checksum = self._sha256(temporary_path)
                size_bytes = temporary_path.stat().st_size
                temporary_path.replace(destination)
                temporary_path = None

                artifact = Artifact(
                    artifact_id=artifact_id,
                    path=destination.resolve(),
                    kind=kind,
                    metadata=artifact_metadata,
                    checksum_sha256=checksum,
                    size_bytes=size_bytes,
                )
                self._artifacts[artifact_id] = artifact
                try:
                    self._write_manifest()
                except Exception:
                    self._artifacts.pop(artifact_id, None)
                    destination.unlink(missing_ok=True)
                    raise
                return artifact
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def get(self, artifact_id: str) -> Artifact:
        with self._lock:
            try:
                return self._artifacts[artifact_id]
            except KeyError as exc:
                raise KeyError(f"Artifact {artifact_id!r} was not found") from exc

    def list(self) -> list[Artifact]:
        with self._lock:
            return list(self._artifacts.values())

    def _load_manifest(self) -> dict[str, Artifact]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Invalid artifact manifest: {self.manifest_path}")

        artifacts: dict[str, Artifact] = {}
        for item in payload:
            artifact = Artifact(
                artifact_id=item["artifact_id"],
                path=(self.root / item["path"]).resolve(),
                kind=item["kind"],
                metadata=item.get("metadata", {}),
                checksum_sha256=item["checksum_sha256"],
                size_bytes=item["size_bytes"],
            )
            if artifact.artifact_id in artifacts:
                raise ValueError(f"Duplicate artifact ID in manifest: {artifact.artifact_id!r}")
            if not artifact.path.is_relative_to(self.root):
                raise ValueError(f"Artifact path escapes store root: {artifact.path}")
            artifacts[artifact.artifact_id] = artifact
        return artifacts

    def _write_manifest(self) -> None:
        payload = [artifact.to_dict(relative_to=self.root) for artifact in self._artifacts.values()]
        fd, raw_path = tempfile.mkstemp(dir=self.root, prefix=".manifest-", suffix=".json")
        temporary_path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(self.manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_artifact_id(artifact_id: str) -> None:
        if not _SAFE_ID.fullmatch(artifact_id):
            raise ValueError("artifact_id must contain only letters, numbers, '.', '_' or '-'")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
