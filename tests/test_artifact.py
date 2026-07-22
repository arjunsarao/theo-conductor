import hashlib
import json

import pytest

from theo_conductor.artifact import ArtifactStore


def test_publish_copies_file_and_persists_manifest(tmp_path):
    source = tmp_path / "results.csv"
    source.write_text("value\n42\n", encoding="utf-8")
    store = ArtifactStore("run-1", root=tmp_path / "runs")

    artifact = store.publish(
        source,
        artifact_id="simulation-results",
        kind="table",
        metadata={"rows": 1},
    )

    assert artifact.path.read_text(encoding="utf-8") == "value\n42\n"
    assert artifact.path != source
    assert artifact.size_bytes == source.stat().st_size
    assert artifact.checksum_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact.metadata == {"rows": 1}

    reloaded = ArtifactStore("run-1", root=tmp_path / "runs")
    assert reloaded.get("simulation-results") == artifact
    assert json.loads(reloaded.manifest_path.read_text(encoding="utf-8"))[0]["path"].startswith("files/")


def test_publish_rejects_duplicate_and_unsafe_ids(tmp_path):
    source = tmp_path / "result.txt"
    source.write_text("result", encoding="utf-8")
    store = ArtifactStore("run-1", root=tmp_path / "runs")
    store.publish(source, artifact_id="result", kind="text")

    with pytest.raises(ValueError, match="already exists"):
        store.publish(source, artifact_id="result", kind="text")
    with pytest.raises(ValueError, match="artifact_id"):
        store.publish(source, artifact_id="../escape", kind="text")


def test_publish_rejects_missing_source(tmp_path):
    store = ArtifactStore("run-1", root=tmp_path / "runs")

    with pytest.raises(FileNotFoundError):
        store.publish(tmp_path / "missing.csv", artifact_id="missing", kind="table")
