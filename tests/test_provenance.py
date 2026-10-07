import hashlib
import json

import pytest

from src.provenance import MANIFEST, verify_manifest, write_manifest


@pytest.fixture
def project(tmp_path):
    for directory in ["src", "data/raw", "outputs/tables"]:
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "src/example.py").write_bytes(b"x = 1\n")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    raw = b"value\r\n1\r\n"
    (tmp_path / "data/raw/example.csv").write_bytes(raw)
    (tmp_path / "data/raw/manifest.json").write_text(
        json.dumps(
            [{"file": "example.csv", "sha256": hashlib.sha256(raw).hexdigest()}]
        ),
        encoding="utf-8",
    )
    (tmp_path / "outputs/tables/result.csv").write_text("value\n2\n", encoding="utf-8")
    write_manifest(tmp_path)
    return tmp_path


def test_manifest_detects_changed_missing_and_added_files(project):
    assert verify_manifest(project)["status"] == "verified"
    (project / "src/example.py").write_text("x = 2\n", encoding="utf-8")
    (project / "outputs/tables/result.csv").unlink()
    (project / "outputs/tables/new.csv").write_text("value\n3\n", encoding="utf-8")
    result = verify_manifest(project)
    assert result["status"] == "stale"
    assert len(result["issues"]) == 3


def test_text_newlines_are_portable_but_raw_bytes_are_protected(project):
    (project / "src/example.py").write_bytes(b"x = 1\r\n")
    assert verify_manifest(project)["status"] == "verified"
    (project / "data/raw/example.csv").write_bytes(b"value\n1\n")
    assert verify_manifest(project)["status"] == "stale"


def test_manifest_rejects_external_paths_and_invalid_json(project):
    path = project / MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inputs"]["../outside.txt"] = "abc"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "leaves the project" in str(verify_manifest(project)["issues"])
    path.write_text("broken", encoding="utf-8")
    assert verify_manifest(project)["status"] == "stale"
