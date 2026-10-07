"""Detect changed, missing or unrecorded analysis inputs and generated artifacts."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .data_loader import ROOT, verify_raw

MANIFEST = Path("outputs/artifact_manifest.json")
TEXT_SUFFIXES = {".py", ".csv", ".json", ".md", ".txt", ".toml"}


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path leaves the project directory")
    return path


def digest(root, relative):
    path = checked_path(root, relative)
    content = path.read_bytes()
    # Git changes text newlines across Windows/Linux; original raw CSVs stay byte-exact.
    raw_csv = relative.startswith("data/raw/") and path.suffix == ".csv"
    if path.suffix in TEXT_SUFFIXES and not raw_csv:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def inventory(root):
    inputs = sorted(
        {
            p.relative_to(root).as_posix()
            for pattern in [
                "src/*.py",
                "data/raw/*.csv",
                "data/raw/manifest.json",
                "requirements.txt",
                "pyproject.toml",
            ]
            for p in root.glob(pattern)
            if p.is_file()
        }
    )
    artifacts = sorted(
        {
            p.relative_to(root).as_posix()
            for pattern in [
                "outputs/tables/*.csv",
                "outputs/tables/*.json",
                "outputs/charts/*.png",
                "data/processed/fights.csv",
                "RESEARCH.md",
                "FINDINGS.md",
            ]
            for p in root.glob(pattern)
            if p.is_file()
        }
    )
    return {"inputs": inputs, "artifacts": artifacts}


def write_manifest(root=ROOT):
    """Called only after a successful full pipeline, not after isolated refreshes."""
    root = Path(root)
    verify_raw(root / "data/raw")
    files = inventory(root)
    packages = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            package = line.split("==", 1)[0]
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not installed"
    manifest = {
        "schema_version": 1,
        "hash_policy": "SHA-256; normalize CRLF for text except byte-exact raw CSVs",
        "python": platform.python_version(),
        "packages": packages,
        **{
            kind: {name: digest(root, name) for name in names}
            for kind, names in files.items()
        },
    }
    destination = root / MANIFEST
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def verify_manifest(root=ROOT):
    root = Path(root)
    path = root / MANIFEST
    if not path.exists():
        return {
            "status": "missing",
            "checked": 0,
            "issues": ["No full-pipeline artifact manifest exists."],
        }
    issues, checked = [], 0
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported manifest schema")
        current = inventory(root)
        for kind in ["inputs", "artifacts"]:
            recorded = manifest[kind]
            for name, expected in recorded.items():
                target = checked_path(root, name)
                if not target.is_file():
                    issues.append(f"Missing {kind}: {name}")
                elif digest(root, name) != expected:
                    issues.append(f"Changed {kind}: {name}")
                checked += 1
            for name in sorted(set(current[kind]) - set(recorded)):
                issues.append(f"Unrecorded {kind}: {name}")
        verify_raw(root / "data/raw")
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as error:
        issues.append(f"Verification failed: {error}")
    return {
        "status": "stale" if issues else "verified",
        "checked": checked,
        "issues": issues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = verify_manifest()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "verified" else 1)
