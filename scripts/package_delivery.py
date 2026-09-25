"""Assemble a local, allowlisted delivery with hashes; never uploads anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from auto_reliability.config import ProjectPaths
from auto_reliability.releases import publish_release, serving_paths
from auto_reliability.storage import atomic_json


def package(candidate: Path, destination: Path) -> Path:
    """Copy only explicit product inputs, excluding secrets, caches and planning notes."""
    root = ProjectPaths.discover().root
    candidate, destination = candidate.resolve(), destination.resolve()
    if (root / "output").resolve() not in destination.parents or destination.exists():
        raise ValueError("Delivery must be a new directory inside output/")
    if (root / "output").resolve() not in candidate.parents:
        raise ValueError("Candidate must be inside output/")
    release = publish_release(ProjectPaths(candidate))
    serving_paths(ProjectPaths(candidate))  # Verify immutable package before copying.
    destination.mkdir(parents=True)
    for name in ("app.py", "pyproject.toml", "requirements.lock", "README.md", "PRESENTACION_PROYECTO.md"):
        shutil.copy2(root / name, destination / name)
    for directory in ("src", "tests", "scripts", "docs"):
        shutil.copytree(root / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    if (root / ".streamlit/config.toml").exists():
        (destination / ".streamlit").mkdir()
        shutil.copy2(root / ".streamlit/config.toml", destination / ".streamlit/config.toml")
    for directory in ("data", "artifacts", "reports", "releases"):
        shutil.copytree(candidate / directory, destination / directory,
                        ignore=shutil.ignore_patterns("*.lock"))
    for name in ("candidate_protocol.json", "candidate_comparison.json"):
        shutil.copy2(candidate / name, destination / name)
    files = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(destination.rglob("*")) if path.is_file()}
    atomic_json(destination / "delivery_manifest.json", {
        "release": release, "files": files,
        "scope": "Retrospective MVP, not certified mechanical reliability or full original scientific scope.",
        "limitations": ["No verified all-history zero cases", "Unresolved identities remain excluded",
                        "Baseline not surpassed", "Technical coverage ends in 2017"],
    })
    archive = destination.with_suffix(".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(destination))
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Delivery archive integrity failed")
    print(json.dumps({"release": release, "files": len(files), "zip": str(archive),
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.candidate, args.destination)
