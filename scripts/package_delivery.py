"""Prepara una entrega local con lista explícita y hashes; nunca sube archivos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from auto_reliability.config import ProjectPaths
from auto_reliability.releases import serving_paths
from auto_reliability.storage import atomic_json


def package(candidate: Path, destination: Path) -> Path:
    """Copia una publicación existente, sin activar ni modificar modelos o datos."""
    root = ProjectPaths.discover().root
    candidate, destination = candidate.resolve(), destination.resolve()
    if (root / "output").resolve() not in destination.parents or destination.exists():
        raise ValueError("Delivery must be a new directory inside output/")
    if candidate != root and (root / "output").resolve() not in candidate.parents:
        raise ValueError("El origen debe ser la raíz o una candidata dentro de output/.")
    release = json.loads((candidate / "releases/active.json").read_text(encoding="utf-8"))["release"]
    serving_paths(ProjectPaths(candidate))  # Verificar el paquete inmutable antes de copiarlo.
    destination.mkdir(parents=True)
    for name in ("app.py", "pyproject.toml", "requirements.lock", "README.md", "PRESENTACION_PROYECTO.md", ".gitignore", "ENTREGA.md"):
        shutil.copy2(root / name, destination / name)
    for directory in ("src", "tests", "scripts", "docs"):
        shutil.copytree(root / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    if (root / ".streamlit/config.toml").exists():
        (destination / ".streamlit").mkdir()
        shutil.copy2(root / ".streamlit/config.toml", destination / ".streamlit/config.toml")
    if candidate == root:
        # Solo entradas de servicio y evidencias congeladas; no cachés ni temporales.
        for name in ("data/gold/gold_us_car_reliability.parquet", "data/gold/us_car_inference_catalog.parquet",
                     "data/processed/pipeline_manifest.json", "data/processed/target_normalizer.json",
                     "data/processed/exclusion_review/latest.json", "data/processed/inventory_resolution/latest.json",
                     "artifacts/reliability_model.joblib", "artifacts/model_metrics.json",
                     "artifacts/delivery_freeze.json", "artifacts/test_predictions.csv",
                     "artifacts/validation_predictions.csv", "artifacts/feature_attributions.json",
                     "releases/active.json"):
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination / name)
        shutil.copytree(root / "releases" / release, destination / "releases" / release)
        shutil.copytree(root / "artifacts/frozen_evidence", destination / "artifacts/frozen_evidence")
    else:
        for directory in ("data", "artifacts", "reports", "releases"):
            shutil.copytree(candidate / directory, destination / directory,
                            ignore=shutil.ignore_patterns("*.lock"))
    for name in ("candidate_protocol.json", "candidate_comparison.json"):
        if (candidate / name).exists():
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
    parser.add_argument("--candidate", type=Path, default=ProjectPaths.discover().root)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.candidate, args.destination)
