"""Paquetes de servicio inmutables activados tras validación; solo fuentes locales confiables."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .config import ProjectPaths
from .contracts import dataset_fingerprint
from .modeling import load_model_artifact
from .storage import atomic_destination, atomic_json

FILES = ("data/gold/gold_us_car_reliability.parquet", "data/gold/us_car_inference_catalog.parquet",
         "artifacts/reliability_model.joblib", "artifacts/model_metrics.json")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify(directory: Path, hashes: dict[str, str]) -> None:
    if set(hashes) != set(FILES):
        raise ValueError("Release manifest contains unexpected paths")
    for name in FILES:
        if _digest((directory / name).read_bytes()) != hashes[name]:
            raise ValueError(f"Release integrity failure: {name}")


def publish_release(paths: ProjectPaths) -> str:
    """Captura resultados locales confiables; un fallo no cambia el puntero activo. Los hashes
    aportan integridad, no autenticidad: nunca publicar pickle no confiable. El normalizador
    ajustado está integrado en modelo y métricas.
    """
    releases = paths.root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    with FileLock(str(releases / ".publish.lock"), timeout=30):
        contents = {name: (paths.root / name).read_bytes() for name in FILES}
        hashes = {name: _digest(content) for name, content in contents.items()}
        identifier = _digest(json.dumps(hashes, sort_keys=True).encode())
        directory = releases / identifier
        for name, content in contents.items():
            with atomic_destination(directory / name, immutable=True) as temporary:
                temporary.write_bytes(content)
        _verify(directory, hashes)
        bundle_paths = ProjectPaths(directory)
        gold = pd.read_parquet(bundle_paths.gold_path)
        catalog = pd.read_parquet(bundle_paths.inference_catalog_path)
        artifact = load_model_artifact(bundle_paths.model_path)
        metrics = json.loads((directory / "artifacts/model_metrics.json").read_text(encoding="utf-8"))
        fingerprint = dataset_fingerprint(gold)
        if (artifact.metadata.get("dataset_fingerprint") != fingerprint
                or metrics.get("dataset_fingerprint") != fingerprint
                or metrics.get("target_normalizer") != artifact.metadata.get("target_normalizer")):
            raise ValueError("Release contains inconsistent model, metrics or Gold")
        if catalog.id_vehiculo_ano.duplicated().any():
            raise ValueError("Release catalog contains duplicate identities")
        features = list(artifact.feature_columns)
        indexed = catalog.set_index("id_vehiculo_ano")
        reference = gold.set_index("id_vehiculo_ano")
        if not reference.index.isin(indexed.index).all():
            raise ValueError("Release catalog omits labelled identities")
        pd.testing.assert_frame_equal(reference[features].sort_index(),
                                      indexed.loc[reference.index, features].sort_index(), check_dtype=False)
        # Detectar modificaciones concurrentes de las fuentes durante la captura.
        if any(_digest((paths.root / name).read_bytes()) != hashes[name] for name in FILES):
            raise ValueError("Source files changed during publication; retry")
        atomic_json(directory / "manifest.json", {"files": hashes}, immutable=True)
        atomic_json(releases / "active.json", {"release": identifier})
    return identifier


def serving_paths(paths: ProjectPaths) -> ProjectPaths:
    """Fija cada servicio a una publicación inmutable validada. Proyectos antiguos sin
    publicación usan archivos de trabajo. Una publicación corrupta falla explícitamente, sin
    respaldo silencioso.
    """
    pointer = paths.root / "releases/active.json"
    if not pointer.exists():
        return paths
    identifier = json.loads(pointer.read_text(encoding="utf-8"))["release"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{64}", identifier):
        raise ValueError("Invalid active release identifier")
    directory = paths.root / "releases" / identifier
    hashes = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["files"]
    if _digest(json.dumps(hashes, sort_keys=True).encode()) != identifier:
        raise ValueError("Release manifest identity mismatch")
    _verify(directory, hashes)
    return ProjectPaths(directory)
