"""Ventanas retrospectivas congeladas, sin publicación de modelos ni ajuste sobre test."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectPaths
from .contracts import dataset_fingerprint
from .model_audit import audit_frozen_model
from .modeling import train_and_select_model
from .storage import atomic_json


@dataclass(frozen=True)
class ExperimentWindow:
    """Cortes de calendario previos al embargo de madurez, con fin de test acotado."""

    train_end: int
    validation_end: int
    test_end: int


# Fijado antes de la primera ejecución. Los períodos de test no se solapan;
# el entrenamiento posterior puede usar resultados previos en una ventana expansiva.
WINDOWS = (ExperimentWindow(2001, 2005, 2008), ExperimentWindow(2004, 2008, 2011),
           ExperimentWindow(2007, 2011, 2017))


def experiment_protocol(windows: tuple[ExperimentWindow, ...] = WINDOWS) -> dict[str, Any]:
    """Valida intervalos de test cronológicos y no solapados."""
    previous_test_end: int | None = None
    if not windows:
        raise ValueError("At least one window is required")
    for window in windows:
        if not window.train_end < window.validation_end < window.test_end:
            raise ValueError("Window cutoffs must be strictly increasing")
        if previous_test_end is not None and window.validation_end < previous_test_end:
            raise ValueError("Test calendar intervals must not overlap")
        previous_test_end = window.test_end
    return {"version": "retrospective-expanding-v1", "windows": [asdict(window) for window in windows],
            "random_state": 73, "advanced_improvement_threshold": 0.10, "observation_window_years": 3,
            "interpretation": "Exploratory retrospective evaluation on already inspected data, not untouched final tests.",
            "aggregation": "No pooled MAE: each window uses its own train-only target normalizer.",
            "serving_publication": False}


def run_temporal_experiments(paths: ProjectPaths, *, windows: tuple[ExperimentWindow, ...] = WINDOWS) -> Path:
    """Publica el protocolo antes del entrenamiento y resultados aislados inmutables después.
    Registra explícitamente ventanas fallidas o adaptadas; no las presenta como evaluaciones
    de cortes fijos correctas. Conserva intactos los archivos de servicio.
    """
    gold = pd.read_parquet(paths.gold_path)
    protocol = {**experiment_protocol(windows), "dataset_fingerprint": dataset_fingerprint(gold)}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    directory = paths.reports_dir / "temporal_experiments" / digest
    protocol_path = directory / "protocol.json"
    atomic_json(protocol_path, protocol, immutable=True)
    if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing experiment protocol differs")
    destination = directory / "results.json"
    # Nunca repetir silenciosamente un experimento congelado; preservar su evidencia.
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("protocol") != protocol:
            raise ValueError("Existing experiment results have a different protocol")
        return destination
    results: list[dict[str, Any]] = []
    for window in windows:
        subset = gold.loc[gold.ano_fabricacion <= window.test_end].copy()
        try:
            training = train_and_select_model(subset, persist=False, train_end_year=window.train_end,
                validation_end_year=window.validation_end, random_state=protocol["random_state"],
                advanced_improvement_threshold=protocol["advanced_improvement_threshold"],
                observation_window_years=protocol["observation_window_years"])
            split = training.metrics["split"]
            if (split["configured_train_end_year"] != window.train_end
                    or split["configured_validation_end_year"] != window.validation_end
                    or not split["strategy"].startswith("documented_fixed_years")
                    or not split["validation"]["rows"] or not split["test"]["rows"]):
                raise ValueError("Insufficient data for the frozen fixed-calendar split")
            audit = audit_frozen_model(subset, training.artifact)
            results.append({"window": asdict(window), "status": "evaluated", "metrics": training.metrics,
                            "audit": audit, "model_created_at": training.artifact.created_at})
        except ValueError as exc:
            results.append({"window": asdict(window), "status": "not_evaluable", "reason": str(exc)})
    atomic_json(destination, {"protocol": protocol, "windows": results}, immutable=True)
    return destination
