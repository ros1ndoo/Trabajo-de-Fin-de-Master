"""Descriptive subgroup audit of a frozen model; never fit or select models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .contracts import PREDICTION_FEATURES, dataset_fingerprint, require_columns
from .modeling import (
    BrandSegmentMeanBaseline,
    ReliabilityModelArtifact,
    TemporalTargetNormalizer,
    load_model_artifact,
)
from .storage import atomic_json


def error_summary(frame: pd.DataFrame, *, minimum_group_size: int = 30) -> dict[str, Any]:
    """Describe paired errors, retaining small groups without claiming precision.

    The minimum is a reporting flag, not a statistical significance threshold.
    Positive bias means the index is overestimated (lower apparent recall risk).
    """
    if minimum_group_size < 1:
        raise ValueError("minimum_group_size must be positive")
    require_columns(frame, ("observed", "predicted"), context="Audit errors")
    values = frame[["observed", "predicted"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Audit values must be finite")
    n = len(values)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "small_group": True}
    residual = values[:, 1] - values[:, 0]
    return {"n": n, "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())), "bias": float(residual.mean()),
            "small_group": n < minimum_group_size}


def audit_frozen_model(gold: pd.DataFrame, artifact: ReliabilityModelArtifact,
                       *, minimum_group_size: int = 30) -> dict[str, Any]:
    """Audit only the recorded test years using the persisted target scale.

    Fail closed on stale inputs or a partition that cannot be reconstructed.
    This reuses already inspected test data: it is not a new independent test.
    """
    require_columns(gold, ("id_vehiculo_ano", "ano_fabricacion", *PREDICTION_FEATURES), context="Audit Gold")
    if gold["id_vehiculo_ano"].isna().any() or gold["id_vehiculo_ano"].duplicated().any():
        raise ValueError("Gold identifiers must be unique and non-null")
    fingerprint = dataset_fingerprint(gold)
    if artifact.metadata.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Model/Gold fingerprint mismatch")
    recorded = artifact.metadata["split"]["test"]
    if not recorded["rows"]:
        raise ValueError("The artifact has no recorded test partition")
    years = pd.to_numeric(gold["ano_fabricacion"], errors="raise")
    if not np.isfinite(years).all() or not years.eq(years.astype(int)).all():
        raise ValueError("Model years must be finite integers")
    test = gold.loc[years.between(recorded["min_year"], recorded["max_year"])].copy()
    if len(test) != recorded["rows"]:
        raise ValueError("Test partition count differs from the recorded experiment")
    maturity_columns = [column for column in ("es_cohorte_completa", "cohorte_completa") if column in test]
    if not maturity_columns:
        raise ValueError("Explicit cohort maturity evidence is required")
    for column in maturity_columns:
        if not test[column].eq(True).fillna(False).all():
            raise ValueError("Test contains immature or unverified cohorts")
    available = artifact.metadata.get("prediction_available_from_year")
    if available is None or int(available) > int(test.ano_fabricacion.min()):
        raise ValueError("Model availability does not precede the test launches")
    normalizer = TemporalTargetNormalizer.from_dict(artifact.metadata["target_normalizer"])
    if normalizer.train_end_year + normalizer.observation_window_years > int(test.ano_fabricacion.min()):
        raise ValueError("Target normalizer includes labels unavailable at test launch")
    test["observed"] = normalizer.transform(test)
    # Explicit feature allowlist: outcomes never reach the estimator.
    test["predicted"] = artifact.predict(test.loc[:, list(PREDICTION_FEATURES)])
    dimensions = ["marca", "categoria_vehiculo", "ano_fabricacion"]
    if isinstance(artifact.estimator, BrandSegmentMeanBaseline):
        _, levels = artifact.estimator.predict_with_sources(test.loc[:, list(PREDICTION_FEATURES)])
        test["support_level"] = levels
        dimensions.append("support_level")
        test["serving_support"] = np.where(test["support_level"].isin(["marca_y_segmento", "marca"]),
                                           "represented_brand", "unrepresented_brand")
        dimensions.append("serving_support")
    groups = {
        dimension: [{"group": str(key), **error_summary(group, minimum_group_size=minimum_group_size)}
                    for key, group in test.groupby(dimension, dropna=False, sort=True)]
        for dimension in dimensions
    }
    return {
        "protocol": "frozen-model-subgroups-v1", "model": artifact.model_name,
        "fuente_demo": bool(artifact.metadata.get("fuente_demo", False)),
        "dataset_fingerprint": fingerprint, "test_partition": recorded,
        "minimum_group_size": minimum_group_size,
        "interpretation": "Descriptive reanalysis of previously inspected test data; no fitting or model selection.",
        "limitations": ["Recall-bearing matched population; not representative of all vehicles.",
                        "MAE is not an individual confidence interval.",
                        "Small-group threshold is a reporting convention, not a significance test.",
                        "Subgroup errors do not establish causality or homogeneous performance."],
        "overall": error_summary(test, minimum_group_size=minimum_group_size), "groups": groups,
        "predictions": test[["id_vehiculo_ano", "observed", "predicted"]].to_dict(orient="records"),
    }


def publish_model_audit(paths: ProjectPaths, *, minimum_group_size: int = 30) -> Path:
    """Publish a content-addressed JSON report without touching serving files."""
    # A changed model during loading must not acquire the wrong provenance hash.
    model_hash = hashlib.sha256(paths.model_path.read_bytes()).hexdigest()
    artifact = load_model_artifact(paths.model_path)
    gold = pd.read_parquet(paths.gold_path)
    report = audit_frozen_model(gold, artifact, minimum_group_size=minimum_group_size)
    if hashlib.sha256(paths.model_path.read_bytes()).hexdigest() != model_hash:
        raise ValueError("Model changed during audit; retry with a stable artifact")
    report["model_sha256"] = model_hash
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    destination = paths.reports_dir / "model_audit" / f"{digest}.json"
    atomic_json(destination, report, immutable=True)
    # Detect an existing corrupt report instead of silently accepting it.
    if json.loads(destination.read_text(encoding="utf-8")) != report:
        raise ValueError("Existing content-addressed report is inconsistent")
    return destination
