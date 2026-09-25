"""Sensibilidad descriptiva del objetivo construido, no nueva precisión predictiva."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .contracts import dataset_fingerprint
from .evidence import observation_evidence
from .matching import accepted_matches
from .modeling import fit_temporal_target_normalizer, load_model_artifact
from .releases import serving_paths
from .storage import atomic_json
from .transform import prepare_recall_events

SCENARIOS = {
    "documented": {"critical": 3., "moderate": 1.5, "low": 1., "unclassified": 1.},
    "equal_weights": {"critical": 1., "moderate": 1., "low": 1., "unclassified": 1.},
    "critical_double": {"critical": 6., "moderate": 1.5, "low": 1., "unclassified": 1.},
    "unclassified_excluded": {"critical": 3., "moderate": 1.5, "low": 1., "unclassified": 0.},
}


def severity_counts(gold: pd.DataFrame, matches: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Cuenta campañas deduplicadas en los tres años naturales de cada modelo."""
    if gold.id_vehiculo_ano.duplicated().any():
        raise ValueError("Duplicate Gold identity")
    mapping = accepted_matches(matches)[["id_vehiculo_ano", "nhtsa_vehicle_id"]]
    keys = gold[["id_vehiculo_ano", "ano_fabricacion"]].merge(mapping, validate="one_to_one")
    if len(keys) != len(gold):
        raise ValueError("Gold lacks accepted identity evidence")
    events = prepare_recall_events(records).merge(keys, on="nhtsa_vehicle_id", suffixes=("_source", ""))
    year = events.fecha_reporte_normalizada.dt.year
    events = events.loc[year.ge(events.ano_fabricacion) & year.lt(events.ano_fabricacion + 3)]
    counts = pd.crosstab(events.id_vehiculo_ano, events.severidad_recall)
    return counts.reindex(index=gold.id_vehiculo_ano, columns=list(SCENARIOS["documented"]), fill_value=0)


def selection_coverage(technical: pd.DataFrame, gold: pd.DataFrame) -> dict[str, Any]:
    """Publica denominadores de inclusión sin etiquetar excluidos como ceros."""
    if technical.id_vehiculo_ano.duplicated().any() or gold.id_vehiculo_ano.duplicated().any():
        raise ValueError("Coverage requires unique identities")
    if not gold.id_vehiculo_ano.isin(technical.id_vehiculo_ano).all():
        raise ValueError("Gold contains identities absent from technical universe")
    frame = technical.copy()
    frame["included"] = frame.id_vehiculo_ano.isin(gold.id_vehiculo_ano)
    def summary(group: pd.DataFrame) -> dict[str, Any]:
        included = int(group.included.sum())
        return {"technical": len(group), "included": included, "excluded": len(group) - included,
                "inclusion_rate": included / len(group) if len(group) else None}
    return {"overall": summary(frame), "groups": {
        column: [{"group": str(key), **summary(group)} for key, group in frame.groupby(column, dropna=False)]
        for column in ("marca", "categoria_vehiculo", "ano_fabricacion")}}


def sensitivity(gold: pd.DataFrame, counts: pd.DataFrame, *, train_max: int) -> dict[str, Any]:
    """Reajusta solo estadísticas de escala con las mismas filas históricas de entrenamiento.
    Las escalas alternativas definen objetivos diferentes: sus desplazamientos NO son
    cambios de MAE predictivo. No modifica estimadores, hiperparámetros ni puntuaciones
    servidas.
    """
    aligned = counts.reindex(gold.id_vehiculo_ano)
    if (aligned.isna().any().any() or not np.isfinite(aligned.to_numpy()).all()
            or (aligned < 0).any().any() or (aligned % 1 != 0).any().any()):
        raise ValueError("Invalid campaign counts")
    targets = {}
    normalizers = {}
    for name, weights in SCENARIOS.items():
        frame = gold.copy()
        frame["score_recalls_bruto"] = sum(aligned[key].to_numpy() * value for key, value in weights.items())
        if name == "documented" and not np.allclose(frame.score_recalls_bruto, gold.score_recalls_bruto):
            raise ValueError("Reconstructed campaign weights disagree with serving Gold")
        train = frame.loc[frame.ano_fabricacion <= train_max]
        normalizer = fit_temporal_target_normalizer(train, raw_score_column="score_recalls_bruto")
        normalizers[name] = normalizer.to_dict()
        targets[name] = normalizer.transform(frame)
    baseline = targets["documented"]
    result = {}
    # Informar cohortes posteriores por separado; no ajustar la escala con estas filas.
    for partition, mask in {"all_gold": gold.ano_fabricacion.notna(), "after_train": gold.ano_fabricacion > train_max}.items():
        reference = baseline.loc[mask]
        rows = []
        for name, target in targets.items():
            values = target.loc[mask]
            delta = (values - reference).abs()
            correlation = reference.rank().corr(values.rank()) if reference.nunique() > 1 and values.nunique() > 1 else None
            rows.append({"scenario": name, "n": len(values), "mean_absolute_target_shift": float(delta.mean()) if len(delta) else None,
                         "max_absolute_target_shift": float(delta.max()) if len(delta) else None, "rank_correlation": correlation,
                         "changed_visual_band": int((np.digitize(values, [45, 70]) != np.digitize(reference, [45, 70])).sum())})
        result[partition] = rows
    result["unclassified_campaign_vehicle_pairs_in_window"] = int(aligned.unclassified.sum())
    result["train_only_normalizers"] = normalizers
    return result


def run_scientific_audit(paths: ProjectPaths) -> Path:
    """Congela un protocolo y audita instantáneas locales sin API ni entrenamiento de modelos."""
    active = serving_paths(paths)
    gold = pd.read_parquet(active.gold_path)
    artifact = load_model_artifact(active.model_path)
    if artifact.metadata.get("dataset_fingerprint") != dataset_fingerprint(gold):
        raise ValueError("Serving model and Gold disagree")
    database = paths.processed_dir / "auto_reliability.sqlite"
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest = json.loads((paths.processed_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
    observation_date = str(manifest["generated_at"])[:10]
    protocol = {"version": "scientific-audit-v5", "observation_date": observation_date,
                "weights": SCENARIOS, "database_sha256": digest,
                "gold_fingerprint": dataset_fingerprint(gold), "train_max": artifact.metadata["split"]["train"]["max_year"],
                "window": "model year through model year+2; not exact launch dates",
                "interpretation": "Exploratory target sensitivity; no causal, calibration or new predictive accuracy claim."}
    identifier = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    directory = paths.reports_dir / "scientific_audit" / identifier
    atomic_json(directory / "protocol.json", protocol, immutable=True)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        technical = pd.read_sql_query("SELECT * FROM technical_specs", connection)
        matches = pd.read_sql_query("SELECT * FROM fuzzy_matches", connection)
        records = pd.read_sql_query("SELECT * FROM nhtsa_recalls", connection)
        queries = pd.read_sql_query("SELECT * FROM nhtsa_vehicle_index", connection)
    if hashlib.sha256(database.read_bytes()).hexdigest() != digest:
        raise ValueError("Source database changed during audit")
    counts = severity_counts(gold, matches, records)
    evidence = observation_evidence(technical, matches, queries, gold, as_of_date=observation_date)
    report = {"protocol": protocol, "coverage": selection_coverage(technical, gold),
              "observation_evidence": {
                  "interpretation": "Recorded query states, not an independent certification of source coverage. Unknown is never zero.",
                  "counts": {column: {str(k): int(v) for k, v in evidence[column].value_counts().items()}
                             for column in ("primary_reason", "identity_status", "query_status", "window_status", "window_outcome")},
                  "vehicles": evidence.drop(columns="nhtsa_vehicle_id").to_dict(orient="records")},
              "sensitivity": sensitivity(gold, counts, train_max=protocol["train_max"]),
              "date_quality": {
                  "source_records": len(records),
                  "missing_or_invalid_report_dates": int(pd.to_datetime(records.fecha_reporte, errors="coerce").isna().sum()),
                  "interpretation": "Missing dates are excluded, not evidence of no campaign in the observation window."},
              "limitations": ["Selection bias not corrected by this report.",
                  "No sales/exposure denominator; segment normalization is not a sales adjustment.",
                  "Missing dates and uncertain identities are not repaired by sensitivity analysis.",
                  "Historical attributes and report corrections are not reconstructed as-of each original launch date.",
                  "Rank correlations are descriptive; no significance or independence assumed."]}
    destination = directory / "results.json"
    atomic_json(destination, report, immutable=True)
    if json.loads(destination.read_text(encoding="utf-8")) != report:
        raise ValueError("Existing audit differs; do not overwrite scientific evidence")
    return destination
