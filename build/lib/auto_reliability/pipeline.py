"""End-to-end, auditable Raw -> Processed -> Gold orchestration.

This module intentionally has no hidden downloads. A caller supplies the
licensed/public CooperUnion CSV, and the NHTSA client persists every response
once in the immutable Raw layer before any transformation happens.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import OBSERVATION_WINDOW_YEARS, ProjectPaths
from .data_sources import (
    HP_TO_CV,
    NHTSABatchResult,
    NHTSACatalogBatchResult,
    NHTSARecallClient,
    ingest_cooperunion_csv,
)
from .matching import (
    MatchSummary,
    accepted_matches,
    apply_manual_match_decisions,
    match_technical_to_recalls,
    summarise_matches,
    write_fuzzy_audit,
)
from .storage import atomic_destination, atomic_json
from .transform import (
    ReliabilityNormalizer,
    build_gold_dataset,
    build_inference_catalog,
    prepare_technical_specs,
    quality_check_technical_specs,
    write_normalizer_metadata,
    write_processed_sqlite,
)


@dataclass
class PipelineResult:
    """Inspectable output of one pipeline run, including non-fatal warnings."""

    technical: pd.DataFrame
    nhtsa_catalog: pd.DataFrame
    nhtsa_records: pd.DataFrame
    nhtsa_vehicle_index: pd.DataFrame
    matches: pd.DataFrame
    gold: pd.DataFrame
    inference_catalog: pd.DataFrame
    match_summary: MatchSummary
    normalizer: ReliabilityNormalizer
    paths: ProjectPaths
    manifest_path: Path
    warnings: list[str] = field(default_factory=list)

    @property
    def contingency_recommended(self) -> bool:
        """Expose the documented >40% matching-loss warning."""

        return self.match_summary.contingency_recommended


def select_technical_scope(
    technical: pd.DataFrame,
    *,
    makes: Sequence[str] | None = None,
    model_years: Sequence[int] | None = None,
    max_vehicles: int | None = None,
) -> pd.DataFrame:
    """Select a reproducible, temporally broad real-data integration scope.

    A bounded run is a safety feature for public APIs.  Unlike ``head(n)``,
    the selector round-robins chronological model-year strata, so a 120-row
    smoke test preserves historical breadth instead of accidentally querying a
    single alphabetically early make or year.
    """

    required = {"marca", "modelo", "ano_fabricacion"}
    missing = required.difference(technical.columns)
    if missing:
        raise ValueError(f"Technical scope requires columns: {', '.join(sorted(missing))}.")
    selected = technical.copy()
    if makes:
        wanted_makes = {str(make).strip().casefold() for make in makes if str(make).strip()}
        selected = selected.loc[selected["marca"].astype(str).str.casefold().isin(wanted_makes)].copy()
    if model_years:
        wanted_years = {int(year) for year in model_years}
        selected = selected.loc[pd.to_numeric(selected["ano_fabricacion"], errors="coerce").isin(wanted_years)].copy()
    selected = selected.sort_values(["ano_fabricacion", "marca", "modelo"], kind="stable").reset_index(drop=True)
    if selected.empty:
        raise ValueError("The requested make/year filters left no technical vehicles to query.")
    if max_vehicles is None or len(selected) <= max_vehicles:
        return selected
    if max_vehicles < 1:
        raise ValueError("max_vehicles must be positive when it is provided.")

    by_year = {}
    for year, group in selected.groupby("ano_fabricacion", sort=True):
        group = group.copy()
        group["_make_rank"] = group.groupby("marca").cumcount()
        by_year[int(year)] = group.sort_values(["_make_rank", "marca", "modelo"], kind="stable").drop(
            columns="_make_rank"
        ).reset_index(drop=True)
    years = sorted(by_year)
    if max_vehicles < len(years):
        indexes = [round(i * (len(years) - 1) / max(1, max_vehicles - 1)) for i in range(max_vehicles)]
        years = [years[index] for index in indexes]
    picked: list[pd.Series] = []
    offset = 0
    while len(picked) < max_vehicles:
        progressed = False
        for year in years:
            group = by_year[year]
            if offset < len(group):
                picked.append(group.iloc[offset])
                progressed = True
                if len(picked) == max_vehicles:
                    break
        if not progressed:
            break
        offset += 1
    return pd.DataFrame(picked).reset_index(drop=True)


def stage_technical_csv(
    source_csv: str | Path,
    paths: ProjectPaths,
    *,
    filename: str = "cooperunion_car_features.csv",
) -> Path:
    """Copy the supplied source to Raw exactly once, without overwriting it.

    Raw is an evidence layer. If a file with the target name already exists
    and differs from the supplied CSV, the caller must choose a distinct name
    or explicitly manage the raw inputs; this function never replaces it.
    """

    source = Path(source_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Technical CSV does not exist: {source}")
    if not filename or Path(filename).name != filename or any(char in filename for char in ("/", "\\", ":")):
        raise ValueError("raw_filename must be a simple filename inside data/raw.")
    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("raw_filename must end in .csv.")
    paths.ensure_runtime_directories()
    destination = paths.raw_dir / filename
    if destination.exists() and source == destination.resolve():
        return destination
    if destination.exists():
        if _sha256(source) == _sha256(destination):
            return destination
        raise FileExistsError(
            f"Refusing to overwrite immutable Raw input {destination}. "
            "Choose a different --raw-filename or keep the existing evidence."
        )
    shutil.copy2(source, destination)
    return destination


def run_pipeline(
    technical_csv: str | Path,
    *,
    paths: ProjectPaths | None = None,
    as_of_date: date | datetime | str | None = None,
    train_end_year: int = 2018,
    max_vehicles: int | None = None,
    manual_decisions_csv: str | Path | None = None,
    raw_filename: str = "cooperunion_car_features.csv",
    request_delay_seconds: float = 0.20,
    makes: Sequence[str] | None = None,
    model_years: Sequence[int] | None = None,
    snapshot_date: date | str | None = None,
    recall_source: str = "api",
) -> PipelineResult:
    """Run the official pipeline and persist all writable layer artefacts.

    ``max_vehicles`` is deliberately provided for a safe smoke test. Omit it
    for a full historic execution; NHTSA queries will be rate-limited and raw
    JSON responses are cache-first on later runs.
    """

    project_paths = paths or ProjectPaths.discover()
    project_paths.ensure_runtime_directories()
    raw_csv = stage_technical_csv(technical_csv, project_paths, filename=raw_filename)
    raw_technical = ingest_cooperunion_csv(raw_csv)
    full_technical = prepare_technical_specs(raw_technical)
    full_technical_rows = len(full_technical)
    technical = select_technical_scope(
        full_technical, makes=makes, model_years=model_years, max_vehicles=max_vehicles
    )
    quality = quality_check_technical_specs(technical, source_rows=len(raw_technical))

    snapshot = date.fromisoformat(str(snapshot_date)) if snapshot_date is not None else datetime.now(timezone.utc).date()
    snapshot_dir = project_paths.raw_dir / "nhtsa" / snapshot.isoformat()
    client = NHTSARecallClient(
        snapshot_dir,
        request_delay_seconds=request_delay_seconds,
    )
    if recall_source == "bulk":
        from .bulk_recalls import BulkSnapshotClient
        client = BulkSnapshotClient(project_paths)
        snapshot_dir = project_paths.raw_dir / "nhtsa_bulk" / client.store.metadata()["snapshot_date"]
    elif recall_source != "api":
        raise ValueError("recall_source must be 'bulk' or 'api'.")
    catalog_batch = client.fetch_independent_vehicle_catalog(technical)
    matches = match_technical_to_recalls(technical, catalog_batch.vehicles, require_catalog_verified=True)
    if manual_decisions_csv is not None:
        decisions_path = Path(manual_decisions_csv)
        if not decisions_path.is_file():
            raise FileNotFoundError(f"Manual fuzzy-match decisions do not exist: {decisions_path}")
        decisions = pd.read_csv(decisions_path)
        matches = apply_manual_match_decisions(matches, decisions)
    batch: NHTSABatchResult = client.fetch_catalog_verified_recalls(accepted_matches(matches))
    query_status = batch.vehicle_index.set_index("nhtsa_vehicle_id")["query_status"].to_dict()
    for row_index, row in accepted_matches(matches).iterrows():
        status = query_status.get(row["nhtsa_vehicle_id"], "query_failed")
        if status not in {"valid_with_recalls", "valid_zero_recalls"}:
            matches.at[row_index, "match_status"] = status
            matches.at[row_index, "audit_required"] = True
            matches.at[row_index, "recomendacion_revision"] = "No transferir una etiqueta: consulta fallida o cero sin verificar."
    write_fuzzy_audit(matches, project_paths.audit_path)
    summary = summarise_matches(matches)

    gold, normalizer = build_gold_dataset(
        technical,
        matches,
        batch.records,
        as_of_date=as_of_date,
        train_end_year=train_end_year,
        observation_window_years=OBSERVATION_WINDOW_YEARS,
        return_normalizer=True,
    )
    # This catalog intentionally includes historic and recent technical rows.
    # Only Gold contains the selected vehicle's recall-derived label.
    inference_catalog = build_inference_catalog(
        full_technical,
        gold,
        as_of_date=as_of_date,
        observation_window_years=OBSERVATION_WINDOW_YEARS,
        include_complete_cohorts=True,
    )
    match_metadata = matches[["id_vehiculo_ano", "match_status", "match_score", "catalog_source"]].copy()
    match_metadata["coincidencia_recall_aceptada"] = match_metadata["match_status"].isin(["auto_accepted", "manual_accepted"])
    inference_catalog = inference_catalog.merge(match_metadata, on="id_vehiculo_ano", how="left", validate="one_to_one")
    inference_catalog["match_status"] = inference_catalog["match_status"].fillna("not_queried")
    inference_catalog["coincidencia_recall_aceptada"] = inference_catalog["coincidencia_recall_aceptada"].fillna(False).astype(bool)
    gold["fuente_demo"] = False
    inference_catalog["fuente_demo"] = False
    with atomic_destination(project_paths.gold_path) as temporary:
        gold.to_parquet(temporary, index=False)
    with atomic_destination(project_paths.inference_catalog_path) as temporary:
        inference_catalog.to_parquet(temporary, index=False)
    write_normalizer_metadata(normalizer, project_paths.processed_dir / "target_normalizer.json")
    catalog_batch.vehicles.to_parquet(project_paths.processed_dir / "nhtsa_vehicle_catalog.parquet", index=False)
    write_processed_sqlite(
        project_paths.sqlite_path,
        technical=technical,
        nhtsa_records=batch.records,
        nhtsa_vehicle_index=batch.vehicle_index,
        matches=matches,
        gold=gold,
        inference_catalog=inference_catalog,
    )

    warnings: list[str] = []
    if catalog_batch.failures:
        warnings.append(f"{len(catalog_batch.failures)} consultas de catálogo NHTSA fallaron; cobertura parcial.")
    unverified_zeros = int(batch.vehicle_index["query_status"].eq("unverified_zero_result").sum())
    if unverified_zeros:
        warnings.append(
            f"{unverified_zeros} respuestas vacías de modelos exclusivos de vPIC no prueban ausencia de recalls; "
            "se excluyen de Gold. Esta exclusión puede introducir sesgo de selección."
        )
    if batch.failures:
        warnings.append(f"{len(batch.failures)} consultas NHTSA fallaron; sus vehículos no entraron en Gold.")
    if recall_source == "bulk":
        warnings.append(
            "Ingesta masiva oficial sin límite de vehículos. El fichero contiene vehículos con campañas; "
            "su ausencia no acredita cero recalls. Las exclusiones y los cruces dudosos se conservan en la auditoría."
        )
    if summary.contingency_recommended:
        warnings.append(
            "La pérdida de matches supera el 40%. Revise audit_fuzzy_matches.csv; "
            "el plan de contingencia de depreciación sigue fuera del alcance del MVP."
        )
    if max_vehicles is not None and full_technical_rows > len(technical):
        warnings.append(
            f"Entrenamiento basado en una ingesta limitada a {len(technical)} de {full_technical_rows} vehículos técnicos. "
            "El selector sí incluye el catálogo técnico completo; los vehículos sin cruce usan una referencia histórica si existe."
        )
    manifest_path = _write_manifest(
        project_paths,
        raw_csv=raw_csv,
        raw_quality=raw_technical.attrs.get("quality_report", {}),
        processed_quality=quality.as_dict(),
        batch=batch,
        catalog_batch=catalog_batch,
        snapshot_dir=snapshot_dir,
        summary=summary,
        normalizer=normalizer,
        gold_rows=len(gold),
        inference_rows=len(inference_catalog),
        warnings=warnings,
    )
    return PipelineResult(
        technical=technical,
        nhtsa_catalog=catalog_batch.vehicles,
        nhtsa_records=batch.records,
        nhtsa_vehicle_index=batch.vehicle_index,
        matches=matches,
        gold=gold,
        inference_catalog=inference_catalog,
        match_summary=summary,
        normalizer=normalizer,
        paths=project_paths,
        manifest_path=manifest_path,
        warnings=warnings,
    )


def refresh_inference_catalog(technical_csv: str | Path, *, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Expose ALL technical makes/models, regardless of an API sampling limit.

    This operation is offline: missing recall evidence stays explicitly
    unknown. It never downloads, creates labels, or silently retrains a model.
    """
    project_paths = paths or ProjectPaths.discover()
    technical = prepare_technical_specs(ingest_cooperunion_csv(technical_csv))
    if project_paths.gold_path.exists():
        gold = pd.read_parquet(project_paths.gold_path)
        if "fuente_demo" in gold and gold["fuente_demo"].fillna(False).any():
            raise ValueError("No se puede mezclar un catálogo real con Gold sintético. Ejecute pipeline primero.")
    else:
        gold = pd.DataFrame(columns=["marca", "ano_fabricacion", "score_recalls_bruto"])
    catalog = build_inference_catalog(technical, gold, include_complete_cohorts=True)
    if project_paths.audit_path.exists():
        audit = pd.read_csv(project_paths.audit_path)
        keep = [c for c in ("id_vehiculo_ano", "match_status", "match_score", "catalog_source") if c in audit]
        catalog = catalog.merge(audit[keep], on="id_vehiculo_ano", how="left", validate="one_to_one")
    if "match_status" not in catalog:
        catalog["match_status"] = "not_queried"
    catalog["match_status"] = catalog["match_status"].fillna("not_queried")
    catalog["coincidencia_recall_aceptada"] = catalog["match_status"].isin(["auto_accepted", "manual_accepted"])
    catalog["fuente_demo"] = False
    project_paths.ensure_runtime_directories()
    catalog.to_parquet(project_paths.inference_catalog_path, index=False)
    # Keep this derived SQLite view aligned with the GUI catalogue as well.
    import sqlite3

    from .transform import _sqlite_value
    with sqlite3.connect(project_paths.sqlite_path) as connection:
        safe_catalog = catalog.astype(object).map(_sqlite_value)
        safe_catalog.where(pd.notna(safe_catalog), None).to_sql("inference_catalog", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_inference_vehicle ON inference_catalog(id_vehiculo_ano)")
    manifest_path = project_paths.processed_dir / "pipeline_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inference_catalog_rows"] = len(catalog)
        manifest["inference_catalog_makes"] = int(catalog.marca.nunique())
        manifest["catalog_refreshed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["warnings"] = [
            (f"Ingesta NHTSA limitada a {manifest.get('processed_quality', {}).get('output_rows', '?')} "
             f"vehículos. El catálogo web sí contiene los {len(catalog)} modelos-año de las "
             f"{catalog.marca.nunique()} marcas disponibles; no todos tienen etiquetas NHTSA verificadas.")
            if warning.startswith("Ejecución limitada a") else warning
            for warning in manifest.get("warnings", [])
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def _write_manifest(
    paths: ProjectPaths,
    *,
    raw_csv: Path,
    raw_quality: Any,
    processed_quality: dict[str, Any],
    batch: NHTSABatchResult,
    catalog_batch: NHTSACatalogBatchResult,
    snapshot_dir: Path,
    summary: MatchSummary,
    normalizer: ReliabilityNormalizer,
    gold_rows: int,
    inference_rows: int,
    warnings: list[str],
) -> Path:
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raw_technical_csv": str(raw_csv),
        "raw_technical_sha256": _sha256(raw_csv),
        "technical_transform": {
            "version": "2-observed-only-cv",
            "power_unit": "CV",
            "hp_interpretation": "US mechanical horsepower (source column convention)",
            "hp_to_cv_factor": HP_TO_CV,
            "conversion_reference": "NIST SP 811 Appendix B.9: 745.6999 W/HP; 735.4988 W/CV",
            "aggregation": "median of observed valid trims within each make/model/year",
            "imputation": "none before split; fitted on training rows by estimator only",
        },
        "raw_quality": raw_quality,
        "processed_quality": processed_quality,
        "nhtsa": {
            "retrieval_mode": "bulk" if catalog_batch.vehicles.get("catalog_source", pd.Series(dtype=str)).eq("nhtsa_bulk").any() else "api",
            "snapshot_directory": str(snapshot_dir),
            "catalog_rows": len(catalog_batch.vehicles),
            "catalog_failures": catalog_batch.failures,
            "unverified_zero_queries": int(batch.vehicle_index["query_status"].eq("unverified_zero_result").sum()),
            "recall_records": len(batch.records),
            "vehicle_queries_succeeded": len(batch.vehicle_index),
            "vehicle_queries_failed": len(batch.failures),
            "failures": batch.failures,
        },
        "fuzzy_matching": summary.as_dict(),
        "gold_rows": int(gold_rows),
        "inference_catalog_rows": int(inference_rows),
        "normalizer": normalizer.to_dict(),
        "outputs": {
            "gold": str(paths.gold_path),
            "inference_catalog": str(paths.inference_catalog_path),
            "sqlite": str(paths.sqlite_path),
            "fuzzy_audit": str(paths.audit_path),
        },
        "warnings": warnings,
    }
    destination = paths.processed_dir / "pipeline_manifest.json"
    atomic_json(destination, manifest)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PipelineResult", "run_pipeline", "select_technical_scope", "stage_technical_csv"]
