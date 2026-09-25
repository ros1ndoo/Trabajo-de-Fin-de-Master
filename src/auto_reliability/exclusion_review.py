"""Evidencia independiente de identidad y clasificación de consultas, nunca etiquetas
automáticas.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from filelock import FileLock

from .config import ProjectPaths
from .data_sources import (
    DataSourceError,
    NHTSARecallClient,
    comparison_make,
    comparison_model,
    normalise_nhtsa_response,
)
from .fuel_economy import digest, read_inventory
from .releases import serving_paths
from .storage import atomic_json


def independent_identities(technical: pd.DataFrame, inventory: pd.DataFrame) -> list[dict[str, Any]]:
    """Resuelve familia/año por nombres normalizados exactos, sin alias difusos ni de acabados.
    Conserva variantes EPA como identificadores de fuente, no evidencia independiente de
    campañas. Colisiones de nombres requieren revisión; ausencia EPA no prueba inexistencia
    del vehículo.
    """
    if technical.id_vehiculo_ano.isna().any() or technical.id_vehiculo_ano.duplicated().any():
        raise ValueError("Technical identities must be unique and non-null")
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for item in inventory.to_dict("records"):
        key = (comparison_make(item["make"]), comparison_model(item["model"]), int(item["year"]))
        groups.setdefault(key, []).append(item)
    result = []
    for item in technical.to_dict("records"):
        key = (comparison_make(item["marca"]), comparison_model(item["modelo"]), int(item["ano_fabricacion"]))
        candidates = groups.get(key, [])
        names = sorted({(str(row["make"]), str(row["model"])) for row in candidates})
        status = "exact_independent_identity" if len(names) == 1 else "ambiguous_independent_identity" if names else "absent_from_independent_inventory"
        result.append({"id_vehiculo_ano": item["id_vehiculo_ano"], "year": key[2],
                       "independent_identity_status": status,
                       "source_ids": sorted({str(row["id"]) for row in candidates}),
                       "query_make": names[0][0] if len(names) == 1 else None,
                       "query_model": names[0][1] if len(names) == 1 else None,
                       "recall_query_status": "not_queried", "label_eligible": False})
    return result


def query_identity(client: NHTSARecallClient, row: dict[str, Any]) -> dict[str, Any]:
    """Conserva hashes de respuestas y distingue vacío de recuperación fallida. Ni respuestas
    positivas ni vacías aprueban el vínculo técnico con recalls: Gold sigue requiriendo
    cruce revisado y cobertura de ventana.
    """
    result = dict(row)
    if row["independent_identity_status"] != "exact_independent_identity":
        return result
    try:
        response = client.fetch_vehicle(row["query_make"], row["query_model"], row["year"])
        records = normalise_nhtsa_response(response.payload, make=row["query_make"],
                                           model=row["query_model"], year=row["year"])
        count = response.result_count
        result.update({"recall_query_status": "campaigns_returned_review_required" if count else "empty_response_identity_only",
                       "response_count": count, "parsed_records": len(records),
                       "response_sha256": digest(response.cache_path),
                       "response_path": str(response.cache_path), "from_cache": response.from_cache,
                       "endpoint": client.endpoint,
                       "assessed_at": datetime.now(timezone.utc).isoformat()})
    except (DataSourceError, requests.RequestException, ValueError) as exc:
        result.update({"recall_query_status": "query_failed", "error": str(exc),
                       "assessed_at": datetime.now(timezone.utc).isoformat()})
    return result


def review_exclusions(paths: ProjectPaths, *, snapshot_date: str, query_limit: int = 0) -> Path:
    """Audita todas las identidades y permite consultar un lote acotado reanudable. Límite cero
    no usa red; repeticiones consultan pendientes y conservan fallos y observaciones para
    revisión explícita. Los hashes separan versiones para evitar mezclar evidencia.
    """
    if query_limit < 0:
        raise ValueError("Query limit must be nonnegative")
    date.fromisoformat(snapshot_date)
    source = paths.raw_dir / "fuel_economy" / snapshot_date
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    archive = source / "vehicles.xml.zip"
    source_hash = digest(archive)
    if source_hash != manifest["sha256"]:
        raise ValueError("Independent source hash mismatch")
    active = serving_paths(paths)
    catalog = pd.read_parquet(active.inference_catalog_path)
    gold = pd.read_parquet(active.gold_path)
    protocol = {"version": "independent-exclusions-v1", "epa_sha256": source_hash,
                "catalog_sha256": digest(active.inference_catalog_path), "gold_sha256": digest(active.gold_path),
                "source_url": manifest["source_url"], "source_retrieved_at": manifest["retrieved_at_utc"],
                "zero_policy": "Independent identity plus empty response does not establish recall-namespace coverage; not Gold eligible."}
    identifier = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    destination = paths.reports_dir / "exclusion_review" / identifier / "results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock", timeout=30):
        if destination.exists():
            report = json.loads(destination.read_text(encoding="utf-8"))
            if report["protocol"] != protocol:
                raise ValueError("Stored review protocol mismatch")
            rows = report["vehicles"]
        else:
            rows = independent_identities(catalog, read_inventory(archive))
            included = set(gold.id_vehiculo_ano)
            for row in rows:
                row["included_in_gold"] = row["id_vehiculo_ano"] in included
        def save() -> None:
            frame = pd.DataFrame(rows)
            excluded = frame.loc[~frame.included_in_gold]
            summary = {"protocol": protocol,
                "excluded_rows": len(excluded),
                "independent_identity_counts": excluded.independent_identity_status.value_counts().to_dict(),
                "query_counts": excluded.recall_query_status.value_counts().to_dict(),
                "labels_added_to_gold": 0}
            atomic_json(destination, {**summary, "vehicles": rows})
            atomic_json(paths.processed_dir / "exclusion_review/latest.json", summary)
        save()
        client = NHTSARecallClient(paths.raw_dir / "exclusion_queries" / datetime.now(timezone.utc).date().isoformat(),
                                  timeout_seconds=12, max_retries=1, request_delay_seconds=.3)
        pending = [index for index, row in enumerate(rows) if not row["included_in_gold"]
                   and row["independent_identity_status"] == "exact_independent_identity"
                   and row["recall_query_status"] == "not_queried"]
        for index in pending[:query_limit]:
            rows[index] = query_identity(client, rows[index])
            save()  # Reanudar sin perder observaciones ya completadas.
    return destination
