"""Auditable EPA/NHTSA name alignment, separate from labels and prediction.

Only unique full-name equivalences are linked automatically. Fuzzy and trim
similarities are review suggestions, never authority to transfer campaigns.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from filelock import FileLock

from .bulk_recalls import BulkRecallStore
from .config import ProjectPaths
from .contracts import require_columns
from .data_sources import DataSourceError, comparison_make, comparison_model
from .fuel_economy import digest, read_inventory
from .matching import token_set_ratio
from .storage import atomic_destination, atomic_json

POLICY_VERSION = "epa-nhtsa-names-v1"
KEYS = ["make", "model", "year"]
NHTSA_KEYS = ["nhtsa_vehicle_id", "marca", "modelo", "ano_fabricacion"]
LINKED = {"exact_name", "normalized_name"}


def _literal(value: object) -> str:
    """Compare case/spacing only; preserve model numbers and suffixes."""
    return " ".join(str(value).casefold().split())


def _validate_keys(frame: pd.DataFrame, columns: list[str], year: str) -> None:
    """Reject invalid identities rather than silently truncating model years."""
    require_columns(frame, columns, context="Inventory resolution")
    if frame[columns].isna().any().any() or frame[columns].astype(str).apply(
        lambda col: col.str.strip().eq("")
    ).any().any():
        raise DataSourceError("Inventory resolution requires non-empty identity fields.")
    years = pd.to_numeric(frame[year], errors="coerce")
    if years.isna().any() or years.mod(1).ne(0).any() or not years.between(1995, 2100).all():
        raise DataSourceError("Invalid model year in inventory resolution.")


def resolve_inventory(inventory: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Return one audit row per EPA variant, retaining unresolved candidates.

    A linked name means only family/year alignment. It does not assert that a
    recall applies to every trim/VIN, that a three-year label is mature, or that
    the technical inputs suffice for inference. Unmatched means unknown.
    """
    _validate_keys(inventory, ["id", *KEYS], "year")
    _validate_keys(catalog, NHTSA_KEYS, "ano_fabricacion")
    if inventory.id.duplicated().any():
        raise DataSourceError("Duplicate EPA source IDs.")
    candidates = catalog[NHTSA_KEYS].drop_duplicates().copy()
    if candidates.nhtsa_vehicle_id.duplicated().any():
        raise DataSourceError("Conflicting NHTSA identities for the same ID.")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.to_dict("records"):
        groups[(comparison_make(row["marca"]), int(row["ano_fabricacion"]))].append(row)
    rows: list[dict[str, Any]] = []
    for make, model, year in inventory[KEYS].drop_duplicates().itertuples(index=False, name=None):
        pool = groups[(comparison_make(make), int(year))]
        exact = [r for r in pool if _literal(r["modelo"]) == _literal(model)
                 and _literal(r["marca"]) == _literal(make)]
        normalized = [r for r in pool if comparison_model(r["modelo"]) == comparison_model(model)]
        matches = exact or normalized
        selected = matches[0] if len(matches) == 1 else None
        if selected:
            status = "exact_name" if exact else "normalized_name"
            suggestions = matches
        elif len(matches) > 1:
            status, suggestions = "ambiguous_name", matches
        elif not pool:
            status, suggestions = "no_same_make_year", []
        else:
            scored = [(token_set_ratio(model, r["modelo"]), r) for r in pool]
            suggestions = [dict(r, similarity=score) for score, r in sorted(
                scored, key=lambda item: (-item[0], str(item[1]["nhtsa_vehicle_id"]))
            ) if score >= 75]
            status = "manual_review" if suggestions else "no_model_match"
        rows.append({"make": make, "model": model, "year": year,
                     "identity_status": status,
                     "nhtsa_vehicle_id": selected["nhtsa_vehicle_id"] if selected else None,
                     "nhtsa_make": selected["marca"] if selected else None,
                     "nhtsa_model": selected["modelo"] if selected else None,
                     "candidate_count": len(suggestions),
                     "candidates_json": json.dumps(sorted(suggestions, key=lambda r: str(r["nhtsa_vehicle_id"])), ensure_ascii=False),
                     "recall_status": "not_evaluated", "prediction_eligible": False,
                     "resolution_policy": POLICY_VERSION})
    audit_columns = [*KEYS, "identity_status", "nhtsa_vehicle_id", "nhtsa_make", "nhtsa_model",
                     "candidate_count", "candidates_json", "recall_status", "prediction_eligible", "resolution_policy"]
    alignment = pd.DataFrame(rows, columns=audit_columns)
    source = inventory.drop(columns=[c for c in audit_columns if c not in KEYS and c in inventory], errors="ignore")
    return source.merge(alignment, on=KEYS, how="left", validate="many_to_one")


def resolution_summary(audit: pd.DataFrame) -> dict[str, Any]:
    """Count variants and distinct source names separately, without label claims."""
    names = audit.drop_duplicates(KEYS)

    def counts(frame: pd.DataFrame) -> dict[str, int]:
        return {str(k): int(v) for k, v in frame.identity_status.value_counts().sort_index().items()}

    return {
        "policy_version": POLICY_VERSION, "variants": len(audit), "source_names": len(names),
        "linked_variants": int(audit.identity_status.isin(LINKED).sum()),
        "linked_source_names": int(names.identity_status.isin(LINKED).sum()),
        "unique_linked_nhtsa_identities": int(audit.nhtsa_vehicle_id.nunique()),
        "variant_statuses": counts(audit), "name_statuses": counts(names),
        "by_make": {str(k): counts(g) for k, g in names.groupby("make")},
        "by_year": {str(k): counts(g) for k, g in names.groupby("year")},
        "warning": "Alineación de nombres, no validación predictiva ni aplicabilidad a cada versión/VIN. Ausencias no equivalen a cero recalls.",
    }


def publish_resolution(paths: ProjectPaths, snapshot_date: str) -> dict[str, Any]:
    """Rebuild from verified raw EPA and publish a content-addressed audit bundle.

    A completion pointer is written last. Interrupted runs never replace the
    active predictor or expose an incomplete bundle as completed.
    """
    if date.fromisoformat(snapshot_date).isoformat() != snapshot_date:
        raise ValueError("Snapshot date must be YYYY-MM-DD.")
    source_dir = paths.raw_dir / "fuel_economy" / snapshot_date
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    archive = source_dir / "vehicles.xml.zip"
    sha256 = digest(archive)
    if sha256 != manifest["sha256"]:
        raise DataSourceError("EPA source hash mismatch.")
    inventory = read_inventory(archive)
    catalog = BulkRecallStore(paths).catalog()
    canonical = catalog[NHTSA_KEYS].sort_values(NHTSA_KEYS).to_json(orient="records")
    catalog_sha = hashlib.sha256(canonical.encode()).hexdigest()
    bundle_id = hashlib.sha256(f"{POLICY_VERSION}:{sha256}:{catalog_sha}".encode()).hexdigest()
    root = paths.processed_dir / "inventory_resolution"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "publication.lock"), timeout=30):
        output = root / bundle_id
        audit = resolve_inventory(inventory, catalog)
        audit["epa_source_sha256"] = sha256
        audit["nhtsa_catalog_sha256"] = catalog_sha
        audit["epa_retrieved_at"] = manifest["retrieved_at_utc"]
        with atomic_destination(output / "audit.parquet", immutable=True) as temporary:
            audit.to_parquet(temporary, index=False)
        # Save the exact name catalog used, independent of later index refreshes.
        with atomic_destination(output / "nhtsa_catalog.parquet", immutable=True) as temporary:
            catalog[NHTSA_KEYS].to_parquet(temporary, index=False)
        summary = resolution_summary(audit)
        summary.update({"bundle_id": bundle_id, "epa_snapshot_date": snapshot_date,
                        "epa_sha256": sha256, "nhtsa_catalog_sha256": catalog_sha,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "audit_sha256": digest(output / "audit.parquet"),
                        "catalog_file_sha256": digest(output / "nhtsa_catalog.parquet")})
        summary_path = output / "summary.json"
        atomic_json(summary_path, summary, immutable=True)
        # On reruns, retain the original completion time and verify stored files.
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved["audit_sha256"] != summary["audit_sha256"] or saved["catalog_file_sha256"] != summary["catalog_file_sha256"]:
            raise DataSourceError("Existing resolution bundle failed integrity verification.")
        atomic_json(root / "latest.json", saved)
        return saved
