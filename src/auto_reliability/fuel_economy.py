"""Inventario independiente EPA/DOE, nunca promovido silenciosamente a Gold. Las clases de
tamaño EPA no equivalen a carrocerías CooperUnion ni aportan potencia general de motor.
Ambas carencias permanecen explícitas para revisión.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import ParseError

import pandas as pd
import requests
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import iterparse
from filelock import FileLock

from .config import ProjectPaths
from .data_sources import DataSourceError
from .storage import atomic_destination, atomic_json

SOURCE_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.xml.zip"
SCHEMA_URL = "https://www.fueleconomy.gov/feg/ws/"
TERMS_URL = "https://www.fueleconomy.gov/feg/ORNL-disclaimer.htm"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_XML_BYTES = 512 * 1024 * 1024
FIELDS = ("id", "make", "model", "year", "cylinders", "VClass", "fuelType1", "createdOn", "modifiedOn")


def digest(path: Path) -> str:
    """Calcula la huella por bloques sin cargar el archivo entero en memoria."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            result.update(chunk)
    return result.hexdigest()


def read_inventory(archive_path: Path, *, min_year: int = 1995) -> pd.DataFrame:
    """Valida XML acotado y conserva identidades y atributos ausentes. La presencia no acredita
    equivalencia NHTSA, cero recalls ni elegibilidad de entrenamiento. No infiere cilindros
    de eléctricos si faltan en la fuente.
    """
    rows: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != 1 or not members[0].filename.lower().endswith(".xml"):
                raise DataSourceError("Expected one XML member in EPA archive.")
            if members[0].file_size > MAX_XML_BYTES:
                raise DataSourceError("EPA XML exceeds the uncompressed size limit.")
            with archive.open(members[0]) as stream:
                for _, element in iterparse(stream, events=("end",), forbid_dtd=True):
                    if element.tag == "vehicle":
                        rows.append({field: (element.findtext(field) or "").strip() for field in FIELDS})
                        element.clear()
    except (zipfile.BadZipFile, ParseError, DefusedXmlException) as exc:
        raise DataSourceError("Invalid or unsafe EPA XML archive.") from exc
    frame = pd.DataFrame(rows, columns=FIELDS)
    if frame.empty or frame[["id", "make", "model", "year"]].eq("").any().any():
        raise DataSourceError("EPA inventory is empty or has missing identity fields.")
    years = pd.to_numeric(frame.year, errors="coerce")
    if years.isna().any() or years.mod(1).ne(0).any() or not years.between(1900, 2100).all():
        raise DataSourceError("EPA inventory contains invalid model years.")
    if frame.id.duplicated().any():
        raise DataSourceError("EPA inventory contains duplicate source IDs.")
    frame["year"] = years.astype(int)
    frame = frame.loc[frame.year.ge(min_year)].copy()
    raw_cylinders = frame.cylinders.copy()
    cylinders = pd.to_numeric(raw_cylinders, errors="coerce")
    valid = cylinders.between(0, 32) & cylinders.mod(1).eq(0)
    frame["cylinders_original"] = raw_cylinders
    frame["cylinders"] = cylinders.where(valid).astype("Int64")
    frame["cylinders_status"] = "missing"
    frame.loc[raw_cylinders.ne("") & ~valid, "cylinders_status"] = "invalid"
    frame.loc[valid, "cylinders_status"] = "observed"
    frame["power_cv"] = float("nan")
    frame["power_status"] = "not_provided_by_source"
    frame["source"] = "epa_doe_fueleconomy"
    frame["identity_status"] = "pending_nhtsa_resolution"
    frame["recall_status"] = "not_queried"
    frame["prediction_eligible"] = False
    return frame.reset_index(drop=True)


def acquire_inventory(paths: ProjectPaths) -> dict[str, object]:
    """Obtiene la fuente inmutable del día y publica un inventario auditado separado. Expone
    fallos HTTP sin reintentos ciegos; adquisiciones repetidas correctas verifican y
    reutilizan la instantánea sin red.
    """
    now = datetime.now(timezone.utc)
    directory = paths.raw_dir / "fuel_economy" / now.date().isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "vehicles.xml.zip"
    manifest_path = directory / "manifest.json"
    with FileLock(str(directory / "snapshot.lock"), timeout=30):
        if archive_path.exists():
            if not manifest_path.exists():
                raise DataSourceError("EPA snapshot has no manifest; preserve it for recovery review.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if digest(archive_path) != manifest.get("sha256"):
                raise DataSourceError("EPA snapshot hash mismatch; source was not overwritten.")
            frame = read_inventory(archive_path)
        else:
            if manifest_path.exists():
                raise DataSourceError("EPA manifest exists without its source; recovery review required.")
            with atomic_destination(archive_path, immutable=True) as temporary:
                with requests.get(SOURCE_URL, timeout=(10, 60), stream=True) as response:
                    response.raise_for_status()
                    size = 0
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            if size > MAX_ARCHIVE_BYTES:
                                raise DataSourceError("EPA archive exceeds download limit.")
                            stream.write(chunk)
                frame = read_inventory(temporary)
                sha256 = digest(temporary)
            manifest = {
                "source_url": SOURCE_URL, "schema_url": SCHEMA_URL,
                "terms_url": TERMS_URL, "retrieved_at_utc": now.isoformat(),
                "sha256": sha256, "bytes": size,
                "use_scope": "Non-commercial scientific/educational use; no vehicle photographs.",
            }
            atomic_json(manifest_path, manifest, immutable=True)
        # Los artefactos ligados a una instantánea no sustituyen Gold ni el modelo productivo.
        output = paths.processed_dir / "fuel_economy" / now.date().isoformat()
        frame["source_sha256"] = manifest["sha256"]
        frame["source_retrieved_at"] = manifest["retrieved_at_utc"]
        with atomic_destination(output / "inventory.parquet") as temporary:
            frame.to_parquet(temporary, index=False)
        coverage = {
            "source_manifest": str(manifest_path), "variants": len(frame),
            "make_model_years": len(frame[["make", "model", "year"]].drop_duplicates()),
            "makes": int(frame.make.nunique()),
            "min_year": int(frame.year.min()) if len(frame) else None,
            "max_year": int(frame.year.max()) if len(frame) else None,
            "missing_power": len(frame), "missing_cylinders": int(frame.cylinders.isna().sum()),
            "by_year": {str(year): len(group) for year, group in frame.groupby("year")},
            "by_class": {str(key): len(group) for key, group in frame.groupby("VClass")},
            "by_make": {
                str(make): {"variants": len(group),
                            "make_model_years": len(group[["model", "year"]].drop_duplicates()),
                            "missing_cylinders": int(group.cylinders.isna().sum()),
                            "missing_power": len(group)}
                for make, group in frame.groupby("make")
            },
            "warnings": ["Not a complete universe of all US vehicles.",
                         "EPA classes are not mapped to CooperUnion body styles.",
                         "No recall labels inferred; no automatic promotion to Gold.",
                         "Current snapshot does not prove attributes were known at launch."],
        }
        atomic_json(output / "coverage.json", coverage)
        return coverage
