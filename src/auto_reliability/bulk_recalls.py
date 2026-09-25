"""Instantáneas oficiales NHTSA indexadas sin interpretar ausencias como ceros. Esquema:
https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt (29 campos, mayo de 2025). Se requieren
PRE_2010 y POST_2010: la partición corresponde a informes, no a años-modelo. Explorador e
ingesta comparten evidencia inmutable, nunca etiquetas futuras como características
predictoras.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import zipfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from filelock import FileLock

from .config import ProjectPaths
from .contracts import vehicle_id
from .data_sources import (
    DataSourceError,
    NHTSACatalogBatchResult,
    NHTSAFetchResult,
    NHTSARecallClient,
    comparison_make,
    comparison_model,
)
from .storage import atomic_destination, atomic_json

BASE_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/"
ARCHIVES = ("FLAT_RCL_PRE_2010.zip", "FLAT_RCL_POST_2010.zip")
FIELDS = (
    "record_id", "campaign", "make", "model", "year", "mfr_campaign", "component",
    "manufacturer", "begin_manufacture", "end_manufacture", "recall_type", "affected",
    "owner_date", "influenced_by", "mfr_text", "received_date", "created_date", "part",
    "fmvss", "summary", "consequence", "remedy", "notes", "component_id",
    "mfr_component_name", "mfr_component_desc", "mfr_component_part", "do_not_drive", "park_outside",
)


def download_snapshot(paths: ProjectPaths, *, snapshot_date: str | None = None) -> Path:
    """Descarga una vez ambos ZIP con tamaño acotado, sin publicar descargas parciales. Permite
    reintentar el archivo fallido sin repetir el correcto; publica el manifiesto solo tras
    validar ambos.
    """
    day = snapshot_date or datetime.now(timezone.utc).date().isoformat()
    datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)  # Rechazar rutas y fechas inválidas.
    directory = paths.raw_dir / "nhtsa_bulk" / day
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"snapshot_date": day, "files": []}
    with FileLock(str(directory / "download.lock"), timeout=30):
        for name in ARCHIVES:
            destination = directory / name
            if not destination.exists():
                with requests.get(BASE_URL + name, stream=True, timeout=(10, 45)) as response:
                    response.raise_for_status()
                    with atomic_destination(destination, immutable=True) as temporary:
                        size = 0
                        with temporary.open("wb") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                size += len(chunk)
                                if size > 100 * 1024 * 1024:
                                    raise DataSourceError("NHTSA archive exceeds the 100 MiB safety bound.")
                                stream.write(chunk)
                        _validate_archive(temporary)
            _validate_archive(destination)
            manifest["files"].append({"name": name, "url": BASE_URL + name,
                                      "sha256": _sha256(destination),
                                      "bytes": destination.stat().st_size})
        atomic_json(directory / "manifest.json", manifest, immutable=True)
    return directory


def _validate_archive(path: Path) -> None:
    """Rechaza miembros, tamaños o CRC inesperados sin extraer rutas del ZIP."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or not members[0].filename.lower().endswith((".txt", ".lst")):
            raise DataSourceError(f"Unexpected NHTSA ZIP members: {path.name}")
        if members[0].file_size > 800 * 1024 * 1024:
            raise DataSourceError("Uncompressed NHTSA archive exceeds safety bound.")
        if archive.testzip() is not None:
            raise DataSourceError(f"Corrupt NHTSA ZIP: {path.name}")


def read_archive(path: Path) -> pd.DataFrame:
    """Lee el esquema tabulado documentado y falla ante cambios de estructura. Conserva
    caracteres históricos Windows con latin-1, una correspondencia sin pérdida de bytes; no
    descarta silenciosamente filas malformadas.
    """
    _validate_archive(path)
    rows: list[list[str]] = []
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as raw:
        for line_number, row in enumerate(csv.reader(io.TextIOWrapper(raw, encoding="latin-1"),
                                                     delimiter="\t", quoting=csv.QUOTE_NONE), 1):
            if not row:
                continue
            # Algunas exportaciones terminan cada registro con un delimitador adicional.
            if len(row) == len(FIELDS) + 1 and row[-1] == "":
                row.pop()
            if len(row) != len(FIELDS):
                raise DataSourceError(f"{path.name}:{line_number}: expected 29 fields, got {len(row)}")
            rows.append([value.strip() for value in row])
    return pd.DataFrame(rows, columns=FIELDS)


def build_bulk_index(paths: ProjectPaths, snapshot: Path) -> dict[str, Any]:
    """Indexa atómicamente una instantánea completa para selectores y consultas locales.
    Conserva todas las marcas y años-modelo desde 1995 disponibles en la fuente oficial, sin
    restringir el explorador a CooperUnion.
    """
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if {entry["name"] for entry in manifest["files"]} != set(ARCHIVES):
        raise DataSourceError("Both recall archives are required for a complete snapshot.")
    frames = []
    for entry in manifest["files"]:
        source = snapshot / entry["name"]
        if _sha256(source) != entry["sha256"]:
            raise DataSourceError(f"Raw snapshot hash mismatch: {source.name}")
        frame = read_archive(source)
        frame["source_archive"] = source.name
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    total_rows = len(records)
    year = pd.to_numeric(records["year"], errors="coerce")
    records = records.loc[year.between(1995, datetime.now(timezone.utc).year + 1)
                          & records.recall_type.eq("V")].copy()
    records["year"] = records.year.astype(int)
    records["make_key"] = records.make.map(comparison_make)
    records["model_key"] = records.model.map(comparison_model)
    records["vehicle_id"] = [vehicle_id(m, n, y) for m, n, y in
                               records[["make", "model", "year"]].itertuples(index=False, name=None)]
    # Conservar componentes originales; deduplicar campañas solo al normalizar.
    vehicles = records[["vehicle_id", "make", "model", "year", "make_key", "model_key"]].drop_duplicates("vehicle_id")
    metadata = {"snapshot_date": manifest["snapshot_date"], "source_rows": total_rows,
                "vehicle_recall_rows": len(records), "vehicles": len(vehicles),
                "makes": int(vehicles.make_key.nunique()), "archives": manifest["files"],
                "built_at": datetime.now(timezone.utc).isoformat(),
                "notice": "Recall-bearing vehicles only; absence from this index is not evidence of zero recalls."}
    destination = paths.processed_dir / "nhtsa_bulk.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (FileLock(str(destination) + ".lock", timeout=30),
          atomic_destination(destination) as temporary,
          closing(sqlite3.connect(temporary)) as connection):
        records.to_sql("recalls", connection, index=False)
        vehicles.to_sql("vehicles", connection, index=False)
        connection.execute("CREATE INDEX idx_recall_vehicle ON recalls(vehicle_id)")
        connection.execute("CREATE INDEX idx_vehicle_key ON vehicles(make_key, model_key, year)")
        connection.execute("CREATE TABLE metadata (payload TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        connection.commit()
    return metadata


class BulkRecallStore:
    """Acceso parametrizado de solo lectura al último índice publicado completamente."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.path = paths.processed_dir / "nhtsa_bulk.sqlite"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Windows no permite reemplazar la base mientras un lector mantiene abierto el archivo.
        with (FileLock(str(self.path) + ".lock", timeout=30),
              closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as connection):
            yield connection

    def metadata(self) -> dict[str, Any]:
        """Devuelve la procedencia de la instantánea, no la hora actual como fecha de descarga."""
        with self._connect() as connection:
            return json.loads(connection.execute("SELECT payload FROM metadata").fetchone()[0])

    def catalog(self) -> pd.DataFrame:
        """Devuelve candidatos oficiales con campañas, NO un universo independiente del
        resultado.
        """
        with self._connect() as connection:
            frame = pd.read_sql_query("SELECT * FROM vehicles ORDER BY make, model, year", connection)
        frame = frame.rename(columns={"vehicle_id": "nhtsa_vehicle_id", "make": "marca", "model": "modelo",
                                      "year": "ano_fabricacion", "make_key": "marca_normalizada",
                                      "model_key": "modelo_normalizado"})
        frame["catalog_verified"] = True
        frame["catalog_source"] = "nhtsa_bulk"
        frame["catalog_cache_path"] = str(self.path)
        frame["catalog_desde_cache"] = True
        return frame

    def payload(self, make: str, model: str, year: int) -> dict[str, Any] | None:
        """Devuelve una respuesta con formato API o None para desconocidos, NUNCA un cero
        falso.
        """
        with self._connect() as connection:
            identity = connection.execute(
                "SELECT vehicle_id FROM vehicles WHERE make=? COLLATE NOCASE AND model=? COLLATE NOCASE AND year=?",
                (make.strip(), model.strip(), int(year)),
            ).fetchall()
            if not identity:
                identity = connection.execute(
                    "SELECT vehicle_id FROM vehicles WHERE make_key=? AND model_key=? AND year=?",
                    (comparison_make(make), comparison_model(model), int(year)),
                ).fetchall()
            if len(identity) != 1:
                return None
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM recalls WHERE vehicle_id=?", (identity[0][0],)).fetchall()
        results = [{"NHTSACampaignNumber": row["campaign"], "Component": row["component"],
                    "Summary": row["summary"], "Consequence": row["consequence"], "Remedy": row["remedy"],
                    "ReportReceivedDate": _iso_date(row["received_date"]),
                    "Notes": row["notes"], "DoNotDrive": row["do_not_drive"], "ParkOutside": row["park_outside"]}
                   for row in rows]
        return {"Count": len(results), "results": results}


def _iso_date(value: str) -> str:
    """Rechaza fechas ausentes en vez de generar silenciosamente etiquetas cero falsas."""
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc).date().isoformat()
    except ValueError as exc:
        raise DataSourceError(f"Invalid NHTSA received date: {value!r}") from exc


def _sha256(path: Path) -> str:
    """Calcula la huella por bloques, compatible con Python 3.10."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BulkSnapshotClient(NHTSARecallClient):
    """Adaptador de ingesta: sustituye miles de GET por la instantánea oficial completa. Los
    vehículos desconocidos quedan sin cruce; su ausencia en una fuente de campañas no genera
    etiquetas cero.
    """

    def __init__(self, paths: ProjectPaths) -> None:
        super().__init__(paths.raw_dir / "nhtsa_bulk", request_delay_seconds=0)
        self.store = BulkRecallStore(paths)

    def fetch_independent_vehicle_catalog(self, vehicles: Any, *, continue_on_error: bool = True) -> NHTSACatalogBatchResult:
        """Expone todas las identidades oficiales; el cruce exige misma marca y año."""
        return NHTSACatalogBatchResult(self.store.catalog(), [])

    def fetch_vehicle(self, make: object, model: object, year: int) -> NHTSAFetchResult:
        """Lee evidencia de campañas sin consultar la red ni fabricar ceros."""
        payload = self.store.payload(str(make), str(model), year)
        if payload is None:
            raise DataSourceError("Vehicle absent or ambiguous in the official bulk snapshot; recalls unknown.")
        return NHTSAFetchResult(str(make), str(model), year, payload, self.store.path, True)
