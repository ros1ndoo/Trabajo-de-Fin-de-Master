"""Descarga la publicación original de CooperUnion sin credenciales ni servidores espejo."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .data_sources import DataSourceError

COOPERUNION_URL = "https://www.kaggle.com/api/v1/datasets/download/CooperUnion/cardataset"
COOPERUNION_PAGE = "https://www.kaggle.com/datasets/CooperUnion/cardataset"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


def download_cooperunion(raw_directory: Path, *, session=None) -> Path:
    """Conserva un CSV inmutable y su manifiesto de procedencia desde el ZIP oficial. No extrae
    rutas del archivo: lee únicamente el CSV y verifica tamaño y esquema antes de crear
    destinos. Reutiliza los bytes existentes sin sobrescribirlos.
    """
    destination = Path(raw_directory) / "cooperunion_car_features.csv"
    if destination.exists():
        return destination
    client = session or requests.Session()
    response = client.get(COOPERUNION_URL, timeout=(10, 60), stream=True)
    response.raise_for_status()
    chunks, size = [], 0
    for chunk in response.iter_content(chunk_size=65536):
        size += len(chunk)
        if size > MAX_DOWNLOAD_BYTES:
            raise DataSourceError("CooperUnion archive exceeds the download size limit.")
        chunks.append(chunk)
    archive_bytes = b"".join(chunks)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
            if len(members) != 1 or members[0].file_size > MAX_DOWNLOAD_BYTES:
                raise DataSourceError("Expected exactly one bounded CSV in the CooperUnion release.")
            csv_bytes = archive.read(members[0])
    except zipfile.BadZipFile as exc:
        raise DataSourceError("Kaggle did not return a ZIP. Supply a downloaded CSV with --technical-csv.") from exc
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    if not {"Make", "Model", "Year", "Engine HP", "Engine Cylinders"}.issubset(frame.columns):
        raise DataSourceError("Downloaded CooperUnion data does not match the documented schema.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(csv_bytes)
    provenance = {
        "source": COOPERUNION_PAGE, "download_url": COOPERUNION_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(), "rows": len(frame),
        "min_year": int(frame.Year.min()), "max_year": int(frame.Year.max()),
        "note": "Static technical specifications; not a current vehicle catalogue. Retain original source attribution.",
    }
    with destination.with_suffix(".provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False)
    return destination
