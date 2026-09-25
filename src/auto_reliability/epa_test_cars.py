"""Evidencia de potencia de ensayos EPA aislada del modelo productivo. Las filas son pruebas,
no versiones comerciales. Conserva la potencia declarada; no infiere potencia híbrida total,
CV, carrocería ni cobertura de todas las versiones.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from .config import ProjectPaths
from .contracts import require_columns
from .data_sources import DataSourceError
from .fuel_economy import digest
from .storage import atomic_destination, atomic_json

SOURCE_PAGE = "https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy"
YEAR_URLS = {
    2018: "https://www.epa.gov/sites/default/files/2018-10/18tstcar-2018-10-24.xlsx",
    2019: "https://www.epa.gov/sites/default/files/2020-10/19tstcar-2020-10-02.xlsx",
    2020: "https://www.epa.gov/sites/default/files/2021-03/20tstcar-2021-03-02.xlsx",
    2021: "https://www.epa.gov/system/files/documents/2022-04/21-tstcar-2022-04-15.xlsx",
    2022: "https://www.epa.gov/system/files/documents/2023-06/22-testcar-2023-06-13.xlsx",
    2023: "https://www.epa.gov/system/files/documents/2024-05/23-testcar-2024-05-17_0.xlsx",
    2024: "https://www.epa.gov/system/files/documents/2025-05/24-testcar-2025-05.xlsx",
    2025: "https://www.epa.gov/system/files/documents/2026-01/25-testcar-2026-01-21.xlsx",
    2026: "https://www.epa.gov/system/files/documents/2026-01/26-testcar-2026-01-21.xlsx",
}
COLUMNS = {
    "Model Year": "year", "Represented Test Veh Make": "make",
    "Represented Test Veh Model": "model", "Test Vehicle ID": "test_vehicle_id",
    "Test Veh Configuration #": "configuration", "Rated Horsepower": "rated_hp_original",
    "# of Cylinders and Rotors": "cylinders_or_rotors_original",
    "Vehicle Type": "vehicle_type", "Test Number": "test_number",
    "Test Fuel Type Description": "test_fuel", "Test Category": "test_category",
}


def normalize_tests(source: pd.DataFrame, year: int) -> pd.DataFrame:
    """Conserva cada ensayo y señala potencia inutilizable sin fabricar claves de versiones."""
    require_columns(source, tuple(COLUMNS), context="EPA test car workbook")
    frame = source[list(COLUMNS)].rename(columns=COLUMNS).copy()
    frame.insert(0, "source_excel_row", np.arange(len(frame)) + 2)
    years = pd.to_numeric(frame.year, errors="coerce")
    if frame.empty or not years.eq(year).all():
        raise DataSourceError("EPA test workbook has missing or unexpected model years.")
    for key in ("make", "model", "test_vehicle_id", "test_number", "configuration"):
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise DataSourceError(f"EPA test workbook has missing {key}.")
        frame[key] = frame[key].astype(str).str.strip()
    frame["year"] = years.astype(int)
    power = pd.to_numeric(frame.rated_hp_original, errors="coerce")
    frame["rated_hp"] = power.where(np.isfinite(power) & power.gt(0))
    frame["power_status"] = np.where(frame.rated_hp.notna(), "source_rated_hp", "missing_or_invalid")
    frame["prediction_eligible"] = False
    # Conservar tipos de celdas mixtos como texto para estabilizar el esquema Parquet.
    for column in ("rated_hp_original", "cylinders_or_rotors_original"):
        frame[column] = frame[column].astype("string")
    return frame.reset_index(drop=True)


def read_tests(path: Path, year: int) -> pd.DataFrame:
    """Lee valores de celdas y limita tamaño XLSX expandido antes de asignar memoria al lector."""
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 128 * 1024 * 1024:
            raise DataSourceError("EPA workbook exceeds expanded size limit.")
    return normalize_tests(pd.read_excel(path, engine="openpyxl"), year)


def configuration_evidence(tests: pd.DataFrame) -> pd.DataFrame:
    """Deduplica ciclos por vehículo/configuración y expone contradicciones. Una configuración
    inequívoca no representa la mediana comercial completa; potencias contradictorias quedan
    ausentes para revisión.
    """
    keys = ["year", "make", "model", "test_vehicle_id", "configuration"]
    result = tests.groupby(keys, as_index=False, dropna=False).agg(
        test_rows=("source_excel_row", "size"), observed_power_values=("rated_hp", "nunique"),
        rated_hp=("rated_hp", "first"),
    )
    result.loc[result.observed_power_values.ne(1), "rated_hp"] = np.nan
    result["power_status"] = np.select(
        [result.observed_power_values.eq(0), result.observed_power_values.gt(1)],
        ["missing", "conflicting_tests"], default="unique_test_configuration_power",
    )
    result["prediction_eligible"] = False
    return result


def acquire_test_cars(paths: ProjectPaths, years: list[int]) -> list[dict[str, object]]:
    """Descarga publicaciones autorizadas con manifiestos inmutables y conserva éxitos
    parciales. No promueve a Gold ni transfiere valores automáticamente entre fuentes. Al
    repetir, verifica hashes y reutiliza años descargados.
    """
    if not years or set(years) - YEAR_URLS.keys():
        raise ValueError("Supported EPA test years: 2018–2026.")
    summaries: list[dict[str, object]] = []
    snapshot = datetime.now(timezone.utc).date().isoformat()
    for year in sorted(set(years)):
        directory = paths.raw_dir / "epa_test_cars" / snapshot / str(year)
        directory.mkdir(parents=True, exist_ok=True)
        path, manifest_path = directory / "tests.xlsx", directory / "manifest.json"
        with FileLock(str(directory / "download.lock"), timeout=30):
            if path.exists():
                if not manifest_path.exists():
                    raise DataSourceError("EPA test snapshot lacks provenance; review interrupted acquisition.")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if digest(path) != manifest["sha256"]:
                    raise DataSourceError("EPA test snapshot hash mismatch.")
                tests = read_tests(path, year)
            else:
                if manifest_path.exists():
                    raise DataSourceError("EPA test manifest without source; recovery review required.")
                with atomic_destination(path, immutable=True) as temporary:
                    with requests.get(YEAR_URLS[year], stream=True, timeout=(10, 60)) as response:
                        response.raise_for_status()
                        size = 0
                        with temporary.open("wb") as stream:
                            for chunk in response.iter_content(65536):
                                size += len(chunk)
                                if size > 16 * 1024 * 1024:
                                    raise DataSourceError("EPA test workbook exceeds download limit.")
                                stream.write(chunk)
                    tests = read_tests(temporary, year)
                manifest = {"url": YEAR_URLS[year], "source_page": SOURCE_PAGE,
                            "sha256": digest(path), "year": year,
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "note": "Test configurations only; rated HP unit preserved without conversion. Redistribution review pending."}
                atomic_json(manifest_path, manifest, immutable=True)
            tests["source_sha256"] = manifest["sha256"]
            output = paths.processed_dir / "epa_test_cars" / snapshot / str(year)
            configs = configuration_evidence(tests)
            for filename, frame in (("tests.parquet", tests), ("configurations.parquet", configs)):
                with atomic_destination(output / filename) as temporary:
                    frame.to_parquet(temporary, index=False)
            summary = {"year": year, "test_rows": len(tests), "configurations": len(configs),
                       "models": len(tests[["make", "model"]].drop_duplicates()),
                       "usable_power_configurations": int(configs.rated_hp.notna().sum()),
                       "conflicting_configurations": int(configs.power_status.eq("conflicting_tests").sum()),
                       "source_manifest": str(manifest_path), "production_promoted": False}
            atomic_json(output / "coverage.json", summary)
            summaries.append(summary)
    return summaries
