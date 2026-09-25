"""Ingesta de fuentes de vehículos estadounidenses. Separa adquisición y transformación:
conserva respuestas NHTSA como JSON originales antes de normalizar, permitiendo repetir la
limpieza sin volver a consultar la API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from filelock import FileLock, Timeout

from .config import MIN_MODEL_YEAR
from .contracts import canonical_text, vehicle_id
from .request_state import RequestState
from .storage import atomic_json

LOGGER = logging.getLogger(__name__)
NHTSA_RECALLS_ENDPOINT = "https://api.nhtsa.gov/recalls/recallsByVehicle"
# El catálogo de recalls de NHTSA sirve para explorar campañas,
# pero no certifica vehículos sin recalls: issueType=r puede omitirlos.
# vPIC es un catálogo oficial independiente de marca, modelo y año;
# acredita identidad, no cobertura completa ni una etiqueta cero por sí solo.
NHTSA_RECALL_MODELS_ENDPOINT = "https://api.nhtsa.gov/products/vehicle/models"
NHTSA_VPIC_MODELS_ENDPOINT = "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear"
# NIST SP 811, apéndice B.9: HP mecánico estadounidense y caballo métrico (CV).
HP_TO_CV = 745.6999 / 735.4988
HP_ALIASES = frozenset({"engine_hp", "engine_horsepower", "horsepower", "hp"})


class DataSourceError(RuntimeError):
    """Indica que una fuente no puede satisfacer el contrato de datos."""


class NHTSARequestError(DataSourceError):
    """Fallo estructurado de consulta: un error HTTP nunca representa cero recalls."""

    def __init__(self, message: str, *, status: int | None, attempts: int, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts
        self.retry_after = retry_after
        self.transient = status is None or status == 429 or status >= 500


def retry_after_seconds(value: str | None) -> float | None:
    """Interpreta Retry-After como demora o fecha, ignorando valores malformados o no finitos."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date_value = parsedate_to_datetime(value)
            if date_value.tzinfo is None:
                date_value = date_value.replace(tzinfo=timezone.utc)
            seconds = (date_value - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    import math
    return max(0., seconds) if math.isfinite(seconds) else None


@dataclass(frozen=True)
class NHTSAFetchResult:
    """Resultado original inmutable de una consulta NHTSA de vehículo."""

    make: str
    model: str
    year: int
    payload: dict[str, Any]
    cache_path: Path
    from_cache: bool
    catalog_verified: bool | None = None
    query_status: str = "unverified"

    @property
    def result_count(self) -> int:
        """Devuelve el conteo de la API tolerando variantes históricas de mayúsculas."""

        value = self.payload.get("Count", self.payload.get("count"))
        try:
            return int(value)
        except (TypeError, ValueError):
            records = self.payload.get("Results", self.payload.get("results", []))
            return len(records) if isinstance(records, list) else 0


@dataclass
class NHTSABatchResult:
    """Registros normalizados y universo consultado. vehicle_index conserva respuestas sin
    campañas; faltar un resultado no demuestra ausencia ni un fallo de cruce técnico. Su
    elegibilidad para Gold depende de identidad y observación acreditadas, no solo del
    conteo.
    """

    records: pd.DataFrame
    vehicle_index: pd.DataFrame
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class NHTSACatalogFetchResult:
    """Instantánea inmutable del catálogo NHTSA de marca, modelo y año."""

    make: str
    year: int
    payload: dict[str, Any]
    cache_path: Path
    from_cache: bool
    source: str


@dataclass
class NHTSACatalogBatchResult:
    """Catálogo independiente de modelos NHTSA y fallos recuperables del lote."""

    vehicles: pd.DataFrame
    failures: list[dict[str, str]] = field(default_factory=list)


def _normalised_column_name(name: object) -> str:
    return canonical_text(name).replace(" ", "_")


# El archivo de CooperUnion utiliza nombres como Engine HP y Market Category.
# Los alias permiten también leer exportaciones con columnas en castellano
# sin requerir un cuaderno de preprocesamiento específico.
COOPERUNION_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "marca": ("make", "manufacturer", "brand", "marca", "fabricante"),
    "modelo": ("model", "modelo"),
    "ano_fabricacion": (
        "year",
        "model_year",
        "modelyear",
        "ano",
        "ano_fabricacion",
        "año",
    ),
    "potencia_cv": (
        "engine_hp",
        "engine_horsepower",
        "horsepower",
        "hp",
        "potencia_cv",
        "potencia",
    ),
    "cilindros": (
        "engine_cylinders",
        "cylinders",
        "cylinder_count",
        "cilindros",
    ),
    "categoria_vehiculo": (
        # Vehicle Style es el segmento más cercano a la carrocería comercial
        # en CooperUnion (por ejemplo, 4dr SUV); tiene prioridad frente a
        # Market Category, más amplio y con varias etiquetas, si ambos existen.
        "vehicle_style",
        "estilo_vehiculo",
        "market_category",
        "marketcategory",
        "vehicle_category",
        "vehicle_type",
        "category",
        "categoria_vehiculo",
        "categoria",
        "segment",
    ),
    "msrp": ("msrp", "manufacturer_suggested_retail_price", "precio_msrp"),
    "combustible": ("engine_fuel_type", "fuel_type", "combustible"),
    "transmision": ("transmission_type", "transmission", "transmision"),
    "tamano_vehiculo": ("vehicle_size", "tamano_vehiculo"),
    "estilo_vehiculo": ("vehicle_style", "estilo_vehiculo"),
}


def _first_non_empty(series: pd.Series) -> object:
    """Devuelve el primer valor significativo entre alias duplicados de fuente."""

    for value in series:
        if pd.notna(value) and str(value).strip():
            return value
    return pd.NA


def _coalesce_alias_columns(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    # Respetar la prioridad de los alias, no el orden de las columnas del CSV.
    # Es especialmente importante para el respaldo Vehicle Style -> Market Category.
    existing = [alias for alias in aliases if alias in frame.columns]
    if not existing:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    if len(existing) == 1:
        return frame[existing[0]].copy()
    return frame[existing].apply(_first_non_empty, axis=1)


def _normalise_source_text(value: object) -> str | pd._libs.missing.NAType:
    """Normaliza texto visible sin convertir ausencias en la cadena nan."""

    if value is None or pd.isna(value):
        return pd.NA
    cleaned = " ".join(str(value).strip().split())
    return cleaned if cleaned else pd.NA


def normalise_cooperunion_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Adapta un CSV CooperUnion/Kaggle al contrato de fuente. Conserva la granularidad
    original de versión/motor; prepare_technical_specs agrega por modelo-año y los
    estimadores imputan solo con entrenamiento. Interpreta HP estadounidense como potencia
    mecánica, no CV métrico. Tolera campos técnicos opcionales, pero rechaza ausencia de
    marca, modelo o año porque esas claves no admiten reparación justificable.
    """

    if source.empty:
        raise DataSourceError("The CooperUnion CSV is empty.")

    frame = source.copy()
    frame.columns = [_normalised_column_name(column) for column in frame.columns]

    output = pd.DataFrame(index=frame.index)
    for target, aliases in COOPERUNION_COLUMN_ALIASES.items():
        output[target] = _coalesce_alias_columns(frame, aliases)

    # Convertir cada alias antes de combinarlo: una exportación mixta HP/CV
    # no debe aplicar el factor HP a valores ya expresados en CV.
    output["potencia_cv"] = float("nan")
    output["potencia_original"] = float("nan")
    output["columna_potencia_origen"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    output["unidad_potencia_origen"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for alias in COOPERUNION_COLUMN_ALIASES["potencia_cv"]:
        if alias not in frame:
            continue
        values = pd.to_numeric(frame[alias], errors="coerce")
        selected = output["potencia_cv"].isna() & values.notna()
        is_hp = alias in HP_ALIASES
        output.loc[selected, "potencia_original"] = values.loc[selected]
        output.loc[selected, "columna_potencia_origen"] = alias
        output.loc[selected, "unidad_potencia_origen"] = "hp_mechanical" if is_hp else "CV"
        output.loc[selected, "potencia_cv"] = values.loc[selected] * (HP_TO_CV if is_hp else 1.0)
    output["unidad_potencia"] = "CV"

    required = ("marca", "modelo", "ano_fabricacion")
    missing = [column for column in required if output[column].isna().all()]
    if missing:
        available = ", ".join(map(str, source.columns))
        raise DataSourceError(
            "CooperUnion CSV is missing required key columns: "
            f"{', '.join(missing)}. Available columns: {available}"
        )

    for column in ("marca", "modelo", "categoria_vehiculo", "combustible", "transmision", "tamano_vehiculo", "estilo_vehiculo"):
        output[column] = output[column].map(_normalise_source_text)

    output["ano_fabricacion"] = pd.to_numeric(output["ano_fabricacion"], errors="coerce").astype("Int64")
    for column in ("potencia_cv", "cilindros", "msrp"):
        output[column] = pd.to_numeric(output[column], errors="coerce")

    # Conservar la posición original para trazar cada fila hasta su fuente.
    output.insert(0, "fila_origen", range(len(output)))
    output["fuente_tecnica"] = "cooperunion"
    return output


# Normalizar variantes americanas/británicas y alias habituales solo
# al comparar. La interfaz conserva el texto original limpio.
MAKE_ALIASES: dict[str, str] = {
    "vw": "volkswagen",
    "volkswagen": "volkswagen",
    "mercedes benz": "mercedes benz",
    "mercedesbenz": "mercedes benz",
    "chevy": "chevrolet",
}


def comparison_make(value: object) -> str:
    """Devuelve una clave conservadora y estable de marca para cruces."""

    key = canonical_text(value)
    return MAKE_ALIASES.get(key, key)


def comparison_model(value: object) -> str:
    """Devuelve una clave de modelo insensible a puntuación para cruce difuso."""

    key = canonical_text(value)
    # Los separadores cosméticos no cambian la familia alfanumérica:
    # F-150 == F150, CX-5 == CX5. Los números siguen siendo significativos.
    return re.sub(r"(?<=[a-z])\s+(?=\d)|(?<=\d)\s+(?=[a-z])", "", key)


def ingest_cooperunion_csv(
    csv_path: str | Path,
    *,
    min_year: int = MIN_MODEL_YEAR,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Lee y valida mínimamente Car Features and MSRP de CooperUnion. Las fuentes son de solo
    lectura. Excluye claves inválidas y años anteriores a 1995, informándolos en atributos
    del dataframe para conservar un informe de calidad explícito.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CooperUnion CSV does not exist: {path}")

    try:
        source = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # Las exportaciones de Kaggle suelen usar UTF-8; latin-1 permite leer
        # copias locales alternativas sin modificar el archivo original.
        source = pd.read_csv(path, encoding="latin-1")

    frame = normalise_cooperunion_columns(source)
    initial_rows = len(frame)
    valid_key = frame["marca"].notna() & frame["modelo"].notna() & frame["ano_fabricacion"].notna()
    valid_year = frame["ano_fabricacion"].ge(min_year)
    filtered = frame.loc[valid_key & valid_year].copy()
    filtered["ano_fabricacion"] = filtered["ano_fabricacion"].astype(int)
    filtered["marca_normalizada"] = filtered["marca"].map(comparison_make)
    filtered["modelo_normalizado"] = filtered["modelo"].map(comparison_model)
    filtered.attrs["quality_report"] = {
        "source_rows": initial_rows,
        "kept_rows": len(filtered),
        "dropped_invalid_key_or_year": int(initial_rows - len(filtered)),
        "dropped_missing_identity_or_year": int((~valid_key).sum()),
        "dropped_before_min_year": int((valid_key & ~valid_year).sum()),
        "min_year": min_year,
        "source_path": str(path),
    }
    return filtered.reset_index(drop=True)


def _safe_filename_part(value: object) -> str:
    """Crea un componente de nombre de archivo legible y seguro."""

    value = canonical_text(value).replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", value) or "unknown"


def _query_digest(*parts: object) -> str:
    """Evita colisiones de nombres sin introducir texto no confiable en rutas."""

    source = "\x1f".join(str(part).strip().casefold() for part in parts)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def nhtsa_cache_filename(make: object, model: object, year: int) -> str:
    """Devuelve un nombre de caché original seguro y resistente a colisiones para una consulta."""

    return (
        f"nhtsa_recalls_{_safe_filename_part(make)}_{_safe_filename_part(model)}_"
        f"{int(year)}_{_query_digest(make, model, year)}.json"
    )


def nhtsa_catalog_cache_filename(make: object, year: int, *, source: str = "vpic") -> str:
    """Devuelve un nombre inmutable seguro para una respuesta de catálogo NHTSA."""

    clean_source = _safe_filename_part(source)
    return (
        f"nhtsa_{clean_source}_models_{_safe_filename_part(make)}_{int(year)}_"
        f"{_query_digest(source, make, year)}.json"
    )


class NHTSARecallClient:
    """Cliente de la API NHTSA con limitación de solicitudes y prioridad de caché. Permite
    inyectar una sesión compatible con requests.Session y una función de espera para probar
    reintentos deterministas. Nunca sobrescribe JSON originales: son evidencia inmutable.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        endpoint: str = NHTSA_RECALLS_ENDPOINT,
        vpic_models_endpoint: str = NHTSA_VPIC_MODELS_ENDPOINT,
        recall_models_endpoint: str = NHTSA_RECALL_MODELS_ENDPOINT,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
        request_delay_seconds: float = 0.20,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        coordination_dir: str | Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.endpoint = endpoint
        self.vpic_models_endpoint = vpic_models_endpoint.rstrip("/")
        self.recall_models_endpoint = recall_models_endpoint
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.base_backoff_seconds = max(0.0, float(base_backoff_seconds))
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.sleeper = sleeper
        self.clock = clock
        self.coordination_dir = (Path(coordination_dir) if coordination_dir is not None else
                                 self.raw_dir.parent if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.raw_dir.name)
                                 else self.raw_dir)
        self.request_state = RequestState(self.coordination_dir / ".nhtsa-operations.sqlite")
        self._request_attempts = 0

    def cache_path_for(self, make: object, model: object, year: int) -> Path:
        return self.raw_dir / nhtsa_cache_filename(make, model, year)

    def catalog_cache_path_for(self, make: object, year: int, *, source: str = "vpic") -> Path:
        """Devuelve la ubicación inmutable de una solicitud de catálogo."""

        return self.raw_dir / nhtsa_catalog_cache_filename(make, year, source=source)

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataSourceError(
                f"Raw NHTSA cache is unreadable and will not be overwritten: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise DataSourceError(f"Raw NHTSA cache has an invalid payload shape: {path}")
        return payload

    @staticmethod
    def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
        """Crea JSON original atómicamente, conservando evidencia existente."""

        atomic_json(path, payload, immutable=True)

    def _request_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, object] | None,
        request_label: str,
    ) -> dict[str, Any]:
        """Consulta JSON por GET con reintentos exponenciales acotados ante límites NHTSA."""

        last_error: Exception | None = None
        response_status: int | None = None
        retry_after: float | None = None
        for attempt in range(self.max_retries + 1):
            response = None
            retry_after = None
            self._request_attempts = attempt + 1
            try:
                response = self.session.get(
                    endpoint,
                    params=dict(params or {}),
                    timeout=self.timeout_seconds,
                    headers={"Accept": "application/json", "User-Agent": "auto-reliability-thesis/0.1"},
                )
                status = getattr(response, "status_code", None)
                response_status = status
                retry_after = retry_after_seconds(getattr(response, "headers", {}).get("Retry-After"))
                if status in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"Transient NHTSA response: HTTP {status}", response=response)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise DataSourceError("NHTSA returned a non-object JSON payload.")
                return payload
            except (requests.RequestException, ValueError, DataSourceError) as exc:
                last_error = exc
                # Las consultas inválidas o desconocidas son errores permanentes. Reintentar
                # solo fallos de conexión y respuestas transitorias o de límite de solicitudes.
                response_status = getattr(getattr(exc, "response", None), "status_code", None)
                if response_status is None and response is not None:
                    response_status = getattr(response, "status_code", None)
                if response_status is not None and 400 <= response_status < 500 and response_status != 429:
                    break
                if attempt >= self.max_retries:
                    break
                # Respetar esperas largas del servidor devolviendo un error recuperable,
                # sin reintentar antes de tiempo ni bloquear indefinidamente la sesión.
                if retry_after is not None and retry_after > 30:
                    break
                delay = max(retry_after or 0., self.base_backoff_seconds * (2**attempt))
                delay += random.uniform(0, min(.5, delay * .1))
                LOGGER.warning(
                    "NHTSA request failed for %s (attempt %s/%s); retrying in %.2fs: %s",
                    request_label,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                    exc,
                )
                self.sleeper(delay)
            finally:
                if response is not None and callable(getattr(response, "close", None)):
                    response.close()
        raise NHTSARequestError(
            f"NHTSA request failed after {attempt + 1} attempt(s) for {request_label}: {last_error}",
            status=response_status, attempts=attempt + 1, retry_after=retry_after,
        )

    def _fetch_cached_json(
        self, cache_path: Path, endpoint: str, *, params: Mapping[str, object] | None,
        request_label: str,
    ) -> tuple[dict[str, Any], bool]:
        """Convierte fallos de bloqueo o almacenamiento en errores recuperables de fuente."""
        try:
            return self._fetch_cached_json_locked(cache_path, endpoint, params=params, request_label=request_label)
        except (Timeout, sqlite3.Error, OSError) as exc:
            raise DataSourceError("NHTSA request coordination unavailable; retry later.") from exc

    def _fetch_cached_json_locked(
        self,
        cache_path: Path,
        endpoint: str,
        *,
        params: Mapping[str, object] | None,
        request_label: str,
    ) -> tuple[dict[str, Any], bool]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # El sistema operativo libera los bloqueos al terminar el proceso. Revisar
        # la caché tras adquirir el bloqueo evita solicitudes simultáneas duplicadas.
        with FileLock(str(cache_path) + ".lock", timeout=30):
            if cache_path.exists():
                payload = self._read_cache(cache_path)
                _payload_results(payload)
                return payload, True
            self.coordination_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self.coordination_dir / ".nhtsa-request.lock"), timeout=30):
                remaining = self.request_state.remaining(endpoint, self.clock())
                if remaining > 0:
                    raise NHTSARequestError("NHTSA shared cooldown active; use cached evidence or retry later.",
                                            status=429, attempts=0, retry_after=remaining)
                if self.request_delay_seconds:
                    self.sleeper(self.request_delay_seconds)
                try:
                    payload = self._request_json(endpoint, params=params, request_label=request_label)
                    _payload_results(payload)
                except NHTSARequestError as exc:
                    # Un tiempo de espera agotado no demuestra limitación de peticiones del servidor.
                    delay = (max(exc.retry_after or 0., self.base_backoff_seconds)
                             if exc.transient and (exc.retry_after is not None or exc.status in {429, 503}) else 0.)
                    self.request_state.record(endpoint, params, now=self.clock(),
                        state="retryable" if exc.transient else "permanent_failure", status=exc.status,
                        attempts=exc.attempts, delay=delay, error_type=type(exc).__name__)
                    raise
                except DataSourceError as exc:
                    self.request_state.record(endpoint, params, now=self.clock(), state="permanent_failure",
                                              status=200, attempts=self._request_attempts, error_type=type(exc).__name__)
                    raise
                self.request_state.record(endpoint, params, now=self.clock(), state="succeeded", status=200,
                                          attempts=self._request_attempts)
            self._write_json_once(cache_path, payload)
            return payload, False

    def fetch_vehicle(self, make: object, model: object, year: int) -> NHTSAFetchResult:
        """Obtiene la respuesta original de recalls priorizando la caché inmutable."""

        make_text, model_text, model_year = str(make).strip(), str(model).strip(), int(year)
        if not make_text or not model_text:
            raise ValueError("NHTSA requests require non-empty make and model.")
        if model_year < MIN_MODEL_YEAR:
            raise ValueError(f"NHTSA requests are restricted to {MIN_MODEL_YEAR}+ model years.")

        cache_path = self.cache_path_for(make_text, model_text, model_year)
        params = {"make": make_text, "model": model_text, "modelYear": model_year}
        payload, from_cache = self._fetch_cached_json(
            cache_path,
            self.endpoint,
            params=params,
            request_label=f"recalls {make_text} {model_text} {model_year}",
        )
        return NHTSAFetchResult(
            make=make_text,
            model=model_text,
            year=model_year,
            payload=payload,
            cache_path=cache_path,
            from_cache=from_cache,
        )

    def fetch_vpic_models_for_make_year(self, make: object, year: int) -> NHTSACatalogFetchResult:
        """Obtiene una instantánea independiente de vPIC. Enumera marcas/modelos/años sin
        depender de campañas actuales; su presencia aporta identidad, no equivalencia de
        nombres con el servicio de recalls ni prueba de cero.
        """

        make_text, model_year = str(make).strip(), int(year)
        if not make_text:
            raise ValueError("NHTSA catalogue requests require a non-empty make.")
        if model_year < MIN_MODEL_YEAR:
            raise ValueError(f"NHTSA catalogue requests are restricted to {MIN_MODEL_YEAR}+ model years.")
        cache_path = self.catalog_cache_path_for(make_text, model_year, source="vpic")
        endpoint = f"{self.vpic_models_endpoint}/make/{quote(make_text, safe='')}/modelyear/{model_year}"
        payload, from_cache = self._fetch_cached_json(
            cache_path,
            endpoint,
            params={"format": "json"},
            request_label=f"vPIC catalogue {make_text} {model_year}",
        )
        return NHTSACatalogFetchResult(
            make=make_text,
            year=model_year,
            payload=payload,
            cache_path=cache_path,
            from_cache=from_cache,
            source="vpic",
        )

    def fetch_recall_models_for_make_year(self, make: object, year: int) -> NHTSACatalogFetchResult:
        """Obtiene nombres oficiales del catálogo de recalls. Sirve para descubrir
        nomenclaturas asociadas a campañas, no para probar ausencia de recalls en vehículos
        omitidos. vPIC aporta identidad por separado, no etiquetas cero.
        """

        make_text, model_year = str(make).strip(), int(year)
        if not make_text:
            raise ValueError("NHTSA recall-model requests require a non-empty make.")
        cache_path = self.catalog_cache_path_for(make_text, model_year, source="recall_catalog")
        payload, from_cache = self._fetch_cached_json(
            cache_path,
            self.recall_models_endpoint,
            params={"modelYear": model_year, "make": make_text, "issueType": "r"},
            request_label=f"recall model catalogue {make_text} {model_year}",
        )
        return NHTSACatalogFetchResult(
            make=make_text,
            year=model_year,
            payload=payload,
            cache_path=cache_path,
            from_cache=from_cache,
            source="recall_catalog",
        )

    # Alias legible para cuadernos y documentación anterior.
    fetch_recalls = fetch_vehicle

    def fetch_independent_vehicle_catalog(
        self,
        vehicles: pd.DataFrame | Iterable[Mapping[str, object] | Sequence[object]],
        *,
        continue_on_error: bool = True,
    ) -> NHTSACatalogBatchResult:
        """Construye catálogos independientes de productos de recalls y vPIC. Consulta una vez
        por marca/año, no por versión técnica; el proceso es acotado y reanudable al
        conservar cada JSON original una vez en data/raw.
        """

        make_year_queries = _normalise_make_year_queries(vehicles)
        catalog_frames: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        for position, query in enumerate(make_year_queries):
            # El catálogo de recalls aporta los nombres que espera su servicio.
            # vPIC complementa identidades fuera de ese catálogo; una respuesta vacía
            # basada solo en vPIC sigue siendo incierta y nunca genera una etiqueta.
            for fetcher in (self.fetch_recall_models_for_make_year, self.fetch_vpic_models_for_make_year):
                try:
                    snapshot = fetcher(query["marca"], int(query["ano_fabricacion"]))
                    catalog_frames.append(
                        normalise_nhtsa_catalog_response(
                            snapshot.payload,
                            make=snapshot.make,
                            year=snapshot.year,
                            source=snapshot.source,
                            cache_path=snapshot.cache_path,
                            from_cache=snapshot.from_cache,
                        )
                    )
                except (DataSourceError, ValueError, requests.RequestException) as exc:
                    failure = {
                        "marca": str(query["marca"]),
                        "ano_fabricacion": str(query["ano_fabricacion"]),
                        "source": fetcher.__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    LOGGER.error("Skipping failed NHTSA catalogue query: %s", failure)
                    if not continue_on_error:
                        raise
                if self.request_delay_seconds:
                    self.sleeper(self.request_delay_seconds)
            if position < len(make_year_queries) - 1 and self.request_delay_seconds:
                self.sleeper(self.request_delay_seconds)
        catalog = (
            pd.concat(catalog_frames, ignore_index=True)
            if catalog_frames
            else empty_nhtsa_catalog_frame()
        )
        if not catalog.empty:
            # Ante nombres equivalentes, priorizar la nomenclatura del catálogo de recalls.
            catalog = catalog.drop_duplicates(
                ["marca_normalizada", "modelo_normalizado", "ano_fabricacion"], keep="first"
            ).reset_index(drop=True)
        return NHTSACatalogBatchResult(catalog, failures)

    # Alias breve para llamadas desde el servicio y los cuadernos.
    fetch_vehicle_catalog = fetch_independent_vehicle_catalog

    def fetch_catalog_verified_recalls(
        self,
        catalog_vehicles: pd.DataFrame | Iterable[Mapping[str, object]],
        *,
        continue_on_error: bool = True,
    ) -> NHTSABatchResult:
        """Consulta campañas de vehículos verificados en un catálogo independiente. Solo
        etiqueta valid_zero_recalls si también existe la identidad en la nomenclatura de
        productos de recalls. vPIC por sí solo no acredita esa equivalencia; así se evita
        convertir nombres mal escritos en falsos ceros.
        """

        queries = _normalise_catalog_vehicle_queries(catalog_vehicles)
        records: list[pd.DataFrame] = []
        index_rows: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []
        for position, query in enumerate(queries):
            try:
                result = self.fetch_vehicle(query["marca"], query["modelo"], int(query["ano_fabricacion"]))
                nhtsa_id = str(query["nhtsa_vehicle_id"])
                normalised = normalise_nhtsa_response(
                    result.payload,
                    make=query["marca"],
                    model=query["modelo"],
                    year=int(query["ano_fabricacion"]),
                    nhtsa_vehicle_id=nhtsa_id,
                )
                if not normalised.empty:
                    records.append(normalised)
                result_count = result.result_count
                zero_verified = result_count == 0 and query.get("catalog_source") == "recall_catalog"
                index_rows.append(
                    {
                        "nhtsa_vehicle_id": nhtsa_id,
                        "marca": query["marca"],
                        "modelo": query["modelo"],
                        "ano_fabricacion": int(query["ano_fabricacion"]),
                        "marca_normalizada": comparison_make(query["marca"]),
                        "modelo_normalizado": comparison_model(query["modelo"]),
                        "numero_recalls_api": result_count,
                        "cache_path": str(result.cache_path),
                        "desde_cache": result.from_cache,
                        "catalog_verified": True,
                        "catalog_source": query.get("catalog_source", "vpic"),
                        "query_status": "valid_with_recalls" if result_count else (
                            "valid_zero_recalls" if zero_verified else "unverified_zero_result"
                        ),
                        "resultado_cero_validado": zero_verified,
                    }
                )
            except Exception as exc:
                failure = {
                    "marca": str(query["marca"]),
                    "modelo": str(query["modelo"]),
                    "ano_fabricacion": str(query["ano_fabricacion"]),
                    "error": str(exc),
                }
                failures.append(failure)
                LOGGER.error("Skipping failed verified NHTSA recall query: %s", failure)
                if not continue_on_error:
                    raise
            if position < len(queries) - 1 and self.request_delay_seconds:
                self.sleeper(self.request_delay_seconds)
        record_frame = pd.concat(records, ignore_index=True) if records else empty_nhtsa_records_frame()
        index_frame = pd.DataFrame(index_rows, columns=NHTSA_VEHICLE_INDEX_COLUMNS)
        return NHTSABatchResult(record_frame, index_frame, failures)

    fetch_verified_recall_batch = fetch_catalog_verified_recalls

    def fetch_vehicle_batch(
        self,
        vehicles: pd.DataFrame | Iterable[Mapping[str, object] | Sequence[object]],
        *,
        continue_on_error: bool = True,
    ) -> NHTSABatchResult:
        """Consulta vehículos únicos secuencialmente con una pausa respetuosa. La secuencia
        limita solicitudes al servicio público y la caché inmutable acelera ejecuciones
        posteriores sin concurrencia.
        """

        queries = _normalise_vehicle_queries(vehicles)
        records: list[pd.DataFrame] = []
        index_rows: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []

        for position, query in enumerate(queries):
            try:
                result = self.fetch_vehicle(query["marca"], query["modelo"], int(query["ano_fabricacion"]))
                nhtsa_id = vehicle_id(result.make, result.model, result.year)
                normalised = normalise_nhtsa_response(
                    result.payload,
                    make=result.make,
                    model=result.model,
                    year=result.year,
                    nhtsa_vehicle_id=nhtsa_id,
                )
                if not normalised.empty:
                    records.append(normalised)
                index_rows.append(
                    {
                        "nhtsa_vehicle_id": nhtsa_id,
                        "marca": result.make,
                        "modelo": result.model,
                        "ano_fabricacion": result.year,
                        "marca_normalizada": comparison_make(result.make),
                        "modelo_normalizado": comparison_model(result.model),
                        "numero_recalls_api": len(normalised),
                        "cache_path": str(result.cache_path),
                        "desde_cache": result.from_cache,
                        # Las consultas directas son útiles para explorar, pero Count=0
                        # no distingue un error en el nombre de una ausencia real de campañas.
                        "catalog_verified": False,
                        "catalog_source": pd.NA,
                        "query_status": "unverified_with_recalls" if result.result_count else "unverified_zero_result",
                        "resultado_cero_validado": False,
                    }
                )
            except Exception as exc:  # conservar el lote tras una consulta fallida
                failure = {
                    "marca": str(query["marca"]),
                    "modelo": str(query["modelo"]),
                    "ano_fabricacion": str(query["ano_fabricacion"]),
                    "error": str(exc),
                }
                failures.append(failure)
                LOGGER.error("Skipping failed NHTSA query: %s", failure)
                if not continue_on_error:
                    raise

            if position < len(queries) - 1 and self.request_delay_seconds:
                self.sleeper(self.request_delay_seconds)

        empty_records = empty_nhtsa_records_frame()
        record_frame = pd.concat(records, ignore_index=True) if records else empty_records
        index_frame = pd.DataFrame(index_rows, columns=NHTSA_VEHICLE_INDEX_COLUMNS)
        return NHTSABatchResult(record_frame, index_frame, failures)


def _normalise_vehicle_queries(
    vehicles: pd.DataFrame | Iterable[Mapping[str, object] | Sequence[object]],
) -> list[dict[str, object]]:
    """Convierte formatos de entrada admitidos en consultas sin duplicados."""

    if isinstance(vehicles, pd.DataFrame):
        make_col = "marca" if "marca" in vehicles.columns else "make"
        model_col = "modelo" if "modelo" in vehicles.columns else "model"
        year_col = "ano_fabricacion" if "ano_fabricacion" in vehicles.columns else "year"
        missing = [column for column in (make_col, model_col, year_col) if column not in vehicles.columns]
        if missing:
            raise ValueError("Vehicle batch must provide marca/modelo/ano_fabricacion (or make/model/year).")
        iterable: Iterable[object] = vehicles[[make_col, model_col, year_col]].itertuples(index=False, name=None)
    else:
        iterable = vehicles

    result: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in iterable:
        if isinstance(item, Mapping):
            make = item.get("marca", item.get("make"))
            model = item.get("modelo", item.get("model"))
            year = item.get("ano_fabricacion", item.get("year"))
        else:
            try:
                make, model, year = item  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError("Each vehicle query must be a mapping or (make, model, year) sequence.") from exc
        if pd.isna(make) or pd.isna(model) or pd.isna(year):
            continue
        key = (comparison_make(make), comparison_model(model), int(year))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        result.append({"marca": str(make).strip(), "modelo": str(model).strip(), "ano_fabricacion": int(year)})
    return result


def _normalise_make_year_queries(
    vehicles: pd.DataFrame | Iterable[Mapping[str, object] | Sequence[object]],
) -> list[dict[str, object]]:
    """Extrae pares únicos marca/año de filas técnicas o diccionarios."""

    if isinstance(vehicles, pd.DataFrame):
        make_col = "marca" if "marca" in vehicles.columns else "make"
        year_col = "ano_fabricacion" if "ano_fabricacion" in vehicles.columns else "year"
        if make_col not in vehicles.columns or year_col not in vehicles.columns:
            raise ValueError("Vehicle catalogue scope requires marca and ano_fabricacion (or make and year).")
        iterable: Iterable[object] = vehicles[[make_col, year_col]].itertuples(index=False, name=None)
    else:
        iterable = vehicles
    output: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for item in iterable:
        if isinstance(item, Mapping):
            make = item.get("marca", item.get("make"))
            year = item.get("ano_fabricacion", item.get("year"))
        else:
            try:
                # Aceptar tanto la clave natural (marca, año) como
                # (marca, modelo, año), utilizada en los lotes de consultas directas.
                make, year = item[0], item[-1]  # type: ignore[index]
            except (TypeError, ValueError) as exc:
                raise ValueError("Each catalogue query must provide (make, year).") from exc
        if pd.isna(make) or pd.isna(year):
            continue
        key = (comparison_make(make), int(year))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        output.append({"marca": str(make).strip(), "ano_fabricacion": int(year)})
    return output


def _normalise_catalog_vehicle_queries(
    vehicles: pd.DataFrame | Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Convierte filas del catálogo independiente o cruces aceptados en consultas de recalls."""

    if isinstance(vehicles, pd.DataFrame):
        make_col = "marca_nhtsa" if "marca_nhtsa" in vehicles.columns else ("marca" if "marca" in vehicles.columns else "make")
        model_col = "modelo_nhtsa" if "modelo_nhtsa" in vehicles.columns else ("modelo" if "modelo" in vehicles.columns else "model")
        year_col = "ano_fabricacion"
        required = [make_col, model_col, year_col]
        if any(column not in vehicles.columns for column in required):
            raise ValueError("Verified recall queries require NHTSA make, model, and ano_fabricacion.")
        id_col = "nhtsa_vehicle_id" if "nhtsa_vehicle_id" in vehicles.columns else None
        source_col = "catalog_source" if "catalog_source" in vehicles.columns else None
        iterable = vehicles.to_dict("records")
    else:
        iterable = list(vehicles)
        make_col, model_col, year_col, id_col, source_col = "marca", "modelo", "ano_fabricacion", "nhtsa_vehicle_id", "catalog_source"
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in iterable:
        if not isinstance(record, Mapping):
            raise TypeError("Verified recall queries must be mappings or a dataframe.")
        if record.get("catalog_verified") is not True:
            raise ValueError("Verified recall queries require explicit independent catalog_verified=True.")
        # Los registros pueden utilizar nombres alternativos de columnas de cruce aceptado.
        make = record.get(make_col, record.get("marca_nhtsa", record.get("marca", record.get("make"))))
        model = record.get(model_col, record.get("modelo_nhtsa", record.get("modelo", record.get("model"))))
        year = record.get(year_col, record.get("ano_fabricacion", record.get("year")))
        if pd.isna(make) or pd.isna(model) or pd.isna(year):
            continue
        make_text, model_text, model_year = str(make).strip(), str(model).strip(), int(year)
        identifier = record.get(id_col) if id_col else record.get("nhtsa_vehicle_id")
        if identifier is None or pd.isna(identifier) or not str(identifier).strip():
            identifier = vehicle_id(make_text, model_text, model_year)
        identifier = str(identifier)
        if identifier in seen:
            continue
        seen.add(identifier)
        source = record.get(source_col) if source_col else record.get("catalog_source", "vpic")
        output.append(
            {
                "nhtsa_vehicle_id": identifier,
                "marca": make_text,
                "modelo": model_text,
                "ano_fabricacion": model_year,
                "catalog_source": "vpic" if source is None or pd.isna(source) else str(source),
            }
        )
    return output


NHTSA_RECORD_COLUMNS: tuple[str, ...] = (
    "nhtsa_vehicle_id",
    "marca",
    "modelo",
    "ano_fabricacion",
    "marca_normalizada",
    "modelo_normalizado",
    "campana_nhtsa",
    "fecha_reporte",
    "componente",
    "resumen",
    "consecuencia",
    "remedio",
    "descripcion_recall",
)

NHTSA_VEHICLE_INDEX_COLUMNS: tuple[str, ...] = (
    "nhtsa_vehicle_id",
    "marca",
    "modelo",
    "ano_fabricacion",
    "marca_normalizada",
    "modelo_normalizado",
    "numero_recalls_api",
    "cache_path",
    "desde_cache",
    "catalog_verified",
    "catalog_source",
    "query_status",
    "resultado_cero_validado",
)

NHTSA_CATALOG_COLUMNS: tuple[str, ...] = (
    "nhtsa_vehicle_id",
    "marca",
    "modelo",
    "ano_fabricacion",
    "marca_normalizada",
    "modelo_normalizado",
    "catalog_verified",
    "catalog_source",
    "catalog_cache_path",
    "catalog_desde_cache",
)


def empty_nhtsa_records_frame() -> pd.DataFrame:
    """Devuelve un dataframe NHTSA vacío con esquema estable."""

    return pd.DataFrame(columns=NHTSA_RECORD_COLUMNS)


def empty_nhtsa_catalog_frame() -> pd.DataFrame:
    """Devuelve el esquema estable de un catálogo independiente vacío."""

    return pd.DataFrame(columns=NHTSA_CATALOG_COLUMNS)


def _payload_results(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_results = payload.get("results", payload.get("Results"))
    if not isinstance(raw_results, list):
        raise DataSourceError("NHTSA payload has a missing or non-list Results field; not a zero result.")
    if any(not isinstance(result, Mapping) for result in raw_results):
        raise DataSourceError("NHTSA payload contains invalid result records.")
    declared_count = payload.get("Count", payload.get("count"))
    if declared_count is not None:
        try:
            valid_count = int(declared_count) == len(raw_results)
        except (TypeError, ValueError):
            valid_count = False
        if not valid_count:
            raise DataSourceError("NHTSA payload count disagrees with returned result records.")
    return raw_results


def _result_value(result: Mapping[str, Any], *names: str) -> object:
    for name in names:
        if name in result and result[name] is not None:
            return result[name]
        # Tolerar cambios de mayúsculas de la API sin enumerar todas las variantes.
        lowered = {str(key).lower(): value for key, value in result.items()}
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return pd.NA


def normalise_nhtsa_catalog_response(
    payload: Mapping[str, Any],
    *,
    make: object,
    year: int,
    source: str = "vpic",
    cache_path: str | Path | None = None,
    from_cache: bool = False,
) -> pd.DataFrame:
    """Normaliza catálogos oficiales a candidatos de cruce. vPIC usa Make_Name/Model_Name; el
    descubrimiento de recalls usa make/model. Se admiten ambos. vPIC aporta identidad
    independiente porque el catálogo de campañas puede omitir vehículos sin recalls; por sí
    solo no acredita una etiqueta cero.
    """

    if not isinstance(payload, Mapping):
        raise DataSourceError("NHTSA catalogue payload must be a JSON object.")
    default_make, default_year = str(make).strip(), int(year)
    rows: list[dict[str, object]] = []
    for result in _payload_results(payload):
        result_make = _result_value(result, "Make_Name", "make", "Make")
        result_model = _result_value(result, "Model_Name", "model", "Model")
        result_year = _result_value(result, "modelYear", "ModelYear", "model_year")
        make_text = default_make if pd.isna(result_make) or not str(result_make).strip() else str(result_make).strip()
        model_text = "" if pd.isna(result_model) else str(result_model).strip()
        if not make_text or not model_text:
            continue
        numeric_year = pd.to_numeric(result_year, errors="coerce")
        model_year = default_year if pd.isna(numeric_year) else int(numeric_year)
        # La solicitud está delimitada por marca y año. No incorporar al catálogo
        # independiente una respuesta malformada que indique otro año.
        if model_year != default_year or comparison_make(make_text) != comparison_make(default_make):
            continue
        rows.append(
            {
                "nhtsa_vehicle_id": vehicle_id(make_text, model_text, model_year),
                "marca": make_text,
                "modelo": model_text,
                "ano_fabricacion": model_year,
                "marca_normalizada": comparison_make(make_text),
                "modelo_normalizado": comparison_model(model_text),
                "catalog_verified": source in {"vpic", "recall_catalog"},
                "catalog_source": source,
                "catalog_cache_path": str(cache_path) if cache_path is not None else pd.NA,
                "catalog_desde_cache": bool(from_cache),
            }
        )
    if not rows:
        return empty_nhtsa_catalog_frame()
    return pd.DataFrame(rows, columns=NHTSA_CATALOG_COLUMNS).drop_duplicates("nhtsa_vehicle_id", keep="first")


def normalise_nhtsa_response(
    payload: Mapping[str, Any],
    *,
    make: object,
    model: object,
    year: int,
    nhtsa_vehicle_id: str | None = None,
) -> pd.DataFrame:
    """Convierte una respuesta NHTSA original en una fila canónica por campaña."""

    if not isinstance(payload, Mapping):
        raise DataSourceError("NHTSA payload must be a JSON object.")
    make_text, model_text, model_year = str(make).strip(), str(model).strip(), int(year)
    identifier = nhtsa_vehicle_id or vehicle_id(make_text, model_text, model_year)
    rows: list[dict[str, object]] = []
    for result in _payload_results(payload):
        component = _result_value(result, "Component", "component")
        summary = _result_value(result, "Summary", "summary")
        consequence = _result_value(result, "Consequence", "consequence")
        remedy = _result_value(result, "Remedy", "remedy")
        description = " ".join(
            str(value).strip()
            for value in (component, summary, consequence, remedy)
            if pd.notna(value) and str(value).strip()
        )
        rows.append(
            {
                "nhtsa_vehicle_id": identifier,
                "marca": make_text,
                "modelo": model_text,
                "ano_fabricacion": model_year,
                "marca_normalizada": comparison_make(make_text),
                "modelo_normalizado": comparison_model(model_text),
                "campana_nhtsa": _result_value(
                    result,
                    "NHTSACampaignNumber",
                    "CampaignNumber",
                    "campaign_number",
                ),
                "fecha_reporte": _result_value(
                    result,
                    "ReportReceivedDate",
                    "report_received_date",
                    "ReportDate",
                ),
                "componente": component,
                "resumen": summary,
                "consecuencia": consequence,
                "remedio": remedy,
                "descripcion_recall": description,
            }
        )
    if not rows:
        return empty_nhtsa_records_frame()
    frame = pd.DataFrame(rows, columns=NHTSA_RECORD_COLUMNS)
    # El servicio NHTSA publica ReportReceivedDate como DD/MM/YYYY
    # (campaña 19V859000: 04/12/2019 corresponde al 4 de diciembre).
    # Interpretar ese formato antes que ISO o variantes antiguas; pandas
    # podría convertirlo silenciosamente en el 12 de abril.
    raw_dates = frame["fecha_reporte"]
    parsed_dates = pd.to_datetime(raw_dates, format="%d/%m/%Y", errors="coerce")
    unresolved = parsed_dates.isna() & raw_dates.notna()
    if unresolved.any():
        try:
            parsed_dates.loc[unresolved] = pd.to_datetime(
                raw_dates.loc[unresolved], format="mixed", errors="coerce"
            )
        except (TypeError, ValueError):  # versiones de pandas anteriores a format='mixed'
            parsed_dates.loc[unresolved] = pd.to_datetime(raw_dates.loc[unresolved], errors="coerce")
    frame["fecha_reporte"] = parsed_dates
    return frame


__all__ = [
    "COOPERUNION_COLUMN_ALIASES",
    "NHTSA_CATALOG_COLUMNS",
    "NHTSA_RECALLS_ENDPOINT",
    "NHTSA_RECALL_MODELS_ENDPOINT",
    "NHTSA_VEHICLE_INDEX_COLUMNS",
    "NHTSA_VPIC_MODELS_ENDPOINT",
    "DataSourceError",
    "NHTSABatchResult",
    "NHTSACatalogBatchResult",
    "NHTSACatalogFetchResult",
    "NHTSAFetchResult",
    "NHTSARecallClient",
    "comparison_make",
    "comparison_model",
    "empty_nhtsa_catalog_frame",
    "empty_nhtsa_records_frame",
    "ingest_cooperunion_csv",
    "nhtsa_cache_filename",
    "nhtsa_catalog_cache_filename",
    "normalise_cooperunion_columns",
    "normalise_nhtsa_catalog_response",
    "normalise_nhtsa_response",
]
