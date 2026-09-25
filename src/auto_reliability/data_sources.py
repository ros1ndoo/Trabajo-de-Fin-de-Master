"""Source ingestion for the US vehicle-reliability data pipeline.

This module deliberately keeps acquisition separate from transformation.  In
particular, NHTSA responses are cached as their original JSON payloads before
they are normalised, so a pipeline run is reproducible and does not need to
re-query a public API merely to re-run a cleaning step.
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
# NHTSA's recalls discovery endpoint is useful for recall-only exploration,
# but cannot certify a *zero*-recall vehicle: ``issueType=r`` can omit it.
# vPIC is a separate official NHTSA make/model/year catalogue and is the
# authority used to distinguish an API-valid zero from an unverified query.
NHTSA_RECALL_MODELS_ENDPOINT = "https://api.nhtsa.gov/products/vehicle/models"
NHTSA_VPIC_MODELS_ENDPOINT = "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear"
# NIST SP 811, Appendix B.9; US mechanical HP and metric horsepower (CV).
HP_TO_CV = 745.6999 / 735.4988
HP_ALIASES = frozenset({"engine_hp", "engine_horsepower", "horsepower", "hp"})


class DataSourceError(RuntimeError):
    """Raised when an input source cannot satisfy the data contract."""


class NHTSARequestError(DataSourceError):
    """Structured request failure; an HTTP error never represents zero recalls."""

    def __init__(self, message: str, *, status: int | None, attempts: int, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts
        self.retry_after = retry_after
        self.transient = status is None or status == 429 or status >= 500


def retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After delay/date, ignoring malformed or non-finite values."""
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
    """The immutable raw result of one NHTSA vehicle request."""

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
        """Return API recall count despite NHTSA's historical field casing."""

        value = self.payload.get("Count", self.payload.get("count"))
        try:
            return int(value)
        except (TypeError, ValueError):
            records = self.payload.get("Results", self.payload.get("results", []))
            return len(records) if isinstance(records, list) else 0


@dataclass
class NHTSABatchResult:
    """Normalised records plus the requested vehicle universe.

    ``vehicle_index`` includes vehicles for which NHTSA returned zero recalls.
    Keeping those rows is essential: a missing NHTSA result is not evidence of
    an unmatched technical specification, and zero-recall vehicles must remain
    eligible for the gold layer.
    """

    records: pd.DataFrame
    vehicle_index: pd.DataFrame
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class NHTSACatalogFetchResult:
    """One immutable NHTSA make/model/year catalogue snapshot."""

    make: str
    year: int
    payload: dict[str, Any]
    cache_path: Path
    from_cache: bool
    source: str


@dataclass
class NHTSACatalogBatchResult:
    """Independent NHTSA model catalogue plus any non-fatal batch failures."""

    vehicles: pd.DataFrame
    failures: list[dict[str, str]] = field(default_factory=list)


def _normalised_column_name(name: object) -> str:
    return canonical_text(name).replace(" ", "_")


# CooperUnion's published file uses e.g. ``Engine HP`` and ``Market Category``.
# The aliases also make a Spanish-labelled classroom export usable without a
# one-off preprocessing notebook.
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
        # ``Vehicle Style`` is the most direct consumer-facing segment in the
        # CooperUnion source (e.g. "4dr SUV"), so it wins over the broader,
        # multi-label Market Category whenever both are present.
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
    """Return the first meaningful value from duplicate source aliases."""

    for value in series:
        if pd.notna(value) and str(value).strip():
            return value
    return pd.NA


def _coalesce_alias_columns(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    # Respect alias priority rather than CSV source-column order.  This is
    # specifically important for Vehicle Style -> Market Category fallback.
    existing = [alias for alias in aliases if alias in frame.columns]
    if not existing:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    if len(existing) == 1:
        return frame[existing[0]].copy()
    return frame[existing].apply(_first_non_empty, axis=1)


def _normalise_source_text(value: object) -> str | pd._libs.missing.NAType:
    """Normalise display text without turning missing values into the string 'nan'."""

    if value is None or pd.isna(value):
        return pd.NA
    cleaned = " ".join(str(value).strip().split())
    return cleaned if cleaned else pd.NA


def normalise_cooperunion_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Map a CooperUnion/Kaggle CSV to the project's source-level contract.

    The returned dataframe remains at its original trim/engine-row granularity;
    :func:`auto_reliability.transform.prepare_technical_specs` performs
    model-year aggregation later, leaving imputation to train-only estimators.
    US-labelled HP is interpreted as mechanical horsepower, not metric CV.
    It is intentionally tolerant
    of optional technical fields, but rejects a file without make, model, or
    year because there is no defensible way to repair those keys.
    """

    if source.empty:
        raise DataSourceError("The CooperUnion CSV is empty.")

    frame = source.copy()
    frame.columns = [_normalised_column_name(column) for column in frame.columns]

    output = pd.DataFrame(index=frame.index)
    for target, aliases in COOPERUNION_COLUMN_ALIASES.items():
        output[target] = _coalesce_alias_columns(frame, aliases)

    # Convert each alias before coalescing: mixed HP/CV exports must not apply
    # the HP factor to values already supplied in canonical CV.
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

    # Preserve the original row position for source-to-processed traceability.
    output.insert(0, "fila_origen", range(len(output)))
    output["fuente_tecnica"] = "cooperunion"
    return output


# American / British spellings and a few common source aliases are normalised
# only for comparisons.  We retain the cleaned original text for the dashboard.
MAKE_ALIASES: dict[str, str] = {
    "vw": "volkswagen",
    "volkswagen": "volkswagen",
    "mercedes benz": "mercedes benz",
    "mercedesbenz": "mercedes benz",
    "chevy": "chevrolet",
}


def comparison_make(value: object) -> str:
    """Return a conservative, stable make key used for joins."""

    key = canonical_text(value)
    return MAKE_ALIASES.get(key, key)


def comparison_model(value: object) -> str:
    """Return a punctuation-insensitive model key used by fuzzy matching."""

    key = canonical_text(value)
    # Cosmetic separators in alphanumeric model names do not change family:
    # F-150 == F150, CX-5 == CX5. The numbers themselves remain significant.
    return re.sub(r"(?<=[a-z])\s+(?=\d)|(?<=\d)\s+(?=[a-z])", "", key)


def ingest_cooperunion_csv(
    csv_path: str | Path,
    *,
    min_year: int = MIN_MODEL_YEAR,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Read and minimally validate a CooperUnion ``Car Features and MSRP`` CSV.

    Source files are read only.  Rows with invalid keys or years before the
    documented 1995 boundary are excluded here and reported through dataframe
    attributes, allowing callers to persist an explicit quality report.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CooperUnion CSV does not exist: {path}")

    try:
        source = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # Kaggle exports are normally UTF-8, but latin-1 is a safe read-only
        # fallback for local classroom copies.
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
    """Make a readable, filesystem-safe filename component."""

    value = canonical_text(value).replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", value) or "unknown"


def _query_digest(*parts: object) -> str:
    """Prevent slug collisions without placing untrusted query text in paths."""

    source = "\x1f".join(str(part).strip().casefold() for part in parts)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def nhtsa_cache_filename(make: object, model: object, year: int) -> str:
    """Return a safe, collision-resistant raw-cache filename for a vehicle query."""

    return (
        f"nhtsa_recalls_{_safe_filename_part(make)}_{_safe_filename_part(model)}_"
        f"{int(year)}_{_query_digest(make, model, year)}.json"
    )


def nhtsa_catalog_cache_filename(make: object, year: int, *, source: str = "vpic") -> str:
    """Return a safe immutable filename for one NHTSA catalogue response."""

    clean_source = _safe_filename_part(source)
    return (
        f"nhtsa_{clean_source}_models_{_safe_filename_part(make)}_{int(year)}_"
        f"{_query_digest(source, make, year)}.json"
    )


class NHTSARecallClient:
    """Rate-limited and cache-first client for the public NHTSA recalls API.

    A client is injectable with a ``requests.Session``-compatible object and a
    sleeper, making retry behaviour deterministic in unit tests.  Cached JSON
    is never overwritten: raw input is treated as immutable evidence.
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
        """Return the immutable snapshot location for one catalogue request."""

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
        """Atomically create a raw JSON file while preserving existing evidence."""

        atomic_json(path, payload, immutable=True)

    def _request_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, object] | None,
        request_label: str,
    ) -> dict[str, Any]:
        """GET JSON with bounded exponential backoff for NHTSA rate limits."""

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
                # Invalid/unknown vehicle queries are permanent errors. Retry
                # only connectivity and transient/rate-limit responses.
                response_status = getattr(getattr(exc, "response", None), "status_code", None)
                if response_status is None and response is not None:
                    response_status = getattr(response, "status_code", None)
                if response_status is not None and 400 <= response_status < 500 and response_status != 429:
                    break
                if attempt >= self.max_retries:
                    break
                # Respect long server cooldowns by failing recoverably, not by
                # retrying early or blocking an interactive session indefinitely.
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
        """Convert lock/storage failures to recoverable source errors."""
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
        # Locks are OS-backed and released on process death. Recheck the cache
        # after acquisition so simultaneous sessions make only one request.
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
                    # A timeout for one vehicle is not proof of host throttling.
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
        """Fetch one vehicle's raw recalls response, preferring immutable cache."""

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
        """Fetch an independent NHTSA vPIC vehicle catalogue snapshot.

        vPIC lists NHTSA vehicle make/model/year entries irrespective of whether
        the model currently has a recall. Presence establishes identity evidence,
        not equivalence with recall-service names or proof of a zero label.
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
        """Fetch NHTSA's official recall-model discovery response.

        This endpoint is useful for discovering model spellings associated with
        recall issues.  It must not be used as proof that an absent vehicle has
        zero recalls. vPIC supplies separate identity evidence, not zero labels.
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

    # A readable alias for notebooks and earlier documentation drafts.
    fetch_recalls = fetch_vehicle

    def fetch_independent_vehicle_catalog(
        self,
        vehicles: pd.DataFrame | Iterable[Mapping[str, object] | Sequence[object]],
        *,
        continue_on_error: bool = True,
    ) -> NHTSACatalogBatchResult:
        """Build independent recall-product and vPIC make/model/year catalogues.

        Only one catalogue request is made per make/year rather than per
        technical trim.  The result is both bounded and resumable because each
        original vPIC JSON response is cached in ``data/raw`` once.
        """

        make_year_queries = _normalise_make_year_queries(vehicles)
        catalog_frames: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        for position, query in enumerate(make_year_queries):
            # Recall discovery supplies the actual names expected by the recall
            # endpoint. vPIC supplements vehicles outside that namespace, but a
            # vPIC-only zero response is kept uncertain, never used as a label.
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
            # Equivalent model spellings always prefer the recalls namespace.
            catalog = catalog.drop_duplicates(
                ["marca_normalizada", "modelo_normalizado", "ano_fabricacion"], keep="first"
            ).reset_index(drop=True)
        return NHTSACatalogBatchResult(catalog, failures)

    # Shorter, discoverable alias for service and notebook callers.
    fetch_vehicle_catalog = fetch_independent_vehicle_catalog

    def fetch_catalog_verified_recalls(
        self,
        catalog_vehicles: pd.DataFrame | Iterable[Mapping[str, object]],
        *,
        continue_on_error: bool = True,
    ) -> NHTSABatchResult:
        """Fetch recalls only for independently catalogue-verified vehicles.

        A zero-result recall response is labelled ``valid_zero_recalls`` only
        when independently present in the recalls product namespace. Presence
        in vPIC alone does not establish equivalence with recall model names.
        This prevents a misspelled direct query from being promoted to a false
        zero-recall label.
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
        """Fetch unique vehicle queries sequentially with a respectful delay.

        Sequential batches are intentional: the public service is rate-limited,
        and immutable caching makes subsequent runs fast without concurrency.
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
                        # Direct calls are convenient for exploratory work but
                        # a Count=0 cannot distinguish typo from true absence.
                        "catalog_verified": False,
                        "catalog_source": pd.NA,
                        "query_status": "unverified_with_recalls" if result.result_count else "unverified_zero_result",
                        "resultado_cero_validado": False,
                    }
                )
            except Exception as exc:  # keep a long batch usable after one bad query
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
    """Coerce supported batch input shapes to a de-duplicated query list."""

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
    """Extract unique make/year pairs from technical rows or loose mappings."""

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
                # Accept either the natural (make, year) form or the
                # (make, model, year) form used by direct recall batches.
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
    """Coerce independent catalogue rows (or accepted matches) to recall queries."""

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
        # Dataframe records can have alternate accepted-match column names.
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
    """Return an empty NHTSA dataframe with its stable downstream schema."""

    return pd.DataFrame(columns=NHTSA_RECORD_COLUMNS)


def empty_nhtsa_catalog_frame() -> pd.DataFrame:
    """Return the stable schema for an empty independent vehicle catalogue."""

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
        # Be tolerant of API casing changes without spelling out every variant.
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
    """Normalise an official NHTSA model catalogue to matching candidates.

    ``vpic`` records use ``Make_Name`` / ``Model_Name`` whereas the recalls
    discovery API uses lower-case ``make`` / ``model``.  Both are handled, but
    callers should use vPIC for validity because recall-only discovery can omit
    genuine zero-recall vehicles.
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
        # The request itself is make/year scoped.  Do not admit a malformed API
        # record that reports a different year into the independent catalogue.
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
    """Convert a raw NHTSA response into one canonical row per campaign."""

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
    # The live NHTSA recalls service exports ReportReceivedDate as DD/MM/YYYY
    # (e.g. campaign 19V859000: ``04/12/2019`` is 4 December).  Parse that
    # explicitly before accepting ISO or legacy variants; pandas' default can
    # otherwise silently turn it into 12 April.
    raw_dates = frame["fecha_reporte"]
    parsed_dates = pd.to_datetime(raw_dates, format="%d/%m/%Y", errors="coerce")
    unresolved = parsed_dates.isna() & raw_dates.notna()
    if unresolved.any():
        try:
            parsed_dates.loc[unresolved] = pd.to_datetime(
                raw_dates.loc[unresolved], format="mixed", errors="coerce"
            )
        except (TypeError, ValueError):  # pandas versions before format='mixed'
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
