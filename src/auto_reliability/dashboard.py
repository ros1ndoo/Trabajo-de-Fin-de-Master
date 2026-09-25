"""Streamlit presentation layer for the vehicle-recall reliability project.

The dashboard deliberately keeps presentation concerns separate from the data and
model layers. ``ReliabilityService`` is optional at import time so the explorer
and honest empty states remain available while model artefacts are being built.

The score shown here is always a *recall-propensity proxy*: it is not a measure
of general mechanical reliability.
"""

from __future__ import annotations

import inspect
import math
import os
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

try:  # Keep module imports useful in lightweight/test environments.
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only without project deps
    pd = None  # type: ignore[assignment]


APP_TITLE = "AutoReliability"
PLACEHOLDER = "— Selecciona una opción —"
MIN_YEAR = 1995
TARGET_COLUMN = "indice_fiabilidad_100"

REQUIRED_CATALOG_COLUMNS = (
    "marca",
    "modelo",
    "ano_fabricacion",
    "categoria_vehiculo",
    "mediana_cilindros",
    "mediana_cv",
    "hist_fiabilidad_marca",
    TARGET_COLUMN,
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "marca": ("marca", "make", "manufacturer", "brand"),
    "modelo": ("modelo", "model", "model_name"),
    "ano_fabricacion": ("ano_fabricacion", "año_fabricacion", "year", "model_year"),
    "categoria_vehiculo": (
        "categoria_vehiculo",
        "vehicle_size",
        "vehicle_style",
        "category",
        "segment",
    ),
    "mediana_cilindros": ("mediana_cilindros", "engine_cylinders", "cylinders"),
    "mediana_cv": ("mediana_cv", "engine_hp", "engine hp", "horsepower", "cv"),
    "hist_fiabilidad_marca": (
        "hist_fiabilidad_marca",
        "brand_reliability_history",
        "historial_marca",
    ),
    TARGET_COLUMN: (TARGET_COLUMN, "reliability_index", "target", "score"),
}


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "El dashboard requiere pandas. Instala las dependencias del proyecto antes de ejecutarlo."
        )
    return pd


def _streamlit() -> Any:
    """Import Streamlit only when the web application is actually started."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Streamlit no está instalado. Ejecuta `pip install -r requirements.txt` y "
            "después `streamlit run app.py`."
        ) from exc
    return st


def _canonical(text: object) -> str:
    """Return an accent/case-insensitive identifier for columns and filters."""

    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower().strip()


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert common service result types into a plain mapping."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {"value": dumped}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {"value": dumped}
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"value": value}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else default


def _text(value: Any, default: str = "No disponible") -> str:
    if value is None:
        return default
    rendered = str(value).strip()
    return rendered if rendered and rendered.lower() not in {"nan", "none", "<na>"} else default


def _as_bool(value: Any) -> bool:
    """Interpret nullable service/data flags without treating every string as true."""

    if value is None:
        return False
    try:
        if pd is not None and bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "si", "sí", "demo"}
    return bool(value)


def _service_status(service: Any | None) -> dict[str, Any]:
    """Read the optional status contract without making the dashboard fragile."""

    getter = getattr(service, "status", None) if service is not None else None
    if not callable(getter):
        return {}
    try:
        return _as_mapping(getter())
    except Exception:
        return {}


def project_root() -> Path:
    """Resolve the repository root without relying on the current shell directory."""

    return Path(__file__).resolve().parents[2]


def normalize_catalog(frame: Any) -> Any:
    """Normalize several likely Gold/catalogue schemas to the UI data contract.

    It accepts a DataFrame or a tabular object that pandas can convert.  Unknown
    columns are retained because services may expose additional context fields.
    """

    pandas = _require_pandas()
    if frame is None:
        return pandas.DataFrame(columns=REQUIRED_CATALOG_COLUMNS)
    if not isinstance(frame, pandas.DataFrame):
        frame = pandas.DataFrame(frame)
    data = frame.copy()

    normalized_columns = {_canonical(column): column for column in data.columns}
    rename: dict[str, str] = {}
    for canonical_name, aliases in _COLUMN_ALIASES.items():
        if canonical_name in data.columns:
            continue
        for alias in aliases:
            existing = normalized_columns.get(_canonical(alias))
            if existing is not None:
                rename[existing] = canonical_name
                break
    if rename:
        data = data.rename(columns=rename)

    for column in REQUIRED_CATALOG_COLUMNS:
        if column not in data.columns:
            data[column] = pandas.NA

    for column in ("marca", "modelo", "categoria_vehiculo"):
        data[column] = data[column].astype("string").str.strip()
        data.loc[data[column].isin(["", "nan", "None", "<NA>"]), column] = pandas.NA

    data["ano_fabricacion"] = pandas.to_numeric(data["ano_fabricacion"], errors="coerce")
    data["mediana_cilindros"] = pandas.to_numeric(data["mediana_cilindros"], errors="coerce")
    data["mediana_cv"] = pandas.to_numeric(data["mediana_cv"], errors="coerce")
    # Compatibility imports can provide HP instead of canonical CV. Convert
    # only a renamed HP column, never an already canonical catalogue value.
    if any(target == "mediana_cv" and _canonical(original) in {"engine_hp", "engine hp", "horsepower"}
           for original, target in rename.items()):
        from .data_sources import HP_TO_CV

        data["mediana_cv"] = data["mediana_cv"] * HP_TO_CV
    data["hist_fiabilidad_marca"] = pandas.to_numeric(
        data["hist_fiabilidad_marca"], errors="coerce"
    )
    data[TARGET_COLUMN] = pandas.to_numeric(data[TARGET_COLUMN], errors="coerce")

    data = data.dropna(subset=["marca", "modelo", "ano_fabricacion"]).copy()
    if data.empty:
        return data
    data = data.loc[data["ano_fabricacion"].between(MIN_YEAR, datetime.now(timezone.utc).year + 1)].copy()
    data["ano_fabricacion"] = data["ano_fabricacion"].astype(int)
    data["marca_sort"] = data["marca"].map(_canonical)
    data["modelo_sort"] = data["modelo"].map(_canonical)
    data = data.sort_values(["marca_sort", "modelo_sort", "ano_fabricacion"]).drop(
        columns=["marca_sort", "modelo_sort"]
    )
    return data.reset_index(drop=True)


def _gold_candidates() -> list[Path]:
    configured = os.environ.get("AUTO_RELIABILITY_GOLD_PATH")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            project_root() / "data" / "gold" / "gold_us_car_reliability.parquet",
            project_root() / "data" / "gold" / "gold_us_car_reliability.csv",
        ]
    )
    return candidates


def load_catalog(service: Any | None = None) -> tuple[Any, str | None]:
    """Load the catalogue from the service first, then from the Gold artefact.

    Returns a normalized DataFrame and an optional diagnostic that is suitable
    for an end-user warning (rather than exposing a raw traceback).
    """

    pandas = _require_pandas()
    service_error: Exception | None = None
    if service is not None:
        loader = getattr(service, "load_catalog", None)
        if callable(loader):
            try:
                return normalize_catalog(loader()), None
            except Exception as exc:  # Service failure should not blank the UI.
                service_error = exc

    file_error: Exception | None = None
    for candidate in _gold_candidates():
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".csv":
                return normalize_catalog(pandas.read_csv(candidate)), None
            return normalize_catalog(pandas.read_parquet(candidate)), None
        except Exception as exc:
            file_error = exc

    if service_error is not None or file_error is not None:
        detail = "No se pudo cargar el catálogo analítico."
        if service_error is not None:
            detail += " El servicio de predicción no respondió al cargarlo."
        if file_error is not None:
            detail += " El artefacto Gold encontrado no pudo leerse."
        return pandas.DataFrame(columns=REQUIRED_CATALOG_COLUMNS), detail
    return (
        pandas.DataFrame(columns=REQUIRED_CATALOG_COLUMNS),
        "Aún no existe la capa Gold. Ejecuta el pipeline de datos antes de usar el predictor.",
    )


def resolve_default_service() -> Any | None:
    """Best-effort discovery of the backend without making it a UI dependency."""

    candidates = (
        ("auto_reliability.service", "ReliabilityService"),
        ("auto_reliability.inference", "ReliabilityService"),
    )
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            service_class = getattr(module, class_name)
            return service_class()
        except (ImportError, AttributeError, TypeError):
            continue
        except Exception:
            # A model/artifact can be unavailable during first setup. The UI
            # shows an empty state; it never constructs a score on its own.
            continue
    return None


def _call_prediction(service: Any, marca: str, modelo: str, ano_fabricacion: int) -> dict[str, Any]:
    """Invoke the documented service contract with minor naming tolerance."""

    predictor = (getattr(service, "predict_on_demand", None) or getattr(service, "predict", None)
                 or getattr(service, "predict_vehicle", None))
    if not callable(predictor):
        raise TypeError("El servicio no expone un método predict().")

    # Prefer the agreed Spanish contract.  Inspecting the signature prevents a
    # harmless mismatch from being reported as an unavailable model.
    try:
        parameter_names = set(inspect.signature(predictor).parameters)
    except (TypeError, ValueError):
        parameter_names = set()
    if {"marca", "modelo", "ano_fabricacion"}.issubset(parameter_names) or not parameter_names:
        response = predictor(marca=marca, modelo=modelo, ano_fabricacion=ano_fabricacion)
    elif {"make", "model", "year"}.issubset(parameter_names):
        response = predictor(make=marca, model=modelo, year=ano_fabricacion)
    elif {"brand", "model", "year"}.issubset(parameter_names):
        response = predictor(brand=marca, model=modelo, year=ano_fabricacion)
    else:
        response = predictor(marca, modelo, ano_fabricacion)
    result = _as_mapping(response)
    if not result:
        raise ValueError("El servicio devolvió una predicción vacía.")
    return result


def observation_notice(row: Mapping[str, Any]) -> str:
    """Describe stored observation evidence, never the reliability of a car.

    Old release catalogs lack the new evidence contract. Their matching status
    can explain an exclusion but cannot certify an empty historical response.
    """
    reason = str(row.get("primary_reason", ""))
    if reason in {"", "nan", "None", "<NA>"}:
        reason = str(row.get("match_status", ""))
    messages = {
        "manual_review": "Identidad pendiente de revisión: no se atribuyen campañas del candidato a este vehículo.",
        "rejected": "Cruce de identidad rechazado: no hay un historial oficial atribuible por esta ruta.",
        "manual_rejected": "Cruce de identidad rechazado en revisión: no se atribuye el historial del candidato.",
        "no_candidate": "Sin candidato oficial compatible en el catálogo utilizado. Esto no demuestra ausencia de recalls.",
        "not_in_matching_scope": "Vehículo fuera del alcance de la ingesta registrada; no hay observación validada.",
        "not_queried": "Sin consulta validada registrada en esta versión.",
        "not_recorded": "No consta una respuesta oficial validada; no se interpreta como cero recalls.",
        "query_failed": "La consulta registrada falló. La falta de respuesta no es evidencia de fiabilidad.",
        "unverified_zero_result": "Respuesta vacía sin evidencia suficiente para acreditar cero recalls.",
        "incomplete_window": "La ventana de observación aún no está completa; no hay una etiqueta final comparable.",
    }
    message = messages.get(reason)
    if message is None and reason == "included":
        outcome = str(row.get("window_outcome", "unknown"))
        if outcome == "zero_in_window":
            message = "Cero campañas observadas en la ventana de tres años; no significa cero en todo el histórico ni fiabilidad garantizada."
        elif outcome == "positive_in_window":
            message = "Campañas observadas dentro de la ventana de tres años, según la evidencia del conjunto publicado."
    if message is None:
        message = "Esta versión no aporta un diagnóstico detallado de observación para este vehículo; no se deduce ausencia de recalls."
    observed_at = str(row.get("evidence_as_of", ""))
    if observed_at and observed_at not in {"nan", "None", "<NA>"}:
        message += f" Fecha de evaluación de la ventana: {observed_at}."
    return message + " Este estado es independiente de una posible referencia predictiva de marca/categoría y de la consulta oficial actual."


def _select_rows(catalog: Any, marca: str, modelo: str, ano_fabricacion: int) -> Any:
    pandas = _require_pandas()
    if catalog is None or catalog.empty:
        return pandas.DataFrame(columns=REQUIRED_CATALOG_COLUMNS)
    brand_mask = catalog["marca"].map(_canonical) == _canonical(marca)
    model_mask = catalog["modelo"].map(_canonical) == _canonical(modelo)
    year_mask = catalog["ano_fabricacion"] == int(ano_fabricacion)
    return catalog.loc[brand_mask & model_mask & year_mask].copy()


def _feature_context(catalog: Any, marca: str, modelo: str, ano_fabricacion: int) -> dict[str, Any]:
    """Build display-only technical context from the selected catalogue row."""

    pandas = _require_pandas()
    selected = _select_rows(catalog, marca, modelo, ano_fabricacion)
    if selected.empty:
        return {"vehicle_features": {}, "segment_features": {}, "feature_ranges": {}, "brand_history": []}

    row = selected.iloc[0]
    category = _text(row.get("categoria_vehiculo"), "Sin segmento")
    past_segment = catalog.loc[
        (catalog["categoria_vehiculo"].map(_canonical) == _canonical(category))
        & (catalog["ano_fabricacion"] < int(ano_fabricacion))
    ].copy()
    # No future/coincident launches are borrowed for an empty historical group.

    feature_columns = {
        "Cilindros": "mediana_cilindros",
        "Potencia (CV)": "mediana_cv",
        "Historial previo disponible": "hist_fiabilidad_marca",
    }
    vehicle_features: dict[str, float] = {}
    segment_features: dict[str, float] = {}
    feature_ranges: dict[str, tuple[float, float]] = {}
    for label, column in feature_columns.items():
        vehicle_value = _safe_float(row.get(column))
        segment_values = pandas.to_numeric(past_segment.get(column), errors="coerce").dropna()
        if vehicle_value is not None:
            vehicle_features[label] = vehicle_value
        if not segment_values.empty:
            segment_features[label] = float(segment_values.mean())
            low = min(float(segment_values.min()), vehicle_value if vehicle_value is not None else float(segment_values.min()))
            high = max(float(segment_values.max()), vehicle_value if vehicle_value is not None else float(segment_values.max()))
            feature_ranges[label] = (low, high)

    historical_brand = catalog.loc[
        (catalog["marca"].map(_canonical) == _canonical(marca))
        & (catalog["ano_fabricacion"] <= int(ano_fabricacion) - 3)
        & catalog[TARGET_COLUMN].notna()
    ]
    history: list[dict[str, float | int]] = []
    if not historical_brand.empty:
        grouped = historical_brand.groupby("ano_fabricacion", as_index=False)[TARGET_COLUMN].mean()
        history = [
            {"ano_fabricacion": int(item["ano_fabricacion"]), TARGET_COLUMN: float(item[TARGET_COLUMN])}
            for _, item in grouped.iterrows()
        ]

    return {
        "categoria_vehiculo": category,
        "vehicle_features": vehicle_features,
        "segment_features": segment_features,
        "feature_ranges": feature_ranges,
        "brand_history": history,
        "hist_fiabilidad_marca": _safe_float(row.get("hist_fiabilidad_marca")),
    }


def normalize_prediction(
    raw_result: Any,
    catalog: Any,
    marca: str,
    modelo: str,
    ano_fabricacion: int,
) -> dict[str, Any]:
    """Make backend outputs safe and complete for the rendering layer."""

    result = _as_mapping(raw_result)
    aliases = {
        "prediccion_indice_100": ("prediccion_indice_100", "prediction", "score", TARGET_COLUMN),
        "mae": ("mae", "mae_test", "error_medio_absoluto"),
        "modelo_usado": ("modelo_usado", "model_name", "model"),
        "explicacion_factores": ("explicacion_factores", "explanation", "narrative"),
        "es_baseline": ("es_baseline", "fallback", "is_baseline"),
    }
    normalized: dict[str, Any] = dict(result)
    for destination, possible_names in aliases.items():
        if destination in normalized and normalized[destination] is not None:
            continue
        for name in possible_names:
            if name in result and result[name] is not None:
                normalized[destination] = result[name]
                break

    score = _safe_float(normalized.get("prediccion_indice_100"))
    if score is None or not 0 <= score <= 100:
        raise ValueError("La predicción no contiene un índice numérico 0–100.")
    normalized["prediccion_indice_100"] = score
    # Absence of an evaluation metric must remain explicit.  Rendering a zero
    # MAE would misleadingly imply perfect model validation.
    mae = _safe_float(normalized.get("mae"))
    normalized["mae"] = mae if mae is not None and mae >= 0 else None
    # A response for another vehicle must never be placed beside these filters.
    for key, expected in (("marca", marca), ("modelo", modelo)):
        if normalized.get(key) is not None and _canonical(normalized[key]) != _canonical(expected):
            raise ValueError("La respuesta corresponde a otro vehículo.")
        normalized[key] = expected
    response_year = _safe_int(normalized.get("ano_fabricacion"), int(ano_fabricacion))
    if response_year != int(ano_fabricacion):
        raise ValueError("La respuesta corresponde a otro año.")
    normalized["ano_fabricacion"] = response_year
    normalized["modelo_usado"] = _text(normalized.get("modelo_usado"), "Modelo predictivo")
    normalized["es_baseline"] = _as_bool(normalized.get("es_baseline", False))
    normalized["fuente_demo"] = _as_bool(normalized.get("fuente_demo", False))
    normalized["fecha_ejecucion"] = _text(
        normalized.get("fecha_ejecucion"), datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    context = _feature_context(catalog, marca, modelo, ano_fabricacion)
    for key, value in context.items():
        existing = normalized.get(key)
        # Avoid equality comparisons between pandas/NumPy containers and None.
        if existing is None or (isinstance(existing, (str, Mapping, Sequence)) and len(existing) == 0):
            normalized[key] = value
    normalized["categoria_vehiculo"] = _text(
        normalized.get("categoria_vehiculo"), _text(context.get("categoria_vehiculo"), "Sin segmento")
    )
    return normalized


def make_radar_figure(
    vehicle_features: Mapping[str, Any],
    segment_features: Mapping[str, Any],
    feature_ranges: Mapping[str, Sequence[float]] | None = None,
) -> Any | None:
    """Create a unitless 0–100 radar from technical features and segment means."""

    usable = [
        label
        for label in vehicle_features
        if _safe_float(vehicle_features.get(label)) is not None
        and _safe_float(segment_features.get(label)) is not None
    ]
    if len(usable) < 2:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - project requirements include Plotly
        return None

    def relative_value(label: str, value: float) -> float:
        supplied = feature_ranges.get(label) if feature_ranges else None
        if supplied and len(supplied) >= 2:
            low, high = _safe_float(supplied[0]), _safe_float(supplied[1])
        else:
            other = _safe_float(segment_features.get(label), value) or value
            low, high = min(0.0, value, other), max(value, other)
        if low is None or high is None or math.isclose(low, high):
            return 50.0
        return max(0.0, min(100.0, (value - low) * 100 / (high - low)))

    vehicle_raw = [_safe_float(vehicle_features[label], 0.0) or 0.0 for label in usable]
    segment_raw = [_safe_float(segment_features[label], 0.0) or 0.0 for label in usable]
    vehicle_scaled = [relative_value(label, value) for label, value in zip(usable, vehicle_raw)]
    segment_scaled = [relative_value(label, value) for label, value in zip(usable, segment_raw)]
    labels = usable + [usable[0]]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=vehicle_scaled + [vehicle_scaled[0]],
            theta=labels,
            fill="toself",
            name="Vehículo seleccionado",
            line={"color": "#4ea1ff", "width": 3},
            fillcolor="rgba(78, 161, 255, .22)",
            customdata=vehicle_raw + [vehicle_raw[0]],
            hovertemplate="%{theta}: %{customdata:.1f}<extra>Vehículo</extra>",
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=segment_scaled + [segment_scaled[0]],
            theta=labels,
            fill="toself",
            name="Media histórica del segmento",
            line={"color": "#a9b6c8", "width": 2, "dash": "dot"},
            fillcolor="rgba(169, 182, 200, .08)",
            customdata=segment_raw + [segment_raw[0]],
            hovertemplate="%{theta}: %{customdata:.1f}<extra>Segmento</extra>",
        )
    )
    figure.update_layout(
        height=390,
        margin={"l": 42, "r": 42, "t": 24, "b": 24},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d7e0ec", "family": "Inter, system-ui, sans-serif"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.18, "xanchor": "center", "x": 0.5},
        polar={
            "bgcolor": "rgba(20, 31, 47, .35)",
            "radialaxis": {"visible": True, "range": [0, 100], "showticklabels": False, "gridcolor": "#2c3b52"},
            "angularaxis": {"gridcolor": "#2c3b52", "linecolor": "#2c3b52"},
        },
    )
    return figure


def _factor_table(result: Mapping[str, Any]) -> Any:
    """Turn structured service explanations into a concise, safe factor table."""

    pandas = _require_pandas()
    raw_factors = result.get("factores") or result.get("feature_importance") or result.get("factors")
    rows: list[dict[str, str]] = []
    if isinstance(raw_factors, Mapping):
        raw_factors = [
            {"factor": factor, "impacto": impact} for factor, impact in raw_factors.items()
        ]
    if isinstance(raw_factors, Sequence) and not isinstance(raw_factors, (str, bytes)):
        for item in raw_factors:
            if isinstance(item, Mapping):
                factor = _text(item.get("factor") or item.get("feature") or item.get("nombre"))
                factor = {
                    "mediana_cv": "Potencia mediana (CV)",
                    "mediana_cilindros": "Cilindros",
                    "hist_fiabilidad_marca": "Historial previo disponible",
                    "marca": "Marca",
                    "marca_y_segmento": "Marca y segmento",
                    "segmento": "Segmento",
                    "media_global": "Media global",
                    "categoria_vehiculo": "Segmento",
                    "ano_fabricacion": "Año de lanzamiento",
                }.get(factor, factor)
                value = _text(item.get("valor", item.get("value")), "—")
                impact = _safe_float(item.get("impacto", item.get("impact", item.get("contribution"))))
                direction = _text(item.get("direccion") or item.get("direction"), "")
                direction = {"aumenta_indice": "Eleva el índice", "reduce_indice": "Reduce el índice",
                             "sin_efecto_apreciable": "Sin efecto apreciable"}.get(direction, direction)
                if not direction and impact is not None:
                    direction = "Eleva el índice" if impact > 0 else "Reduce el índice" if impact < 0 else "Sin efecto neto"
                rows.append(
                    {
                        "Factor": factor,
                        "Valor": value,
                        "Influencia estimada": (
                            f"{impact:+.2f} puntos frente a la media global de entrenamiento"
                            if impact is not None and item.get("method") == "media_grupo_vs_media_global_train"
                            else f"{direction} ({impact:+.2f} puntos)" if impact is not None
                            else direction or "Disponible en el modelo"
                        ),
                    }
                )
            else:
                rows.append({"Factor": _text(item), "Valor": "—", "Influencia estimada": "Disponible en el modelo"})

    if not rows:
        vehicle = result.get("vehicle_features") if isinstance(result.get("vehicle_features"), Mapping) else {}
        segment = result.get("segment_features") if isinstance(result.get("segment_features"), Mapping) else {}
        for label, value in vehicle.items():
            numeric = _safe_float(value)
            mean = _safe_float(segment.get(label))
            if numeric is None:
                continue
            comparison = "sin media de segmento"
            if mean is not None:
                delta = numeric - mean
                comparison = f"{abs(delta):.1f} {'por encima' if delta >= 0 else 'por debajo'} de la media"
            rows.append({"Factor": str(label), "Valor": f"{numeric:.1f}", "Influencia estimada": comparison})
        if not rows:
            rows.append(
                {
                    "Factor": "Especificaciones e historial previo de marca",
                    "Valor": "—",
                    "Influencia estimada": "El servicio no entregó atribuciones detalladas.",
                }
            )
    return pandas.DataFrame(rows)


def _narrative(result: Mapping[str, Any]) -> str:
    explanation = result.get("explicacion_factores")
    if isinstance(explanation, str) and explanation.strip():
        return explanation.strip()
    return (
        "La estimación se fundamenta únicamente en especificaciones disponibles antes del lanzamiento "
        "y en el historial previo de recalls de la marca. La comparación describe patrones del "
        "dataset; no demuestra una relación causal."
    )


def summary_frame(result: Mapping[str, Any]) -> Any:
    """Create the downloadable, tabular prediction summary."""

    pandas = _require_pandas()
    features = result.get("vehicle_features") if isinstance(result.get("vehicle_features"), Mapping) else {}
    row = {
        "marca": result.get("marca"),
        "modelo": result.get("modelo"),
        "ano_fabricacion": result.get("ano_fabricacion"),
        "categoria_vehiculo": result.get("categoria_vehiculo"),
        "prediccion_indice_100": result.get("prediccion_indice_100"),
        "mae": result.get("mae"),
        "modelo_usado": result.get("modelo_usado"),
        "es_baseline": result.get("es_baseline"),
        "fuente_demo": _as_bool(result.get("fuente_demo")),
        "evaluacion_retrospectiva": _as_bool(result.get("evaluacion_retrospectiva")),
        "limitaciones": result.get("mensaje"),
        "naturaleza_indice": "Proxy de recalls de seguridad; no mide fiabilidad mecánica general.",
        "hist_fiabilidad_marca": result.get("hist_fiabilidad_marca"),
        "fecha_ejecucion": result.get("fecha_ejecucion"),
        "version_modelo": result.get("version_modelo"),
        "version_datos": result.get("version_datos"),
        "version_escala": result.get("version_escala"),
        "evidencia_oficial": str(result.get("evidencia_oficial", {})),
        "explicacion_factores": _narrative(result),
    }
    for label, value in features.items():
        row[f"especificacion_{_canonical(label).replace(' ', '_')}"] = value
    return pandas.DataFrame([row])


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        :root { --ar-bg: #080d18; --ar-panel: #0f1729; --ar-panel-2: #172035;
          --ar-border: #2a3b54; --ar-text: #eff6ff; --ar-muted: #9fb0c5;
          --ar-primary: #4ea1ff; --ar-green: #35d49a; --ar-orange: #f8b84e; --ar-red: #fb7185; }
        .stApp { background: radial-gradient(circle at 14% -20%, #193b66 0, transparent 33%), var(--ar-bg); color: var(--ar-text); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1240px; padding-top: 2.3rem; padding-bottom: 3rem; }
        .ar-nav { display:flex; align-items:center; justify-content:space-between; gap:1rem; border-bottom:1px solid var(--ar-border); padding:0 0 1.05rem; margin:0 0 1.8rem; }
        .ar-brand { color:var(--ar-text); font-size:1.12rem; font-weight:800; letter-spacing:-.03em; }
        .ar-brand span { color:var(--ar-primary); }
        .ar-nav-tags { display:flex; gap:.45rem; flex-wrap:wrap; justify-content:flex-end; }
        .ar-nav-tag, .ar-proxy-badge { border:1px solid rgba(78,161,255,.38); color:#a8d3ff; background:rgba(78,161,255,.10); border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.03em; padding:.3rem .68rem; }
        .ar-proxy-badge { display:inline-block; color:#fcd34d; border-color:rgba(252,211,77,.36); background:rgba(245,158,11,.10); margin:.1rem 0 .75rem; }
        .ar-demo-alert { display:flex; align-items:center; gap:.8rem; border:2px solid #fb7185; background:linear-gradient(90deg, rgba(190,24,93,.28), rgba(251,113,133,.13)); color:#ffe4e6; border-radius:14px; padding:.82rem 1rem; margin:1rem 0 1.2rem; font-weight:700; box-shadow:0 0 24px rgba(251,113,133,.15); }
        .ar-demo-alert small { color:#fecdd3; font-weight:500; }
        .ar-kicker { color: #70b7ff; font-size: .75rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .45rem; }
        .ar-title { color: var(--ar-text); font-size: clamp(2rem, 4vw, 3.45rem); font-weight: 800; line-height: 1.03; letter-spacing: -.055em; margin: 0; }
        .ar-subtitle { color: var(--ar-muted); font-size: 1.05rem; max-width: 760px; line-height: 1.65; margin: 1rem 0 1.35rem; }
        .ar-card { background: linear-gradient(145deg, rgba(25, 40, 61, .96), rgba(17, 28, 45, .96)); border: 1px solid var(--ar-border); border-radius: 18px; padding: 1.35rem; box-shadow: 0 16px 40px rgba(0, 0, 0, .16); }
        .ar-card h3 { color: var(--ar-text); font-size: 1.05rem; margin: 0 0 .4rem; }
        .ar-card p { color: var(--ar-muted); margin-bottom: 0; }
        .ar-disclaimer { border-left: 4px solid #4ea1ff; background: rgba(78, 161, 255, .10); color: #d5e9ff; border-radius: 8px; padding: .85rem 1rem; margin: 1rem 0; line-height: 1.5; }
        .ar-score { border-radius: 18px; border: 1px solid var(--ar-border); background: linear-gradient(125deg, #14253b, #102035); padding: 1.35rem; min-height: 170px; }
        .ar-score-label { color: var(--ar-muted); font-size: .82rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
        .ar-score-value { font-size: 4rem; font-weight: 800; letter-spacing: -.07em; line-height: 1; margin: .35rem 0; }
        .ar-score-caption { color: var(--ar-muted); font-size: .9rem; }
        .ar-score-hero { min-height: 100%; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:1.5rem; background:linear-gradient(155deg, rgba(18,34,55,.98), rgba(12,24,40,.98)); border:1px solid var(--ar-border); border-radius:18px; box-shadow:0 18px 45px rgba(0,0,0,.20); }
        .ar-ring { --score: 0; --ring-color: #4ea1ff; width:180px; height:180px; border-radius:50%; background:conic-gradient(var(--ring-color) calc(var(--score) * 1%), rgba(255,255,255,.07) 0); display:grid; place-items:center; margin:.5rem auto 1rem; box-shadow:0 0 0 6px rgba(78,161,255,.06), 0 0 34px color-mix(in srgb, var(--ring-color) 25%, transparent); }
        .ar-ring-inner { width:142px; height:142px; border-radius:50%; background:#101c2e; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:inset 0 0 0 1px rgba(255,255,255,.08); }
        .ar-ring-value { color:var(--ring-color); font-size:3.35rem; font-weight:800; line-height:1; letter-spacing:-.07em; }
        .ar-ring-unit { color:var(--ar-muted); font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-top:.18rem; }
        .ar-score-band { width:100%; height:8px; border-radius:999px; background:linear-gradient(90deg, var(--ar-red), var(--ar-orange) 47%, var(--ar-green)); margin:1rem 0 .5rem; position:relative; }
        .ar-score-marker { position:absolute; top:50%; left:calc(var(--score) * 1%); transform:translate(-50%, -50%); width:16px; height:16px; border-radius:50%; background:#fff; border:3px solid var(--ring-color); box-shadow:0 0 0 4px rgba(255,255,255,.12); }
        .ar-stat-line { display:flex; justify-content:space-between; width:100%; gap:1rem; margin-top:.75rem; text-align:left; color:var(--ar-muted); font-size:.82rem; }
        .ar-stat-line strong { color:var(--ar-text); white-space:nowrap; }
        .ar-chip { display: inline-block; border: 1px solid #36577d; border-radius: 999px; padding: .27rem .62rem; color: #bfdbfe; background: rgba(59,130,246,.11); font-size: .78rem; margin: .18rem .22rem .18rem 0; }
        .ar-section-title { color: var(--ar-text); font-size: 1.3rem; font-weight: 750; margin: 1.8rem 0 .7rem; letter-spacing: -.02em; }
        .stButton > button, .stDownloadButton > button { border-radius: 9px; font-weight: 650; min-height: 2.65rem; }
        .stButton > button[kind="primary"] { background: linear-gradient(135deg, #3b82f6, #6366f1); border: 0; }
        [data-testid="stMetric"] { background: rgba(22, 37, 57, .75); border: 1px solid var(--ar-border); border-radius: 14px; padding: .9rem 1rem; }
        [data-testid="stMetricLabel"] { color: var(--ar-muted); }
        [data-testid="stMetricValue"] { color: var(--ar-text); }
        .stSelectbox label, .stMarkdown p, .stCaption { color: var(--ar-muted); }
        [data-testid="stDataFrame"] { border: 1px solid var(--ar-border); border-radius: 12px; overflow: hidden; }
        @media (max-width: 950px) {
          [data-testid="stHorizontalBlock"] { flex-direction:column; gap:1rem; }
          [data-testid="stColumn"] { width:100% !important; flex:1 1 100% !important; min-width:0 !important; }
          .stButton > button { white-space:normal; word-break:normal; }
        }
        @media (max-width: 700px) {
          .block-container { padding: 1.25rem .9rem 2rem; }
          .ar-nav { align-items:flex-start; flex-direction:column; margin-bottom:1.25rem; }
          .ar-nav-tags { justify-content:flex-start; }
          .ar-title { font-size: 2.15rem; }
          .ar-score-value { font-size: 3.25rem; }
          .ar-ring { width:154px; height:154px; }
          .ar-ring-inner { width:122px; height:122px; }
          .ar-ring-value { font-size:2.85rem; }
          .ar-card, .ar-score { padding: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _score_status(score: float) -> tuple[str, str, str]:
    if score >= 70:
        return "#35d49a", "Tramo superior del índice", "Los cortes 45/70 son convenciones visuales, no umbrales de seguridad validados."
    if score >= 45:
        return "#f8b84e", "Tramo intermedio del índice", "Los cortes 45/70 son convenciones visuales, no umbrales de seguridad validados."
    return "#fb7185", "Tramo inferior del índice", "Los cortes 45/70 son convenciones visuales, no umbrales de seguridad validados."


def _mae_label(value: Any) -> str:
    mae = _safe_float(value)
    return f"{mae:.1f} pts" if mae is not None and mae >= 0 else "No disponible"


def _is_demo_result(result: Mapping[str, Any] | None, status: Mapping[str, Any] | None = None) -> bool:
    return _as_bool((status or {}).get("demo_mode")) or _as_bool((result or {}).get("fuente_demo"))


def _render_demo_indicator(
    st: Any, status: Mapping[str, Any], result: Mapping[str, Any] | None = None
) -> None:
    """Make synthetic data impossible to confuse with an NHTSA-backed result."""

    if _is_demo_result(result, status):
        st.markdown(
            "<div class='ar-demo-alert'><span style='font-size:1.3rem'>⚠</span><span>"
            "MODO DEMOSTRACIÓN — el catálogo del predictor y sus predicciones son sintéticos. "
            "<small>No representan evidencia NHTSA real y no deben emplearse para decisiones de compra.</small>"
            "</span></div>",
            unsafe_allow_html=True,
        )


def _render_score_ring(st: Any, result: Mapping[str, Any] | None, *, status: Mapping[str, Any]) -> None:
    """Render the hero score card; it intentionally has a useful empty state."""

    if result is None:
        score, color = 0.0, "#4ea1ff"
        heading = "Tu estimación aparecerá aquí"
        method, mae = "Selecciona un vehículo", "No disponible"
        caption = "El índice se muestra en una escala de 0 a 100."
    else:
        score = _safe_float(result.get("prediccion_indice_100"), 0.0) or 0.0
        color, heading, caption = _score_status(score)
        method = _text(result.get("modelo_usado"), "Modelo predictivo")
        mae = _mae_label(result.get("mae"))
    value = f"{score:.1f}" if result is not None else "—"
    demo_note = (
        "<p style='color:#fda4af;font-size:.78rem;font-weight:700;margin:.45rem 0 0'>"
        "Resultado sintético de demostración</p>"
        if _is_demo_result(result, status)
        else ""
    )
    st.markdown(
        f"""
        <div class="ar-score-hero">
          <div class="ar-score-label">Índice de propensión a recalls</div>
          <div class="ar-ring" style="--score:{score:.2f};--ring-color:{color}"><div class="ar-ring-inner">
            <div class="ar-ring-value">{value}</div><div class="ar-ring-unit">sobre 100</div>
          </div></div>
          <strong style="color:{color};font-size:1rem">{escape(heading)}</strong>
          <div class="ar-score-band" style="--score:{score:.2f};--ring-color:{color}"><span class="ar-score-marker"></span></div>
          <div class="ar-score-caption">0 = mayor propensión · 100 = menor propensión</div>
          <div class="ar-stat-line"><span>Error esperado (MAE)</span><strong>{escape(mae)}</strong></div>
          <div class="ar-stat-line"><span>Método</span><strong>{escape(method)}</strong></div>
          <p class="ar-score-caption" style="margin:.9rem 0 0">{escape(caption)}</p>{demo_note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_current_result(st: Any) -> None:
    """Changing a selector invalidates the displayed estimate, not comparisons."""

    st.session_state.pop("ar_current_result", None)
    st.session_state.pop("ar_last_error", None)


def _reset_downstream(st: Any) -> None:
    _clear_current_result(st)
    st.session_state["ar_modelo"] = PLACEHOLDER
    st.session_state["ar_ano"] = PLACEHOLDER


def _reset_year(st: Any) -> None:
    _clear_current_result(st)
    st.session_state["ar_ano"] = PLACEHOLDER


def _reset_dashboard(st: Any) -> None:
    for key in list(st.session_state):
        if key.startswith("ar_pdf_"):
            st.session_state.pop(key, None)
    for key in (
        "ar_marca",
        "ar_modelo",
        "ar_ano",
        "ar_current_result",
        "ar_comparison",
        "ar_last_error",
    ):
        st.session_state.pop(key, None)


def _render_selector(st: Any, catalog: Any) -> tuple[str | None, str | None, int | None, bool]:
    """Render cascading brand/model/year controls and return the action state."""

    # A real Streamlit container is required here. An opening HTML <div>
    # cannot wrap later Streamlit components and would render as an empty card.
    with st.container(border=True):
        st.markdown("#### Consulta un lanzamiento")
        st.caption("Mercado estadounidense · vehículos desde 1995 · los filtros se actualizan en cascada")
        brands = (
            sorted(catalog["marca"].dropna().unique().tolist(), key=_canonical)
            if not catalog.empty
            else []
        )
        left, middle, right = st.columns(3)
        with left:
            brand = st.selectbox(
                "Marca", [PLACEHOLDER, *brands], key="ar_marca", on_change=_reset_downstream, args=(st,)
            )
        selected_brand = None if brand == PLACEHOLDER else str(brand)
        brand_rows = (
            catalog.loc[catalog["marca"].map(_canonical) == _canonical(selected_brand)]
            if selected_brand
            else catalog.iloc[0:0]
        )
        models = sorted(brand_rows["modelo"].dropna().unique().tolist(), key=_canonical)
        with middle:
            model = st.selectbox(
                "Modelo",
                [PLACEHOLDER, *models],
                key="ar_modelo",
                on_change=_reset_year,
                args=(st,),
                disabled=not selected_brand,
            )
        selected_model = None if model == PLACEHOLDER else str(model)
        model_rows = (
            brand_rows.loc[brand_rows["modelo"].map(_canonical) == _canonical(selected_model)]
            if selected_model
            else brand_rows.iloc[0:0]
        )
        years = sorted(
            {
                _safe_int(year)
                for year in model_rows["ano_fabricacion"].dropna().tolist()
                if _safe_int(year) is not None
            },
            reverse=True,
        )
        with right:
            selected_year_raw = st.selectbox(
                "Año de fabricación", [PLACEHOLDER, *years], key="ar_ano", disabled=not selected_model,
                on_change=_clear_current_result, args=(st,),
            )
        selected_year = _safe_int(selected_year_raw) if selected_year_raw != PLACEHOLDER else None

        action_left, action_right = st.columns([3, 1])
        with action_left:
            evaluate = st.button("Estimar propensión a recalls", type="primary", width="stretch")
        with action_right:
            st.button("Reiniciar", width="stretch", on_click=_reset_dashboard, args=(st,))
    return selected_brand, selected_model, selected_year, evaluate


def _result_key(result: Mapping[str, Any]) -> str:
    return "|".join(
        _canonical(result.get(key, "")) for key in ("marca", "modelo", "ano_fabricacion")
    )


def _comparison_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    mae = _safe_float(result.get("mae"))
    return {
        "id": _result_key(result),
        "Vehículo": f"{_text(result.get('marca'))} {_text(result.get('modelo'))} ({_text(result.get('ano_fabricacion'))})",
        "Índice (0–100)": round(_safe_float(result.get("prediccion_indice_100"), 0.0) or 0.0, 1),
        "MAE": round(mae, 1) if mae is not None else "No disponible",
        "Segmento": _text(result.get("categoria_vehiculo")),
        "Método": _text(result.get("modelo_usado")),
        "Origen": "Demostración sintética" if _as_bool(result.get("fuente_demo")) else "Datos reales",
        "Retrospectiva": _as_bool(result.get("evaluacion_retrospectiva")),
        "Reserva sin validar": _as_bool(result.get("fallback")),
        "Versión modelo": result.get("version_modelo"),
        "Versión datos": result.get("version_datos"),
        "Escala": result.get("version_escala"),
        "Fecha": result.get("fecha_ejecucion"),
        "Limitaciones": result.get("mensaje", ""),
        "Evidencia oficial": str(result.get("evidencia_oficial", {})),
    }


def comparison_warnings(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Flag non-comparable estimates; never derive a purchase ranking."""
    warnings = []
    for field in ("Versión modelo", "Versión datos", "Escala"):
        if any(not entry.get(field) for entry in entries):
            warnings.append(f"{field}: falta trazabilidad; no se puede confirmar comparabilidad.")
        elif len({entry[field] for entry in entries}) > 1:
            warnings.append(f"{field}: los resultados proceden de versiones diferentes; vuelva a calcularlos.")
    if any(entry.get("Reserva sin validar") for entry in entries):
        warnings.append("Hay referencias de reserva sin evaluación propia: no equivalen al modelo validado.")
    if any(entry.get("Retrospectiva") for entry in entries):
        warnings.append("Hay consultas retrospectivas: no representan predicciones disponibles al lanzamiento.")
    if len({entry.get("Origen") for entry in entries}) > 1:
        warnings.append("Se mezclan datos reales y sintéticos; no interprete sus diferencias como evidencia.")
    if len({entry.get("Segmento") for entry in entries}) > 1:
        warnings.append("Son segmentos distintos: el índice es relativo a su segmento, no riesgo absoluto comparable.")
    return warnings


def _add_to_comparison(st: Any, result: Mapping[str, Any]) -> None:
    entries = list(st.session_state.get("ar_comparison", []))
    entry = _comparison_entry(result)
    entries = [existing for existing in entries if existing.get("id") != entry["id"]]
    entries.append(entry)
    # Two cards make the comparison legible on mobile and desktop.  A newly
    # added third result replaces the oldest one.
    st.session_state["ar_comparison"] = entries[-2:]


def _render_comparison(st: Any) -> None:
    entries = st.session_state.get("ar_comparison", [])
    if not entries:
        return
    pandas = _require_pandas()
    with st.expander(f"Comparación guardada ({len(entries)}/2)", expanded=len(entries) == 2):
        st.caption("Guarda otro resultado para contrastarlo. 100 indica menor propensión histórica a recalls.")
        st.info("Comparación descriptiva, no recomendación de compra. El MAE no es un intervalo individual "
                "y una diferencia de puntuación no demuestra superioridad estadística.")
        for warning in comparison_warnings(entries):
            st.warning(warning)
        comparison = pandas.DataFrame(entries).drop(columns=["id"], errors="ignore")
        st.dataframe(comparison, width="stretch", hide_index=True)
        if st.button("Borrar comparación", key="ar_clear_comparison"):
            st.session_state["ar_comparison"] = []
            st.rerun()


def _render_brand_history(st: Any, result: Mapping[str, Any]) -> None:
    history = result.get("brand_history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        st.info("No hay suficientes cohortes anteriores de la marca para mostrar la evolución histórica.")
        return
    pandas = _require_pandas()
    try:
        import plotly.express as px
    except ImportError:  # pragma: no cover
        st.info("Instala Plotly para visualizar la evolución de la marca.")
        return
    data = pandas.DataFrame(history)
    if "ano_fabricacion" not in data or TARGET_COLUMN not in data:
        return
    figure = px.line(
        data,
        x="ano_fabricacion",
        y=TARGET_COLUMN,
        markers=True,
        labels={"ano_fabricacion": "Año", TARGET_COLUMN: "Índice histórico (0–100)"},
    )
    figure.update_traces(line={"color": "#4ea1ff", "width": 3}, marker={"size": 7})
    figure.update_layout(
        height=300,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,31,47,.35)",
        font={"color": "#d7e0ec"},
        xaxis={"gridcolor": "#2c3b52"},
        yaxis={"gridcolor": "#2c3b52", "range": [0, 100]},
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _pdf_hook(service: Any | None, result: Mapping[str, Any]) -> tuple[bytes | None, str | None]:
    """Consume, but never implement, the backend's optional PDF-report hook."""

    hook = getattr(service, "build_prediction_report_pdf", None) if service is not None else None
    if not callable(hook):
        try:
            from auto_reliability.reporting import build_prediction_report_pdf

            hook = build_prediction_report_pdf
        except (ImportError, AttributeError):
            return None, None
    try:
        try:
            generated = hook(result)
        except TypeError:
            generated = hook(prediction=result)
    except Exception:
        return None, None
    filename = f"autoreliability_{_result_key(result).replace('|', '_')}.pdf"
    if isinstance(generated, tuple) and generated:
        payload = generated[0]
        if len(generated) > 1 and generated[1]:
            filename = str(generated[1])
    else:
        payload = generated
    if isinstance(payload, (str, Path)):
        possible_path = Path(payload)
        if possible_path.is_file():
            return possible_path.read_bytes(), filename
        return None, None
    if isinstance(payload, bytes):
        return payload, filename
    return None, None


def _render_downloads(st: Any, service: Any | None, result: Mapping[str, Any]) -> None:
    st.markdown("<div class='ar-section-title'>Resumen para compartir</div>", unsafe_allow_html=True)
    left, middle, _ = st.columns([1.2, 1.2, 2.6])
    csv_data = summary_frame(result).to_csv(index=False).encode("utf-8-sig")
    file_stem = _result_key(result).replace("|", "_")
    with left:
        st.download_button(
            "Descargar resumen CSV",
            data=csv_data,
            file_name=f"autoreliability_{file_stem}.csv",
            mime="text/csv",
            width="stretch",
        )
    pdf_state_key = f"ar_pdf_{file_stem}"
    with middle:
        if pdf_state_key in st.session_state:
            pdf_data, pdf_name = st.session_state[pdf_state_key]
            st.download_button(
                "Descargar informe PDF",
                data=pdf_data,
                file_name=pdf_name,
                mime="application/pdf",
                width="stretch",
            )
        elif service is not None and (
            callable(getattr(service, "build_prediction_report_pdf", None))
            or _has_reporting_hook()
        ):
            if st.button("Preparar informe PDF", key=f"prepare_{pdf_state_key}", width="stretch"):
                with st.spinner("Preparando el informe…"):
                    pdf_data, pdf_name = _pdf_hook(service, result)
                if pdf_data:
                    st.session_state[pdf_state_key] = (pdf_data, pdf_name or "informe.pdf")
                    st.rerun()
                else:
                    st.warning("El informe PDF no está disponible para este resultado.")


def _has_reporting_hook() -> bool:
    try:
        from auto_reliability.reporting import build_prediction_report_pdf  # noqa: F401

        return True
    except ImportError:
        return False


def _render_result(st: Any, service: Any | None, result: Mapping[str, Any]) -> None:
    score = _safe_float(result.get("prediccion_indice_100"), 0.0) or 0.0
    mae = _safe_float(result.get("mae"))
    color, status, caption = _score_status(score)
    vehicle_name = f"{_text(result.get('marca'))} {_text(result.get('modelo'))} {_text(result.get('ano_fabricacion'))}"

    st.markdown("<div class='ar-section-title'>Resultado de la estimación</div>", unsafe_allow_html=True)
    chips = "".join(
        f"<span class='ar-chip'>{escape(value)}</span>"
        for value in (
            vehicle_name,
            _text(result.get("categoria_vehiculo")),
            _text(result.get("modelo_usado")),
        )
    )
    st.markdown(chips, unsafe_allow_html=True)
    if _as_bool(result.get("fuente_demo")):
        st.error(
            "MODO DEMOSTRACIÓN: este resultado procede de datos sintéticos; no es evidencia NHTSA real."
        )
    message = result.get("mensaje")
    if result.get("es_baseline") or result.get("fallback"):
        st.warning(
            _text(message, "Se muestra una referencia baseline histórica; no es la salida del modelo validado.")
        )
    elif message:
        st.info(_text(message))

    official_evidence = result.get("evidencia_oficial")
    if isinstance(official_evidence, Mapping):
        if official_evidence.get("status") == "unavailable":
            st.warning("Consulta oficial no disponible. El índice procede del baseline entrenado; "
                       "no se ha supuesto que el vehículo tenga cero recalls.")
        else:
            evidence_source = ("Snapshot masivo NHTSA (reserva ante fallo de API)"
                               if "/odi/ffdd/" in str(official_evidence.get("source", "")) else "API NHTSA")
            st.caption(
                f"{evidence_source} · fecha de la evidencia: {official_evidence.get('fetched_at', 'No disponible')} · "
                f"{official_evidence.get('count', '—')} campañas en la consulta actual. "
                "Estos registros no intervienen como variables predictoras."
            )
            if official_evidence.get("identity_status") == "technical_name_unverified_in_nhtsa":
                st.warning("Denominación sin equivalencia NHTSA confirmada. La respuesta, incluso vacía, "
                           "no certifica ausencia de campañas. Verifique su VIN.")

    mae_column, context_column = st.columns(2)
    with mae_column:
        st.metric("Error esperado (MAE)", _mae_label(mae))
        if mae is None:
            st.caption("El servicio no publicó una métrica de evaluación para esta estimación.")
        else:
            st.caption("Error absoluto medio en evaluación temporal. No es un intervalo de confianza individual.")
    with context_column:
        history_score = _safe_float(result.get("hist_fiabilidad_marca"))
        history_source = _text(result.get("origen_hist_fiabilidad_marca"))
        market_history = history_source == "global_cohortes_completadas_previas"
        brand_history = history_source == "marca_cohortes_completadas_previas"
        st.metric(
            "Historial previo del mercado" if market_history else
            "Historial previo de marca" if brand_history else "Historial previo disponible",
            f"{history_score:.2f} pts" if history_score is not None else "No disponible",
        )
        st.caption("Media de cohortes maduras de lanzamiento entre año−5 y año−3; "
                   "no equivale al conteo de campañas de los tres años calendario previos. "
                   "No incorpora recalls posteriores del vehículo consultado.")
        if market_history:
            st.caption("Sin historial suficiente de esta marca: se utiliza la referencia del mercado disponible.")

    st.markdown(f"<p style='color:{color}; font-weight:700; margin:.8rem 0 .15rem'>{escape(status)}</p>", unsafe_allow_html=True)
    st.caption(caption)
    st.markdown(
        "<div class='ar-disclaimer'><strong>Cómo interpretar este resultado.</strong> "
        "Es un proxy basado en registros oficiales de recalls de seguridad de la NHTSA. "
        "No mide directamente la fiabilidad mecánica general ni predice una avería concreta.</div>",
        unsafe_allow_html=True,
    )

    action_left, action_right, _ = st.columns([1.35, 1.5, 3])
    with action_left:
        if st.button("Añadir a comparación", key=f"compare_{_result_key(result)}", width="stretch"):
            _add_to_comparison(st, result)
            st.rerun()
    with action_right:
        st.caption("Guarda hasta dos resultados y compáralos debajo del formulario.")

    technical_column, factors_column = st.columns([1.15, 1])
    with technical_column:
        st.markdown("<div class='ar-section-title'>Perfil técnico frente al segmento</div>", unsafe_allow_html=True)
        chart = make_radar_figure(
            result.get("vehicle_features") if isinstance(result.get("vehicle_features"), Mapping) else {},
            result.get("segment_features") if isinstance(result.get("segment_features"), Mapping) else {},
            result.get("feature_ranges") if isinstance(result.get("feature_ranges"), Mapping) else None,
        )
        if chart is None:
            st.info("No hay suficientes especificaciones comparables para generar el radar.")
        else:
            st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
        st.caption("Radar relativo para facilitar la comparación visual; los valores exactos aparecen al pasar el cursor.")
    with factors_column:
        st.markdown("<div class='ar-section-title'>Factores explicativos</div>", unsafe_allow_html=True)
        st.dataframe(_factor_table(result), width="stretch", hide_index=True)
        st.markdown(f"<div class='ar-card'><h3>Lectura del resultado</h3><p>{escape(_narrative(result))}</p></div>", unsafe_allow_html=True)
        st.caption(_text(result.get("tipo_explicacion"), "Resumen de las atribuciones del modelo; no establece causalidad."))

    st.markdown("<div class='ar-section-title'>Contexto histórico de la marca</div>", unsafe_allow_html=True)
    _render_brand_history(st, result)
    _render_downloads(st, service, result)


def _coverage_notice(st: Any, status: Mapping[str, Any], catalog: Any = None) -> None:
    """Keep the bounded technical sample from being mistaken for a current forecast feed."""

    message = _text(status.get("message"), "")
    max_year = _safe_int(
        status.get("technical_year_max")
        or status.get("technical_coverage_max_year")
        or status.get("coverage_max_year")
        or status.get("max_year")
    )
    if max_year is None and catalog is not None and not catalog.empty:
        max_year = _safe_int(catalog["ano_fabricacion"].max())
    if max_year is not None and max_year < datetime.now(timezone.utc).year and not _as_bool(status.get("demo_mode")):
        st.warning(
            f"Cobertura técnica verificada hasta {max_year}. El predictor no debe interpretarse como una "
            "previsión real para lanzamientos actuales o posteriores a esa cobertura."
        )
    if message and not _as_bool(status.get("demo_mode")):
        st.caption(f"Estado de datos: {message}")
    alignment = status.get("inventory_alignment")
    if isinstance(alignment, Mapping):
        with st.expander("Inventario complementario EPA/NHTSA · en validación"):
            if alignment.get("error"):
                st.warning(_text(alignment["error"]))
            else:
                st.write(
                    f"{alignment.get('variants')} versiones EPA; "
                    f"{alignment.get('linked_source_names')} de {alignment.get('source_names')} "
                    "nombres marca-modelo-año alineados con el catálogo NHTSA."
                )
                st.caption(f"Snapshot EPA: {alignment.get('epa_snapshot_date')} · "
                           f"Regla de cruce: {alignment.get('policy_version')}")
                st.info("Este inventario no amplía todavía el predictor: falta potencia y validar "
                        "las categorías. Coincidir en nombre no acredita aplicabilidad a cada "
                        "versión o VIN; no encontrar una coincidencia no significa cero recalls.")
    warnings = status.get("ingestion_warnings", ())
    if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
        messages = [item for item in dict.fromkeys(_text(item, "") for item in warnings) if item]
        if len(messages) > 2:
            st.warning(f"La ingesta tiene {len(messages)} limitaciones. Revisa el detalle antes de interpretar resultados.")
            with st.expander("Limitaciones de la cobertura y del cruce de datos"):
                for warning in messages:
                    st.write(warning)
        else:
            for warning in messages:
                st.warning(warning)


def _records_from_response(response: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize the optional NHTSA explorer response for display only."""

    payload = _as_mapping(response)
    candidates = payload.get("records") or payload.get("results") or payload.get("recalls") or []
    if isinstance(candidates, Mapping):
        candidates = [candidates]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        candidates = []
    records = [_as_mapping(item) for item in candidates]
    return records, payload


def _render_recall_records(st: Any, response: Any) -> None:
    records, metadata = _records_from_response(response)
    if _as_bool(metadata.get("fuente_demo")) or _as_bool(metadata.get("demo_mode")):
        st.error("Esta respuesta está marcada como demostración; no se presenta como un recall oficial.")
        return
    source = _text(metadata.get("source"), "NHTSA")
    from_cache = _as_bool(metadata.get("from_cache"))
    fetched_at = _text(metadata.get("fetched_at"), "ahora")
    count = _safe_int(metadata.get("count"), len(records)) or 0
    st.success(
        f"Consulta completada: {count} recall(s) devuelto(s) por {source}"
        + (" (resultado en caché)." if from_cache else ".")
    )
    st.caption(f"Consulta realizada/actualizada: {fetched_at}. Estos registros no recalculan el índice predictivo.")
    if metadata.get("notice"):
        st.info(_text(metadata["notice"]))
    if not records:
        st.info(
            "La fuente no devolvió recalls para esta consulta. Esto no demuestra ausencia de riesgos de seguridad "
            "ni equivale a una puntuación de fiabilidad."
        )
        return
    for position, record in enumerate(records, start=1):
        campaign = _text(record.get("campana_nhtsa") or record.get("NHTSACampaignNumber") or record.get("campaign"), f"Recall {position}")
        component = _text(record.get("componente") or record.get("Component"), "Componente no especificado")
        report_date = _text(record.get("fecha_reporte") or record.get("ReportReceivedDate"), "Fecha no disponible")
        with st.expander(f"{campaign} · {component}", expanded=position == 1):
            st.caption(f"Fecha de reporte: {report_date}")
            for label, candidates in (
                ("Resumen", ("resumen", "Summary")),
                ("Consecuencia", ("consecuencia", "Consequence")),
                ("Remedio", ("remedio", "Remedy")),
            ):
                text = next((_text(record.get(key), "") for key in candidates if _text(record.get(key), "")), "")
                if text:
                    st.markdown(f"**{label}:** {text}")


def _render_real_recalls_tab(st: Any, service: Any | None) -> None:
    """Render an explicit-action explorer for official recall records.

    It intentionally has no automatic fetch: entering a make/model/year never
    sends a request.  This tab is a source explorer, not a reliability score.
    """

    st.markdown("<div class='ar-section-title'>Consulta de recalls oficiales</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ar-disclaimer'><strong>Explorador separado del predictor.</strong> "
        "Consulta los registros que devuelva la fuente oficial para una combinación concreta. "
        "La consulta no modifica el índice ni convierte «sin resultados» en una garantía de seguridad.</div>",
        unsafe_allow_html=True,
    )
    explorer = getattr(service, "real_recalls", None) if service is not None else None
    if not callable(explorer):
        st.info("La conexión con el explorador de recalls oficiales todavía no está configurada.")
        return

    local_catalog = None
    catalog_getter = getattr(service, "real_recalls_catalog", None) if service is not None else None
    if callable(catalog_getter):
        try:
            local_catalog = normalize_catalog(catalog_getter())
        except Exception:
            local_catalog = None
        if local_catalog is not None and not local_catalog.empty:
            st.caption(
                f"Catálogo oficial local: {len(local_catalog):,} combinaciones. "
                "La fuente masiva enumera vehículos con campañas; una ausencia no prueba cero recalls."
            )
    if local_catalog is None or local_catalog.empty:
        st.info("Catálogo oficial pendiente. Ejecute auto-reliability download-recalls para cargar los selectores.")
        return

    def clear_official_result(*dependent_keys: str) -> None:
        for key in ("ar_real_response", "ar_real_error", "ar_real_query", *dependent_keys):
            st.session_state.pop(key, None)

    # Outside st.form: dependent options must rerun immediately on selection.
    first, second, third = st.columns([1, 1.35, .72])
    with first:
        make = st.selectbox("Marca (NHTSA)", [PLACEHOLDER, *sorted(local_catalog.marca.unique())],
                            key="ar_real_make", on_change=clear_official_result,
                            args=("ar_real_model", "ar_real_year"))
    models = sorted(local_catalog.loc[local_catalog.marca.eq(make), "modelo"].unique())
    with second:
        model = st.selectbox("Modelo (NHTSA)", [PLACEHOLDER, *models], key="ar_real_model",
                             disabled=not models, on_change=clear_official_result, args=("ar_real_year",))
    years = sorted(local_catalog.loc[local_catalog.marca.eq(make) & local_catalog.modelo.eq(model),
                                     "ano_fabricacion"].unique(), reverse=True)
    with third:
        year = st.selectbox("Año (NHTSA)", [PLACEHOLDER, *map(int, years)], key="ar_real_year",
                            disabled=not years, on_change=clear_official_result)
    submitted = st.button("Consultar fuente oficial NHTSA", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("ar_real_response", None)
        st.session_state.pop("ar_real_error", None)
        if make == PLACEHOLDER or model == PLACEHOLDER or year == PLACEHOLDER:
            st.session_state["ar_real_error"] = "Selecciona marca, modelo y año antes de consultar la fuente oficial."
        else:
            try:
                with st.spinner("Consultando registros oficiales de recalls…"):
                    response = explorer(make.strip(), model.strip(), int(year))
                st.session_state["ar_real_response"] = response
                st.session_state["ar_real_query"] = f"{make.strip()} {model.strip()} · {int(year)}"
            except Exception:
                st.session_state["ar_real_error"] = (
                    "La fuente oficial no pudo responder a esta consulta. Inténtalo de nuevo más tarde."
                )
    if st.session_state.get("ar_real_error"):
        st.error(st.session_state["ar_real_error"])
    if "ar_real_response" in st.session_state:
        st.markdown(f"**Resultados para {escape(st.session_state['ar_real_query'])}**")
        _render_recall_records(st, st.session_state["ar_real_response"])


def _get_service(st: Any, explicit_service: Any | None) -> Any | None:
    if explicit_service is not None:
        return explicit_service
    if "ar_default_service_resolved" not in st.session_state:
        st.session_state["ar_default_service"] = resolve_default_service()
        st.session_state["ar_default_service_resolved"] = True
    return st.session_state.get("ar_default_service")


def _get_catalog(st: Any, service: Any | None) -> tuple[Any, str | None]:
    """Cache catalogue data per Streamlit session, without serialising a service."""

    revision = service.data_revision() if service is not None and callable(getattr(service, "data_revision", None)) else None
    if "ar_catalog" not in st.session_state or revision != st.session_state.get("ar_catalog_revision"):
        catalog, issue = load_catalog(service)
        st.session_state["ar_catalog"] = catalog
        st.session_state["ar_catalog_issue"] = issue
        st.session_state["ar_catalog_revision"] = revision
        st.session_state.pop("ar_current_result", None)
        st.session_state.pop("ar_comparison", None)
        for key in list(st.session_state):
            if key.startswith("ar_pdf_"):
                st.session_state.pop(key, None)
    return st.session_state["ar_catalog"], st.session_state.get("ar_catalog_issue")


def run_dashboard(service: Any | None = None) -> None:
    """Run the responsive Streamlit dashboard.

    Backend integration contract (implemented outside this UI module):

    * ``load_catalog() -> pandas.DataFrame``
    * ``predict(marca, modelo, ano_fabricacion) -> Mapping | dataclass``
    * optional ``build_prediction_report_pdf(prediction) -> bytes | Path``

    A prediction mapping should include ``prediccion_indice_100``, ``mae`` and
    ``modelo_usado``.  Optional keys are rendered when supplied and otherwise
    derived safely from Gold.
    """

    st = _streamlit()
    st.set_page_config(page_title=f"{APP_TITLE} · Predictor", page_icon="🚗", layout="wide")
    _inject_styles(st)
    active_service = _get_service(st, service)
    status = _service_status(active_service)
    catalog, catalog_issue = _get_catalog(st, active_service)

    st.markdown(
        "<div class='ar-nav'><div class='ar-brand'>◈ Auto<span>Reliability</span></div>"
        "<div class='ar-nav-tags'><span class='ar-nav-tag'>NHTSA · CooperUnion</span>"
        "<span class='ar-nav-tag'>EE. UU. · desde 1995</span></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='ar-kicker'>Mercado estadounidense · Desde 1995</div>", unsafe_allow_html=True)
    st.markdown("<h1 class='ar-title'>Conoce el historial.<br><span style='color:#7db6ff'>Anticipa los recalls.</span></h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='ar-subtitle'>Estima la propensión a <em>recalls</em> de seguridad de un lanzamiento "
        "a partir de sus especificaciones y del historial previo de su fabricante.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ar-proxy-badge'>Índice basado en recalls NHTSA - proxy, no fiabilidad mecánica directa</div>",
        unsafe_allow_html=True,
    )
    _render_demo_indicator(st, status)
    _coverage_notice(st, status, catalog)

    prediction_tab, recalls_tab = st.tabs(["Predicción de propensión", "Recalls oficiales NHTSA"])
    with prediction_tab:
        if catalog_issue:
            st.warning(catalog_issue)
        if catalog.empty:
            st.markdown(
                "<div class='ar-card'><h3>El predictor está esperando datos</h3><p>Cuando la capa Gold esté "
                "disponible, aquí aparecerán las marcas, los modelos y sus años de fabricación. El explorador "
                "de recalls oficiales sigue disponible en la otra pestaña.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            selector_column, score_column = st.columns([1.12, .88], gap="large")
            with selector_column:
                marca, modelo, ano_fabricacion, evaluate = _render_selector(st, catalog)
                if marca and modelo and ano_fabricacion is not None:
                    selected_evidence = _select_rows(catalog, marca, modelo, ano_fabricacion)
                    if not selected_evidence.empty:
                        st.info(observation_notice(selected_evidence.iloc[0].to_dict()))
            if evaluate:
                if not marca or not modelo or ano_fabricacion is None:
                    st.session_state["ar_last_error"] = "Selecciona marca, modelo y año antes de solicitar una estimación."
                elif active_service is None:
                    st.session_state.pop("ar_current_result", None)
                    st.session_state["ar_last_error"] = "El servicio predictivo no está disponible todavía."
                else:
                    try:
                        with st.spinner("Consultando el modelo y construyendo el contexto…"):
                            raw_result = _call_prediction(active_service, marca, modelo, ano_fabricacion)
                        st.session_state["ar_current_result"] = normalize_prediction(
                            raw_result, catalog, marca, modelo, ano_fabricacion
                        )
                        pdf_key = _result_key(st.session_state["ar_current_result"]).replace("|", "_")
                        st.session_state.pop(f"ar_pdf_{pdf_key}", None)
                        st.session_state.pop("ar_last_error", None)
                    except Exception as exc:
                        st.session_state.pop("ar_current_result", None)
                        if exc.__class__.__name__ == "InsufficientEvidenceError":
                            st.session_state["ar_last_error"] = (
                                f"No se puede emitir una estimación para esta consulta. {exc}"
                            )
                        else:
                            st.session_state["ar_last_error"] = (
                                "No se pudo obtener una estimación válida para esta consulta. Revisa los datos o "
                                "elige otro vehículo."
                            )
            current_result = st.session_state.get("ar_current_result")
            with score_column:
                _render_score_ring(
                    st,
                    current_result if isinstance(current_result, Mapping) else None,
                    status=status,
                )
            if st.session_state.get("ar_last_error"):
                st.error(st.session_state["ar_last_error"])
            if isinstance(current_result, Mapping) and _as_bool(current_result.get("fuente_demo")) and not _as_bool(status.get("demo_mode")):
                _render_demo_indicator(st, {}, current_result)
            _render_comparison(st)
            if isinstance(current_result, Mapping):
                _render_result(st, active_service, current_result)
    with recalls_tab:
        _render_real_recalls_tab(st, active_service)


__all__ = [
    "load_catalog",
    "make_radar_figure",
    "normalize_catalog",
    "normalize_prediction",
    "project_root",
    "run_dashboard",
    "summary_frame",
]
