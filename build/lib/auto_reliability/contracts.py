"""Data contracts and validation helpers shared across project layers."""

from __future__ import annotations

import hashlib
import json
import numbers
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

GOLD_REQUIRED_COLUMNS: tuple[str, ...] = (
    "id_vehiculo_ano",
    "marca",
    "modelo",
    "ano_fabricacion",
    "categoria_vehiculo",
    "mediana_cilindros",
    "mediana_cv",
    "score_recalls_bruto",
    "hist_fiabilidad_marca",
    "indice_fiabilidad_100",
)

PREDICTION_FEATURES: tuple[str, ...] = (
    "marca",
    "categoria_vehiculo",
    "mediana_cilindros",
    "mediana_cv",
    "hist_fiabilidad_marca",
)


def canonical_text(value: object) -> str:
    """Produce a stable lowercase comparison key without losing source labels."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def vehicle_id(make: object, model: object, year: object) -> str:
    """Build the canonical traceability key used across all project layers."""

    make_key = canonical_text(make).replace(" ", "_").upper()
    model_key = canonical_text(model).replace(" ", "_").upper()
    return f"{make_key}_{model_key}_{int(year)}"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    """Raise a concise diagnostic if a dataframe violates its schema contract."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def validate_gold_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate fundamental gold-layer invariants without mutating caller data."""

    require_columns(frame, GOLD_REQUIRED_COLUMNS, context="Gold dataset")
    if frame.empty:
        raise ValueError("Gold dataset is empty; train only after a successful pipeline run.")
    if frame["id_vehiculo_ano"].isna().any() or frame["id_vehiculo_ano"].duplicated().any():
        raise ValueError("Gold dataset requires unique, non-null id_vehiculo_ano values.")
    if not frame["indice_fiabilidad_100"].dropna().between(0, 100).all():
        raise ValueError("indice_fiabilidad_100 must remain within the documented 0-100 range.")
    return frame


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Bind a model to its actual source rows, independent of storage dtypes.

    Derived target scales are deliberately omitted: training fits its own
    scale. Raw labels, feature values and provenance must still agree.
    """
    columns = sorted(set(GOLD_REQUIRED_COLUMNS + ("fuente_demo",)).intersection(frame.columns)
                     - {"indice_fiabilidad_100"})
    data = frame[columns].copy()
    if "id_vehiculo_ano" in data:
        data = data.sort_values("id_vehiculo_ano")
    def scalar(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, (bool, numbers.Number)):
            return float(value)
        return str(value)
    payload = {"columns": columns, "rows": [[scalar(v) for v in row]
               for row in data.itertuples(index=False, name=None)]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
