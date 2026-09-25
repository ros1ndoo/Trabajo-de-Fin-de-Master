"""Limpieza, controles de calidad, puntuación y construcción de Gold. Regla central: los
resultados de recalls son etiquetas, nunca características del propio vehículo. El historial
de marca solo utiliza cohortes previas completas y la normalización se ajusta al período de
entrenamiento indicado.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MIN_MODEL_YEAR,
    OBSERVATION_WINDOW_YEARS,
    RECALL_SEVERITY_KEYWORDS,
    RECALL_SEVERITY_WEIGHTS,
)
from .contracts import (
    GOLD_REQUIRED_COLUMNS,
    canonical_text,
    require_columns,
    validate_gold_dataset,
    vehicle_id,
)
from .data_sources import comparison_make, comparison_model
from .matching import MATCH_COLUMNS, accepted_matches


class DataQualityError(ValueError):
    """Indica que un conjunto no puede promoverse con seguridad a la siguiente capa."""


@dataclass(frozen=True)
class QualityReport:
    """Resumen compacto y serializable de calidad para manifiestos."""

    source_rows: int
    output_rows: int
    duplicate_keys: int
    null_keys: int
    null_numeric_values: int
    min_year: int | None
    max_year: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "source_rows": self.source_rows,
            "output_rows": self.output_rows,
            "duplicate_keys": self.duplicate_keys,
            "null_keys": self.null_keys,
            "null_numeric_values": self.null_numeric_values,
            "min_year": self.min_year,
            "max_year": self.max_year,
        }


def _display_or_key(value: object, default: str = "sin_categoria") -> str:
    if value is None or pd.isna(value):
        return default
    normalized = canonical_text(value)
    return normalized or default


def _first_mode(values: pd.Series) -> str:
    """Elige una categoría determinista si los acabados discrepan sobre el segmento."""

    clean = values.dropna().astype(str)
    if clean.empty:
        return "sin_categoria"
    counts = clean.value_counts()
    max_count = counts.max()
    return min(counts[counts == max_count].index)


def prepare_technical_specs(
    raw_technical: pd.DataFrame,
    *,
    min_year: int = MIN_MODEL_YEAR,
) -> pd.DataFrame:
    """Agrega especificaciones observadas sin imputación entre vehículos. Los nulos permanecen
    hasta la imputación ajustada solo con entrenamiento. Usar vehículos de test/validación,
    incluso del mismo año, haría transductiva la evaluación. Los conteos muestran cobertura
    parcial de acabados.
    """

    require_columns(raw_technical, ("marca", "modelo", "ano_fabricacion"), context="Raw technical data")
    frame = raw_technical.copy()
    frame["marca"] = frame["marca"].map(comparison_make)
    frame["modelo"] = frame["modelo"].map(comparison_model)
    frame["categoria_vehiculo"] = (
        frame["categoria_vehiculo"].map(_display_or_key)
        if "categoria_vehiculo" in frame.columns
        else "sin_categoria"
    )
    frame["ano_fabricacion"] = pd.to_numeric(frame["ano_fabricacion"], errors="coerce")
    frame = frame.loc[
        frame["marca"].ne("")
        & frame["modelo"].ne("")
        & frame["ano_fabricacion"].notna()
        & frame["ano_fabricacion"].ge(min_year)
    ].copy()
    if frame.empty:
        raise DataQualityError(f"No valid technical rows remain for model years >= {min_year}.")
    frame["ano_fabricacion"] = frame["ano_fabricacion"].astype(int)

    for numeric_column in ("potencia_cv", "cilindros"):
        if numeric_column not in frame.columns:
            frame[numeric_column] = np.nan
        numeric = pd.to_numeric(frame[numeric_column], errors="coerce").astype(float)
        valid = np.isfinite(numeric) & (numeric.ge(0) if numeric_column == "cilindros" else numeric.gt(0))
        frame[numeric_column] = numeric.where(valid)

    group_columns = ["marca", "modelo", "ano_fabricacion"]
    aggregated = (
        frame.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            categoria_vehiculo=("categoria_vehiculo", _first_mode),
            mediana_cilindros=("cilindros", "median"),
            mediana_cv=("potencia_cv", "median"),
            n_versiones_tecnicas=("modelo", "size"),
            n_potencia_observada=("potencia_cv", "count"),
            n_cilindros_observados=("cilindros", "count"),
        )
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )
    aggregated["mediana_cilindros"] = pd.to_numeric(aggregated["mediana_cilindros"], errors="raise")
    aggregated["mediana_cv"] = pd.to_numeric(aggregated["mediana_cv"], errors="raise")
    aggregated["unidad_potencia"] = "CV"
    aggregated.insert(
        0,
        "id_vehiculo_ano",
        [
            vehicle_id(make, model, year)
            for make, model, year in aggregated[["marca", "modelo", "ano_fabricacion"]].itertuples(index=False, name=None)
        ],
    )
    if aggregated["id_vehiculo_ano"].duplicated().any():
        raise DataQualityError("Technical aggregation produced duplicate id_vehiculo_ano values.")
    report = quality_check_technical_specs(aggregated, source_rows=len(raw_technical), min_year=min_year)
    aggregated.attrs["quality_report"] = report.as_dict()
    return aggregated


def quality_check_technical_specs(
    technical: pd.DataFrame,
    *,
    source_rows: int | None = None,
    min_year: int = MIN_MODEL_YEAR,
) -> QualityReport:
    """Valida la capa técnica procesada y devuelve recuentos auditables."""

    require_columns(
        technical,
        ("id_vehiculo_ano", "marca", "modelo", "ano_fabricacion", "categoria_vehiculo", "mediana_cilindros", "mediana_cv"),
        context="Processed technical data",
    )
    key_columns = ("id_vehiculo_ano", "marca", "modelo", "ano_fabricacion")
    null_keys = int(technical[list(key_columns)].isna().any(axis=1).sum())
    duplicates = int(technical["id_vehiculo_ano"].duplicated().sum())
    numeric_nulls = int(technical[["mediana_cilindros", "mediana_cv"]].isna().sum().sum())
    years = pd.to_numeric(technical["ano_fabricacion"], errors="coerce")
    if null_keys or duplicates or years.lt(min_year).any():
        raise DataQualityError(
            "Processed technical quality failure: "
            f"null_keys={null_keys}, duplicate_keys={duplicates}, null_numeric_values={numeric_nulls}, "
            f"years_before_{min_year}={int(years.lt(min_year).sum())}."
        )
    return QualityReport(
        source_rows=len(technical) if source_rows is None else source_rows,
        output_rows=len(technical),
        duplicate_keys=duplicates,
        null_keys=null_keys,
        null_numeric_values=numeric_nulls,
        min_year=int(years.min()) if not years.empty else None,
        max_year=int(years.max()) if not years.empty else None,
    )


def classify_recall_severity(description: object) -> str:
    """Clasifica campañas con el diccionario ordenado documentado. Términos críticos prevalecen
    sobre moderados y leves. Sin coincidencias se conserva la campaña con peso
    bajo/predeterminado, sin eliminar evidencia oficial silenciosamente.
    """

    text = canonical_text(description)
    for severity in ("critical", "moderate", "low"):
        if any(re.search(r"\b" + re.escape(keyword) + r"s?\b", text.replace("air bag", "airbag"))
               for keyword in RECALL_SEVERITY_KEYWORDS[severity]):
            return severity
    return "unclassified"


def score_recall_severity(description: object) -> float:
    """Devuelve peso de severidad 3,0 / 1,5 / 1,0 para una descripción NHTSA."""

    severity = classify_recall_severity(description)
    return float(RECALL_SEVERITY_WEIGHTS.get(severity, RECALL_SEVERITY_WEIGHTS["low"]))


def prepare_recall_events(recall_records: pd.DataFrame) -> pd.DataFrame:
    """Añade severidad y fechas canónicas a campañas normalizadas."""

    if recall_records.empty:
        columns = list(recall_records.columns) + [
            column
            for column in ("fecha_reporte_normalizada", "severidad_recall", "peso_severidad")
            if column not in recall_records.columns
        ]
        return pd.DataFrame(columns=columns)
    require_columns(recall_records, ("nhtsa_vehicle_id",), context="NHTSA recall records")
    frame = recall_records.copy()
    description_columns = [
        column
        for column in ("descripcion_recall", "componente", "resumen", "consecuencia", "remedio")
        if column in frame.columns
    ]
    if not description_columns:
        raise DataQualityError("NHTSA recall records need a description, component, summary, consequence, or remedy.")
    if "descripcion_recall" not in frame.columns:
        frame["descripcion_recall"] = frame[description_columns].fillna("").astype(str).agg(" ".join, axis=1)
    if "fecha_reporte" not in frame.columns:
        frame["fecha_reporte"] = pd.NaT
    frame["fecha_reporte_normalizada"] = pd.to_datetime(frame["fecha_reporte"], errors="coerce")
    frame["severidad_recall"] = frame["descripcion_recall"].map(classify_recall_severity)
    frame["peso_severidad"] = frame["descripcion_recall"].map(score_recall_severity).astype(float)

    # Una campaña puede aparecer en varios componentes. Contarla una sola vez
    # por vehículo con la severidad máxima evita que respuestas extensas
    # inflen artificialmente las etiquetas objetivo.
    if "campana_nhtsa" not in frame.columns:
        frame["campana_nhtsa"] = pd.NA
    missing_campaign = frame["campana_nhtsa"].isna() | frame["campana_nhtsa"].astype(str).str.strip().eq("")
    fallback_key = frame["descripcion_recall"].fillna("").map(canonical_text)
    frame["_campaign_key"] = frame["campana_nhtsa"].astype("string").fillna("")
    frame.loc[missing_campaign, "_campaign_key"] = "description:" + fallback_key[missing_campaign]
    frame = frame.sort_values(
        ["nhtsa_vehicle_id", "_campaign_key", "peso_severidad"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates(["nhtsa_vehicle_id", "_campaign_key"], keep="first")
    return frame.drop(columns="_campaign_key").reset_index(drop=True)


def _accepted_mapping(matches: pd.DataFrame) -> pd.DataFrame:
    matches = accepted_matches(matches)
    if matches["id_vehiculo_ano"].duplicated().any():
        raise DataQualityError("Accepted fuzzy matches must be one-to-one per id_vehiculo_ano.")
    if matches["nhtsa_vehicle_id"].isna().any():
        raise DataQualityError("Accepted fuzzy matches require a NHTSA vehicle identifier.")
    return matches[["id_vehiculo_ano", "nhtsa_vehicle_id"]].copy()


def aggregate_recall_scores(
    technical: pd.DataFrame,
    matches: pd.DataFrame,
    recall_records: pd.DataFrame,
    *,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
) -> pd.DataFrame:
    """Calcula etiquetas ponderadas en la primera ventana fija. Incluye tres años-modelo: para
    2018 cuenta informes de 2018, 2019 y 2020. No adivina fechas ausentes para colocarlas en
    la ventana; las informa por separado.
    """

    if observation_window_years < 1:
        raise ValueError("observation_window_years must be at least one.")
    require_columns(technical, ("id_vehiculo_ano", "ano_fabricacion"), context="Technical specifications")
    mapping = _accepted_mapping(matches)
    technical_keys = technical[["id_vehiculo_ano", "ano_fabricacion"]].copy()
    technical_keys["ano_fabricacion"] = pd.to_numeric(technical_keys["ano_fabricacion"], errors="raise").astype(int)
    if technical_keys["id_vehiculo_ano"].duplicated().any():
        raise DataQualityError("Technical specifications must have one row per id_vehiculo_ano.")

    result = technical_keys.merge(mapping, on="id_vehiculo_ano", how="left", validate="one_to_one")
    result["coincidencia_recall_aceptada"] = result["nhtsa_vehicle_id"].notna()
    result["score_recalls_bruto"] = 0.0
    result["numero_recalls_en_ventana"] = 0
    result["numero_recalls_sin_fecha"] = 0
    result["numero_recalls_fuera_ventana"] = 0

    if recall_records.empty or mapping.empty:
        return result.drop(columns="nhtsa_vehicle_id")
    events = prepare_recall_events(recall_records)
    # Varias familias técnicas pueden corresponder legítimamente a un modelo
    # oficial. Cada una recibe la campaña una vez, sin rechazo arbitrario.
    events = events.merge(mapping, on="nhtsa_vehicle_id", how="inner", validate="many_to_many")
    if events.empty:
        return result.drop(columns="nhtsa_vehicle_id")
    launch_years = technical_keys.set_index("id_vehiculo_ano")["ano_fabricacion"]
    events["_launch_year"] = events["id_vehiculo_ano"].map(launch_years)
    events["_report_year"] = events["fecha_reporte_normalizada"].dt.year
    events["_has_date"] = events["_report_year"].notna()
    events["_inside_window"] = (
        events["_has_date"]
        & events["_report_year"].ge(events["_launch_year"])
        & events["_report_year"].lt(events["_launch_year"] + observation_window_years)
    )
    in_window = events.loc[events["_inside_window"]]
    if not in_window.empty:
        scores = in_window.groupby("id_vehiculo_ano", as_index=False).agg(
            score_recalls_bruto=("peso_severidad", "sum"),
            numero_recalls_en_ventana=("peso_severidad", "size"),
        )
        result = result.drop(columns=["score_recalls_bruto", "numero_recalls_en_ventana"]).merge(
            scores,
            on="id_vehiculo_ano",
            how="left",
            validate="one_to_one",
        )
        result["score_recalls_bruto"] = result["score_recalls_bruto"].fillna(0.0).astype(float)
        result["numero_recalls_en_ventana"] = result["numero_recalls_en_ventana"].fillna(0).astype(int)
    diagnostics = events.groupby("id_vehiculo_ano", as_index=False).agg(
        numero_recalls_sin_fecha=("_has_date", lambda values: int((~values).sum())),
        numero_recalls_fuera_ventana=("_inside_window", lambda values: int((values.notna() & ~values).sum())),
    )
    result = result.drop(columns=["numero_recalls_sin_fecha", "numero_recalls_fuera_ventana"]).merge(
        diagnostics,
        on="id_vehiculo_ano",
        how="left",
        validate="one_to_one",
    )
    for column in ("numero_recalls_sin_fecha", "numero_recalls_fuera_ventana"):
        result[column] = result[column].fillna(0).astype(int)
    return result.drop(columns="nhtsa_vehicle_id")


def _as_date(value: date | datetime | str | None) -> date:
    if value is None:
        # La completitud depende de la fecha real de corte de observación, no del
        # final de un año futuro fijo. Se puede proporcionar una fecha explícita
        # de instantánea para reproducir un experimento histórico.
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid as_of_date: {value!r}")
    return parsed.date()


def cohort_is_complete(
    model_year: float,
    *,
    as_of_date: date | datetime | str | None = None,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
) -> bool:
    """Indica si han transcurrido todos los años naturales de la ventana de lanzamiento."""

    cutoff = _as_date(as_of_date)
    year = int(model_year)
    return cutoff >= date(year + observation_window_years, 1, 1)


@dataclass(frozen=True)
class ReliabilityNormalizer:
    """Estadísticas de segmento de entrenamiento para la escala z-score documentada."""

    segment_statistics: dict[str, dict[str, float]]
    global_mean: float
    global_std: float
    train_end_year: int

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Añade un índice acotado 0–100; mayor valor representa menos recalls relativos."""

        require_columns(frame, ("categoria_vehiculo", "score_recalls_bruto"), context="Reliability target input")
        output = frame.copy()
        means: list[float] = []
        stds: list[float] = []
        sources: list[str] = []
        for category in output["categoria_vehiculo"].map(_display_or_key):
            stats = self.segment_statistics.get(category)
            if stats is None:
                means.append(self.global_mean)
                stds.append(self.global_std)
                sources.append("global_train_fallback")
            else:
                means.append(stats["mean"])
                stds.append(stats["std"])
                sources.append("segment_train")
        raw = pd.to_numeric(output["score_recalls_bruto"], errors="raise").astype(float)
        z_score = (raw - np.asarray(means)) / np.asarray(stds)
        # 50 representa la media del segmento de entrenamiento y cada desviación
        # típica cambia el índice en 10 puntos. La dirección se invierte porque
        # una mayor carga de recalls corresponde a un índice menor.
        output["indice_fiabilidad_100"] = np.clip(50.0 - 10.0 * z_score, 0.0, 100.0)
        output["origen_normalizacion_indice"] = sources
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "train_end_year": self.train_end_year,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "segment_statistics": self.segment_statistics,
            "formula": "clip(50 - 10 * ((score_recalls_bruto - mean_train_segment) / std_train_segment), 0, 100)",
            "semantics": "100 means lower propensity to NHTSA safety recalls; this is not general mechanical reliability.",
        }


def fit_reliability_normalizer(
    completed_cohorts: pd.DataFrame,
    *,
    train_end_year: int,
) -> ReliabilityNormalizer:
    """Ajusta parámetros de escala con cohortes completas de entrenamiento exclusivamente."""

    require_columns(
        completed_cohorts,
        ("ano_fabricacion", "categoria_vehiculo", "score_recalls_bruto"),
        context="Completed cohorts",
    )
    training = completed_cohorts.loc[
        pd.to_numeric(completed_cohorts["ano_fabricacion"], errors="coerce").le(train_end_year)
    ].copy()
    if training.empty:
        raise DataQualityError(
            f"No completed cohorts at or before train_end_year={train_end_year}; "
            "choose an earlier split boundary or provide historical data."
        )
    training["categoria_vehiculo"] = training["categoria_vehiculo"].map(_display_or_key)
    values = pd.to_numeric(training["score_recalls_bruto"], errors="raise").astype(float)
    global_mean = float(values.mean())
    global_std = float(values.std(ddof=0))
    if not np.isfinite(global_std) or global_std == 0:
        global_std = 1.0
    statistics: dict[str, dict[str, float]] = {}
    for category, group in training.groupby("categoria_vehiculo", sort=True):
        group_values = pd.to_numeric(group["score_recalls_bruto"], errors="raise").astype(float)
        std = float(group_values.std(ddof=0))
        # Un segmento constante o de una fila no define su propio z-score.
        # La media sigue siendo específica del segmento; la escala no nula
        # procede de la distribución global de entrenamiento exclusivamente.
        if not np.isfinite(std) or std == 0:
            std = global_std
        statistics[str(category)] = {"mean": float(group_values.mean()), "std": std}
    return ReliabilityNormalizer(statistics, global_mean, global_std, int(train_end_year))


def _brand_history_for_queries(
    queries: pd.DataFrame,
    completed_reference: pd.DataFrame,
    *,
    lookback_years: int,
    observation_window_years: int,
) -> pd.DataFrame:
    """Calcula historial de marca sin información futura para los años consultados."""

    require_columns(queries, ("id_vehiculo_ano", "marca", "ano_fabricacion"), context="Brand-history queries")
    require_columns(completed_reference, ("marca", "ano_fabricacion", "score_recalls_bruto"), context="Brand-history reference")
    reference = completed_reference[["marca", "ano_fabricacion", "score_recalls_bruto"]].copy()
    reference["marca"] = reference["marca"].map(comparison_make)
    reference["ano_fabricacion"] = pd.to_numeric(reference["ano_fabricacion"], errors="raise").astype(int)
    reference["score_recalls_bruto"] = pd.to_numeric(reference["score_recalls_bruto"], errors="raise").astype(float)
    # Evitar sesgo por número de modelos: cada marca/año aporta una única media.
    brand_year = reference.groupby(["marca", "ano_fabricacion"], as_index=False)["score_recalls_bruto"].mean()

    rows: list[dict[str, object]] = []
    for query in queries[["id_vehiculo_ano", "marca", "ano_fabricacion"]].itertuples(index=False, name=None):
        identifier, make, launch_year = query
        normalized_make = comparison_make(make)
        launch_year = int(launch_year)
        # La ventana cubre año_lanzamiento .. año_lanzamiento+2. Solo se conoce
        # antes de la consulta si lanzamiento_previo + ventana <= lanzamiento_consultado.
        latest_known_launch = launch_year - observation_window_years
        earliest_launch = latest_known_launch - lookback_years + 1
        recent = brand_year.loc[
            brand_year["ano_fabricacion"].between(earliest_launch, latest_known_launch)
        ]
        brand_recent = recent.loc[recent["marca"].eq(normalized_make), "score_recalls_bruto"]
        if not brand_recent.empty:
            score, source = float(brand_recent.mean()), "marca_cohortes_completadas_previas"
        elif not recent.empty:
            # Este respaldo sigue siendo temporalmente válido: usa solo resultados
            # completados antes del lanzamiento y recurre al contexto del mercado.
            score, source = float(recent["score_recalls_bruto"].mean()), "global_cohortes_completadas_previas"
        else:
            score, source = np.nan, "sin_historial_previo"
        rows.append(
            {
                "id_vehiculo_ano": identifier,
                "hist_fiabilidad_marca": score,
                "origen_hist_fiabilidad_marca": source,
            }
        )
    return pd.DataFrame(rows, columns=["id_vehiculo_ano", "hist_fiabilidad_marca", "origen_hist_fiabilidad_marca"])


def compute_brand_recall_history(
    completed_cohorts: pd.DataFrame,
    *,
    lookback_years: int = 3,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
) -> pd.DataFrame:
    """Añade la característica de marca de las tres cohortes previas completas sin fuga."""

    if lookback_years < 1:
        raise ValueError("lookback_years must be at least one.")
    history = _brand_history_for_queries(
        completed_cohorts,
        completed_cohorts,
        lookback_years=lookback_years,
        observation_window_years=observation_window_years,
    )
    return completed_cohorts.merge(history, on="id_vehiculo_ano", how="left", validate="one_to_one")


def build_gold_dataset(
    technical: pd.DataFrame,
    matches: pd.DataFrame,
    recall_records: pd.DataFrame,
    *,
    as_of_date: date | datetime | str | None = None,
    train_end_year: int = 2018,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
    return_normalizer: bool = False,
    observation_index: pd.DataFrame | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, ReliabilityNormalizer]:
    """Construye Gold de cohortes completas para modelado temporal. Excluye cohortes
    recientes/incompletas; build_inference_catalog permite consultarlas sin transferir sus
    resultados propios a las características.
    """

    require_columns(
        technical,
        ("id_vehiculo_ano", "marca", "modelo", "ano_fabricacion", "categoria_vehiculo", "mediana_cilindros", "mediana_cv"),
        context="Processed technical data",
    )
    # Un cruce difuso fallido significa falta de evidencia, no cero recalls.
    # Conservarlo premiaría sistemáticamente registros que no se pudieron
    # vincular a NHTSA. Por ello solo cruces autorizados entran en la capa
    # Gold etiquetada. El catálogo técnico completo sigue disponible para
    # inferencia, donde puede recibir una referencia baseline explícita.
    authorised_ids = set(accepted_matches(matches)["id_vehiculo_ano"].astype(str))
    if observation_index is not None:
        require_columns(observation_index, ("nhtsa_vehicle_id", "query_status"), context="Observation evidence")
        if observation_index.nhtsa_vehicle_id.isna().any() or observation_index.nhtsa_vehicle_id.duplicated().any():
            raise DataQualityError("Observation evidence requires unique official identities")
        statuses = observation_index.set_index("nhtsa_vehicle_id").query_status
        authorised = accepted_matches(matches)
        states = authorised.nhtsa_vehicle_id.map(statuses)
        if not states.isin(["valid_with_recalls", "valid_zero_recalls"]).all():
            raise DataQualityError("Accepted identity lacks a valid observation; missing evidence is not zero")
        observed_ids = set(recall_records.get("nhtsa_vehicle_id", pd.Series(dtype=str)))
        has_events = authorised.nhtsa_vehicle_id.isin(observed_ids)
        if (states.eq("valid_with_recalls") & ~has_events).any() or (states.eq("valid_zero_recalls") & has_events).any():
            raise DataQualityError("Recorded query status contradicts recall events")
    matched_technical = technical.loc[
        technical["id_vehiculo_ano"].astype(str).isin(authorised_ids)
    ].copy()
    if matched_technical.empty:
        raise DataQualityError(
            "No technical vehicles have an accepted NHTSA match; Gold cannot treat unmatched rows as zero recalls."
        )
    scores = aggregate_recall_scores(
        matched_technical,
        matches,
        recall_records,
        observation_window_years=observation_window_years,
    )
    frame = matched_technical.merge(
        scores, on=["id_vehiculo_ano", "ano_fabricacion"], how="left", validate="one_to_one"
    )
    frame["score_recalls_bruto"] = frame["score_recalls_bruto"].fillna(0.0).astype(float)
    frame["ano_fin_ventana"] = frame["ano_fabricacion"].astype(int) + observation_window_years - 1
    cutoff = _as_date(as_of_date)
    frame["es_cohorte_completa"] = frame["ano_fabricacion"].map(
        lambda year: cohort_is_complete(
            int(year), as_of_date=cutoff, observation_window_years=observation_window_years
        )
    )
    complete = frame.loc[frame["es_cohorte_completa"]].copy()
    if complete.empty:
        raise DataQualityError(
            "No complete three-year cohorts are available as of "
            f"{cutoff.isoformat()}; Gold labels cannot be built yet."
        )
    complete = compute_brand_recall_history(
        complete,
        observation_window_years=observation_window_years,
    )
    normalizer = fit_reliability_normalizer(complete, train_end_year=train_end_year)
    gold = normalizer.transform(complete)
    # Colocar primero el contrato documentado y después los diagnósticos.
    contract_columns = list(GOLD_REQUIRED_COLUMNS)
    extras = [column for column in gold.columns if column not in contract_columns]
    gold = gold[contract_columns + extras].sort_values("id_vehiculo_ano", kind="stable").reset_index(drop=True)
    validate_gold_dataset(gold)
    gold.attrs["normalizer"] = normalizer.to_dict()
    gold.attrs["as_of_date"] = cutoff.isoformat()
    gold.attrs["excluded_incomplete_cohorts"] = int((~frame["es_cohorte_completa"]).sum())
    return (gold, normalizer) if return_normalizer else gold


def build_inference_catalog(
    technical: pd.DataFrame,
    completed_gold: pd.DataFrame,
    *,
    as_of_date: date | datetime | str | None = None,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
    include_complete_cohorts: bool = False,
) -> pd.DataFrame:
    """Construye filas de características para lanzamientos recientes sin filtrar recalls
    propios.
    """

    require_columns(
        technical,
        ("id_vehiculo_ano", "marca", "modelo", "ano_fabricacion", "categoria_vehiculo", "mediana_cilindros", "mediana_cv"),
        context="Processed technical data",
    )
    require_columns(completed_gold, ("marca", "ano_fabricacion", "score_recalls_bruto"), context="Gold data")
    cutoff = _as_date(as_of_date)
    catalog = technical.copy()
    catalog["es_cohorte_completa"] = catalog["ano_fabricacion"].map(
        lambda year: cohort_is_complete(
            int(year), as_of_date=cutoff, observation_window_years=observation_window_years
        )
    )
    if not include_complete_cohorts:
        catalog = catalog.loc[~catalog["es_cohorte_completa"]].copy()
    history = _brand_history_for_queries(
        catalog,
        completed_gold,
        lookback_years=3,
        observation_window_years=observation_window_years,
    )
    catalog = catalog.merge(history, on="id_vehiculo_ano", how="left", validate="one_to_one")
    catalog["estado_catalogo"] = np.where(
        catalog["es_cohorte_completa"], "cohorte_completa_sin_etiqueta", "cohorte_incompleta_prediccion"
    )
    catalog["fecha_corte_datos"] = cutoff.isoformat()
    # No incluir score_recalls_bruto ni indice_fiabilidad_100 deliberadamente:
    # son etiquetas del vehículo seleccionado y provocarían fuga en inferencia.
    forbidden = ["score_recalls_bruto", "indice_fiabilidad_100"]
    catalog = catalog.drop(columns=[column for column in forbidden if column in catalog.columns])
    return catalog.sort_values("id_vehiculo_ano", kind="stable").reset_index(drop=True)


def write_processed_sqlite(
    database_path: str | Path,
    *,
    technical: pd.DataFrame,
    nhtsa_records: pd.DataFrame,
    nhtsa_vehicle_index: pd.DataFrame,
    matches: pd.DataFrame,
    gold: pd.DataFrame | None = None,
    inference_catalog: pd.DataFrame | None = None,
) -> Path:
    """Guarda tablas procesadas en SQLite para cruces inspeccionables. Se regeneran
    transaccionalmente desde memoria; las fuentes no cambian. Es una caché intermedia, no un
    sustituto de los originales JSON/CSV inmutables.
    """

    require_columns(technical, ("id_vehiculo_ano",), context="Technical specifications")
    require_columns(matches, MATCH_COLUMNS, context="Fuzzy matches")
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        "technical_specs": technical,
        "nhtsa_recalls": nhtsa_records,
        "nhtsa_vehicle_index": nhtsa_vehicle_index,
        "fuzzy_matches": matches,
    }
    if gold is not None:
        tables["gold_reliability"] = gold
    if inference_catalog is not None:
        tables["inference_catalog"] = inference_catalog

    with sqlite3.connect(database) as connection:
        for table_name, dataframe in tables.items():
            # SQLite no admite de forma uniforme los objetos NA anulables de pandas
            # en todas las versiones compatibles; tampoco acepta directamente
            # Timestamp. Convertir fechas de evidencia a texto ISO sin modificar
            # el dataframe original destinado a Parquet y a la salida.
            sqlite_frame = dataframe.copy().astype(object).map(_sqlite_value)
            sqlite_frame = sqlite_frame.where(pd.notna(sqlite_frame), None)
            sqlite_frame.to_sql(table_name, connection, if_exists="replace", index=False)
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_technical_vehicle ON technical_specs(id_vehiculo_ano);
            CREATE INDEX IF NOT EXISTS idx_nhtsa_vehicle ON nhtsa_recalls(nhtsa_vehicle_id);
            CREATE INDEX IF NOT EXISTS idx_nhtsa_index_vehicle ON nhtsa_vehicle_index(nhtsa_vehicle_id);
            CREATE INDEX IF NOT EXISTS idx_fuzzy_vehicle ON fuzzy_matches(id_vehiculo_ano);
            """
        )
        if gold is not None:
            connection.execute("CREATE INDEX IF NOT EXISTS idx_gold_vehicle ON gold_reliability(id_vehiculo_ano)")
        if inference_catalog is not None:
            connection.execute("CREATE INDEX IF NOT EXISTS idx_inference_vehicle ON inference_catalog(id_vehiculo_ano)")
    return database


def _sqlite_value(value: object) -> object:
    """Convierte escalares pandas/NumPy a valores auditables compatibles con SQLite."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_normalizer_metadata(normalizer: ReliabilityNormalizer, path: str | Path) -> Path:
    """Guarda parámetros de escala de entrenamiento junto a los datos procesados."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(normalizer.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


__all__ = [
    "DataQualityError",
    "QualityReport",
    "ReliabilityNormalizer",
    "aggregate_recall_scores",
    "build_gold_dataset",
    "build_inference_catalog",
    "classify_recall_severity",
    "cohort_is_complete",
    "compute_brand_recall_history",
    "fit_reliability_normalizer",
    "prepare_recall_events",
    "prepare_technical_specs",
    "quality_check_technical_specs",
    "score_recall_severity",
    "write_normalizer_metadata",
    "write_processed_sqlite",
]
