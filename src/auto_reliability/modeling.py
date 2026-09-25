"""Modelado temporal del índice con prevención de fuga. Gold es la única entrada de
entrenamiento. score_recalls_bruto no es una característica porque construye la etiqueta;
hist_fiabilidad_marca debe haberse calculado antes del lanzamiento. train_and_select_model
separa entrenamiento, validación y test, compara baseline, Ridge y Random Forest y exige a
este último una reducción de MAE de validación superior al 10% frente a Ridge para ser
elegible.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CURRENT_YEAR, MIN_MODEL_YEAR, OBSERVATION_WINDOW_YEARS, ProjectPaths
from .contracts import PREDICTION_FEATURES, dataset_fingerprint, require_columns

TARGET_COLUMN = "indice_fiabilidad_100"
YEAR_COLUMN = "ano_fabricacion"
ID_COLUMN = "id_vehiculo_ano"
RAW_RECALL_SCORE_COLUMNS: tuple[str, ...] = (
    "score_recalls_bruto",
    "score_recalls_ponderado",
)
MODEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(PREDICTION_FEATURES)
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "mediana_cilindros",
    "mediana_cv",
    "hist_fiabilidad_marca",
)
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = ("marca", "categoria_vehiculo")

MODEL_FILENAME = "reliability_model.joblib"
METRICS_FILENAME = "model_metrics.json"
VALIDATION_PREDICTIONS_FILENAME = "validation_predictions.csv"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"
ATTRIBUTIONS_FILENAME = "feature_attributions.json"


@dataclass
class TemporalSplit:
    """Partición cronológica sin solapamiento de filas ni años."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    train_end_year: int | None
    validation_end_year: int | None
    train_embargoed: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_embargoed: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_maturity_years: int = OBSERVATION_WINDOW_YEARS

    def metadata(self, *, year_column: str = YEAR_COLUMN) -> dict[str, Any]:
        """Devuelve datos de partición serializables a JSON para el informe."""

        def partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
            if frame.empty:
                return {"rows": 0, "min_year": None, "max_year": None}
            years = pd.to_numeric(frame[year_column], errors="coerce").dropna()
            return {
                "rows": len(frame),
                "min_year": int(years.min()) if not years.empty else None,
                "max_year": int(years.max()) if not years.empty else None,
            }

        return {
            "strategy": self.strategy,
            "configured_train_end_year": self.train_end_year,
            "configured_validation_end_year": self.validation_end_year,
            "train": partition_summary(self.train),
            "validation": partition_summary(self.validation),
            "test": partition_summary(self.test),
            "label_maturity_years": self.label_maturity_years,
            "rows_embargoed_before_validation": len(self.train_embargoed),
            "rows_embargoed_before_test": len(self.validation_embargoed),
        }


@dataclass(frozen=True)
class TemporalTargetNormalizer:
    """Estadísticas del objetivo ajustadas en entrenamiento efectivo. Gold puede incluir un
    índice visual calculado con otro corte, que no debe reutilizarse si cambia la partición.
    Reconstruye el objetivo 0–100 desde la carga bruta usando únicamente entrenamiento
    disponible antes del lanzamiento de validación.
    """

    raw_score_column: str
    segment_statistics: dict[str, dict[str, float]]
    global_mean: float
    global_std: float
    train_end_year: int
    observation_window_years: int = OBSERVATION_WINDOW_YEARS

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        """Devuelve el índice documentado con estadísticas de segmento de entrenamiento."""

        require_columns(
            frame,
            ("categoria_vehiculo", self.raw_score_column),
            context="Temporal target input",
        )
        categories = _normalized_categories(frame["categoria_vehiculo"])
        raw = pd.to_numeric(frame[self.raw_score_column], errors="coerce").astype(float)
        if raw.isna().any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
            raise ValueError(f"{self.raw_score_column} must contain finite values for target construction.")
        means = np.asarray(
            [self.segment_statistics.get(category, {}).get("mean", self.global_mean) for category in categories],
            dtype=float,
        )
        scales = np.asarray(
            [self.segment_statistics.get(category, {}).get("std", self.global_std) for category in categories],
            dtype=float,
        )
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, self.global_std)
        score = np.clip(50.0 - 10.0 * ((raw.to_numpy(dtype=float) - means) / scales), 0.0, 100.0)
        return pd.Series(score, index=frame.index, name=TARGET_COLUMN, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve metadatos del modelo compatibles con JSON."""

        return {
            "raw_score_column": self.raw_score_column,
            "segment_statistics": self.segment_statistics,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "train_end_year": self.train_end_year,
            "observation_window_years": self.observation_window_years,
            "formula": "clip(50 - 10 * ((raw_recall_score - mean_train_segment) / std_train_segment), 0, 100)",
            "semantics": "100 indicates lower estimated propensity to NHTSA safety recalls, not general mechanical reliability.",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TemporalTargetNormalizer:
        """Reconstruye metadatos persistidos para el gráfico histórico del servicio."""

        return cls(
            raw_score_column=str(payload["raw_score_column"]),
            segment_statistics={
                str(key): {"mean": float(value["mean"]), "std": float(value["std"])}
                for key, value in dict(payload["segment_statistics"]).items()
            },
            global_mean=float(payload["global_mean"]),
            global_std=float(payload["global_std"]),
            train_end_year=int(payload["train_end_year"]),
            observation_window_years=int(payload.get("observation_window_years", OBSERVATION_WINDOW_YEARS)),
        )


@dataclass
class ReliabilityModelArtifact:
    """Modelo serializable y transformación del objetivo ajustada solo en entrenamiento. Los
    estimadores usan un objetivo estandarizado para estabilidad numérica; media y desviación
    proceden de sus filas de ajuste y la salida vuelve a la escala 0–100.
    """

    estimator: Any
    model_name: str
    target_mean: float
    target_std: float
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS
    feature_references: dict[str, Any] = field(default_factory=dict)
    expected_mae: float | None = None
    train_year_max: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Estima el proxy con salida acotada entre 0 y 100."""

        features = prepare_feature_frame(frame, feature_columns=self.feature_columns)
        if features.empty:
            return np.asarray([], dtype=float)
        standardized = np.asarray(self.estimator.predict(features), dtype=float).reshape(-1)
        values = standardized * float(self.target_std) + float(self.target_mean)
        if not np.isfinite(values).all():
            raise ValueError("The fitted model returned non-finite predictions.")
        return np.clip(values, 0.0, 100.0)


@dataclass
class TrainingResult:
    """Resultado en memoria de un experimento de entrenamiento temporal."""

    artifact: ReliabilityModelArtifact
    metrics: dict[str, Any]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    artifact_paths: dict[str, Path] = field(default_factory=dict)


class BrandSegmentMeanBaseline(BaseEstimator, RegressorMixin):
    """Baseline estacionario de medias de entrenamiento con respaldos para grupos escasos. En
    validación/test utiliza marca/categoria_vehiculo de entrenamiento; si falta el grupo,
    recurre a media de marca, segmento y global, en ese orden. Nunca lee particiones
    posteriores.
    """

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> BrandSegmentMeanBaseline:
        frame = _baseline_keys(X)
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(frame) != len(values) or len(values) == 0:
            raise ValueError("Baseline requires a non-empty X/y pair of equal length.")

        fit_frame = frame.copy()
        fit_frame["_target"] = values
        self.global_mean_ = float(np.mean(values))
        self.brand_segment_means_ = {
            (str(make), str(category)): float(value)
            for (make, category), value in fit_frame.groupby(
                ["marca", "categoria_vehiculo"], dropna=False
            )["_target"].mean().items()
        }
        self.brand_means_ = {
            str(make): float(value)
            for make, value in fit_frame.groupby("marca", dropna=False)["_target"].mean().items()
        }
        self.category_means_ = {
            str(category): float(value)
            for category, value in fit_frame.groupby("categoria_vehiculo", dropna=False)[
                "_target"
            ].mean().items()
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values, _ = self.predict_with_sources(X)
        return values

    def predict_with_sources(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Devuelve estimaciones y nivel de respaldo utilizado por fila."""

        if not hasattr(self, "global_mean_"):
            raise ValueError("Baseline has not been fitted yet.")
        frame = _baseline_keys(X)
        estimates: list[float] = []
        sources: list[str] = []
        for row in frame.itertuples(index=False):
            pair = (str(row.marca), str(row.categoria_vehiculo))
            if pair in self.brand_segment_means_:
                estimates.append(self.brand_segment_means_[pair])
                sources.append("marca_y_segmento")
            elif str(row.marca) in self.brand_means_:
                estimates.append(self.brand_means_[str(row.marca)])
                sources.append("marca")
            elif str(row.categoria_vehiculo) in self.category_means_:
                estimates.append(self.category_means_[str(row.categoria_vehiculo)])
                sources.append("segmento")
            else:
                estimates.append(self.global_mean_)
                sources.append("media_global")
        return np.asarray(estimates, dtype=float), sources


def temporal_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    year_column: str = YEAR_COLUMN,
    train_end_year: int | None = 2018,
    validation_end_year: int | None = 2021,
    adaptive_fallback: bool = True,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
) -> TemporalSplit:
    """Divide cronológicamente, con respaldo por años distintos. Utiliza los cortes
    documentados 1995–2018 / 2019–2021 / 2022+ cuando hay tres particiones. En extractos
    menores no mezcla filas: asigna años tempranos a entrenamiento, después validación y
    test. Antes del ajuste, el embargo de madurez de tres años elimina etiquetas no
    disponibles en el siguiente lanzamiento.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Temporal splitting requires a pandas DataFrame.")
    require_columns(frame, (year_column,), context="Temporal split input")
    if (train_end_year is None) != (validation_end_year is None):
        raise ValueError("Provide both temporal boundaries or neither of them.")
    if observation_window_years < 1:
        raise ValueError("observation_window_years must be at least one.")
    if (
        train_end_year is not None
        and validation_end_year is not None
        and train_end_year >= validation_end_year
    ):
        raise ValueError("train_end_year must be strictly earlier than validation_end_year.")

    ordered = frame.copy()
    years_numeric = pd.to_numeric(ordered[year_column], errors="coerce")
    if ordered.empty:
        raise ValueError("Temporal split input has no valid manufacturing years.")
    if (years_numeric.isna().any() or not np.isfinite(years_numeric).all()
            or not np.equal(years_numeric, np.floor(years_numeric)).all()):
        raise ValueError("Temporal split years must be finite whole calendar years.")
    ordered[year_column] = years_numeric.astype(int)
    ordered = ordered.sort_values(year_column, kind="stable")

    if train_end_year is not None and validation_end_year is not None:
        train = ordered.loc[ordered[year_column] <= train_end_year].copy()
        validation = ordered.loc[
            (ordered[year_column] > train_end_year)
            & (ordered[year_column] <= validation_end_year)
        ].copy()
        test = ordered.loc[ordered[year_column] > validation_end_year].copy()
        if not train.empty and not validation.empty and not test.empty:
            return _apply_label_maturity_embargo(
                TemporalSplit(
                train=train,
                validation=validation,
                test=test,
                strategy="documented_fixed_years",
                train_end_year=int(train_end_year),
                validation_end_year=int(validation_end_year),
                label_maturity_years=observation_window_years,
                ),
                year_column=year_column,
            )
        if not adaptive_fallback:
            return _apply_label_maturity_embargo(
                TemporalSplit(
                train=train,
                validation=validation,
                test=test,
                strategy="documented_fixed_years_incomplete",
                train_end_year=int(train_end_year),
                validation_end_year=int(validation_end_year),
                label_maturity_years=observation_window_years,
                ),
                year_column=year_column,
            )

    years = sorted(int(year) for year in ordered[year_column].unique())
    count = len(years)
    if count == 1:
        return _apply_label_maturity_embargo(
            TemporalSplit(
                train=ordered.copy(),
                validation=ordered.iloc[0:0].copy(),
                test=ordered.iloc[0:0].copy(),
                strategy="single_year_train_only",
                train_end_year=years[0],
                validation_end_year=None,
                label_maturity_years=observation_window_years,
            ),
            year_column=year_column,
        )
    if count == 2:
        train_years, validation_years, test_years = years[:1], years[1:], []
    else:
        train_count = max(1, int(np.floor(count * 0.60)))
        train_count = min(train_count, count - 2)
        validation_count = max(1, int(np.floor(count * 0.20)))
        validation_count = min(validation_count, count - train_count - 1)
        train_years = years[:train_count]
        validation_years = years[train_count : train_count + validation_count]
        test_years = years[train_count + validation_count :]

    train = ordered.loc[ordered[year_column].isin(train_years)].copy()
    validation = ordered.loc[ordered[year_column].isin(validation_years)].copy()
    test = ordered.loc[ordered[year_column].isin(test_years)].copy()
    return _apply_label_maturity_embargo(
        TemporalSplit(
            train=train,
            validation=validation,
            test=test,
            strategy="adaptive_distinct_years",
            train_end_year=max(train_years) if train_years else None,
            validation_end_year=max(validation_years) if validation_years else None,
            label_maturity_years=observation_window_years,
        ),
        year_column=year_column,
    )


def _apply_label_maturity_embargo(
    split: TemporalSplit,
    *,
    year_column: str,
) -> TemporalSplit:
    """Elimina etiquetas aún desconocidas en el siguiente límite temporal. La etiqueta de y
    cubre y hasta y + observation_window_years - 1. Solo ventanas terminadas antes del
    primer lanzamiento de validación son entrenables; la misma regla se aplica al incorporar
    validación antes del test. El test no se filtra por una fecha posterior porque solo se
    evalúa, no ajusta ni selecciona.
    """

    train = split.train.copy()
    validation = split.validation.copy()
    train_embargoed = train.iloc[0:0].copy()
    validation_embargoed = validation.iloc[0:0].copy()

    next_partition = validation if not validation.empty else split.test
    if not next_partition.empty:
        first_validation_year = int(pd.to_numeric(next_partition[year_column], errors="raise").min())
        latest_train_year = first_validation_year - split.label_maturity_years
        eligible = pd.to_numeric(train[year_column], errors="raise").le(latest_train_year)
        train_embargoed = train.loc[~eligible].copy()
        train = train.loc[eligible].copy()

    if not split.test.empty and not validation.empty:
        first_test_year = int(pd.to_numeric(split.test[year_column], errors="raise").min())
        latest_validation_year = first_test_year - split.label_maturity_years
        eligible = pd.to_numeric(validation[year_column], errors="raise").le(latest_validation_year)
        validation_embargoed = validation.loc[~eligible].copy()
        validation = validation.loc[eligible].copy()

    return TemporalSplit(
        train=train,
        validation=validation,
        test=split.test.copy(),
        strategy=f"{split.strategy}_label_maturity_embargo",
        train_end_year=split.train_end_year,
        validation_end_year=split.validation_end_year,
        train_embargoed=train_embargoed,
        validation_embargoed=validation_embargoed,
        label_maturity_years=split.label_maturity_years,
    )


def fit_temporal_target_normalizer(
    training: pd.DataFrame,
    *,
    raw_score_column: str,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
) -> TemporalTargetNormalizer:
    """Ajusta la escala del objetivo con etiquetas elegibles para entrenamiento."""

    require_columns(
        training,
        (YEAR_COLUMN, "categoria_vehiculo", raw_score_column),
        context="Temporal model training data",
    )
    if training.empty:
        raise ValueError("Cannot fit a target normalizer on an empty training partition.")
    values = pd.to_numeric(training[raw_score_column], errors="coerce").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{raw_score_column} must be finite in the training partition.")
    global_mean = float(values.mean())
    global_std = float(values.std(ddof=0))
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 1.0
    categories = _normalized_categories(training["categoria_vehiculo"])
    working = pd.DataFrame({"_category": categories, "_raw": values.to_numpy(dtype=float)})
    stats: dict[str, dict[str, float]] = {}
    for category, group in working.groupby("_category", sort=True):
        group_values = group["_raw"].to_numpy(dtype=float)
        std = float(np.std(group_values, ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = global_std
        stats[str(category)] = {"mean": float(np.mean(group_values)), "std": std}
    years = pd.to_numeric(training[YEAR_COLUMN], errors="raise")
    return TemporalTargetNormalizer(
        raw_score_column=raw_score_column,
        segment_statistics=stats,
        global_mean=global_mean,
        global_std=global_std,
        train_end_year=int(years.max()),
        observation_window_years=observation_window_years,
    )


def _normalized_categories(values: pd.Series) -> pd.Series:
    """Crea claves de segmento estables y no nulas sin aprender de filas posteriores."""

    series = values.astype("object")
    output = series.where(series.notna(), "sin_categoria").astype(str).str.strip().str.lower()
    return output.mask(output.eq(""), "sin_categoria")


def train_and_select_model(
    gold: pd.DataFrame,
    *,
    artifact_dir: str | Path | None = None,
    persist: bool = True,
    train_end_year: int | None = 2018,
    validation_end_year: int | None = 2021,
    advanced_improvement_threshold: float = 0.10,
    random_state: int = 73,
    advanced_min_rows: int = 8,
    observation_window_years: int = OBSERVATION_WINDOW_YEARS,
) -> TrainingResult:
    """Entrena y selecciona en un experimento cronológico estricto. Decide solo con validación;
    después reajusta el estimador seleccionado con entrenamiento y validación y evalúa el
    test separado. Random Forest exige reducción de MAE frente a Ridge estrictamente mayor
    al umbral (10% por defecto). Una ejecución no convierte datos ya examinados en un test
    nuevo intacto.
    """

    if not 0 <= advanced_improvement_threshold < 1:
        raise ValueError("advanced_improvement_threshold must be in [0, 1).")
    if advanced_min_rows < 2:
        raise ValueError("advanced_min_rows must be at least 2.")
    if observation_window_years < 1:
        raise ValueError("observation_window_years must be at least one.")

    # Calcular la huella de la entrada original. Reconstruir el objetivo
    # tras conocer el límite efectivo de entrenamiento; el objetivo de
    # visualización no debe influir en la comprobación de vigencia ni en el ajuste.
    clean_gold, dropped_rows, raw_score_column = _validated_training_frame(gold)
    input_fingerprint = dataset_fingerprint(gold)
    split = temporal_train_validation_test_split(
        clean_gold,
        train_end_year=train_end_year,
        validation_end_year=validation_end_year,
        observation_window_years=observation_window_years,
    )
    warnings: list[str] = []
    if split.train.empty:
        raise ValueError(
            "El embargo de maduración de etiquetas dejó el train vacío. "
            "Se necesitan cohortes con ventana de recalls completa antes de la primera validación."
        )
    target_normalizer = fit_temporal_target_normalizer(
        split.train,
        raw_score_column=raw_score_column,
        observation_window_years=observation_window_years,
    )
    split = _with_train_scaled_target(split, target_normalizer)
    if split.validation.empty:
        warnings.append(
            "No hay cohorte de validación temporal; se mantiene el baseline por prudencia."
        )
    if split.test.empty:
        warnings.append("No hay cohorte de test temporal disponible para la evaluación final.")

    train_x, train_y = _xy(split.train)
    candidate_models: dict[str, _Candidate] = {}
    candidate_models["baseline"] = _fit_candidate("baseline", train_x, train_y)
    candidate_models["ridge"] = _fit_candidate("ridge", train_x, train_y)
    if len(train_x) >= advanced_min_rows:
        candidate_models["random_forest"] = _fit_candidate(
            "random_forest", train_x, train_y, random_state=random_state
        )
    else:
        candidate_models["random_forest"] = _Candidate(
            name="random_forest",
            error=(
                f"No entrenado: {len(train_x)} filas de train; se requieren "
                f"{advanced_min_rows} para el candidato avanzado."
            ),
        )

    validation_x, validation_y = _xy(split.validation, allow_empty=True)
    validation_predictions: dict[str, np.ndarray] = {}
    for candidate in candidate_models.values():
        if not candidate.available:
            continue
        if validation_x.empty:
            candidate.validation_metrics = _empty_metrics()
            continue
        try:
            artifact = candidate.as_artifact(train_x, split)
            prediction = artifact.predict(validation_x)
            validation_predictions[candidate.name] = prediction
            candidate.validation_metrics = regression_metrics(validation_y, prediction)
        except Exception as exc:  # el fallo de un candidato no debe descartar el baseline
            candidate.error = f"Validation failure: {type(exc).__name__}: {exc}"
            candidate.validation_metrics = _empty_metrics()

    selected_name, selection_reason, advanced_reduction = _select_candidate(
        candidate_models,
        has_validation=not validation_x.empty,
        advanced_improvement_threshold=advanced_improvement_threshold,
    )
    final_fit = pd.concat([split.train, split.validation], axis=0, ignore_index=True)
    final_x, final_y = _xy(final_fit)
    try:
        final_candidate = _fit_candidate(
            selected_name,
            final_x,
            final_y,
            random_state=random_state,
        )
        if not final_candidate.available:
            raise RuntimeError(final_candidate.error or "The selected model could not be refitted.")
    except Exception as exc:  # conservar un baseline válido evita inutilizar el producto
        warnings.append(
            f"El modelo seleccionado no pudo reentrenarse ({type(exc).__name__}); "
            "se utiliza el baseline."
        )
        selected_name = "baseline"
        selection_reason = "Fallback al baseline tras un fallo de reentrenamiento."
        final_candidate = _fit_candidate("baseline", final_x, final_y)

    selected_validation_metrics = candidate_models[selected_name].validation_metrics
    expected_mae = _metric_value(selected_validation_metrics, "mae")
    test_x, test_y = _xy(split.test, allow_empty=True)
    final_split = TemporalSplit(
        train=final_fit,
        validation=split.validation.iloc[0:0].copy(),
        test=split.test,
        strategy=split.strategy,
        train_end_year=split.train_end_year,
        validation_end_year=split.validation_end_year,
    )
    artifact = final_candidate.as_artifact(
        final_x,
        final_split,
        expected_mae=expected_mae,
    )
    test_prediction = np.asarray([], dtype=float)
    test_metrics = _empty_metrics()
    if not test_x.empty:
        test_prediction = artifact.predict(test_x)
        test_metrics = regression_metrics(test_y, test_prediction)
        # El MAE del test reservado es la estimación de error mostrada en la interfaz.
        # No interviene en la selección del modelo.
        test_mae = _metric_value(test_metrics, "mae")
        if test_mae is not None:
            artifact.expected_mae = test_mae

    validation_frame = _prediction_frame(
        split.validation,
        validation_predictions.get(selected_name),
        model_name=selected_name,
        partition="validation",
    )
    test_frame = _prediction_frame(
        split.test,
        test_prediction if len(test_prediction) else None,
        model_name=selected_name,
        partition="test",
    )
    metrics = _build_metrics_report(
        split=split,
        candidates=candidate_models,
        selected_name=selected_name,
        selection_reason=selection_reason,
        advanced_reduction=advanced_reduction,
        threshold=advanced_improvement_threshold,
        test_metrics=test_metrics,
        dropped_rows=dropped_rows,
        warnings=warnings,
    )
    metrics["target_normalizer"] = target_normalizer.to_dict()
    metrics["dataset_fingerprint"] = input_fingerprint
    metrics["fuente_demo"] = _contains_demo_rows(gold)
    artifact.metadata = {
        "metrics_filename": METRICS_FILENAME,
        "selection_reason": selection_reason,
        "split": split.metadata(),
        "feature_columns": list(artifact.feature_columns),
        "target_column": TARGET_COLUMN,
        "target_normalizer": target_normalizer.to_dict(),
        "dataset_fingerprint": input_fingerprint,
        "fuente_demo": _contains_demo_rows(gold),
        "expected_mae_source": (
            "held_out_test" if test_metrics["mae"] is not None
            else "temporal_validation" if expected_mae is not None else "unavailable"
        ),
        "prediction_available_from_year": int(final_fit[YEAR_COLUMN].max()) + observation_window_years,
        "label_maturity_embargo": {
            "observation_window_years": observation_window_years,
            "train_labels_known_before_first_validation_launch": True,
            "validation_labels_known_before_first_test_launch": True,
        },
    }
    result = TrainingResult(
        artifact=artifact,
        metrics=metrics,
        validation_predictions=validation_frame,
        test_predictions=test_frame,
    )
    if persist:
        destination = Path(artifact_dir) if artifact_dir is not None else ProjectPaths.discover().artifacts_dir
        result.artifact_paths = persist_training_result(result, destination)
    return result


def persist_training_result(result: TrainingResult, artifact_dir: str | Path) -> dict[str, Path]:
    """Guarda modelo e informes auditables sin alterar fuentes."""

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": destination / MODEL_FILENAME,
        "metrics": destination / METRICS_FILENAME,
        "validation_predictions": destination / VALIDATION_PREDICTIONS_FILENAME,
        "test_predictions": destination / TEST_PREDICTIONS_FILENAME,
        "feature_attributions": destination / ATTRIBUTIONS_FILENAME,
    }
    joblib.dump(result.artifact, paths["model"])
    _write_json(paths["metrics"], result.metrics)
    result.validation_predictions.to_csv(paths["validation_predictions"], index=False)
    result.test_predictions.to_csv(paths["test_predictions"], index=False)
    # Importar bajo demanda para permitir entrenar aunque falte una dependencia
    # opcional de explicabilidad utilizada únicamente por la interfaz.
    try:
        from .explainability import global_feature_attributions

        attributions = global_feature_attributions(result.artifact).to_dict(orient="records")
    except Exception as exc:  # priorizar la conservación del modelo y sus métricas
        attributions = [{"warning": f"No se pudo calcular atribución: {type(exc).__name__}"}]
    _write_json(paths["feature_attributions"], attributions)
    return paths


def load_model_artifact(path: str | Path | None = None) -> ReliabilityModelArtifact:
    """Carga ReliabilityModelArtifact persistido para la interfaz."""

    source = Path(path) if path is not None else ProjectPaths.discover().model_path
    if not source.exists():
        raise FileNotFoundError(f"No trained model found at {source}.")
    artifact = joblib.load(source)
    if not isinstance(artifact, ReliabilityModelArtifact):
        raise TypeError(f"{source} does not contain an AutoReliability model artifact.")
    return artifact


def predict_with_artifact(
    artifact: ReliabilityModelArtifact | str | Path,
    data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    include_explanations: bool = True,
    top_factors: int = 3,
) -> pd.DataFrame:
    """Devuelve predicciones para cohortes completas o recientes. Solo necesita las
    características documentadas, por lo que admite ventanas aún incompletas; nunca consulta
    resultados propios del vehículo.
    """

    resolved = load_model_artifact(artifact) if isinstance(artifact, (str, Path)) else artifact
    if not isinstance(resolved, ReliabilityModelArtifact):
        raise TypeError("artifact must be a ReliabilityModelArtifact or its saved path.")
    frame = _as_dataframe(data)
    predictions = resolved.predict(frame)
    output = frame.copy()
    if ID_COLUMN not in output.columns:
        output[ID_COLUMN] = [f"consulta_{position + 1}" for position in range(len(output))]
    output["prediccion_indice_100"] = np.round(predictions, 2)
    output["mae"] = resolved.expected_mae
    output["mae_esperado"] = resolved.expected_mae
    output["modelo_seleccionado"] = resolved.model_name
    output["fecha_ejecucion"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if include_explanations:
        try:
            from .explainability import explain_prediction, summarize_attributions

            summaries: list[str] = []
            for _, row in output.iterrows():
                factors = explain_prediction(resolved, row.to_frame().T, top_n=top_factors)
                summaries.append(summarize_attributions(factors))
            output["explicacion_factores"] = summaries
        except Exception:
            output["explicacion_factores"] = "Explicación de factores no disponible."
    return output


def prepare_feature_frame(
    data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Valida y tipa características sin ajustar ningún paso dependiente de datos."""

    frame = _as_dataframe(data)
    required = tuple(feature_columns)
    require_columns(frame, required, context="Prediction input")
    features = frame.loc[:, required].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
            present = features[column].dropna()
            if not np.isfinite(present.to_numpy(dtype=float)).all():
                raise ValueError(f"{column} must be finite or null.")
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in features.columns:
            series = features[column].astype("object")
            features[column] = series.where(series.notna(), "__unknown__").astype(str).str.strip().str.lower()
            features.loc[features[column].eq(""), column] = "__unknown__"
    return features


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int | None]:
    """Calcula MAE y RMSE en la escala pública 0–100."""

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return _empty_metrics()
    actual, predicted = actual[valid], predicted[valid]
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "n": len(actual),
    }


@dataclass
class _Candidate:
    name: str
    estimator: Any | None = None
    target_mean: float | None = None
    target_std: float | None = None
    validation_metrics: dict[str, float | int | None] = field(
        default_factory=lambda: {"mae": None, "rmse": None, "n": 0}
    )
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.estimator is not None and self.target_mean is not None and self.target_std is not None

    def as_artifact(
        self,
        fit_x: pd.DataFrame,
        split: TemporalSplit,
        *,
        expected_mae: float | None = None,
    ) -> ReliabilityModelArtifact:
        if not self.available:
            raise RuntimeError(self.error or f"Candidate {self.name} is unavailable.")
        year_max = None
        if YEAR_COLUMN in fit_x.columns and not fit_x.empty:
            years = pd.to_numeric(fit_x[YEAR_COLUMN], errors="coerce").dropna()
            year_max = int(years.max()) if not years.empty else None
        return ReliabilityModelArtifact(
            estimator=self.estimator,
            model_name=self.name,
            target_mean=float(self.target_mean),
            target_std=float(self.target_std),
            feature_references=_feature_references(fit_x),
            expected_mae=expected_mae,
            train_year_max=year_max or split.train_end_year,
        )


def _fit_candidate(
    name: str,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 73,
) -> _Candidate:
    """Ajusta un candidato con transformación de entrenamiento y conserva errores."""

    if x.empty:
        return _Candidate(name=name, error="No hay filas de entrenamiento.")
    mean, standard_deviation = _target_statistics(y)
    standardized_target = (np.asarray(y, dtype=float) - mean) / standard_deviation
    try:
        if name == "baseline":
            estimator: Any = BrandSegmentMeanBaseline()
        elif name == "ridge":
            estimator = _build_ridge_pipeline()
        elif name == "random_forest":
            estimator = _build_random_forest_pipeline(
                random_state=random_state,
                train_rows=len(x),
            )
        else:
            raise ValueError(f"Unknown candidate model: {name}")
        estimator.fit(prepare_feature_frame(x), standardized_target)
        return _Candidate(
            name=name,
            estimator=estimator,
            target_mean=mean,
            target_std=standard_deviation,
        )
    except Exception as exc:
        return _Candidate(name=name, error=f"Training failure: {type(exc).__name__}: {exc}")


class BrandSegmentMedianImputer(TransformerMixin, BaseEstimator):
    """Imputa campos técnicos con medianas de marca/segmento del ajuste. Grupos desconocidos
    recurren a la mediana de segmento y después a la global de SimpleImputer, ambas de
    entrenamiento. El historial de recalls no se completa con otros años antes de este
    preprocesamiento ajustado.
    """

    technical_columns = ("mediana_cilindros", "mediana_cv")

    def fit(self, X: pd.DataFrame, y: Sequence[float] | None = None) -> BrandSegmentMedianImputer:
        features = prepare_feature_frame(X)
        self.group_medians_ = features.groupby(["marca", "categoria_vehiculo"])[list(self.technical_columns)].median()
        self.segment_medians_ = features.groupby("categoria_vehiculo")[list(self.technical_columns)].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = prepare_feature_frame(X)
        keys = pd.MultiIndex.from_frame(output[["marca", "categoria_vehiculo"]])
        for column in self.technical_columns:
            group_values = self.group_medians_[column].reindex(keys).to_numpy()
            segment_values = output["categoria_vehiculo"].map(self.segment_medians_[column]).to_numpy()
            output[column] = output[column].fillna(pd.Series(group_values, index=output.index))
            output[column] = output[column].fillna(pd.Series(segment_values, index=output.index))
        return output


def _build_ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("group_imputer", BrandSegmentMedianImputer()),
            ("preprocessor", _build_preprocessor(scale_numeric=True)),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def _build_random_forest_pipeline(*, random_state: int, train_rows: int) -> Pipeline:
    # Las cohortes pequeñas necesitan hojas menos restrictivas; en cohortes
    # mayores se introduce regularización moderada para limitar el sobreajuste.
    min_leaf = 1 if train_rows < 20 else 2
    return Pipeline(
        steps=[
            ("group_imputer", BrandSegmentMedianImputer()),
            ("preprocessor", _build_preprocessor(scale_numeric=False)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=min_leaf,
                    max_features=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", _numeric_imputer())]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=numeric_steps)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__unknown__")),
            ("encoder", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURE_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _numeric_imputer() -> SimpleImputer:
    # keep_empty_features evita eliminar silenciosamente una característica
    # cuando todos sus valores técnicos faltan en una cohorte pequeña.
    kwargs: dict[str, Any] = {"strategy": "median"}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        kwargs["keep_empty_features"] = True
    return SimpleImputer(**kwargs)


def _one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:  # pragma: no cover - compatibilidad con versiones anteriores de scikit-learn
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def _select_candidate(
    candidates: Mapping[str, _Candidate],
    *,
    has_validation: bool,
    advanced_improvement_threshold: float,
) -> tuple[str, str, float | None]:
    """Selecciona el candidato elegible con menor MAE sin debilitar la regla de Random Forest.
    Su elegibilidad se compara estrictamente con Ridge, aunque el baseline lo supere; además
    debe superar al baseline para ser seleccionado. Un comparador Ridge débil no justifica
    elegir complejidad.
    """

    baseline = candidates["baseline"]
    ridge = candidates["ridge"]
    forest = candidates["random_forest"]
    if not has_validation:
        return "baseline", "Sin validación temporal: se conserva el baseline conservador.", None

    baseline_mae = _metric_value(baseline.validation_metrics, "mae")
    ridge_mae = _metric_value(ridge.validation_metrics, "mae")
    forest_mae = _metric_value(forest.validation_metrics, "mae")
    eligible: dict[str, float] = {}
    if baseline_mae is not None:
        eligible["baseline"] = baseline_mae
    if ridge_mae is not None:
        eligible["ridge"] = ridge_mae

    reduction: float | None = None
    forest_eligible = False
    if forest_mae is not None and ridge_mae is not None and ridge_mae > 0:
        reduction = (ridge_mae - forest_mae) / ridge_mae
        forest_eligible = reduction > advanced_improvement_threshold
        if forest_eligible:
            eligible["random_forest"] = forest_mae

    if not eligible:
        return "baseline", "Ningún candidato produjo una métrica válida; se conserva el baseline.", reduction
    # En caso de empate, priorizar de forma estable el método más sencillo y explicable.
    preference = {"baseline": 0, "ridge": 1, "random_forest": 2}
    selected = min(eligible, key=lambda name: (eligible[name], preference[name]))
    if selected == "random_forest":
        return (
            selected,
            (
                "Random Forest reduce el MAE de Ridge en "
                f"{(reduction or 0.0) * 100:.1f}% y también obtiene el menor MAE "
                "entre los candidatos elegibles."
            ),
            reduction,
        )
    if selected == "ridge":
        if forest_mae is not None and reduction is not None and not forest_eligible:
            return (
                selected,
                (
                    "Se prefiere Ridge por interpretabilidad: la mejora del modelo avanzado "
                    f"({reduction * 100:.1f}%) no supera el umbral de "
                    f"{advanced_improvement_threshold * 100:.1f}%."
                ),
                reduction,
            )
        return selected, "Ridge obtiene el menor MAE entre los candidatos elegibles.", reduction
    if forest_eligible:
        return (
            selected,
            "El baseline conserva el menor MAE aunque Random Forest supera el umbral frente a Ridge.",
            reduction,
        )
    return selected, (
        "El baseline obtiene el menor MAE entre los candidatos elegibles. "
        "Random Forest no cumple el umbral de mejora exigido frente a Ridge."
    ), reduction


def _validated_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    """Rechaza Gold inseguro antes de que la partición oculte problemas. Permite campos
    técnicos ausentes porque la imputación se ajusta dentro del entrenamiento. Etiquetas,
    identificadores o años ausentes/inválidos, duplicados y cohortes incompletas provocan
    error en vez de reparaciones injustificadas.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Training input must be a pandas DataFrame.")
    raw_score_column = next((column for column in RAW_RECALL_SCORE_COLUMNS if column in frame.columns), None)
    if raw_score_column is None:
        expected = " or ".join(RAW_RECALL_SCORE_COLUMNS)
        raise ValueError(f"Gold training dataset requires a raw recall label ({expected}).")
    required = (ID_COLUMN, YEAR_COLUMN, "categoria_vehiculo", raw_score_column, *MODEL_FEATURE_COLUMNS)
    require_columns(frame, required, context="Gold training dataset")
    if frame.empty:
        raise ValueError("Gold training dataset is empty.")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Gold training dataset requires non-null id_vehiculo_ano values.")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Gold training dataset requires unique id_vehiculo_ano values.")

    clean = frame.copy()
    years = pd.to_numeric(clean[YEAR_COLUMN], errors="coerce")
    if years.isna().any() or not np.isfinite(years.to_numpy(dtype=float)).all():
        raise ValueError("ano_fabricacion must contain finite values for every Gold training row.")
    if not np.equal(years.to_numpy(dtype=float), np.floor(years.to_numpy(dtype=float))).all():
        raise ValueError("ano_fabricacion must contain whole calendar years.")
    clean[YEAR_COLUMN] = years.astype(int)
    if not clean[YEAR_COLUMN].between(MIN_MODEL_YEAR, CURRENT_YEAR).all():
        raise ValueError(f"ano_fabricacion must be between {MIN_MODEL_YEAR} and {CURRENT_YEAR}.")

    raw = pd.to_numeric(clean[raw_score_column], errors="coerce")
    if raw.isna().any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError(f"{raw_score_column} must contain finite values for every Gold training row.")
    if raw.lt(0).any():
        raise ValueError(f"{raw_score_column} cannot be negative.")
    clean[raw_score_column] = raw.astype(float)

    for column in NUMERIC_FEATURE_COLUMNS:
        numeric = pd.to_numeric(clean[column], errors="coerce")
        # SimpleImputer trata los nulos usando solo entrenamiento. Los infinitos,
        # a diferencia de los nulos, no admiten una corrección justificable con la mediana.
        non_null = numeric.notna()
        if non_null.any() and not np.isfinite(numeric.loc[non_null].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains a non-finite value; use null for an unknown measurement.")
        clean[column] = numeric

    completeness_columns = [
        column for column in ("es_cohorte_completa", "cohorte_completa") if column in clean.columns
    ]
    for column in completeness_columns:
        complete = clean[column].map(_as_complete_flag)
        if not complete.all():
            raise ValueError(
                f"Gold training dataset contains incomplete or unknown cohorts in {column}; "
                "only completed three-year labels may be trained."
            )
    return clean, 0, raw_score_column


def _as_complete_flag(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "si", "sí"}
    return bool(value)


def _with_train_scaled_target(
    split: TemporalSplit,
    normalizer: TemporalTargetNormalizer,
) -> TemporalSplit:
    """Copia particiones y reconstruye sus objetivos con estadísticas de entrenamiento."""

    def scaled(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if not output.empty:
            output[TARGET_COLUMN] = normalizer.transform(output)
        elif TARGET_COLUMN not in output.columns:
            output[TARGET_COLUMN] = pd.Series(dtype=float)
        return output

    return TemporalSplit(
        train=scaled(split.train),
        validation=scaled(split.validation),
        test=scaled(split.test),
        strategy=split.strategy,
        train_end_year=split.train_end_year,
        validation_end_year=split.validation_end_year,
        train_embargoed=split.train_embargoed.copy(),
        validation_embargoed=split.validation_embargoed.copy(),
        label_maturity_years=split.label_maturity_years,
    )


def _contains_demo_rows(frame: pd.DataFrame) -> bool:
    if "fuente_demo" not in frame.columns:
        return False
    return bool(frame["fuente_demo"].map(_as_complete_flag).any())


def _xy(frame: pd.DataFrame, *, allow_empty: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    if frame.empty:
        if allow_empty:
            return frame.copy(), pd.Series(dtype=float, name=TARGET_COLUMN)
        raise ValueError("Training partition is empty.")
    return frame.copy(), pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(float)


def _target_statistics(y: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(y, dtype=float)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=0))
    # Una cohorte temprana constante es válida. La escala unitaria mantiene
    # estable la transformación inversa y permite evaluar todos los candidatos.
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-12:
        standard_deviation = 1.0
    return mean, standard_deviation


def _feature_references(frame: pd.DataFrame) -> dict[str, Any]:
    """Calcula referencias de entrenamiento para explicaciones independientes del modelo."""

    references: dict[str, Any] = {}
    prepared = prepare_feature_frame(frame)
    for column in NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(prepared[column], errors="coerce").dropna()
        references[column] = float(values.median()) if not values.empty else 0.0
    for column in CATEGORICAL_FEATURE_COLUMNS:
        values = prepared[column].replace("", np.nan).dropna()
        references[column] = str(values.mode().iloc[0]) if not values.empty else "__unknown__"
    return references


def _prediction_frame(
    source: pd.DataFrame,
    prediction: np.ndarray | None,
    *,
    model_name: str,
    partition: str,
) -> pd.DataFrame:
    columns = [
        ID_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        "prediccion_indice_100",
        "error_absoluto",
        "modelo",
        "particion",
    ]
    if source.empty or prediction is None:
        return pd.DataFrame(columns=columns)
    output = source.loc[:, [column for column in (ID_COLUMN, YEAR_COLUMN, TARGET_COLUMN) if column in source]].copy()
    output["prediccion_indice_100"] = np.round(np.asarray(prediction, dtype=float), 4)
    output["error_absoluto"] = np.round(
        np.abs(output[TARGET_COLUMN].astype(float) - output["prediccion_indice_100"]), 4
    )
    output["modelo"] = model_name
    output["particion"] = partition
    return output.reindex(columns=columns)


def _build_metrics_report(
    *,
    split: TemporalSplit,
    candidates: Mapping[str, _Candidate],
    selected_name: str,
    selection_reason: str,
    advanced_reduction: float | None,
    threshold: float,
    test_metrics: Mapping[str, Any],
    dropped_rows: int,
    warnings: Sequence[str],
) -> dict[str, Any]:
    candidate_report: dict[str, Any] = {}
    for name, candidate in candidates.items():
        candidate_report[name] = {
            "available": candidate.available,
            "validation": dict(candidate.validation_metrics),
            "error": candidate.error,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": TARGET_COLUMN,
        "target_scale": "0-100 (100 = menor propensión histórica estimada a recalls)",
        "features": list(MODEL_FEATURE_COLUMNS),
        "excluded_feature": "score_recalls_bruto (outcome leakage prevention)",
        "split": split.metadata(),
        "rows_dropped_invalid_target_or_year": dropped_rows,
        "candidates": candidate_report,
        "selection": {
            "selected_model": selected_name,
            "reason": selection_reason,
            "advanced_improvement_threshold_pct": round(threshold * 100, 2),
            "advanced_reduction_vs_ridge_pct": (
                round(advanced_reduction * 100, 4) if advanced_reduction is not None else None
            ),
            "rule": "Random Forest is selected only when validation MAE improves on Ridge by > threshold.",
        },
        "final_test": dict(test_metrics),
        "warnings": list(warnings),
    }


def _baseline_keys(data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = _as_dataframe(data)
    require_columns(frame, CATEGORICAL_FEATURE_COLUMNS, context="Baseline input")
    output = frame.loc[:, CATEGORICAL_FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURE_COLUMNS:
        series = output[column].astype("object")
        output[column] = series.where(series.notna(), "__unknown__").astype(str).str.strip().str.lower()
        output.loc[output[column].eq(""), column] = "__unknown__"
    return output


def _as_dataframe(
    data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        # La interfaz suele proporcionar diccionarios de escalares. Los diccionarios
        # de vectores también permiten predicciones por lotes y pandas los admite.
        scalar_values = all(
            not isinstance(value, (list, tuple, np.ndarray, pd.Series)) for value in data.values()
        )
        return pd.DataFrame([data]) if scalar_values else pd.DataFrame(data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return pd.DataFrame(list(data))
    raise TypeError("Expected a DataFrame, mapping, or sequence of mappings.")


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None and np.isfinite(value) else None


def _empty_metrics() -> dict[str, float | int | None]:
    return {"mae": None, "rmse": None, "n": 0}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
