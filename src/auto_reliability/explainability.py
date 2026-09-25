"""Atribuciones legibles y auditables. Ridge utiliza contribuciones locales con signo respecto
a referencias de entrenamiento antes de recortar el índice. El árbol usa valores Shapley
exactos sobre cinco características y un único fondo de entrenamiento: evalúa 32 coaliciones
por lote, sin dependencia opcional. La importancia global por impureza se identifica por
separado y nunca se llama SHAP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from .modeling import (
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    BrandSegmentMeanBaseline,
    ReliabilityModelArtifact,
    prepare_feature_frame,
)

SPANISH_FEATURE_NAMES: dict[str, str] = {
    "marca": "la marca",
    "categoria_vehiculo": "la categoría del vehículo",
    "mediana_cilindros": "los cilindros",
    "mediana_cv": "la potencia",
    "hist_fiabilidad_marca": "el historial previo de la marca",
}


def global_feature_attributions(artifact: ReliabilityModelArtifact) -> pd.DataFrame:
    """Devuelve importancia global agrupada al esquema original. Los valores son comparables
    dentro del modelo y suman uno si el estimador aporta importancias nativas. El baseline
    tiene un factor semántico basado en medias históricas de marca/segmento.
    """

    if artifact.model_name == "baseline" or isinstance(artifact.estimator, BrandSegmentMeanBaseline):
        return pd.DataFrame(
            [
                {
                    "feature": "marca_y_categoria",
                    "importance": 1.0,
                    "signed_effect": None,
                    "method": "media_histórica_marca_segmento",
                }
            ]
        )

    names, values, signed = _native_feature_importance(artifact)
    grouped: dict[str, float] = {feature: 0.0 for feature in MODEL_FEATURE_COLUMNS}
    signed_grouped: dict[str, float] = {feature: 0.0 for feature in MODEL_FEATURE_COLUMNS}
    for name, value, direction in zip(names, values, signed, strict=False):
        source = _source_feature(name, artifact.feature_columns)
        grouped[source] = grouped.get(source, 0.0) + abs(float(value))
        signed_grouped[source] = signed_grouped.get(source, 0.0) + float(direction)
    total = sum(grouped.values())
    method = "coeficientes_ridge" if artifact.model_name == "ridge" else "importancia_random_forest"
    rows = [
        {
            "feature": feature,
            "importance": round(value / total, 6) if total else 0.0,
            "signed_effect": round(signed_grouped[feature], 6) if artifact.model_name == "ridge" else None,
            "method": method,
        }
        for feature, value in grouped.items()
    ]
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


def explain_prediction(
    artifact: ReliabilityModelArtifact,
    data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Explica una predicción mediante efectos con signo en puntos del índice. data debe
    contener una fila. Contribuciones positivas se asocian con mayor índice y menor
    propensión estimada; las negativas indican la dirección contraria.
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    features = prepare_feature_frame(data, feature_columns=artifact.feature_columns)
    if len(features) != 1:
        raise ValueError("explain_prediction expects exactly one input row.")
    if isinstance(artifact.estimator, BrandSegmentMeanBaseline):
        return _baseline_explanation(artifact, features)
    if artifact.model_name == "ridge":
        factors = _ridge_local_attributions(artifact, features)
    elif artifact.model_name == "random_forest":
        factors = _exact_reference_shapley(artifact, features)
    else:
        factors = _counterfactual_attributions(artifact, features)
    ordered = sorted(factors, key=lambda item: abs(float(item["contribution"])), reverse=True)
    for factor in ordered:
        value = features.iloc[0].get(factor["feature"])
        factor["value"] = None if pd.isna(value) else value
    return ordered[:top_n]


def _baseline_explanation(artifact: ReliabilityModelArtifact, features: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe la consulta al grupo realmente ajustado, no efectos técnicos contrafactuales."""
    _, levels = artifact.estimator.predict_with_sources(features)
    level = levels[0]
    row = features.iloc[0]
    reference = float(np.clip(artifact.estimator.global_mean_ * artifact.target_std + artifact.target_mean, 0, 100))
    score = float(artifact.predict(features)[0])
    values = {"marca_y_segmento": f"{row['marca']} / {row['categoria_vehiculo']}",
              "marca": str(row["marca"]), "segmento": str(row["categoria_vehiculo"]),
              "media_global": "Conjunto de entrenamiento"}
    return [{"feature": level, "value": values[level], "contribution": score - reference,
             "baseline_score": reference, "prediction_score": score,
             "direction": _direction(score - reference), "method": "media_grupo_vs_media_global_train",
             "causal": False}]


def explain_predictions(
    artifact: ReliabilityModelArtifact,
    data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    top_n: int = 3,
) -> list[list[dict[str, Any]]]:
    """Versión por lotes de explain_prediction."""

    features = prepare_feature_frame(data, feature_columns=artifact.feature_columns)
    return [explain_prediction(artifact, row.to_frame().T, top_n=top_n) for _, row in features.iterrows()]


def summarize_attributions(factors: Sequence[Mapping[str, Any]]) -> str:
    """Genera una explicación breve en castellano para interfaz y PDF."""

    if not factors:
        return "No hay factores explicativos disponibles para esta estimación."
    if factors[0].get("method") == "media_grupo_vs_media_global_train":
        factor = factors[0]
        labels = {"marca_y_segmento": "marca y segmento", "marca": "marca",
                  "segmento": "segmento", "media_global": "media global"}
        return (
            f"El baseline utiliza {labels[str(factor['feature'])]}: {factor['value']}. "
            f"La estimación del grupo es {float(factor['prediction_score']):.2f}/100, frente a "
            f"{float(factor['baseline_score']):.2f}/100 de la media global de entrenamiento "
            f"(diferencia: {float(factor['contribution']):+.2f} puntos). "
            "Potencia, cilindros e historial reciente no intervienen en este baseline; "
            "el perfil técnico se muestra únicamente como contexto. "
            "No distingue modelos dentro del mismo grupo ni demuestra causas de averías."
        )
    clauses: list[str] = []
    for factor in factors:
        feature = str(factor.get("feature", "factor"))
        label = SPANISH_FEATURE_NAMES.get(feature, feature.replace("_", " "))
        contribution = float(factor.get("contribution", 0.0))
        if contribution > 0:
            direction = "eleva"
        elif contribution < 0:
            direction = "reduce"
        else:
            direction = "apenas modifica"
        clauses.append(f"{label.capitalize()} {direction} el índice ({contribution:+.1f} puntos)")
    return (
        "; ".join(clauses)
        + ". Explicación determinista de asociaciones del modelo; no demuestra causalidad."
    )


def _native_feature_importance(
    artifact: ReliabilityModelArtifact) -> tuple[list[str], np.ndarray, np.ndarray]:
    pipeline = artifact.estimator
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Native feature attribution requires a sklearn Pipeline.")
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(regressor, "coef_"):
        signed = np.asarray(regressor.coef_, dtype=float).reshape(-1) * float(artifact.target_std)
        return names, np.abs(signed), signed
    if hasattr(regressor, "feature_importances_"):
        importance = np.asarray(regressor.feature_importances_, dtype=float).reshape(-1)
        return names, importance, np.zeros_like(importance)
    raise TypeError(f"{type(regressor).__name__} exposes no native feature importance.")


def _ridge_local_attributions(
    artifact: ReliabilityModelArtifact, features: pd.DataFrame
) -> list[dict[str, Any]]:
    pipeline = artifact.estimator
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    reference = pd.DataFrame([
        {feature: artifact.feature_references.get(feature, _default_reference(feature))
         for feature in artifact.feature_columns}
    ])
    # Usar exactamente la misma imputación ajustada por grupo que en la predicción.
    # Las canalizaciones antiguas sin este paso siguen siendo legibles.
    group_imputer = pipeline.named_steps.get("group_imputer")
    actual_input = group_imputer.transform(features) if group_imputer is not None else features
    reference_input = group_imputer.transform(reference) if group_imputer is not None else reference
    transformed = preprocessor.transform(actual_input)
    transformed_reference = preprocessor.transform(reference_input)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    if hasattr(transformed_reference, "toarray"):
        transformed_reference = transformed_reference.toarray()
    vector = (
        np.asarray(transformed, dtype=float).reshape(1, -1)[0]
        - np.asarray(transformed_reference, dtype=float).reshape(1, -1)[0]
    )
    coefficient = np.asarray(regressor.coef_, dtype=float).reshape(-1)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    contributions: dict[str, float] = {feature: 0.0 for feature in artifact.feature_columns}
    for name, contribution in zip(names, vector * coefficient * float(artifact.target_std), strict=False):
        source = _source_feature(name, artifact.feature_columns)
        contributions[source] = contributions.get(source, 0.0) + float(contribution)
    return [
        {
            "feature": feature,
            "contribution": round(value, 4),
            "direction": _direction(value),
            "method": "ridge_vs_referencia_train_antes_del_recorte",
            "causal": False,
        }
        for feature, value in contributions.items()
    ]


def _counterfactual_attributions(
    artifact: ReliabilityModelArtifact, features: pd.DataFrame
) -> list[dict[str, Any]]:
    """Sustituye una característica por su referencia de entrenamiento y mide el cambio."""

    original = float(artifact.predict(features)[0])
    factors: list[dict[str, Any]] = []
    for feature in artifact.feature_columns:
        comparison = features.copy()
        comparison.loc[:, feature] = artifact.feature_references.get(feature, _default_reference(feature))
        without_feature = float(artifact.predict(comparison)[0])
        effect = original - without_feature
        factors.append(
            {
                "feature": feature,
                "contribution": round(effect, 4),
                "direction": _direction(effect),
                "method": "contrafactual_referencia_train",
                "causal": False,
            }
        )
    return factors


def _exact_reference_shapley(
    artifact: ReliabilityModelArtifact, features: pd.DataFrame
) -> list[dict[str, Any]]:
    """SHAP aditivo exacto para un juego de referencia única explícita. Cada coalición conserva
    valores de características participantes y sustituye otras por medianas/modas de
    entrenamiento. Es atribución intervencional relativa a ese fondo, no SHAP condicional
    poblacional ni efecto causal. Reparte interacciones con pesos |S|!(n-|S|-1)!/n!.
    Referencia: Lundberg y Lee, NeurIPS 2017,
    https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
    """

    columns = tuple(artifact.feature_columns)
    count = len(columns)
    if count == 0 or count > 10:
        raise ValueError("Exact reference Shapley is limited to 1–10 original features.")
    query = features.iloc[0]
    reference = {
        feature: artifact.feature_references.get(feature, _default_reference(feature))
        for feature in columns
    }
    coalitions = pd.DataFrame([
        {feature: query[feature] if mask & (1 << position) else reference[feature]
         for position, feature in enumerate(columns)}
        for mask in range(1 << count)
    ])
    values = artifact.predict(coalitions)
    factors: list[dict[str, Any]] = []
    for position, feature in enumerate(columns):
        bit = 1 << position
        contribution = sum(
            (values[mask | bit] - values[mask]) / (count * comb(count - 1, mask.bit_count()))
            for mask in range(1 << count) if not mask & bit
        )
        factors.append({
            "feature": feature,
            "contribution": round(float(contribution), 6),
            "direction": _direction(float(contribution)),
            "method": "shapley_exacto_referencia_train",
            "baseline_score": float(values[0]),
            "prediction_score": float(values[-1]),
            "background": "single_train_medians_and_modes",
            "coalitions_evaluated": len(coalitions),
            "causal": False,
        })
    return factors


def _source_feature(transformed_name: str, feature_columns: Sequence[str]) -> str:
    bare = transformed_name.split("__", 1)[-1]
    # Los nombres one-hot siguen feature_category. Ordenar primero los más largos
    # por si el nombre de una característica es prefijo de otro.
    for source in sorted(feature_columns, key=len, reverse=True):
        if bare == source or bare.startswith(f"{source}_"):
            return source
    return bare


def _default_reference(feature: str) -> Any:
    return 0.0 if feature in NUMERIC_FEATURE_COLUMNS else "__unknown__"


def _direction(value: float) -> str:
    if value > 1e-9:
        return "aumenta_indice"
    if value < -1e-9:
        return "reduce_indice"
    return "sin_efecto_apreciable"
