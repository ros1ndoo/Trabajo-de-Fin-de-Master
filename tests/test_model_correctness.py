"""Regression tests for the scientific guarantees, not only the happy path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.contracts import dataset_fingerprint
from auto_reliability.demo_data import make_demo_catalog, write_demo_catalog
from auto_reliability.explainability import explain_prediction, summarize_attributions
from auto_reliability.modeling import (
    BrandSegmentMedianImputer,
    ReliabilityModelArtifact,
    TemporalTargetNormalizer,
    _Candidate,
    _fit_candidate,
    _select_candidate,
    load_model_artifact,
    prepare_feature_frame,
    temporal_train_validation_test_split,
    train_and_select_model,
)


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    return make_demo_catalog(through_year=2026).query("cohorte_completa").copy()


def test_temporal_split_embargoes_unavailable_three_year_labels(gold) -> None:
    split = temporal_train_validation_test_split(gold)
    assert split.train.ano_fabricacion.max() == 2016
    assert set(split.train_embargoed.ano_fabricacion) == {2017, 2018}
    assert set(split.validation.ano_fabricacion) == {2019}
    assert set(split.validation_embargoed.ano_fabricacion) == {2020, 2021}
    assert split.train.ano_fabricacion.max() + 3 <= split.validation.ano_fabricacion.min()
    assert split.validation.ano_fabricacion.max() + 3 <= split.test.ano_fabricacion.min()


def test_embargo_also_applies_when_validation_is_missing() -> None:
    frame = pd.DataFrame({"ano_fabricacion": [2020, 2021, 2023]})
    split = temporal_train_validation_test_split(
        frame, train_end_year=2021, validation_end_year=2022, adaptive_fallback=False
    )
    assert split.validation.empty
    assert list(split.train.ano_fabricacion) == [2020]
    assert list(split.train_embargoed.ano_fabricacion) == [2021]


def test_test_outcomes_and_supplied_target_cannot_change_fitting(gold) -> None:
    first = train_and_select_model(gold, persist=False)
    changed = gold.copy()
    changed.loc[changed.ano_fabricacion >= 2022, "score_recalls_bruto"] += 20
    changed["indice_fiabilidad_100"] = 1_000_000
    second = train_and_select_model(changed, persist=False)
    assert first.artifact.model_name == second.artifact.model_name
    assert first.artifact.metadata["target_normalizer"] == second.artifact.metadata["target_normalizer"]
    assert first.metrics["candidates"] == second.metrics["candidates"]
    np.testing.assert_allclose(first.artifact.predict(gold), second.artifact.predict(gold))
    assert first.artifact.metadata["dataset_fingerprint"] == dataset_fingerprint(gold)
    assert second.artifact.metadata["dataset_fingerprint"] != dataset_fingerprint(gold)
    # Test is evaluated but cannot select or fit the model.
    assert first.metrics["final_test"] != second.metrics["final_test"]


def test_group_medians_learn_only_during_fit(gold) -> None:
    features = prepare_feature_frame(gold.head(8))
    features["mediana_cv"] = [100, 120, np.nan, 140, 160, 180, 200, 220]
    imputer = BrandSegmentMedianImputer().fit(features)
    before = imputer.group_medians_.copy()
    query = features.iloc[[0]].copy()
    query["mediana_cv"] = np.nan
    result = imputer.transform(query)
    assert result.mediana_cv.iloc[0] == pytest.approx(features.mediana_cv.median())
    query["mediana_cv"] = 1_000_000
    imputer.transform(query)
    pd.testing.assert_frame_equal(before, imputer.group_medians_)


def test_advanced_candidate_can_win_even_when_ridge_loses_to_baseline() -> None:
    candidates = {
        name: _Candidate(name=name, validation_metrics={"mae": error, "rmse": error, "n": 10})
        for name, error in {"baseline": 8.0, "ridge": 10.0, "random_forest": 7.0}.items()
    }
    selected, _, reduction = _select_candidate(
        candidates, has_validation=True, advanced_improvement_threshold=0.10
    )
    assert selected == "random_forest"
    assert reduction == pytest.approx(0.3)
    candidates["random_forest"].validation_metrics["mae"] = 9.0
    assert _select_candidate(candidates, has_validation=True, advanced_improvement_threshold=0.10)[0] == "baseline"
    candidates["baseline"].validation_metrics["mae"] = 11.0
    assert _select_candidate(candidates, has_validation=True, advanced_improvement_threshold=0.10)[0] == "ridge"


def test_selection_reason_distinguishes_best_error_from_eligibility() -> None:
    candidates = {
        name: _Candidate(name=name, validation_metrics={"mae": error, "rmse": error, "n": 10})
        for name, error in {"baseline": 9.5, "ridge": 10., "random_forest": 9.2}.items()
    }
    selected, reason, _ = _select_candidate(candidates, has_validation=True, advanced_improvement_threshold=.10)
    assert selected == "baseline"
    assert "elegibles" in reason
    assert "umbral" in reason


@pytest.mark.parametrize("mutation", ["duplicate", "incomplete", "infinite", "negative", "bad_year", "null_label"])
def test_invalid_training_rows_fail_loudly(gold, mutation) -> None:
    frame = gold.copy()
    first = frame.index[0]
    if mutation == "duplicate":
        frame.loc[frame.index[1], "id_vehiculo_ano"] = frame.loc[first, "id_vehiculo_ano"]
    elif mutation == "incomplete":
        frame.loc[first, "cohorte_completa"] = False
    elif mutation == "infinite":
        frame.loc[first, "mediana_cv"] = np.inf
    elif mutation == "negative":
        frame.loc[first, "score_recalls_bruto"] = -1
    elif mutation == "bad_year":
        frame.loc[first, "ano_fabricacion"] = 1900
    else:
        frame.loc[first, "score_recalls_bruto"] = np.nan
    with pytest.raises(ValueError):
        train_and_select_model(frame, persist=False)


def test_empty_input_has_clear_validation_error(gold) -> None:
    with pytest.raises(ValueError, match="empty"):
        train_and_select_model(gold.head(0), persist=False)
    with pytest.raises(TypeError, match="DataFrame"):
        train_and_select_model([], persist=False)


def test_artifact_roundtrip_and_missing_inference_values(gold, tmp_path) -> None:
    result = train_and_select_model(gold, artifact_dir=tmp_path)
    loaded = load_model_artifact(result.artifact_paths["model"])
    query = gold.head(2).copy()
    query["marca"] = "previously unseen manufacturer"
    query["categoria_vehiculo"] = "previously unseen segment"
    query[["mediana_cv", "mediana_cilindros", "hist_fiabilidad_marca"]] = np.nan
    predictions = loaded.predict(query)
    np.testing.assert_allclose(predictions, result.artifact.predict(query))
    assert np.isfinite(predictions).all()
    assert ((0 <= predictions) & (predictions <= 100)).all()
    normalizer = TemporalTargetNormalizer.from_dict(loaded.metadata["target_normalizer"])
    assert normalizer.to_dict() == loaded.metadata["target_normalizer"]


def test_error_estimate_is_unavailable_without_holdout_and_zero_is_valid(gold) -> None:
    single_year = gold.loc[gold.ano_fabricacion == 1995].copy()
    result = train_and_select_model(single_year, persist=False)
    assert result.artifact.model_name == "baseline"
    assert result.artifact.expected_mae is None
    assert result.artifact.metadata["expected_mae_source"] == "unavailable"
    constant = gold.copy()
    constant["score_recalls_bruto"] = 1.0
    result = train_and_select_model(constant, persist=False)
    assert result.artifact.expected_mae == 0.0
    assert result.artifact.metadata["expected_mae_source"] == "held_out_test"


def test_demo_history_never_uses_future_or_immature_outcomes() -> None:
    shorter = make_demo_catalog(through_year=2020).set_index("id_vehiculo_ano")
    longer = make_demo_catalog(through_year=2026).set_index("id_vehiculo_ano")
    pd.testing.assert_series_equal(shorter.hist_fiabilidad_marca, longer.loc[shorter.index, "hist_fiabilidad_marca"])
    earliest = longer.loc[longer.ano_fabricacion < 1998, "hist_fiabilidad_marca"]
    assert earliest.isna().all()
    selected = longer.query("marca == 'ford' and ano_fabricacion == 2010").iloc[0]
    known = longer.query("marca == 'ford' and 2005 <= ano_fabricacion <= 2007")
    assert selected.hist_fiabilidad_marca == pytest.approx(known.score_recalls_bruto.mean())
    assert longer.loc[~longer.cohorte_completa, "score_recalls_bruto"].isna().all()


def test_demo_cannot_overwrite_real_data(gold, tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_directories()
    real = gold.head(2).copy()
    real["fuente_demo"] = False
    real.to_parquet(paths.gold_path, index=False)
    with pytest.raises(ValueError, match="non-demo"):
        write_demo_catalog(paths)
    pd.testing.assert_frame_equal(pd.read_parquet(paths.gold_path), real.reset_index(drop=True))


def test_ridge_explanations_follow_prediction_preprocessing(gold) -> None:
    train = gold.loc[gold.ano_fabricacion < 2017].copy()
    candidate = _fit_candidate("ridge", train, train.indice_fiabilidad_100)
    assert candidate.available, candidate.error
    references = {"marca": "ford", "categoria_vehiculo": "suv", "mediana_cv": 210.0,
                  "mediana_cilindros": 4.0, "hist_fiabilidad_marca": 1.5}
    artifact = ReliabilityModelArtifact(candidate.estimator, "ridge", candidate.target_mean,
                                        candidate.target_std, feature_references=references)
    query = train.iloc[[0]].copy()
    query["mediana_cv"] = np.nan
    factors = explain_prediction(artifact, query, top_n=5)
    expected_difference = (
        candidate.estimator.predict(prepare_feature_frame(query))[0]
        - candidate.estimator.predict(prepare_feature_frame(references))[0]
    ) * artifact.target_std
    assert sum(factor["contribution"] for factor in factors) == pytest.approx(expected_difference, abs=0.001)
    assert all(factor["causal"] is False for factor in factors)
    assert "no demuestra causalidad" in summarize_attributions(factors)


def test_forest_shapley_values_are_additive_and_use_one_batch(gold) -> None:
    train = gold.loc[gold.ano_fabricacion < 2017].copy()
    candidate = _fit_candidate("random_forest", train, train.indice_fiabilidad_100)
    assert candidate.available, candidate.error
    references = {"marca": "ford", "categoria_vehiculo": "suv", "mediana_cv": 210.0,
                  "mediana_cilindros": 4.0, "hist_fiabilidad_marca": 1.5}
    artifact = ReliabilityModelArtifact(candidate.estimator, "random_forest", candidate.target_mean,
                                        candidate.target_std, feature_references=references)
    query = train.iloc[[0]].copy()
    query["mediana_cv"] = np.nan
    factors = explain_prediction(artifact, query, top_n=5)
    expected_difference = artifact.predict(query)[0] - artifact.predict(references)[0]
    assert sum(factor["contribution"] for factor in factors) == pytest.approx(expected_difference, abs=0.00001)
    assert {factor["method"] for factor in factors} == {"shapley_exacto_referencia_train"}
    assert {factor["coalitions_evaluated"] for factor in factors} == {32}
    assert all(factor["baseline_score"] == pytest.approx(artifact.predict(references)[0]) for factor in factors)
    assert all(factor["causal"] is False for factor in factors)
    # A feature identical to the background is a dummy player with zero value.
    matching_reference = explain_prediction(artifact, references, top_n=5)
    assert all(factor["contribution"] == 0 for factor in matching_reference)
