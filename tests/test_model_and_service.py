from __future__ import annotations

from auto_reliability.config import ProjectPaths
from auto_reliability.contracts import PREDICTION_FEATURES
from auto_reliability.demo_data import make_demo_catalog, write_demo_catalog
from auto_reliability.modeling import predict_with_artifact, train_and_select_model
from auto_reliability.service import ReliabilityService


def test_temporal_training_excludes_recall_outcome_from_features() -> None:
    catalog = make_demo_catalog(through_year=2026)
    gold = catalog.loc[catalog["cohorte_completa"]].copy()
    result = train_and_select_model(gold, persist=False)
    assert set(result.artifact.feature_columns) == set(PREDICTION_FEATURES)
    assert "score_recalls_bruto" not in result.artifact.feature_columns
    recent = catalog.loc[~catalog["cohorte_completa"]].head(2)
    prediction = predict_with_artifact(result.artifact, recent)
    assert prediction["prediccion_indice_100"].between(0, 100).all()
    assert prediction["mae"].notna().all()


def test_service_predicts_incomplete_cohort_without_own_target(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    gold, catalog = write_demo_catalog(paths)
    train_and_select_model(gold, artifact_dir=paths.artifacts_dir, persist=True)
    service = ReliabilityService(paths, auto_bootstrap_demo=False)
    recent = catalog.loc[~catalog["cohorte_completa"]].iloc[0]
    result = service.predict(recent["marca"], recent["modelo"], int(recent["ano_fabricacion"]))
    assert result["prediccion_indice_100"] >= 0
    assert result["prediccion_indice_100"] <= 100
    assert "recalls" in result["mensaje"]
    assert result["modelo_usado"] in {"Regresión Ridge", "Random Forest", "Baseline histórico"}
    assert result["factores"]
