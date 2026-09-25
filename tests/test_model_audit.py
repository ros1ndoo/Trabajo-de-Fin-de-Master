"""La auditoría congelada conserva el servicio y rechaza evidencia obsoleta."""

from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.contracts import PREDICTION_FEATURES, dataset_fingerprint
from auto_reliability.model_audit import audit_frozen_model, error_summary, publish_model_audit
from auto_reliability.modeling import (
    BrandSegmentMeanBaseline,
    ReliabilityModelArtifact,
    TemporalTargetNormalizer,
)


def fixture_pair():
    gold = pd.DataFrame([
        {"id_vehiculo_ano": f"v{i}", "marca": make, "modelo": f"m{i}", "ano_fabricacion": year,
         "categoria_vehiculo": "suv", "mediana_cv": 100, "mediana_cilindros": 4,
         "hist_fiabilidad_marca": 0, "score_recalls_bruto": raw,
         "indice_fiabilidad_100": 999, "cohorte_completa": True}
        for i, (make, year, raw) in enumerate([("ford", 2000, 0), ("ford", 2010, 1), ("new", 2011, 2)])
    ])
    estimator = BrandSegmentMeanBaseline().fit(gold.iloc[:1], [50.0])
    normalizer = TemporalTargetNormalizer("score_recalls_bruto", {}, 0.0, 1.0, 2000)
    artifact = ReliabilityModelArtifact(estimator, "baseline", 0, 1, metadata={
        "dataset_fingerprint": dataset_fingerprint(gold),
        "split": {"test": {"rows": 2, "min_year": 2010, "max_year": 2011}},
        "prediction_available_from_year": 2003,
        "target_normalizer": normalizer.to_dict(),
    })
    return gold, artifact


def test_exact_errors_and_empty_group():
    result = error_summary(pd.DataFrame({"observed": [0, 4], "predicted": [2, 2]}))
    assert result == {"n": 2, "mae": 2., "rmse": 2., "bias": 0., "small_group": True}
    assert error_summary(pd.DataFrame(columns=["observed", "predicted"]))["mae"] is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_errors_are_not_silently_dropped(value):
    with pytest.raises(ValueError, match="finite"):
        error_summary(pd.DataFrame({"observed": [value], "predicted": [1]}))


def test_audit_uses_frozen_scale_and_does_not_fit_or_expose_outcomes(monkeypatch):
    gold, artifact = fixture_pair()
    original = gold.copy(deep=True)
    metadata = deepcopy(artifact.metadata)
    predict = artifact.predict

    def checked(frame):
        assert list(frame.columns) == list(PREDICTION_FEATURES)
        return predict(frame)

    monkeypatch.setattr(artifact, "predict", checked)
    monkeypatch.setattr(artifact.estimator, "fit", lambda *args: pytest.fail("Audit must never fit"))
    report = audit_frozen_model(gold, artifact)
    assert report["overall"]["mae"] == 15
    assert report["overall"]["n"] == 2
    assert {r["observed"] for r in report["predictions"]} == {30., 40.}
    assert {r["group"] for r in report["groups"]["serving_support"]} == {"represented_brand", "unrepresented_brand"}
    for groups in report["groups"].values():
        assert sum(group["n"] for group in groups) == 2
    pd.testing.assert_frame_equal(original, gold)
    assert metadata == artifact.metadata


@pytest.mark.parametrize("mutation", ["fingerprint", "count", "availability", "scale", "duplicate", "immature", "unknown_maturity"])
def test_inconsistent_inputs_are_rejected(mutation):
    gold, artifact = fixture_pair()
    if mutation == "fingerprint":
        gold.loc[1, "mediana_cv"] = 123
    elif mutation == "count":
        artifact.metadata["split"]["test"]["rows"] = 3
    elif mutation == "availability":
        artifact.metadata["prediction_available_from_year"] = 2012
    elif mutation == "scale":
        artifact.metadata["target_normalizer"]["train_end_year"] = 2010
    elif mutation == "duplicate":
        gold.loc[1, "id_vehiculo_ano"] = "v0"
    else:
        gold["cohorte_completa"] = gold.cohorte_completa.astype("boolean")
        gold.loc[1, "cohorte_completa"] = False if mutation == "immature" else pd.NA
        artifact.metadata["dataset_fingerprint"] = dataset_fingerprint(gold)
    with pytest.raises(ValueError):
        audit_frozen_model(gold, artifact)


def test_small_group_threshold_only_affects_reporting():
    gold, artifact = fixture_pair()
    report = audit_frozen_model(gold, artifact, minimum_group_size=1)
    assert not report["overall"]["small_group"]
    with pytest.raises(ValueError, match="positive"):
        audit_frozen_model(gold, artifact, minimum_group_size=0)


def test_publication_reuses_report_and_preserves_serving_files(tmp_path):
    gold, artifact = fixture_pair()
    paths = ProjectPaths(tmp_path)
    paths.gold_path.parent.mkdir(parents=True)
    paths.model_path.parent.mkdir(parents=True)
    gold.to_parquet(paths.gold_path)
    joblib.dump(artifact, paths.model_path)
    before = {path: path.read_bytes() for path in (paths.gold_path, paths.model_path)}
    output = publish_model_audit(paths)
    first_bytes = output.read_bytes()
    assert publish_model_audit(paths) == output
    assert first_bytes == output.read_bytes()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(list(output.parent.glob("*.json"))) == 1


def test_demo_audit_is_explicitly_labelled():
    gold, artifact = fixture_pair()
    artifact.metadata["fuente_demo"] = True
    assert audit_frozen_model(gold, artifact)["fuente_demo"] is True
