"""Prediction orchestration must acquire evidence without using its outcome."""
from pathlib import Path

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.service import InsufficientEvidenceError, ReliabilityService


def test_acquisition_precedes_prediction_and_own_recalls_are_not_features(tmp_path, monkeypatch):
    row = {"id_vehiculo_ano": "AUDI_A3_2017", "marca": "audi", "modelo": "a3",
           "ano_fabricacion": 2017, "origen_hist_fiabilidad_marca": "marca_cohortes_completadas_previas"}
    official = pd.DataFrame([{**row, "nhtsa_vehicle_id": "AUDI_A3_2017",
                              "catalog_verified": True, "catalog_source": "nhtsa_bulk"}])
    class Store:
        path: Path = tmp_path
        def __init__(self, _paths):
            pass
        def catalog(self):
            return official
    monkeypatch.setattr("auto_reliability.bulk_recalls.BulkRecallStore", Store)
    service = ReliabilityService(ProjectPaths(tmp_path), auto_bootstrap_demo=False)
    monkeypatch.setattr(service, "load_catalog", lambda: pd.DataFrame([row]))
    sequence = []
    counts = iter([1, 1000])
    def fetch(make, model, year):
        sequence.append("fetch")
        return {"count": next(counts), "fetched_at": "2026-09-18", "source": "NHTSA", "from_cache": False}
    def predict(make, model, year, *, identity_verified):
        sequence.append("predict")
        assert (make, model, year, identity_verified) == ("audi", "a3", 2017, True)
        return {"prediccion_indice_100": 60}
    monkeypatch.setattr(service, "real_recalls", fetch)
    monkeypatch.setattr(service, "predict", predict)
    first = service.predict_on_demand("audi", "a3", 2017)
    second = service.predict_on_demand("audi", "a3", 2017)
    assert sequence == ["fetch", "predict", "fetch", "predict"]
    assert first["prediccion_indice_100"] == second["prediccion_indice_100"]
    assert first["evidencia_oficial"]["count"] != second["evidencia_oficial"]["count"]


def test_default_service_never_bootstraps_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_RELIABILITY_AUTO_DEMO", raising=False)
    service = ReliabilityService(ProjectPaths(tmp_path))
    assert not service.status().demo_mode
    assert not service.status().model_available
    assert not list(tmp_path.iterdir())


def test_on_demand_unknown_identity_does_not_invent_a_score(tmp_path, monkeypatch):
    service = ReliabilityService(ProjectPaths(tmp_path), auto_bootstrap_demo=False)
    row = {"marca": "audi", "modelo": "a3", "ano_fabricacion": 2017}
    monkeypatch.setattr(service, "load_catalog", lambda: pd.DataFrame([row]))
    with pytest.raises(InsufficientEvidenceError, match="catálogo oficial"):
        service.predict_on_demand("audi", "a3", 2017)
