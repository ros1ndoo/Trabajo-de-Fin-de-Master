"""Missing own labels and unused features must not block a fitted group baseline."""

from types import SimpleNamespace

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.data_sources import DataSourceError
from auto_reliability.modeling import BrandSegmentMeanBaseline
from auto_reliability.service import InsufficientEvidenceError, ReliabilityService


def supported_service(tmp_path, monkeypatch, make="ford"):
    row = {"id_vehiculo_ano": "FORD_TEST_2017", "marca": make, "modelo": "test", "ano_fabricacion": 2017,
           "categoria_vehiculo": "suv", "hist_fiabilidad_marca": None,
           "origen_hist_fiabilidad_marca": "sin_historial_previo"}
    estimator = BrandSegmentMeanBaseline().fit(pd.DataFrame([{"marca": "ford", "categoria_vehiculo": "suv"}]), [0.0])
    service = ReliabilityService(ProjectPaths(tmp_path), auto_bootstrap_demo=False)
    monkeypatch.setattr(service, "load_catalog", lambda: pd.DataFrame([row]))
    monkeypatch.setattr(service, "_load_artifact", lambda: SimpleNamespace(estimator=estimator))
    calls = []

    def predict(*args, **kwargs):
        calls.append(kwargs)
        return {"prediccion_indice_100": 61, "mae": 12, "mensaje": "Baseline de prueba"}

    monkeypatch.setattr(service, "predict", predict)
    return service, calls


def test_missing_recent_history_and_recall_identity_do_not_block_known_brand(tmp_path, monkeypatch):
    service, calls = supported_service(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "real_recalls", lambda *a: {"count": 0, "source": "api", "from_cache": False, "fetched_at": "today"})
    result = service.predict_on_demand("ford", "test", 2017)
    assert calls == [{"identity_verified": True}]
    assert result["prediccion_indice_100"] == 61
    assert result["tipo_resultado"] == "baseline_entrenado_de_grupo"
    assert result["nivel_estimacion"] == "marca_y_segmento"
    assert result["evidencia_oficial"]["identity_status"] == "technical_name_unverified_in_nhtsa"
    assert "no distingue modelos" in result["mensaje"]


def test_api_failure_does_not_turn_into_zero_or_change_prediction(tmp_path, monkeypatch):
    service, calls = supported_service(tmp_path, monkeypatch)

    def offline(*args):
        raise DataSourceError("offline fixture")

    monkeypatch.setattr(service, "real_recalls", offline)
    result = service.predict_on_demand("ford", "test", 2017)
    assert result["prediccion_indice_100"] == 61
    assert result["evidencia_oficial"]["count"] is None
    assert result["evidencia_oficial"]["status"] == "unavailable"
    assert len(calls) == 1


def test_unseen_brand_never_receives_global_baseline_as_individual_score(tmp_path, monkeypatch):
    service, calls = supported_service(tmp_path, monkeypatch, make="unknown")
    with pytest.raises(InsufficientEvidenceError, match="marca no está representada"):
        service.predict_on_demand("unknown", "test", 2017)
    assert not calls
