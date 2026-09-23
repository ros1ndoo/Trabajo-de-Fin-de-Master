from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.contracts import canonical_text, dataset_fingerprint
from auto_reliability.service import InsufficientEvidenceError, ReliabilityService


def _row(year=2017):
    return {"id_vehiculo_ano": f"FORD_EXPLORER_{year}", "marca": "ford", "modelo": "explorer",
            "ano_fabricacion": year, "categoria_vehiculo": "suv", "mediana_cv": 290.,
            "mediana_cilindros": 6., "hist_fiabilidad_marca": None,
            "score_recalls_bruto": 3., "indice_fiabilidad_100": 50.}


def test_nullable_text_and_fingerprint_are_stable_across_storage_types():
    assert canonical_text(pd.NA) == ""
    frame = pd.DataFrame([_row(2016), _row(2017)])
    other = frame.iloc[::-1].copy()
    other["ano_fabricacion"] = other.ano_fabricacion.astype(float)
    other["indice_fiabilidad_100"] = 99.  # Display target is not the model's train-fit scale.
    assert dataset_fingerprint(frame) == dataset_fingerprint(other)
    other.loc[:, "score_recalls_bruto"] = 7.
    assert dataset_fingerprint(frame) != dataset_fingerprint(other)


def test_status_is_read_only_without_auto_demo(tmp_path):
    service = ReliabilityService(ProjectPaths(tmp_path), auto_bootstrap_demo=False)
    status = service.status()
    assert status.catalog_rows == 0
    assert not status.model_available
    assert not list(tmp_path.iterdir())


def test_stale_artifact_never_predicts_on_another_dataset(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_directories()
    pd.DataFrame([_row()]).to_parquet(paths.gold_path)
    monkeypatch.setattr("auto_reliability.service.load_model_artifact",
                        lambda _: SimpleNamespace(metadata={"dataset_fingerprint": "wrong"}))
    service = ReliabilityService(paths, auto_bootstrap_demo=False)
    assert not service.status().model_available
    with pytest.raises(InsufficientEvidenceError):
        service.predict("ford", "explorer", 2017)


def test_no_held_out_error_is_not_replaced_with_training_dispersion(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_directories()
    pd.DataFrame([_row(2000), _row(2017)]).to_parquet(paths.gold_path)
    service = ReliabilityService(paths, auto_bootstrap_demo=False)
    result = service.predict("ford", "explorer", 2017)
    assert result["fallback"]
    assert result["mae"] is None
    assert result["fuente_demo"] is False


def test_partial_real_catalog_is_never_replaced_by_demo(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_directories()
    feature_only = pd.DataFrame([_row()]).drop(columns=["score_recalls_bruto", "indice_fiabilidad_100"])
    feature_only.to_parquet(paths.inference_catalog_path)
    service = ReliabilityService(paths, auto_bootstrap_demo=True)
    assert len(service.load_catalog()) == 1
    assert not paths.gold_path.exists()
    with pytest.raises(InsufficientEvidenceError):
        service.predict("ford", "explorer", 2017)


def test_market_history_is_not_presented_as_brand_evidence(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_directories()
    pd.DataFrame([_row(2000)]).to_parquet(paths.gold_path)
    query = {**_row(), "marca": "audi", "modelo": "a3", "id_vehiculo_ano": "AUDI_A3_2017",
             "hist_fiabilidad_marca": 3.,
             "origen_hist_fiabilidad_marca": "global_cohortes_completadas_previas",
             "coincidencia_recall_aceptada": False}
    pd.DataFrame([query]).to_parquet(paths.inference_catalog_path)
    result = ReliabilityService(paths, auto_bootstrap_demo=False).predict("audi", "a3", 2017)
    assert result["origen_hist_fiabilidad_marca"] == "global_cohortes_completadas_previas"
    assert "Historial previo de marca" not in result["vehicle_features"]
    assert "no específico de esta marca" in result["mensaje"]


def test_live_recalls_are_real_cache_backed_and_do_not_change_ml(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    evidence = tmp_path / "response.json"
    evidence.write_text('{"results": []}')
    calls = []
    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass
        def fetch_vehicle(self, make, model, year):
            calls.append((make, model, year))
            return SimpleNamespace(payload={"results": [], "Count": 0}, cache_path=evidence, from_cache=True)
    monkeypatch.setattr("auto_reliability.data_sources.NHTSARecallClient", FakeClient)
    service = ReliabilityService(paths, auto_bootstrap_demo=False)
    result = service.real_recalls("Ford", "Explorer", 2020)
    assert result["count"] == 0
    assert not result["fuente_demo"]
    assert "Cero resultados no acredita" in result["notice"]
    assert len(service.real_recalls_catalog()) == 1
    assert not paths.gold_path.exists()
    assert calls == [("Ford", "Explorer", 2020)]
