"""Regresiones deterministas de integración API, sin necesidad de red."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from auto_reliability.config import ProjectPaths
from auto_reliability.data_sources import (
    DataSourceError,
    NHTSARecallClient,
    normalise_nhtsa_response,
)
from auto_reliability.matching import match_technical_to_recalls, write_fuzzy_audit
from auto_reliability.pipeline import (
    refresh_inference_catalog,
    run_pipeline,
    select_technical_scope,
    stage_technical_csv,
)


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self):
        self.calls = []

    def get(self, endpoint, *, params, **kwargs):
        self.calls.append((endpoint, params))
        if "GetModelsForMakeYear" in endpoint:
            models = ["Explorer", "UnknownZero", "ApiFailure"]
            return Response({"Count": len(models), "Results": [{"Make_Name": "Ford", "Model_Name": name} for name in models]})
        if "/products/" in endpoint:
            return Response({"count": 2, "results": [
                {"make": "Ford", "model": "Explorer", "modelYear": "2010"},
                {"make": "Ford", "model": "ApiFailure", "modelYear": "2010"},
            ]})
        if params["model"] == "ApiFailure":
            raise requests.ConnectionError("test offline")
        if params["model"] == "UnknownZero":
            return Response({"Count": 0, "results": []})
        return Response({"Count": 1, "results": [{
            "NHTSACampaignNumber": "11V000001", "ReportReceivedDate": "04/12/2011", "Summary": "brake defect"
        }]})


def test_pipeline_uses_independent_names_and_never_labels_failed_or_unverified_zeros(tmp_path, monkeypatch):
    source = tmp_path / "technical.csv"
    pd.DataFrame([
        {"Make": "Ford", "Model": name, "Year": 2010, "Engine HP": 200,
         "Engine Cylinders": 6, "Vehicle Style": "4dr SUV"}
        for name in ("Explorer", "UnknownZero", "ApiFailure")
    ]).to_csv(source, index=False)
    session = Session()
    monkeypatch.setattr("auto_reliability.pipeline.NHTSARecallClient", lambda raw_dir, **kwargs: NHTSARecallClient(
        raw_dir, session=session, max_retries=0, request_delay_seconds=0, sleeper=lambda _: None
    ))
    result = run_pipeline(source, paths=ProjectPaths(tmp_path / "project"), train_end_year=2010,
                          snapshot_date="2026-09-18", as_of_date="2026-09-18")
    assert result.gold["modelo"].tolist() == ["explorer"]
    evidence = result.inference_catalog.set_index("modelo")
    assert evidence.loc["explorer", "window_outcome"] == "positive_in_window"
    assert evidence.loc["unknownzero", "primary_reason"] == "unverified_zero_result"
    assert evidence.loc["apifailure", "primary_reason"] == "query_failed"
    assert evidence.loc["apifailure", "window_outcome"] == "unknown"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["observation_evidence"]["primary_reason"] == {
        "included": 1, "query_failed": 1, "unverified_zero_result": 1,
    }
    assert result.gold["score_recalls_bruto"].tolist() == [3.0]
    statuses = result.matches.set_index("modelo_tecnico")["match_status"].to_dict()
    assert statuses["unknownzero"] == "unverified_zero_result"
    assert statuses["apifailure"] == "query_failed"
    assert result.inference_catalog["coincidencia_recall_aceptada"].sum() == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["nhtsa"]["unverified_zero_queries"] == 1
    assert manifest["nhtsa"]["vehicle_queries_failed"] == 1
    assert "sesgo de selección" in " ".join(result.warnings)
    assert len(pd.read_csv(result.paths.audit_path)) == 3
    assert not result.gold["fuente_demo"].any()
    assert (result.paths.raw_dir / "nhtsa" / "2026-09-18").is_dir()


@pytest.mark.parametrize("payload", [{"error": "unavailable"}, {"Count": 1, "results": []}, {"Count": 0, "results": None}])
def test_malformed_api_payload_is_never_a_zero_label(tmp_path, payload):
    class BadSession:
        def get(self, *args, **kwargs):
            return Response(payload)
    client = NHTSARecallClient(tmp_path, session=BadSession(), max_retries=0)
    with pytest.raises(DataSourceError):
        client.fetch_vehicle("Ford", "Explorer", 2010)
    assert not list(tmp_path.glob("*.json"))


def test_nhtsa_ambiguous_dates_are_day_first():
    frame = normalise_nhtsa_response({"results": [
        {"ReportReceivedDate": "04/12/2019", "NHTSACampaignNumber": "19V859000"}
    ]}, make="Ford", model="Explorer", year=2020)
    assert frame["fecha_reporte"].iloc[0] == pd.Timestamp("2019-12-04")


@pytest.mark.parametrize(("technical_name", "candidate", "expected"), [
    ("F-150", "F-250", "rejected"),
    ("F-150", "F150", "auto_accepted"),
    ("Explorer", "Explorer Sport", "manual_review"),
    ("Civic", "Civic LX", "auto_accepted"),
])
def test_fuzzy_numeric_family_and_root_guards(technical_name, candidate, expected, tmp_path):
    technical = pd.DataFrame([{"marca": "Ford", "modelo": technical_name, "ano_fabricacion": 2010}])
    catalogue = pd.DataFrame([{"marca": "Ford", "modelo": candidate, "ano_fabricacion": 2010}])
    matches = match_technical_to_recalls(technical, catalogue)
    assert matches["match_status"].iloc[0] == expected
    assert len(pd.read_csv(write_fuzzy_audit(matches, tmp_path / "audit.csv"))) == 1


def test_scope_balances_makes_and_years():
    frame = pd.DataFrame([
        {"marca": make, "modelo": model, "ano_fabricacion": year, "id_vehiculo_ano": f"{make}_{model}_{year}"}
        for year in range(1995, 2018) for make in ("ford", "honda") for model in ("a", "b")
    ])
    bounded = select_technical_scope(frame, max_vehicles=46)
    assert bounded["ano_fabricacion"].nunique() == 23
    assert bounded.groupby("marca").size().to_dict() == {"ford": 23, "honda": 23}
    tiny = select_technical_scope(frame, max_vehicles=3)
    assert tiny["ano_fabricacion"].tolist() == [1995, 2006, 2017]


@pytest.mark.parametrize("filename", ["../README.md", "..\\outside.csv", "C:\\outside.csv", "README.md"])
def test_raw_staging_refuses_path_escape_and_non_csv(tmp_path, filename):
    source = tmp_path / "in.csv"
    source.write_text("Make,Model,Year\nFord,Explorer,2010\n", encoding="utf-8")
    with pytest.raises(ValueError):
        stage_technical_csv(source, ProjectPaths(tmp_path / "project"), filename=filename)


def test_full_catalog_does_not_depend_on_sampled_makes(tmp_path, monkeypatch):
    source = tmp_path / "technical.csv"
    pd.DataFrame([
        {"Make": make, "Model": model, "Year": 2010, "Engine HP": 200,
         "Engine Cylinders": 6, "Vehicle Style": "4dr SUV"}
        for make, model in [("Ford", "Explorer"), ("Toyota", "RAV4"), ("Audi", "Q5")]
    ]).to_csv(source, index=False)
    monkeypatch.setattr("auto_reliability.pipeline.NHTSARecallClient", lambda raw_dir, **kwargs: NHTSARecallClient(
        raw_dir, session=Session(), max_retries=0, request_delay_seconds=0, sleeper=lambda _: None
    ))
    paths = ProjectPaths(tmp_path / "project")
    result = run_pipeline(source, paths=paths, makes=["ford"], max_vehicles=1, train_end_year=2010)
    assert len(result.gold) == 1
    assert set(result.inference_catalog.marca) == {"ford", "toyota", "audi"}
    assert result.inference_catalog.match_status.eq("not_queried").sum() == 2
    refreshed = refresh_inference_catalog(source, paths=paths)
    assert set(refreshed.marca) == {"ford", "toyota", "audi"}
    assert refreshed.coincidencia_recall_aceptada.sum() == 1
