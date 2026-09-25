"""Stored observation evidence is distinct from predictions and live queries."""

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.dashboard import observation_notice
from auto_reliability.service import ReliabilityService


@pytest.mark.parametrize("reason,expected", [
    ("query_failed", "falló"), ("unverified_zero_result", "sin evidencia suficiente"),
    ("manual_review", "pendiente de revisión"), ("no_candidate", "no demuestra ausencia"),
    ("incomplete_window", "no está completa"), ("not_recorded", "no se interpreta como cero"),
])
def test_reason_is_explained_without_claiming_reliability(reason, expected):
    message = observation_notice({"primary_reason": reason})
    assert expected in message
    assert "independiente" in message


def test_legacy_release_does_not_invent_zero_or_query_failure():
    assert "no aporta un diagnóstico" in observation_notice({})
    assert "Cruce de identidad rechazado" in observation_notice({"match_status": "rejected"})
    assert "no aporta un diagnóstico" in observation_notice({"primary_reason": pd.NA})


def test_zero_is_explicitly_scoped_to_window():
    text = observation_notice({"primary_reason": "included", "window_outcome": "zero_in_window",
                               "evidence_as_of": "2026-09-18"})
    assert "no significa cero en todo el histórico" in text
    assert "2026-09-18" in text


def test_service_preserves_diagnostics_for_gold_without_replacing_scores(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_runtime_directories()
    row = {"id_vehiculo_ano": "FORD_GT_2005", "marca": "ford", "modelo": "gt",
           "ano_fabricacion": 2005, "indice_fiabilidad_100": 42.0}
    pd.DataFrame([row]).to_parquet(paths.gold_path)
    pd.DataFrame([{**row, "indice_fiabilidad_100": 99.0, "primary_reason": "included",
                   "window_outcome": "positive_in_window"}]).to_parquet(paths.inference_catalog_path)
    catalog = ReliabilityService(paths, auto_bootstrap_demo=False).load_catalog()
    assert catalog.iloc[0].indice_fiabilidad_100 == 42.0
    assert catalog.iloc[0].window_outcome == "positive_in_window"
