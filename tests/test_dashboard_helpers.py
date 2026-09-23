from __future__ import annotations

import pandas as pd
import pytest

from auto_reliability.dashboard import (
    _factor_table,
    _feature_context,
    make_radar_figure,
    normalize_catalog,
    normalize_prediction,
    summary_frame,
)
from auto_reliability.demo_data import make_demo_catalog


def test_dashboard_catalog_and_radar_helpers_are_renderable() -> None:
    catalog = normalize_catalog(make_demo_catalog(through_year=2026))
    assert not catalog.empty
    figure = make_radar_figure(
        {"Cilindros": 6, "Potencia (CV)": 300, "Historial previo de marca": 1.2},
        {"Cilindros": 4, "Potencia (CV)": 210, "Historial previo de marca": 0.9},
        {
            "Cilindros": (3, 8),
            "Potencia (CV)": (100, 450),
            "Historial previo de marca": (0, 3),
        },
    )
    assert figure is not None
    assert len(figure.data) == 2


def test_prediction_contract_rejects_wrong_vehicle_and_invalid_scores():
    catalog = normalize_catalog(make_demo_catalog(through_year=2026))
    row = catalog.iloc[0]
    for raw in (
        {"prediction": float("nan")},
        {"prediction": 101},
        {"prediction": 50, "marca": "another make"},
        {"prediction": 50, "ano_fabricacion": 1900},
    ):
        with pytest.raises(ValueError):
            normalize_prediction(raw, catalog, row.marca, row.modelo, int(row.ano_fabricacion))


def test_missing_mae_demo_and_retrospective_labels_survive_csv_export():
    catalog = normalize_catalog(make_demo_catalog(through_year=2026))
    row = catalog.iloc[0]
    result = normalize_prediction(
        {"prediction": 50, "fuente_demo": True, "evaluacion_retrospectiva": True},
        catalog, row.marca, row.modelo, int(row.ano_fabricacion),
    )
    assert result["mae"] is None
    exported = summary_frame(result).iloc[0]
    assert pd.isna(exported.mae)
    assert exported.fuente_demo
    assert exported.evaluacion_retrospectiva
    assert "no mide fiabilidad" in exported.naturaleza_indice


def test_no_future_context_is_borrowed_for_earliest_cohort():
    catalog = normalize_catalog(make_demo_catalog(through_year=2026))
    row = catalog.sort_values("ano_fabricacion").iloc[0]
    context = _feature_context(catalog, row.marca, row.modelo, int(row.ano_fabricacion))
    assert not context["segment_features"]
    assert not context["brand_history"]


def test_zero_factor_contribution_and_value_are_retained():
    table = _factor_table({"factores": [{"feature": "mediana_cilindros", "value": 0, "contribution": 0}]})
    assert table.iloc[0]["Valor"] == "0"
    assert table.iloc[0]["Influencia estimada"] == "Sin efecto neto (+0.00 puntos)"
