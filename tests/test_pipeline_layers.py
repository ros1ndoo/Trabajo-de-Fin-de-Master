from __future__ import annotations

import pandas as pd

from auto_reliability.contracts import vehicle_id
from auto_reliability.matching import match_technical_to_recalls
from auto_reliability.transform import (
    build_gold_dataset,
    build_inference_catalog,
    prepare_technical_specs,
    score_recall_severity,
)


def _technical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marca": "Ford",
                "modelo": "Explorer",
                "ano_fabricacion": 2018,
                "categoria_vehiculo": "SUV",
                "potencia_cv": 290,
                "cilindros": 6,
            },
            {
                "marca": "Ford",
                "modelo": "Explorer",
                "ano_fabricacion": 2018,
                "categoria_vehiculo": "SUV",
                "potencia_cv": None,
                "cilindros": 6,
            },
            {
                "marca": "Ford",
                "modelo": "Unknown",
                "ano_fabricacion": 2018,
                "categoria_vehiculo": "SUV",
                "potencia_cv": 170,
                "cilindros": 4,
            },
        ]
    )


def test_gold_excludes_unmatched_technical_rows_and_preserves_zero_recall_matches() -> None:
    technical = prepare_technical_specs(_technical_rows())
    nhtsa_index = pd.DataFrame(
        [
            {
                "nhtsa_vehicle_id": vehicle_id("Ford", "Explorer", 2018),
                "marca": "Ford",
                "modelo": "Explorer",
                "ano_fabricacion": 2018,
            }
        ]
    )
    matches = match_technical_to_recalls(technical, nhtsa_index)
    recalls = pd.DataFrame(
        [
            {
                "nhtsa_vehicle_id": vehicle_id("Ford", "Explorer", 2018),
                "campana_nhtsa": "18V001",
                "fecha_reporte": "2019-06-01",
                "descripcion_recall": "Brake hydraulic issue",
            }
        ]
    )
    gold = build_gold_dataset(
        technical,
        matches,
        recalls,
        as_of_date="2025-01-01",
        train_end_year=2018,
    )
    assert gold["modelo"].tolist() == ["explorer"]
    assert gold["score_recalls_bruto"].iloc[0] == 3.0
    assert gold["indice_fiabilidad_100"].between(0, 100).all()


def test_inference_catalog_never_carries_the_selected_vehicles_recall_label() -> None:
    technical = prepare_technical_specs(_technical_rows())
    completed_gold = pd.DataFrame(
        [
            {
                "id_vehiculo_ano": vehicle_id("Ford", "Explorer", 2018),
                "marca": "ford",
                "modelo": "explorer",
                "ano_fabricacion": 2018,
                "categoria_vehiculo": "suv",
                "mediana_cilindros": 6,
                "mediana_cv": 290,
                "score_recalls_bruto": 3.0,
                "hist_fiabilidad_marca": 2.0,
                "indice_fiabilidad_100": 50.0,
            }
        ]
    )
    catalog = build_inference_catalog(
        technical,
        completed_gold,
        as_of_date="2019-01-01",
        include_complete_cohorts=True,
    )
    assert "score_recalls_bruto" not in catalog
    assert "indice_fiabilidad_100" not in catalog


def test_severity_contract_prioritizes_critical_terms() -> None:
    assert score_recall_severity("Electrical fire in the engine bay") == 3.0
    assert score_recall_severity("Airbag deployment warning") == 1.5
    assert score_recall_severity("Paint finish issue") == 1.0
