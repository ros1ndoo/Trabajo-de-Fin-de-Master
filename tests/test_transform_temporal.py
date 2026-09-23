from __future__ import annotations

import pandas as pd

from auto_reliability.transform import (
    cohort_is_complete,
    prepare_technical_specs,
    score_recall_severity,
)


def test_future_specs_do_not_change_earlier_imputation():
    old = pd.DataFrame([
        {"marca": "ford", "modelo": "a", "ano_fabricacion": 2000, "categoria_vehiculo": "suv", "potencia_cv": None, "cilindros": 6},
        {"marca": "ford", "modelo": "b", "ano_fabricacion": 2000, "categoria_vehiculo": "suv", "potencia_cv": 200, "cilindros": 6},
    ])
    future = old.copy()
    future["ano_fabricacion"] = 2020
    future["potencia_cv"] = 1000
    expected = prepare_technical_specs(old)
    actual = prepare_technical_specs(pd.concat([old, future], ignore_index=True))
    pd.testing.assert_frame_equal(expected, actual.loc[actual.ano_fabricacion.eq(2000)].reset_index(drop=True))


def test_cohort_must_have_elapsed_all_three_calendar_years():
    assert not cohort_is_complete(2023, as_of_date="2025-12-31")
    assert cohort_is_complete(2023, as_of_date="2026-01-01")


def test_severity_keywords_use_word_boundaries():
    assert score_recall_severity("engineered paint finish") == 1.0
    assert score_recall_severity("brakes hydraulic defect") == 3.0
    assert score_recall_severity("air bags failure") == 1.5
