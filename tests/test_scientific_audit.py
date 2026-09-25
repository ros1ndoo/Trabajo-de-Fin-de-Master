"""La sensibilidad no debe ajustarse con resultados futuros ni ocultar exclusiones."""

import numpy as np
import pandas as pd
import pytest

from auto_reliability.matching import MATCH_COLUMNS
from auto_reliability.scientific_audit import selection_coverage, sensitivity, severity_counts


def inputs():
    gold = pd.DataFrame({"id_vehiculo_ano": ["a", "b", "c"], "ano_fabricacion": [2000, 2001, 2010],
                         "categoria_vehiculo": ["sedan"] * 3, "score_recalls_bruto": [0., 3., 1.5]})
    counts = pd.DataFrame({"critical": [0, 1, 0], "moderate": [0, 0, 1], "low": [0, 0, 0],
                           "unclassified": [0, 0, 0]}, index=["a", "b", "c"])
    return gold, counts


def test_future_outcomes_cannot_change_scale_fitting():
    gold, counts = inputs()
    first = sensitivity(gold, counts, train_max=2001)
    gold.loc[2, "score_recalls_bruto"] = 150
    counts.loc["c", "moderate"] = 100
    second = sensitivity(gold, counts, train_max=2001)
    assert first["train_only_normalizers"] == second["train_only_normalizers"]
    assert first["after_train"][0]["mean_absolute_target_shift"] == 0


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, 0.5])
def test_invalid_counts_are_rejected(value):
    gold, counts = inputs()
    counts = counts.astype(float)
    counts.loc["a", "critical"] = value
    with pytest.raises(ValueError):
        sensitivity(gold, counts, train_max=2001)


def test_reconstruction_must_match_original_weights():
    gold, counts = inputs()
    gold.loc[0, "score_recalls_bruto"] = 999
    with pytest.raises(ValueError, match="disagree"):
        sensitivity(gold, counts, train_max=2001)


def test_coverage_preserves_excluded_denominator_and_does_not_modify_input():
    technical = pd.DataFrame({"id_vehiculo_ano": ["a", "b"], "marca": ["ford", "ford"],
                              "categoria_vehiculo": ["sedan"] * 2, "ano_fabricacion": [2010] * 2})
    report = selection_coverage(technical, technical.iloc[:1])
    assert report["overall"] == {"technical": 2, "included": 1, "excluded": 1, "inclusion_rate": .5}
    assert "included" not in technical


def test_campaign_deduplication_and_calendar_boundaries():
    gold = pd.DataFrame({"id_vehiculo_ano": ["a"], "ano_fabricacion": [2010]})
    matches = pd.DataFrame([{**dict.fromkeys(MATCH_COLUMNS), "id_vehiculo_ano": "a",
                             "nhtsa_vehicle_id": "official", "match_status": "auto_accepted"}])
    records = pd.DataFrame([{"nhtsa_vehicle_id": "official", "campana_nhtsa": campaign,
                              "fecha_reporte": date, "descripcion_recall": description}
                             for campaign, date, description in [
                                 ("1", "2010-01-01", "engine"), ("1", "2010-01-01", "engine"),
                                 ("2", "2012-12-31", "airbag"), ("3", "2013-01-01", "engine"),
                                 ("4", None, "engine"), ("5", "2009-12-31", "engine")]])
    result = severity_counts(gold, matches, records)
    assert result.loc["a"].to_dict() == {"critical": 1, "moderate": 1, "low": 0, "unclassified": 0}
