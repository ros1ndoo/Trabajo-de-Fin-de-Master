"""Varios ciclos EPA de una configuración no equivalen a varios acabados."""

import pandas as pd
import pytest

from auto_reliability.data_sources import DataSourceError
from auto_reliability.epa_test_cars import COLUMNS, configuration_evidence, normalize_tests


def source(power=200):
    row = dict.fromkeys(COLUMNS, "fixture")
    row.update({"Model Year": 2026, "Rated Horsepower": power, "Test Veh Configuration #": 0})
    return pd.DataFrame([row])


def test_repeated_tests_do_not_weight_the_same_configuration_twice():
    frame = normalize_tests(pd.concat([source(), source()], ignore_index=True), 2026)
    configs = configuration_evidence(frame)
    assert len(configs) == 1
    assert configs.test_rows.iloc[0] == 2
    assert configs.rated_hp.iloc[0] == 200
    assert not configs.prediction_eligible.any()


def test_conflicting_power_is_exposed_not_averaged():
    configs = configuration_evidence(normalize_tests(pd.concat([source(200), source(300)]), 2026))
    assert pd.isna(configs.rated_hp.iloc[0])
    assert configs.power_status.iloc[0] == "conflicting_tests"


@pytest.mark.parametrize("power", [0, -1, float("inf"), None, "unknown"])
def test_invalid_power_is_not_fabricated(power):
    frame = normalize_tests(source(power), 2026)
    assert pd.isna(frame.rated_hp.iloc[0])


def test_other_model_year_is_rejected():
    with pytest.raises(DataSourceError):
        normalize_tests(source(), 2025)
