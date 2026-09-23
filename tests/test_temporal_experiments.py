"""Frozen protocols and isolated execution are independent of serving files."""

from types import SimpleNamespace

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.temporal_experiments import (
    ExperimentWindow,
    experiment_protocol,
    run_temporal_experiments,
)


@pytest.mark.parametrize("windows", [(), (ExperimentWindow(2000, 1999, 2001),),
    (ExperimentWindow(2000, 2003, 2008), ExperimentWindow(2001, 2007, 2010))])
def test_invalid_protocol_is_rejected(windows):
    with pytest.raises(ValueError):
        experiment_protocol(windows)


def test_protocol_precedes_training_and_repeat_does_not_refit(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    paths.gold_path.parent.mkdir(parents=True)
    gold = pd.DataFrame({"id_vehiculo_ano": ["a", "b"], "ano_fabricacion": [2000, 2015]})
    gold.to_parquet(paths.gold_path)
    before = paths.gold_path.read_bytes()
    calls = []

    def train(frame, **kwargs):
        assert list((paths.reports_dir / "temporal_experiments").glob("*/protocol.json"))
        assert kwargs["persist"] is False
        assert frame.ano_fabricacion.max() == 2000
        calls.append(kwargs)
        return SimpleNamespace(metrics={"split": {"configured_train_end_year": 2000,
            "configured_validation_end_year": 2003, "strategy": "adaptive_distinct_years",
            "validation": {"rows": 1}, "test": {"rows": 1}}})

    monkeypatch.setattr("auto_reliability.temporal_experiments.train_and_select_model", train)
    windows = (ExperimentWindow(2000, 2003, 2006),)
    result = run_temporal_experiments(paths, windows=windows)
    assert '"not_evaluable"' in result.read_text()
    assert run_temporal_experiments(paths, windows=windows) == result
    assert len(calls) == 1
    assert paths.gold_path.read_bytes() == before
    assert not paths.model_path.exists()
