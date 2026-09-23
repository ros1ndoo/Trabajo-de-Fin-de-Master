from __future__ import annotations

import pytest

from auto_reliability.contracts import canonical_text, validate_gold_dataset, vehicle_id
from auto_reliability.demo_data import make_demo_catalog


def test_canonical_text_and_vehicle_id_are_stable() -> None:
    assert canonical_text("  MÁZDA--CX  5 ") == "mazda cx 5"
    assert vehicle_id("Mázda", "CX-5", 2022) == "MAZDA_CX_5_2022"


def test_demo_catalog_keeps_recent_cohorts_unlabelled() -> None:
    catalog = make_demo_catalog(through_year=2026)
    complete = catalog.loc[catalog["cohorte_completa"]]
    incomplete = catalog.loc[~catalog["cohorte_completa"]]
    assert not complete.empty
    assert complete["indice_fiabilidad_100"].between(0, 100).all()
    assert incomplete["indice_fiabilidad_100"].isna().all()
    assert incomplete["ano_fabricacion"].min() > complete["ano_fabricacion"].max()


def test_validate_gold_rejects_duplicate_ids() -> None:
    valid = make_demo_catalog(through_year=2020).query("cohorte_completa").head(2).copy()
    valid.loc[valid.index[1], "id_vehiculo_ano"] = valid.iloc[0]["id_vehiculo_ano"]
    with pytest.raises(ValueError, match="unique"):
        validate_gold_dataset(valid)
