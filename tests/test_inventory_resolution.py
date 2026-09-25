"""Alinear nombres no acredita cero recalls ni suficiencia predictora."""

import hashlib
import json

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.data_sources import DataSourceError
from auto_reliability.inventory_resolution import (
    publish_resolution,
    resolution_summary,
    resolve_inventory,
)


def epa(*models):
    return pd.DataFrame([{"id": str(i), "make": "Ford", "model": model, "year": 2025}
                         for i, model in enumerate(models)])


def nhtsa(*models):
    return pd.DataFrame([{"nhtsa_vehicle_id": str(i), "marca": "FORD", "modelo": model,
                          "ano_fabricacion": 2025} for i, model in enumerate(models)])


def test_exact_and_cosmetic_names_only_are_linked():
    result = resolve_inventory(epa("Focus", "F-150", "Focus AWD"), nhtsa("FOCUS", "F150"))
    assert result.identity_status.tolist() == ["exact_name", "normalized_name", "manual_review"]
    assert pd.isna(result.nhtsa_vehicle_id.iloc[2])
    assert result.recall_status.eq("not_evaluated").all()
    assert not result.prediction_eligible.any()
    assert json.loads(result.candidates_json.iloc[2])[0]["modelo"] == "FOCUS"


def test_ambiguity_never_chooses_first_normalized_candidate():
    result = resolve_inventory(epa("F 150"), nhtsa("F-150", "F150"))
    assert result.identity_status.iloc[0] == "ambiguous_name"
    assert result.candidate_count.iloc[0] == 2
    assert pd.isna(result.nhtsa_vehicle_id.iloc[0])


def test_distinct_numbers_and_years_and_makes_are_never_merged():
    catalog = nhtsa("F250", "Focus", "Focus")
    catalog.loc[1, "ano_fabricacion"] = 2024
    catalog.loc[2, "marca"] = "Other"
    result = resolve_inventory(epa("F150", "Focus"), catalog)
    assert result.nhtsa_vehicle_id.isna().all()
    assert not result.identity_status.isin(["exact_name", "normalized_name"]).any()
    other_year = epa("Focus").assign(year=2026)
    assert resolve_inventory(other_year, catalog).identity_status.iloc[0] == "no_same_make_year"


def test_variants_names_and_nhtsa_families_have_distinct_denominators():
    result = resolve_inventory(epa("Focus", "Focus", "FOCUS"), nhtsa("FOCUS"))
    summary = resolution_summary(result)
    assert summary["variants"] == 3
    assert summary["source_names"] == 2
    assert summary["unique_linked_nhtsa_identities"] == 1
    assert summary["linked_variants"] == 3


def test_repeated_catalog_rows_do_not_create_false_ambiguity():
    catalog = pd.concat([nhtsa("FOCUS")] * 2, ignore_index=True)
    assert resolve_inventory(epa("Focus"), catalog).identity_status.iloc[0] == "exact_name"


@pytest.mark.parametrize("year", [2025.5, None, "bad", float("inf")])
def test_invalid_years_fail_before_matching(year):
    with pytest.raises(DataSourceError):
        resolve_inventory(epa("Focus").assign(year=year), nhtsa("FOCUS"))


def test_conflicting_source_ids_fail_closed():
    with pytest.raises(DataSourceError, match="Duplicate EPA"):
        resolve_inventory(epa("Focus", "Focus").assign(id="1"), nhtsa("FOCUS"))
    with pytest.raises(DataSourceError, match="Conflicting NHTSA"):
        resolve_inventory(epa("Focus"), nhtsa("FOCUS", "F150").assign(nhtsa_vehicle_id="1"))


def test_unknown_is_not_zero_and_input_is_unchanged():
    source = epa("NoSuchModel")
    before = source.copy(deep=True)
    result = resolve_inventory(source, nhtsa("FOCUS"))
    assert result.identity_status.iloc[0] == "no_model_match"
    assert result.recall_status.iloc[0] == "not_evaluated"
    pd.testing.assert_frame_equal(source, before)


def test_publication_is_reproducible_and_detects_corrupted_bundle(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    source = paths.raw_dir / "fuel_economy" / "2026-09-18"
    source.mkdir(parents=True)
    (source / "vehicles.xml.zip").write_bytes(b"fixture")
    (source / "manifest.json").write_text(json.dumps({
        "sha256": hashlib.sha256(b"fixture").hexdigest(), "retrieved_at_utc": "2026-09-18T00:00:00Z"
    }), encoding="utf-8")
    monkeypatch.setattr("auto_reliability.inventory_resolution.read_inventory", lambda _: epa("Focus"))

    class Store:
        def __init__(self, paths):
            pass

        def catalog(self):
            return nhtsa("FOCUS")

    monkeypatch.setattr("auto_reliability.inventory_resolution.BulkRecallStore", Store)
    summary = publish_resolution(paths, "2026-09-18")
    assert publish_resolution(paths, "2026-09-18") == summary
    assert not paths.gold_path.exists()
    assert not paths.inference_catalog_path.exists()
    output = paths.processed_dir / "inventory_resolution"
    (output / summary["bundle_id"] / "audit.parquet").write_bytes(b"broken fixture")
    with pytest.raises(DataSourceError, match="integrity"):
        publish_resolution(paths, "2026-09-18")


def test_invalid_report_is_not_served_as_progress(tmp_path):
    from auto_reliability.service import ReliabilityService

    paths = ProjectPaths(tmp_path)
    service = ReliabilityService(paths, auto_bootstrap_demo=False)
    assert service.inventory_alignment_status() is None
    directory = paths.processed_dir / "inventory_resolution"
    directory.mkdir(parents=True)
    (directory / "latest.json").write_text('{"variants": 1, "source_names": 2, "linked_source_names": 3}')
    assert "error" in service.inventory_alignment_status()
