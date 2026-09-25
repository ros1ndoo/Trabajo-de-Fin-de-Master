"""Las revisiones de prueba son sintéticas; no aprueban alias reales."""

import pandas as pd
import pytest

from auto_reliability.matching import apply_documented_equivalences, match_technical_to_recalls


def inputs():
    technical = pd.DataFrame([{"id_vehiculo_ano": "technical", "marca": "ford", "modelo": "example class", "ano_fabricacion": 2010}])
    catalog = pd.DataFrame([{"nhtsa_vehicle_id": "official", "marca": "ford", "modelo": "example", "ano_fabricacion": 2010,
                             "catalog_verified": True, "catalog_source": "recall_catalog"}])
    matches = match_technical_to_recalls(technical, catalog, require_catalog_verified=True)
    decisions = pd.DataFrame([{"id_vehiculo_ano": "technical", "nhtsa_vehicle_id": "official", "reviewer": "unit-test",
                               "reviewed_at": "2026-01-01", "evidence_url": "https://example.org/test-only",
                               "evidence_sha256": "a" * 64, "justification": "Synthetic fixture, not real evidence"}])
    return matches, decisions, catalog


def test_documented_equivalence_retains_audit_trail():
    matches, decisions, catalog = inputs()
    result = apply_documented_equivalences(matches, decisions, catalog)
    assert result.iloc[0].match_status == "manual_accepted"
    assert result.iloc[0].review_evidence_sha256 == "a" * 64
    assert matches.iloc[0].match_status == "manual_review"


@pytest.mark.parametrize("field,value", [("evidence_sha256", ""), ("reviewer", None),
                                        ("evidence_url", "http://example.org"), ("reviewed_at", "2999-01-01"),
                                        ("nhtsa_vehicle_id", "nonexistent")])
def test_incomplete_or_invalid_review_cannot_approve(field, value):
    matches, decisions, catalog = inputs()
    decisions.loc[0, field] = value
    with pytest.raises(ValueError):
        apply_documented_equivalences(matches, decisions, catalog)


@pytest.mark.parametrize("field,value", [("marca", "toyota"), ("ano_fabricacion", 2011),
                                        ("modelo", "example 250"), ("catalog_verified", False)])
def test_review_cannot_override_identity_guards(field, value):
    matches, decisions, catalog = inputs()
    catalog.loc[0, field] = value
    with pytest.raises(ValueError):
        apply_documented_equivalences(matches, decisions, catalog)
