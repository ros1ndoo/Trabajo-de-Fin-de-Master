"""Los datos sintéticos prueban contratos, no ausencia real de recalls."""

import pandas as pd
import pytest

from auto_reliability.evidence import observation_evidence


def frames():
    technical = pd.DataFrame({"id_vehiculo_ano": ["a", "b", "c", "d", "e"],
                              "ano_fabricacion": [2010] * 5})
    matches = technical[["id_vehiculo_ano"]].assign(
        nhtsa_vehicle_id=["A", "B", "C", "D", "E"],
        match_status=["auto_accepted", "auto_accepted", "query_failed", "manual_review", "unverified_zero_result"],
    )
    queries = pd.DataFrame({"nhtsa_vehicle_id": ["A", "B", "E"],
                            "query_status": ["valid_with_recalls", "valid_zero_recalls", "unverified_zero_result"]})
    gold = pd.DataFrame({"id_vehiculo_ano": ["a", "b"], "numero_recalls_en_ventana": [2, 0]})
    return technical, matches, queries, gold


def test_distinguishes_zero_failure_and_unresolved_without_mutation():
    inputs = frames()
    originals = [frame.copy(deep=True) for frame in inputs]
    result = observation_evidence(*inputs, as_of_date="2026-09-18").set_index("id_vehiculo_ano")
    assert result.window_outcome.to_dict() == {
        "a": "positive_in_window", "b": "zero_in_window", "c": "unknown", "d": "unknown", "e": "unknown",
    }
    assert result.loc["c", "primary_reason"] == "query_failed"
    assert result.loc["d", "primary_reason"] == "manual_review"
    assert result.loc["e", "primary_reason"] == "unverified_zero_result"
    for original, frame in zip(originals, inputs, strict=True):
        pd.testing.assert_frame_equal(original, frame)


@pytest.mark.parametrize("status", ["unverified_zero_result", "query_failed", "not_recorded"])
def test_gold_cannot_claim_zero_without_recorded_valid_query(status):
    technical, matches, queries, gold = frames()
    queries.loc[1, "query_status"] = status
    with pytest.raises(ValueError, match="contradicts"):
        observation_evidence(technical, matches, queries, gold, as_of_date="2026-09-18")


def test_incomplete_cohort_cannot_be_gold():
    technical, matches, queries, gold = frames()
    technical.loc[0, "ano_fabricacion"] = 2025
    with pytest.raises(ValueError, match="contradicts"):
        observation_evidence(technical, matches, queries, gold, as_of_date="2026-09-18")


def test_missing_query_is_not_called_network_failure():
    technical, matches, queries, gold = frames()
    matches.loc[2, "match_status"] = "auto_accepted"
    result = observation_evidence(technical, matches, queries, gold, as_of_date="2026-09-18")
    assert result.loc[2, "primary_reason"] == "not_recorded"
    assert result.loc[2, "window_outcome"] == "unknown"


def test_duplicate_queries_fail_instead_of_multiplying_population():
    technical, matches, queries, gold = frames()
    with pytest.raises(ValueError, match="unique"):
        observation_evidence(technical, matches, pd.concat([queries, queries]), gold, as_of_date="2026-09-18")


def test_rejected_candidate_cannot_transfer_another_vehicles_query():
    technical, matches, queries, gold = frames()
    matches.loc[3, "nhtsa_vehicle_id"] = "A"
    result = observation_evidence(technical, matches, queries, gold, as_of_date="2026-09-18")
    assert result.loc[3, "candidate_query_status"] == "valid_with_recalls"
    assert result.loc[3, "query_status"] == "not_linked_to_verified_identity"
    assert result.loc[3, "window_outcome"] == "unknown"
