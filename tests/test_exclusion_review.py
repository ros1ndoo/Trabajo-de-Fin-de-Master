"""Las respuestas controladas prueban semántica de evidencia, no ceros reales."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from auto_reliability.exclusion_review import independent_identities, query_identity


def identity():
    technical = pd.DataFrame([{"id_vehiculo_ano": "FORD_TEST_2010", "marca": "ford", "modelo": "test", "ano_fabricacion": 2010}])
    inventory = pd.DataFrame([{"id": "1", "make": "Ford", "model": "Test", "year": 2010},
                              {"id": "2", "make": "Ford", "model": "Test", "year": 2010}])
    return technical, inventory


def test_variants_preserve_identity_evidence_without_inventing_outcome():
    technical, inventory = identity()
    row = independent_identities(technical, inventory)[0]
    assert row["source_ids"] == ["1", "2"]
    assert row["independent_identity_status"] == "exact_independent_identity"
    assert row["recall_query_status"] == "not_queried"
    assert row["label_eligible"] is False


def test_absence_and_wrong_year_are_not_zero_or_fuzzy_matches():
    technical, inventory = identity()
    inventory["year"] = 2011
    row = independent_identities(technical, inventory)[0]
    assert row["independent_identity_status"] == "absent_from_independent_inventory"
    assert row["label_eligible"] is False


@pytest.mark.parametrize("count", [0, 1])
def test_valid_response_preserves_hash_but_does_not_authorize_label(tmp_path, count):
    technical, inventory = identity()
    row = independent_identities(technical, inventory)[0]
    payload = {"Count": count, "results": [{"NHTSACampaignNumber": "TEST", "ReportReceivedDate": "01/01/2011"}] if count else []}
    cache = tmp_path / "response.json"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    class Client:
        endpoint = "https://example.org/test-only"

        def fetch_vehicle(self, make, model, year):
            return SimpleNamespace(payload=payload, result_count=count, cache_path=cache, from_cache=True)
    result = query_identity(Client(), row)
    assert result["recall_query_status"] == ("campaigns_returned_review_required" if count else "empty_response_identity_only")
    assert len(result["response_sha256"]) == 64
    assert result["label_eligible"] is False
    assert row["recall_query_status"] == "not_queried"


def test_network_error_never_gets_a_zero_count():
    technical, inventory = identity()
    class Client:
        def fetch_vehicle(self, *args):
            raise requests.Timeout("controlled timeout")
    result = query_identity(Client(), independent_identities(technical, inventory)[0])
    assert result["recall_query_status"] == "query_failed"
    assert "response_count" not in result
    assert result["label_eligible"] is False
