import json

import pytest
import requests

from auto_reliability.narrative import build_grounded_narrative

FACTORS = [{"feature": "mediana_cv", "contribution": -3.5}, {"feature": "marca", "contribution": 1.2}]


class Session:
    def __init__(self, selection=None, fail=False):
        self.selection = selection
        self.fail = fail
        self.calls = []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.fail:
            raise requests.Timeout()
        class Response:
            status_code = 200
            def json(inner_self):
                return {"response": json.dumps({"clauses": self.selection})}
        return Response()


def test_without_model_no_network():
    session = Session()
    result = build_grounded_narrative(FACTORS, "Original", model="", session=session)
    assert result["text"] == "Original"
    assert not session.calls


def test_valid_selection_only_renders_grounded_facts():
    session = Session(["mediana_cv"])
    result = build_grounded_narrative(FACTORS, "Original", model="installed-local-model", session=session)
    assert result["mode"] == "ollama_seleccion_validada"
    assert "reduce" in result["text"] and "3.50" in result["text"]
    assert "no causas" in result["text"]
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.parametrize("selection", [["freno_roto"], ["marca"], ["mediana_cv", "mediana_cv"], []])
def test_unverifiable_output_falls_back(selection):
    assert build_grounded_narrative(FACTORS, "Original", model="local", session=Session(selection))["text"] == "Original"


def test_network_failure_falls_back():
    assert build_grounded_narrative(FACTORS, "Original", model="local", session=Session(fail=True))["text"] == "Original"


def test_remote_provider_is_not_contacted():
    session = Session()
    result = build_grounded_narrative(FACTORS, "Original", model="local", base_url="https://example.com", session=session)
    assert result["mode"] == "determinista"
    assert not session.calls
