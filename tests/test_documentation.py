"""Evita publicar cifras divergentes entre documentos y artefactos reales."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from auto_reliability.documentation import (
    END,
    START,
    check_documentation,
    check_metrics_block,
    render_metrics_table,
    verified_table,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metrics():
    return json.loads((ROOT / "artifacts/model_metrics.json").read_text(encoding="utf-8"))


def test_published_documents_match_the_active_release():
    check_documentation(ROOT)


def test_table_uses_supplied_metrics_without_hardcoded_results(metrics):
    metrics["final_test"]["mae"] = 8.4321
    table = render_metrics_table(metrics)
    assert "| MAE de test | 8,4321 |" in table
    assert "12,1537" not in table
    assert f"{metrics['split']['train']['rows'] + metrics['split']['validation']['rows']} casos" in table


@pytest.mark.parametrize("value", [None, "12.0", True, float("nan"), float("inf"), -1])
def test_invalid_mae_is_not_published_as_a_number(metrics, value):
    metrics["final_test"]["mae"] = value
    with pytest.raises((TypeError, ValueError)):
        render_metrics_table(metrics)


def test_changed_model_or_demo_metrics_require_document_review(metrics):
    changed = deepcopy(metrics)
    changed["selection"]["selected_model"] = "ridge"
    with pytest.raises(ValueError, match="modelo seleccionado"):
        render_metrics_table(changed)
    metrics["fuente_demo"] = True
    with pytest.raises(ValueError, match="demostración"):
        render_metrics_table(metrics)


def test_incomparable_denominators_fail_explicitly(metrics):
    metrics["candidates"]["ridge"]["validation"]["n"] += 1
    with pytest.raises(ValueError, match="no comparable"):
        render_metrics_table(metrics)
    metrics["final_test"]["n"] += 1
    with pytest.raises(ValueError, match="denominador"):
        render_metrics_table(metrics)


def test_block_check_tolerates_crlf_but_rejects_stale_or_duplicate_tables(metrics):
    table = render_metrics_table(metrics)
    check_metrics_block("Introducción\r\n" + table.replace("\n", "\r\n"), table)
    for document in (table.replace("12,1537", "12,1984"), table + table,
                     table.replace(START, ""), END + "\n" + START):
        with pytest.raises(ValueError):
            check_metrics_block(document, table)


def test_metrics_copy_cannot_disagree_with_release(tmp_path):
    # La resolución real ya se valida arriba; aquí se aísla la comparación de copias.
    from unittest.mock import patch

    from auto_reliability.config import ProjectPaths

    active = tmp_path / "active"
    (active / "artifacts").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    (active / "artifacts/model_metrics.json").write_text('{"value": 1}', encoding="utf-8")
    (tmp_path / "artifacts/model_metrics.json").write_text('{"value": 2}', encoding="utf-8")
    with (
        patch("auto_reliability.documentation.serving_paths", return_value=ProjectPaths(active)),
        pytest.raises(ValueError, match="difieren"),
    ):
        verified_table(tmp_path)
