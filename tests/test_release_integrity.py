"""Las publicaciones conservan coherencia cuando falla una activación posterior."""

import json

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.demo_data import write_demo_catalog
from auto_reliability.modeling import train_and_select_model
from auto_reliability.releases import publish_release, serving_paths
from auto_reliability.storage import atomic_json


def test_publication_failure_preserves_previous_release(tmp_path):
    paths = ProjectPaths(tmp_path)
    gold, _ = write_demo_catalog(paths)
    train_and_select_model(gold, artifact_dir=paths.artifacts_dir)
    identifier = publish_release(paths)
    active = serving_paths(paths)
    original = active.gold_path.read_bytes()
    assert identifier == publish_release(paths)
    changed = pd.read_parquet(paths.gold_path)
    changed.loc[0, "score_recalls_bruto"] += 999
    changed.to_parquet(paths.gold_path)
    with pytest.raises(ValueError, match="inconsistent"):
        publish_release(paths)
    assert serving_paths(paths).root == active.root
    assert serving_paths(paths).gold_path.read_bytes() == original
    # Alterar un paquete fijado es un error; nunca recurrir silenciosamente a archivos sueltos.
    with active.model_path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        serving_paths(paths)


def test_invalid_release_pointer_is_rejected(tmp_path):
    paths = ProjectPaths(tmp_path)
    atomic_json(tmp_path / "releases/active.json", {"release": "../../outside"})
    with pytest.raises(ValueError, match="identifier"):
        serving_paths(paths)


def test_comparison_preserves_provenance_and_warns_on_mixed_scales():
    from auto_reliability.dashboard import _comparison_entry, comparison_warnings, summary_frame
    result = {"marca": "ford", "modelo": "test", "ano_fabricacion": 2017,
              "version_modelo": "abc", "version_datos": "def", "version_escala": "ghi",
              "prediccion_indice_100": 50, "mensaje": "Limitación comprobada",
              "evaluacion_retrospectiva": True, "evidencia_oficial": {"count": None}}
    first = _comparison_entry(result)
    second = _comparison_entry({**result, "version_escala": "different", "fallback": True})
    assert first["Limitaciones"] == result["mensaje"]
    messages = comparison_warnings([first, second])
    assert any("Escala" in message for message in messages)
    assert any("retrospectivas" in message for message in messages)
    assert any("reserva" in message for message in messages)
    assert summary_frame(result).iloc[0]["version_modelo"] == "abc"
    json.dumps(first)
