"""Las explicaciones describen el grupo ajustado, nunca características no utilizadas."""

import pandas as pd
import pytest

from auto_reliability.dashboard import _factor_table
from auto_reliability.explainability import explain_prediction, summarize_attributions
from auto_reliability.modeling import BrandSegmentMeanBaseline, ReliabilityModelArtifact
from auto_reliability.narrative import build_grounded_narrative


@pytest.mark.parametrize("make,segment,level", [
    ("ford", "suv", "marca_y_segmento"), ("ford", "van", "marca"),
    ("unknown", "suv", "segmento"), ("unknown", "van", "media_global"),
])
def test_group_explanation_reconstructs_actual_score(make, segment, level):
    training = pd.DataFrame({"marca": ["ford", "ford", "toyota"], "categoria_vehiculo": ["suv", "sedan", "suv"]})
    estimator = BrandSegmentMeanBaseline().fit(training, [-1., 0., 2.])
    artifact = ReliabilityModelArtifact(estimator, "baseline", 50., 10.)
    query = {"marca": make, "categoria_vehiculo": segment, "mediana_cv": 300., "mediana_cilindros": 8,
             "hist_fiabilidad_marca": 10.}
    factors = explain_prediction(artifact, query)
    assert len(factors) == 1
    factor = factors[0]
    assert factor["feature"] == level
    assert factor["baseline_score"] + factor["contribution"] == pytest.approx(artifact.predict(query)[0])
    changed = {**query, "mediana_cv": 999., "mediana_cilindros": 0, "hist_fiabilidad_marca": 0.}
    assert explain_prediction(artifact, changed) == factors
    narrative = summarize_attributions(factors)
    assert "no intervienen" in narrative
    # Activar un LLM opcional no debe convertir diferencias de grupo en efectos individuales.
    assert build_grounded_narrative(factors, narrative, model="enabled")["text"] == narrative
    table = _factor_table({"factores": factors})
    assert table.iloc[0]["Valor"] != "—"
    assert "media global de entrenamiento" in table.iloc[0]["Influencia estimada"]
    assert "aumenta_indice" not in table.to_string()
