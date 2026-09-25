"""Regresiones de unidades explícitas y agregación técnica previa a particiones."""

import numpy as np
import pandas as pd
import pytest

from auto_reliability.data_sources import normalise_cooperunion_columns
from auto_reliability.transform import prepare_technical_specs


def test_dashboard_converts_hp_alias_but_not_canonical_cv():
    from auto_reliability.dashboard import normalize_catalog

    source = pd.DataFrame({"marca": ["Ford"], "modelo": ["a"],
                           "ano_fabricacion": [2010], "Engine HP": [100]})
    result = normalize_catalog(source)
    assert result.mediana_cv.iloc[0] == pytest.approx(101.38696, abs=0.00001)
    canonical = source.rename(columns={"Engine HP": "mediana_cv"})
    assert normalize_catalog(canonical).mediana_cv.iloc[0] == 100


def test_hp_conversion_and_native_cv_fallback_preserve_provenance():
    source = pd.DataFrame({
        "Make": ["Ford"] * 3, "Model": ["a", "b", "c"], "Year": [2010] * 3,
        "Engine HP": [100, None, None], "potencia_cv": [999, 100, None],
    })
    result = normalise_cooperunion_columns(source)
    assert result.potencia_cv.iloc[0] == pytest.approx(101.38696, abs=0.00001)
    assert result.potencia_cv.iloc[1] == 100
    assert pd.isna(result.potencia_cv.iloc[2])
    assert result.potencia_original.iloc[0] == 100
    assert result.unidad_potencia_origen.iloc[:2].tolist() == ["hp_mechanical", "CV"]
    assert result.columna_potencia_origen.iloc[:2].tolist() == ["engine_hp", "potencia_cv"]


def test_other_vehicles_never_impute_missing_trim_specs():
    source = pd.DataFrame({
        "marca": ["ford"] * 4, "modelo": ["a", "a", "b", "c"],
        "ano_fabricacion": [2010] * 4, "categoria_vehiculo": ["suv"] * 4,
        "potencia_cv": [290, None, 170, None], "cilindros": [6, None, 4, 0],
    })
    result = prepare_technical_specs(source).set_index("modelo")
    assert result.loc["a", "mediana_cv"] == 290
    assert result.loc["a", "n_potencia_observada"] == 1
    assert pd.isna(result.loc["c", "mediana_cv"])
    assert result.loc["c", "n_potencia_observada"] == 0
    assert result.loc["c", "mediana_cilindros"] == 0
    changed = source.copy()
    changed.loc[changed.modelo.eq("b"), "potencia_cv"] = 999
    actual = prepare_technical_specs(changed).set_index("modelo")
    pd.testing.assert_frame_equal(result.loc[["a", "c"]], actual.loc[["a", "c"]])


@pytest.mark.parametrize("invalid", [0, -1, np.inf, -np.inf])
def test_invalid_power_is_missing_not_a_valid_observation(invalid):
    source = pd.DataFrame({"marca": ["ford"], "modelo": ["a"], "ano_fabricacion": [2010],
                           "potencia_cv": [invalid], "cilindros": [0]})
    result = prepare_technical_specs(source)
    assert pd.isna(result.mediana_cv.iloc[0])
    assert result.n_potencia_observada.iloc[0] == 0
    assert result.mediana_cilindros.iloc[0] == 0
