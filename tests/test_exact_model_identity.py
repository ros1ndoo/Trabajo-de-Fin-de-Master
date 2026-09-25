"""Full model names must survive trim token removal without weakening ambiguity checks."""

import pandas as pd
import pytest

from auto_reliability.matching import match_technical_to_recalls


@pytest.mark.parametrize("make,model,year", [("ford", "gt", 2005), ("ford", "gt", 2006), ("maserati", "coupe", 2004)])
def test_unique_exact_model_is_not_an_empty_trim(make, model, year):
    technical = pd.DataFrame([{"id_vehiculo_ano": "technical", "marca": make, "modelo": model, "ano_fabricacion": year}])
    catalog = pd.DataFrame([{"nhtsa_vehicle_id": "official", "marca": make, "modelo": model.upper(),
                             "ano_fabricacion": year, "catalog_verified": True, "catalog_source": "nhtsa_bulk"}])
    result = match_technical_to_recalls(technical, catalog, require_catalog_verified=True)
    assert result.iloc[0].match_status == "auto_accepted"
    # Distinct official identities for the same full name still need review.
    duplicate = catalog.copy()
    duplicate["nhtsa_vehicle_id"] = "other"
    ambiguous = match_technical_to_recalls(technical, pd.concat([catalog, duplicate]), require_catalog_verified=True)
    assert ambiguous.iloc[0].match_status == "manual_review"
