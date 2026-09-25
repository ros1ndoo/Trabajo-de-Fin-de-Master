"""Describe observaciones sin fabricar etiquetas ni predicciones."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .transform import cohort_is_complete


def observation_evidence(
    technical: pd.DataFrame,
    matches: pd.DataFrame,
    queries: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    as_of_date: date | str,
) -> pd.DataFrame:
    """Devuelve un diagnóstico por identidad técnica, nunca un cero inferido. Los estados
    proceden de la adquisición, no de una certificación nueva de cobertura. Los conteos de
    ventana solo existen para Gold ya construido. Una consulta ausente es desconocida salvo
    fallo explícitamente registrado.
    """
    key = "id_vehiculo_ano"
    for frame, column in ((technical, key), (matches, key), (gold, key),
                          (queries, "nhtsa_vehicle_id")):
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"Evidence requires unique non-null {column}")
    for frame in (matches, gold):
        if not frame[key].isin(technical[key]).all():
            raise ValueError("Evidence identity outside technical universe")
    result = technical[[key, "ano_fabricacion"]].merge(
        matches[[key, "nhtsa_vehicle_id", "match_status"]], on=key,
        how="left", validate="one_to_one",
    ).merge(
        queries[["nhtsa_vehicle_id", "query_status"]], on="nhtsa_vehicle_id",
        how="left", validate="many_to_one",
    )
    result["match_status"] = result.match_status.fillna("not_in_matching_scope")
    result["query_status"] = result.query_status.fillna("not_recorded")
    result.loc[result.match_status.eq("query_failed"), "query_status"] = "query_failed"
    result.loc[result.match_status.eq("unverified_zero_result"), "query_status"] = "unverified_zero_result"
    result["identity_status"] = "unresolved"
    confirmed = result.match_status.isin([
        "auto_accepted", "manual_accepted", "query_failed", "unverified_zero_result",
    ])
    # Un candidato rechazado puede haber sido consultado para OTRO modelo técnico.
    # Su respuesta no se puede atribuir a esta identidad técnica sin resolver.
    result["candidate_query_status"] = result.query_status
    result.loc[~confirmed, "query_status"] = "not_linked_to_verified_identity"
    result.loc[confirmed, "identity_status"] = "confirmed_by_matching"
    result.loc[result.match_status.eq("manual_review"), "identity_status"] = "requires_review"
    result["window_status"] = result.ano_fabricacion.map(
        lambda year: "complete" if cohort_is_complete(year, as_of_date=as_of_date) else "incomplete"
    )
    result["included_in_gold"] = result[key].isin(gold[key])
    result["primary_reason"] = "unresolved_identity"
    for status in ("rejected", "manual_rejected", "manual_review", "no_candidate", "not_in_matching_scope"):
        result.loc[result.match_status.eq(status), "primary_reason"] = status
    result.loc[confirmed, "primary_reason"] = "eligible_identity_not_in_gold"
    result.loc[confirmed & result.window_status.eq("incomplete"), "primary_reason"] = "incomplete_window"
    for status in ("not_recorded", "query_failed", "unverified_zero_result"):
        result.loc[confirmed & result.query_status.eq(status), "primary_reason"] = status
    result.loc[result.included_in_gold, "primary_reason"] = "included"
    valid = result.query_status.isin(["valid_with_recalls", "valid_zero_recalls"])
    if (result.included_in_gold & (~confirmed | ~valid | result.window_status.ne("complete"))).any():
        raise ValueError("Gold inclusion contradicts recorded observation evidence")
    result["window_outcome"] = "unknown"
    if "numero_recalls_en_ventana" in gold:
        counts = pd.to_numeric(gold["numero_recalls_en_ventana"], errors="coerce")
        if (counts.isna() | counts.lt(0) | counts.mod(1).ne(0)).any():
            raise ValueError("Invalid observed window counts")
        observed = dict(zip(gold[key], counts, strict=True))
        result.loc[result.included_in_gold, "window_outcome"] = result.loc[
            result.included_in_gold, key
        ].map(lambda identity: "zero_in_window" if observed[identity] == 0 else "positive_in_window")
    return result
