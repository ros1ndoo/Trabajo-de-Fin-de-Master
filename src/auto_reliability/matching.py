"""Cruce difuso auditable entre especificaciones y vehículos NHTSA. La política es
conservadora: acepta similitud alta, no une candidatos ambiguos silenciosamente y restringe
decisiones humanas al caso documentado de nombre raíz y sufijo de acabado.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .contracts import require_columns, vehicle_id
from .data_sources import comparison_make, comparison_model

AUTO_ACCEPT_THRESHOLD = 90
MANUAL_REVIEW_THRESHOLD = 75


try:  # TheFuzz es la dependencia principal; el respaldo permite inspeccionar el código.
    from thefuzz.fuzz import token_set_ratio as _thefuzz_token_set_ratio
except ImportError:  # pragma: no cover - solo se ejecuta antes de instalar las dependencias
    _thefuzz_token_set_ratio = None


def token_set_ratio(left: object, right: object) -> int:
    """Devuelve similitud de conjuntos de tokens de TheFuzz con respaldo determinista. Un
    sufijo como Civic LX no penaliza Civic, pero se mantienen las diferencias entre familias
    como F-150 y F-250.
    """

    left_text, right_text = comparison_model(left), comparison_model(right)
    if _thefuzz_token_set_ratio is not None:
        return int(_thefuzz_token_set_ratio(left_text, right_text))
    left_tokens, right_tokens = set(left_text.split()), set(right_text.split())
    if not left_tokens and not right_tokens:
        return 100
    if not left_tokens or not right_tokens:
        return 0
    shared = left_tokens & right_tokens
    left_combined = " ".join(sorted(shared | (left_tokens - shared)))
    right_combined = " ".join(sorted(shared | (right_tokens - shared)))
    shared_text = " ".join(sorted(shared))
    ratios = (
        SequenceMatcher(None, shared_text, left_combined).ratio(),
        SequenceMatcher(None, shared_text, right_combined).ratio(),
        SequenceMatcher(None, left_combined, right_combined).ratio(),
    )
    return round(100 * max(ratios))


# Términos de versión, carrocería o tracción, no del nombre comercial
# principal. Conservar números y raíces significativas: distinguen
# F-150 de F-250 y C-Class de E-Class.
TRIM_TOKENS = frozenset(
    {
        "base",
        "limited",
        "premium",
        "luxury",
        "touring",
        "signature",
        "select",
        "platinum",
        "lariat",
        "xl",
        "xls",
        "xlt",
        "lx",
        "dx",
        "ex",
        "exl",
        "si",
        "se",
        "sel",
        "le",
        "xle",
        "xse",
        "gt",
        "awd",
        "fwd",
        "rwd",
        "4wd",
        "2wd",
        "4x4",
        "4matic",
        "turbo",
        "diesel",
        "sedan",
        "coupe",
        "wagon",
        "convertible",
        "hatchback",
        "crew",
        "cab",
        "extended",
    }
)


def model_root_name(value: object) -> str:
    """Extrae la raíz conservadora del modelo para orientar la revisión manual."""

    tokens = [token for token in comparison_model(value).split() if token not in TRIM_TOKENS]
    return " ".join(tokens)


@dataclass(frozen=True)
class MatchSummary:
    """Métricas de calidad del cruce para decidir si se aplica la contingencia."""

    technical_rows: int
    auto_accepted: int
    manual_review: int
    manually_accepted: int
    rejected: int
    no_candidate: int

    @property
    def accepted(self) -> int:
        return self.auto_accepted + self.manually_accepted

    @property
    def loss_rate(self) -> float:
        return 0.0 if not self.technical_rows else 1 - (self.accepted / self.technical_rows)

    @property
    def contingency_recommended(self) -> bool:
        """Indica si se alcanza el umbral documentado de contingencia del 40% de huérfanos."""

        return self.loss_rate > 0.40

    def as_dict(self) -> dict[str, object]:
        return {
            "technical_rows": self.technical_rows,
            "auto_accepted": self.auto_accepted,
            "manual_review": self.manual_review,
            "manually_accepted": self.manually_accepted,
            "rejected": self.rejected,
            "no_candidate": self.no_candidate,
            "accepted": self.accepted,
            "loss_rate": self.loss_rate,
            "contingency_recommended": self.contingency_recommended,
        }


MATCH_COLUMNS: tuple[str, ...] = (
    "id_vehiculo_ano",
    "marca_tecnica",
    "modelo_tecnico",
    "ano_fabricacion",
    "nhtsa_vehicle_id",
    "marca_nhtsa",
    "modelo_nhtsa",
    "match_score",
    "match_status",
    "root_modelo_tecnico",
    "root_modelo_nhtsa",
    "root_name_matches",
    "recomendacion_revision",
    "decision_manual",
    "audit_required",
    "catalog_verified",
    "catalog_source",
)


def _normalise_technical_keys(technical: pd.DataFrame) -> pd.DataFrame:
    require_columns(technical, ("marca", "modelo", "ano_fabricacion"), context="Technical specifications")
    frame = technical.copy()
    frame["ano_fabricacion"] = pd.to_numeric(frame["ano_fabricacion"], errors="raise").astype(int)
    if "id_vehiculo_ano" not in frame.columns:
        frame["id_vehiculo_ano"] = [
            vehicle_id(make, model, year)
            for make, model, year in frame[["marca", "modelo", "ano_fabricacion"]].itertuples(index=False, name=None)
        ]
    if frame["id_vehiculo_ano"].duplicated().any():
        raise ValueError("Technical specifications must be aggregated to one unique id_vehiculo_ano before matching.")
    frame["_marca_key"] = frame["marca"].map(comparison_make)
    frame["_modelo_key"] = frame["modelo"].map(comparison_model)
    return frame


def _normalise_nhtsa_keys(nhtsa_vehicles: pd.DataFrame) -> pd.DataFrame:
    require_columns(nhtsa_vehicles, ("marca", "modelo", "ano_fabricacion"), context="NHTSA vehicle index")
    frame = nhtsa_vehicles.copy()
    frame["ano_fabricacion"] = pd.to_numeric(frame["ano_fabricacion"], errors="raise").astype(int)
    if "nhtsa_vehicle_id" not in frame.columns:
        frame["nhtsa_vehicle_id"] = [
            vehicle_id(make, model, year)
            for make, model, year in frame[["marca", "modelo", "ano_fabricacion"]].itertuples(index=False, name=None)
        ]
    frame["_marca_key"] = frame["marca"].map(comparison_make)
    frame["_modelo_key"] = frame["modelo"].map(comparison_model)
    # Una consulta repetida no debe crear varios candidatos para un mismo vehículo lógico.
    return frame.drop_duplicates("nhtsa_vehicle_id", keep="first").reset_index(drop=True)


def _review_recommendation(technical_model: object, nhtsa_model: object, score: int) -> str:
    if score < MANUAL_REVIEW_THRESHOLD:
        return "rechazar: puntuacion inferior al umbral de revision"
    technical_root, nhtsa_root = model_root_name(technical_model), model_root_name(nhtsa_model)
    if technical_root and technical_root == nhtsa_root:
        return "revisar: aceptar solo si la diferencia es una subversion o acabado"
    return "revisar: rechazar salvo evidencia documental de que ambos nombres son el mismo modelo"


def match_technical_to_recalls(
    technical: pd.DataFrame,
    nhtsa_vehicles: pd.DataFrame,
    *,
    auto_accept_threshold: int = AUTO_ACCEPT_THRESHOLD,
    manual_review_threshold: int = MANUAL_REVIEW_THRESHOLD,
    require_catalog_verified: bool = False,
    audit_path: str | Path | None = None,
) -> pd.DataFrame:
    """Cruza cada vehículo con una consulta NHTSA de igual marca/año. Política: >=90 acepta;
    75–89 requiere revisión explícita y queda fuera de uniones mientras tanto; <75 rechaza.
    Marca y año nunca se comparan de forma difusa. En producción,
    require_catalog_verified=True impide construir candidatos copiando el CSV técnico a la
    API: ese antipatrón parece emparejar cada fila consigo misma y convierte respuestas
    inválidas vacías en etiquetas falsas.
    """

    if not 0 <= manual_review_threshold <= auto_accept_threshold <= 100:
        raise ValueError("Matching thresholds must satisfy 0 <= manual <= auto <= 100.")
    tech = _normalise_technical_keys(technical)
    recalls = _normalise_nhtsa_keys(nhtsa_vehicles)
    if require_catalog_verified:
        if "catalog_verified" not in recalls.columns:
            raise ValueError(
                "Production fuzzy matching requires an independent NHTSA catalogue with catalog_verified=True."
            )
        verified = recalls["catalog_verified"].fillna(False).astype(bool)
        if not verified.all():
            invalid = int((~verified).sum())
            raise ValueError(
                f"Refusing {invalid} unverified NHTSA candidate(s): fetch a vPIC catalogue before matching."
            )

    candidates_by_key: dict[tuple[str, int], pd.DataFrame] = {
        key: group.sort_values(["_modelo_key", "nhtsa_vehicle_id"], kind="stable")
        for key, group in recalls.groupby(["_marca_key", "ano_fabricacion"], sort=False)
    }
    rows: list[dict[str, object]] = []
    # Usar iterrows deliberadamente: pandas renombra las columnas que empiezan
    # por guion bajo al crear tuplas con nombre, pero esas claves forman
    # parte del contrato interno de este módulo.
    for _, item_dict in tech.iterrows():
        vehicle_identifier = str(item_dict["id_vehiculo_ano"])
        candidates = candidates_by_key.get((item_dict["_marca_key"], int(item_dict["ano_fabricacion"])))
        base: dict[str, object] = {
            "id_vehiculo_ano": vehicle_identifier,
            "marca_tecnica": item_dict["marca"],
            "modelo_tecnico": item_dict["modelo"],
            "ano_fabricacion": int(item_dict["ano_fabricacion"]),
            "root_modelo_tecnico": model_root_name(item_dict["modelo"]),
            "decision_manual": pd.NA,
        }
        if candidates is None or candidates.empty:
            rows.append(
                {
                    **base,
                    "nhtsa_vehicle_id": pd.NA,
                    "marca_nhtsa": pd.NA,
                    "modelo_nhtsa": pd.NA,
                    "match_score": pd.NA,
                    "match_status": "no_candidate",
                    "root_modelo_nhtsa": pd.NA,
                    "root_name_matches": False,
                    "recomendacion_revision": "rechazar: no existe candidato con la misma marca y ano",
                    "audit_required": True,
                }
            )
            continue

        technical_model_key = item_dict["_modelo_key"]
        scored = candidates.assign(
            _score=candidates["_modelo_key"].map(
                lambda candidate, model_key=technical_model_key: token_set_ratio(model_key, candidate)
            )
        )
        scored["_exact"] = scored["_modelo_key"].eq(technical_model_key)
        best = scored.sort_values(
            ["_exact", "_score", "_modelo_key", "nhtsa_vehicle_id"],
            ascending=[False, False, True, True], kind="stable"
        ).iloc[0]
        score = int(best["_score"])
        technical_root = model_root_name(item_dict["modelo"])
        nhtsa_root = model_root_name(best["modelo"])
        root_equal = bool(technical_root and technical_root == nhtsa_root)
        different_number = re.findall(r"\d+", technical_root) != re.findall(r"\d+", nhtsa_root)
        exact_count = int(scored["_exact"].sum())
        unique_exact = bool(best["_exact"]) and exact_count == 1
        ambiguous = exact_count > 1 or (
            not bool(best["_exact"]) and int(scored["_score"].eq(score).sum()) > 1
        )
        if different_number:
            status, audit = "rejected", True
        elif score >= auto_accept_threshold and (root_equal or unique_exact) and not ambiguous:
            status, audit = "auto_accepted", False
        elif score >= manual_review_threshold:
            status, audit = "manual_review", True
        else:
            status, audit = "rejected", True
        rows.append(
            {
                **base,
                "nhtsa_vehicle_id": best["nhtsa_vehicle_id"],
                "marca_nhtsa": best["marca"],
                "modelo_nhtsa": best["modelo"],
                "match_score": score,
                "match_status": status,
                "root_modelo_nhtsa": nhtsa_root,
                "root_name_matches": root_equal,
                "recomendacion_revision": _review_recommendation(item_dict["modelo"], best["modelo"], score),
                "audit_required": audit,
                "catalog_verified": bool(best.get("catalog_verified", False)),
                "catalog_source": best.get("catalog_source", pd.NA),
            }
        )

    result = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    if audit_path is not None:
        write_fuzzy_audit(result, audit_path)
    return result


def _decision_is_accepted(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"accept", "accepted", "approve", "approved", "aceptar", "aceptado", "si", "sí", "yes", "true", "1"}:
        return True
    if text in {"reject", "rejected", "rechazar", "rechazado", "no", "false", "0"}:
        return False
    return None


def apply_manual_match_decisions(
    matches: pd.DataFrame,
    decisions: pd.DataFrame | Mapping[str, object] | Iterable[Mapping[str, object]],
    *,
    enforce_root_name_rule: bool = True,
) -> pd.DataFrame:
    """Aplica decisiones humanas a casos de similitud 75–89. La tabla exige id_vehiculo_ano y
    decision_manual, manual_approved o decision. Por defecto rechaza raíces distintas:
    permite Civic/Civic-LX, no F-150/F-250.
    """

    require_columns(matches, MATCH_COLUMNS, context="Fuzzy match table")
    if isinstance(decisions, Mapping):
        decision_frame = pd.DataFrame(
            [{"id_vehiculo_ano": key, "decision_manual": value} for key, value in decisions.items()]
        )
    elif isinstance(decisions, pd.DataFrame):
        decision_frame = decisions.copy()
    else:
        decision_frame = pd.DataFrame(list(decisions))
    if decision_frame.empty:
        return matches.copy()
    require_columns(decision_frame, ("id_vehiculo_ano",), context="Manual matching decisions")
    decision_column = next(
        (column for column in ("decision_manual", "manual_approved", "decision") if column in decision_frame.columns),
        None,
    )
    if decision_column is None:
        raise ValueError("Manual decisions require decision_manual, manual_approved, or decision.")
    if decision_frame["id_vehiculo_ano"].duplicated().any():
        raise ValueError("Manual matching decisions must contain one row per id_vehiculo_ano.")
    decisions_by_id = decision_frame.set_index("id_vehiculo_ano")[decision_column].to_dict()

    updated = matches.copy()
    for index, match in updated.iterrows():
        raw_decision = decisions_by_id.get(match["id_vehiculo_ano"])
        accepted = _decision_is_accepted(raw_decision)
        if accepted is None or match["match_status"] != "manual_review":
            continue
        if accepted and enforce_root_name_rule and not bool(match["root_name_matches"]):
            updated.at[index, "decision_manual"] = "rechazado_por_regla_root_name"
            updated.at[index, "match_status"] = "manual_rejected"
        elif accepted:
            updated.at[index, "decision_manual"] = "aceptado"
            updated.at[index, "match_status"] = "manual_accepted"
            updated.at[index, "audit_required"] = True
        else:
            updated.at[index, "decision_manual"] = "rechazado"
            updated.at[index, "match_status"] = "manual_rejected"
    return updated


def apply_documented_equivalences(
    matches: pd.DataFrame, decisions: pd.DataFrame, catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Aplica equivalencias revisadas por humanos para un año-modelo y con procedencia. Valida
    el registro de revisión, no la verdad documental. No genera equivalencias; prohíbe
    cambios de marca/año y raíces numéricas. El candidato debe existir en el catálogo
    verificado, no ser un nombre libre.
    """
    fields = ("id_vehiculo_ano", "nhtsa_vehicle_id", "reviewer", "reviewed_at",
              "evidence_url", "evidence_sha256", "justification")
    require_columns(decisions, fields, context="Documented equivalences")
    if decisions.empty:
        return matches.copy()
    if decisions[list(fields)].isna().any().any() or decisions[list(fields)].astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ValueError("Equivalences require complete documentary evidence")
    if decisions.id_vehiculo_ano.duplicated().any() or matches.id_vehiculo_ano.duplicated().any():
        raise ValueError("Equivalences require unique technical identities")
    if catalog.nhtsa_vehicle_id.duplicated().any():
        raise ValueError("Official identities must be unique")
    updated = matches.copy()
    for decision in decisions.to_dict("records"):
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(decision["evidence_sha256"])):
            raise ValueError("Documentary evidence requires SHA-256")
        if not str(decision["evidence_url"]).startswith("https://"):
            raise ValueError("Documentary evidence requires an HTTPS source")
        reviewed = date.fromisoformat(str(decision["reviewed_at"]))
        if reviewed > datetime.now(timezone.utc).date():
            raise ValueError("Review date cannot be in the future")
        selected = updated.loc[updated.id_vehiculo_ano.eq(decision["id_vehiculo_ano"])]
        official = catalog.loc[catalog.nhtsa_vehicle_id.eq(decision["nhtsa_vehicle_id"])]
        if len(selected) != 1 or len(official) != 1:
            raise ValueError("Equivalence references an unknown identity")
        match, candidate = selected.iloc[0], official.iloc[0]
        if match.match_status not in {"manual_review", "rejected", "no_candidate", "manual_rejected"}:
            raise ValueError("Equivalence may only resolve an excluded identity")
        if (comparison_make(match.marca_tecnica) != comparison_make(candidate.marca)
                or int(match.ano_fabricacion) != int(candidate.ano_fabricacion)
                or candidate.get("catalog_verified") not in (True, 1)):
            raise ValueError("Equivalence crosses brand/year or lacks verified catalog identity")
        root = model_root_name(candidate.modelo)
        if re.findall(r"\d+", model_root_name(match.modelo_tecnico)) != re.findall(r"\d+", root):
            raise ValueError("Numeric model identity protection cannot be overridden")
        index = selected.index[0]
        changes = {"nhtsa_vehicle_id": candidate.nhtsa_vehicle_id,
                   "marca_nhtsa": candidate.marca, "modelo_nhtsa": candidate.modelo,
                   "root_modelo_nhtsa": root,
                   "root_name_matches": bool(root and root == model_root_name(match.modelo_tecnico)),
                   "catalog_verified": True, "catalog_source": candidate.get("catalog_source", "unknown"),
                   "match_score": token_set_ratio(match.modelo_tecnico, candidate.modelo),
                   "match_status": "manual_accepted", "decision_manual": "equivalencia_documentada",
                   "audit_required": True}
        for key, value in changes.items():
            updated.at[index, key] = value
        for key in fields[2:]:
            updated.at[index, "review_" + key] = str(decision[key])
    return updated


def accepted_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Devuelve únicamente cruces autorizados a transferir evidencia de campañas."""

    require_columns(matches, MATCH_COLUMNS, context="Fuzzy match table")
    return matches.loc[matches["match_status"].isin(("auto_accepted", "manual_accepted"))].copy()


def summarise_matches(matches: pd.DataFrame) -> MatchSummary:
    """Calcula métricas sin contar como aceptados los casos sin resolver."""

    require_columns(matches, MATCH_COLUMNS, context="Fuzzy match table")
    statuses = matches["match_status"].value_counts()
    rejected = int(sum(statuses.get(status, 0) for status in (
        "rejected", "manual_rejected", "query_failed", "unverified_zero_result"
    )))
    return MatchSummary(
        technical_rows=len(matches),
        auto_accepted=int(statuses.get("auto_accepted", 0)),
        manual_review=int(statuses.get("manual_review", 0)),
        manually_accepted=int(statuses.get("manual_accepted", 0)),
        rejected=rejected,
        no_candidate=int(statuses.get("no_candidate", 0)),
    )


def write_fuzzy_audit(matches: pd.DataFrame, audit_path: str | Path) -> Path:
    """Guarda todas las decisiones, incluidas las aceptadas, para auditoría."""

    require_columns(matches, MATCH_COLUMNS, context="Fuzzy match table")
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = matches.copy()
    audit.sort_values(["match_status", "marca_tecnica", "modelo_tecnico", "ano_fabricacion"], kind="stable").to_csv(
        path,
        index=False,
        encoding="utf-8",
    )
    return path


__all__ = [
    "AUTO_ACCEPT_THRESHOLD",
    "MANUAL_REVIEW_THRESHOLD",
    "MATCH_COLUMNS",
    "MatchSummary",
    "accepted_matches",
    "apply_manual_match_decisions",
    "match_technical_to_recalls",
    "model_root_name",
    "summarise_matches",
    "token_set_ratio",
    "write_fuzzy_audit",
]
