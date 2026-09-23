"""Optional, strictly grounded local-LLM narrative selection.

Set AUTO_RELIABILITY_OLLAMA_MODEL to an already installed Ollama model to opt
in. No model is downloaded, no cloud service is called, and predictions never
depend on this component. The model selects factual clauses, not new facts.
API contract: https://docs.ollama.com/api/generate
"""

from __future__ import annotations

import json
import math
import os
from urllib.parse import urlparse

import requests

LABELS = {"marca": "La marca", "categoria_vehiculo": "La categoría",
          "mediana_cilindros": "Los cilindros", "mediana_cv": "La potencia",
          "hist_fiabilidad_marca": "El historial previo de la marca"}
DISCLAIMER = "Estas atribuciones describen el modelo, no causas de averías. El índice es un proxy de recalls, no fiabilidad mecánica garantizada."


def build_grounded_narrative(factors, fallback_text, *, model=None, base_url=None, session=None):
    """Select up to three approved clauses; fail closed to deterministic text.

    A constrained JSON response may only reorder existing clause identifiers.
    Arbitrary prose, new numbers, invented features and omitted leading effects
    are rejected before anything is displayed to the user.
    """
    fallback = {"text": fallback_text, "mode": "determinista",
                "note": "Resumen determinista de atribuciones; no establece causalidad."}
    if any(factor.get("method") == "media_grupo_vs_media_global_train" for factor in factors):
        return {**fallback, "note": "Regla del baseline entrenado; diferencia de grupo, no efectos individuales ni causales."}
    model = model if model is not None else os.getenv("AUTO_RELIABILITY_OLLAMA_MODEL", "")
    if not model:
        return fallback
    base_url = base_url or os.getenv("AUTO_RELIABILITY_OLLAMA_URL", "http://127.0.0.1:11434")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
        return {**fallback, "note": "IA local no activada: solo se admiten destinos loopback sin credenciales."}
    clauses = {}
    ordered = sorted(factors, key=lambda f: abs(float(f.get("contribution", 0))), reverse=True)
    for factor in ordered:
        name = factor.get("feature")
        effect = float(factor.get("contribution", 0))
        if name not in LABELS or not math.isfinite(effect) or abs(effect) < 0.001:
            continue
        direction = "eleva" if effect > 0 else "reduce"
        clauses[name] = f"{LABELS[name]} {direction} el índice estimado en {abs(effect):.2f} puntos respecto a la referencia de entrenamiento."
    if not clauses:
        return fallback
    schema = {"type": "object", "additionalProperties": False, "required": ["clauses"],
              "properties": {"clauses": {"type": "array", "minItems": 1, "maxItems": 3,
                                          "items": {"type": "string", "enum": list(clauses)}}}}
    try:
        client = session or requests.Session()
        response = client.post(base_url.rstrip("/") + "/api/generate", json={
            "model": model, "stream": False, "format": schema, "options": {"temperature": 0},
            "system": "Selecciona hasta tres IDs para resumir los hechos suministrados. Incluye el primer ID. No añadas texto ni datos.",
            "prompt": json.dumps(clauses, ensure_ascii=False),
        }, timeout=(2, 10), allow_redirects=False)
        if response.status_code != 200:
            raise ValueError("Local provider unavailable")
        payload = json.loads(response.json()["response"])
        selection = payload["clauses"]
        if (set(payload) != {"clauses"} or not isinstance(selection, list) or not 1 <= len(selection) <= 3
                or any(not isinstance(key, str) or key not in clauses for key in selection)
                or len(set(selection)) != len(selection) or next(iter(clauses)) not in selection):
            raise ValueError("Ungrounded selection")
        return {"text": " ".join(clauses[key] for key in selection) + " " + DISCLAIMER,
                "mode": "ollama_seleccion_validada", "note": "IA local: selección de hechos verificados; redacción controlada, sin afirmaciones nuevas."}
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return {**fallback, "note": "IA local no disponible o respuesta no verificable; se muestra el resumen determinista."}
