"""Contrato de cifras publicadas en Markdown con las métricas de la versión activa.

La comprobación es de solo lectura: nunca reentrena ni corrige un documento de forma
silenciosa. Las tablas se generan desde JSON y CI detecta transcripciones obsoletas.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import ProjectPaths
from .releases import serving_paths

START = "<!-- AUTO_RELIABILITY_METRICS:START -->"
END = "<!-- AUTO_RELIABILITY_METRICS:END -->"
DOCUMENTS = ("README.md", "docs/entregas/05_diseno_frontal.md")


def _number(value: Any, *, integer: bool = False) -> str:
    """Formatea cantidades verificables sin convertir ausencias o errores en ceros."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Las métricas publicadas requieren cantidades numéricas explícitas.")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError("La métrica no es finita, positiva o un recuento entero válido.")
    return str(int(value)) if integer else f"{value:.4f}".replace(".", ",")


def render_metrics_table(metrics: Mapping[str, Any]) -> str:
    """Genera la tabla común desde las métricas, no desde valores copiados en código.

    La prosa describe la publicación baseline. Otro ganador exige revisar también
    esa prosa; no basta con regenerar sus cifras automáticamente.
    """
    if metrics.get("fuente_demo") is not False:
        raise ValueError("La documentación final requiere métricas reales, no de demostración.")
    if metrics["selection"]["selected_model"] != "baseline":
        raise ValueError("Cambió el modelo seleccionado: revise la documentación del baseline.")
    split = metrics["split"]
    train, validation, test = (split[key] for key in ("train", "validation", "test"))
    final = metrics["final_test"]
    if final["n"] != test["rows"]:
        raise ValueError("El denominador de test no coincide con la partición.")

    def period(group: Mapping[str, Any]) -> str:
        return f"{_number(group['min_year'], integer=True)}–{_number(group['max_year'], integer=True)}"

    rows = [
        ("Modelo seleccionado", "Baseline de marca y categoría"),
        ("Entrenamiento para selección", f"{_number(train['rows'], integer=True)} casos; {period(train)}"),
        ("Validación", f"{_number(validation['rows'], integer=True)} casos; {period(validation)}"),
        ("Ajuste final sin test", f"{_number(train['rows'] + validation['rows'], integer=True)} casos"),
    ]
    for key, label in (("baseline", "baseline"), ("ridge", "Ridge"),
                       ("random_forest", "Random Forest")):
        candidate = metrics["candidates"][key]
        if not candidate["available"] or candidate["validation"]["n"] != validation["rows"]:
            raise ValueError(f"Evaluación no comparable para {key}.")
        result = candidate["validation"]
        rows.extend((
            (f"MAE de validación {label}", _number(result["mae"])),
            (f"RMSE de validación {label}", _number(result["rmse"])),
        ))
    rows.extend((
        ("Test", f"{_number(final['n'], integer=True)} casos; {period(test)}"),
        ("MAE de test", _number(final["mae"])),
        ("RMSE de test", _number(final["rmse"])),
    ))
    return "\n".join([START, "| Métrica | Resultado |", "|---|---:|",
                       *(f"| {name} | {value} |" for name, value in rows), END])


def check_metrics_block(document: str, expected: str) -> None:
    """Exige un único bloque idéntico al generado, sin depender de finales de línea."""
    normalized = document.replace("\r\n", "\n")
    if normalized.count(START) != 1 or normalized.count(END) != 1:
        raise ValueError("Falta un bloque único de métricas verificadas.")
    start, end = normalized.index(START), normalized.index(END)
    if end < start or normalized[start:end + len(END)] != expected:
        raise ValueError("La tabla de métricas difiere del artefacto activo.")


def verified_table(root: Path) -> str:
    """Lee la publicación con hashes verificados y comprueba su copia de trabajo."""
    paths = ProjectPaths(root.resolve())
    active = serving_paths(paths)
    published = active.artifacts_dir / "model_metrics.json"
    working = paths.artifacts_dir / "model_metrics.json"
    if working.read_bytes() != published.read_bytes():
        raise ValueError("Las métricas de trabajo difieren de la publicación activa.")
    return render_metrics_table(json.loads(published.read_text(encoding="utf-8")))


def check_documentation(root: Path) -> None:
    """Valida los dos documentos de entrega sin escribir ni alterar sus contenidos."""
    expected = verified_table(root)
    for relative in DOCUMENTS:
        try:
            check_metrics_block((root / relative).read_text(encoding="utf-8"), expected)
        except ValueError as exc:
            raise ValueError(f"{relative}: {exc}") from exc


def main() -> None:
    """Muestra la tabla canónica o comprueba la documentación en modo de solo lectura."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ProjectPaths.discover().root)
    parser.add_argument("--show", action="store_true", help="Imprime la tabla, sin editar archivos.")
    arguments = parser.parse_args()
    if arguments.show:
        print(verified_table(arguments.root))
    else:
        check_documentation(arguments.root)
        print("README y Entrega 5: métricas coherentes con la publicación activa.")


if __name__ == "__main__":
    main()
