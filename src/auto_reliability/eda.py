"""Reproducible exploratory analysis for the completed gold cohorts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# This module is also invoked in non-interactive CI and CLI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .contracts import validate_gold_dataset


def eda_summary(gold: pd.DataFrame) -> dict[str, object]:
    """Return concise, serialisable quality and distribution statistics."""

    validate_gold_dataset(gold)
    target = gold["indice_fiabilidad_100"]
    return {
        "filas": len(gold),
        "marcas": int(gold["marca"].nunique()),
        "categorias": int(gold["categoria_vehiculo"].nunique()),
        "periodo": [int(gold["ano_fabricacion"].min()), int(gold["ano_fabricacion"].max())],
        "indice_media": round(float(target.mean()), 2),
        "indice_mediana": round(float(target.median()), 2),
        "nulos": {column: int(count) for column, count in gold.isna().sum().items() if count},
    }


def generate_eda_figures(gold: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Write a small, review-ready EDA set and return exact generated paths."""

    validate_gold_dataset(gold)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(10, 5.5))
    order = gold.groupby("categoria_vehiculo")["indice_fiabilidad_100"].median().sort_values().index
    sns.boxplot(
        data=gold,
        x="categoria_vehiculo",
        y="indice_fiabilidad_100",
        order=order,
        ax=axis,
    )
    axis.set_title("Indice de fiabilidad por categoria")
    axis.set_xlabel("Categoria de vehiculo")
    axis.set_ylabel("Indice 0-100 (alto = menor propension a recalls)")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path = destination / "01_indice_por_categoria.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    trend = (
        gold.groupby("ano_fabricacion", as_index=False)["indice_fiabilidad_100"]
        .mean()
        .sort_values("ano_fabricacion")
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(data=trend, x="ano_fabricacion", y="indice_fiabilidad_100", marker="o", ax=axis)
    axis.set_title("Evolucion media del indice por cohorte")
    axis.set_xlabel("Ano de fabricacion")
    axis.set_ylabel("Indice medio 0-100")
    figure.tight_layout()
    path = destination / "02_evolucion_indice.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    numeric = gold.loc[
        :, ["mediana_cilindros", "mediana_cv", "score_recalls_bruto", "hist_fiabilidad_marca", "indice_fiabilidad_100"]
    ].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(numeric, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axis)
    axis.set_title("Correlaciones de variables numericas")
    figure.tight_layout()
    path = destination / "03_correlaciones.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    return outputs
