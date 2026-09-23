"""Deterministic synthetic data for an honest, runnable product demonstration.

The demo is intentionally labelled synthetic. It validates the end-to-end
software path when the licensed CooperUnion CSV has not yet been placed in
``data/raw``; it must never be presented as NHTSA evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CURRENT_YEAR, MIN_MODEL_YEAR, OBSERVATION_WINDOW_YEARS, ProjectPaths
from .contracts import vehicle_id


@dataclass(frozen=True)
class DemoVehicleFamily:
    make: str
    model: str
    category: str
    cylinders: float
    horsepower: float
    brand_effect: float


_FAMILIES: tuple[DemoVehicleFamily, ...] = (
    DemoVehicleFamily("Ford", "Explorer", "SUV", 6, 300, 0.45),
    DemoVehicleFamily("Ford", "Escape", "SUV", 4, 185, 0.30),
    DemoVehicleFamily("Toyota", "RAV4", "SUV", 4, 203, -0.35),
    DemoVehicleFamily("Toyota", "Camry", "Sedan", 4, 203, -0.45),
    DemoVehicleFamily("Honda", "Civic", "Compact", 4, 180, -0.30),
    DemoVehicleFamily("Honda", "Pilot", "SUV", 6, 285, -0.10),
    DemoVehicleFamily("Chevrolet", "Silverado", "Pickup", 8, 355, 0.50),
    DemoVehicleFamily("Chevrolet", "Malibu", "Sedan", 4, 163, 0.15),
    DemoVehicleFamily("Subaru", "Outback", "Wagon", 4, 182, -0.20),
    DemoVehicleFamily("Nissan", "Rogue", "SUV", 4, 201, 0.25),
    DemoVehicleFamily("Hyundai", "Tucson", "SUV", 4, 187, 0.05),
    DemoVehicleFamily("Mazda", "CX-5", "SUV", 4, 187, -0.15),
)


def make_demo_catalog(*, seed: int = 73, through_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Return a stable catalog with labelled and recent inference-only cohorts."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    if through_year < MIN_MODEL_YEAR + OBSERVATION_WINDOW_YEARS:
        raise ValueError("The demo needs at least one completed three-year cohort.")
    # At the beginning of `through_year`, cohorts through year - 3 have
    # completed their three-calendar-year observation period.
    last_complete_year = through_year - OBSERVATION_WINDOW_YEARS
    for year in range(MIN_MODEL_YEAR, through_year + 1):
        technology_trend = max(0.0, (year - MIN_MODEL_YEAR) / 30)
        for family in _FAMILIES:
            cylinders = max(3.0, family.cylinders + rng.normal(0, 0.22))
            horsepower = max(90.0, family.horsepower + (technology_trend * 35) + rng.normal(0, 12))
            volatility = rng.normal(0, 0.32)
            raw_recall_score = max(
                0.05,
                1.35
                + family.brand_effect
                + (horsepower - 190) / 270
                + (cylinders - 4) / 8
                + technology_trend * 0.22
                + volatility,
            )
            records.append(
                {
                    "id_vehiculo_ano": vehicle_id(family.make, family.model, year),
                    "marca": family.make.lower(),
                    "modelo": family.model.lower(),
                    "ano_fabricacion": year,
                    "categoria_vehiculo": family.category,
                    "mediana_cilindros": round(float(cylinders), 1),
                    "mediana_cv": round(float(horsepower), 1),
                    "score_recalls_bruto": round(float(raw_recall_score), 3),
                    "cohorte_completa": year <= last_complete_year,
                    "fuente_demo": True,
                }
            )

    frame = pd.DataFrame(records)
    # Only the last three *matured* cohorts are available at a launch. Using
    # the final recall totals of y-1 or y-2 here would expose future events.
    frame = frame.sort_values(["marca", "ano_fabricacion", "modelo"]).reset_index(drop=True)
    histories: list[float] = []
    for _, row in frame.iterrows():
        latest_known_cohort = int(row["ano_fabricacion"]) - OBSERVATION_WINDOW_YEARS
        known = frame.loc[
            frame["ano_fabricacion"].between(latest_known_cohort - 2, latest_known_cohort)
            & frame["cohorte_completa"]
        ]
        prior = known.loc[known["marca"] == row["marca"], "score_recalls_bruto"]
        histories.append(
            float(prior.mean()) if not prior.empty
            else float(known["score_recalls_bruto"].mean()) if not known.empty
            else np.nan
        )
    frame["hist_fiabilidad_marca"] = histories

    # The UI semantics are intentionally unambiguous: high score = lower
    # historically observed recall propensity. Even the synthetic fixture
    # fits this target scale on the documented training horizon only, so it
    # cannot conceal a leakage bug during end-to-end tests.
    complete = frame["cohorte_completa"]
    train_reference = frame.loc[complete & frame["ano_fabricacion"].le(2018)].copy()
    global_mean = float(train_reference["score_recalls_bruto"].mean())
    global_std = float(train_reference["score_recalls_bruto"].std(ddof=0))
    if not np.isfinite(global_std) or global_std == 0:
        global_std = 1.0
    segment_stats = train_reference.groupby("categoria_vehiculo")["score_recalls_bruto"].agg(
        mean="mean", std=lambda values: values.std(ddof=0)
    )
    segment_mean = frame.loc[complete, "categoria_vehiculo"].map(segment_stats["mean"]).fillna(global_mean)
    segment_std = frame.loc[complete, "categoria_vehiculo"].map(segment_stats["std"]).fillna(global_std)
    segment_std = segment_std.replace(0, global_std).fillna(global_std)
    z = (frame.loc[complete, "score_recalls_bruto"] - segment_mean) / segment_std
    frame["indice_fiabilidad_100"] = np.nan
    frame.loc[complete, "indice_fiabilidad_100"] = (50 - (z * 10)).clip(0, 100)
    # Do not expose synthetic future outcomes to downstream inference code.
    frame.loc[~complete, "score_recalls_bruto"] = np.nan
    frame["indice_fiabilidad_100"] = frame["indice_fiabilidad_100"].round(2)
    return frame.sort_values(["marca", "modelo", "ano_fabricacion"]).reset_index(drop=True)


def write_demo_catalog(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist labelled gold and full inference catalog without overwriting docs."""

    paths.ensure_runtime_directories()
    for destination in (paths.gold_path, paths.inference_catalog_path):
        if destination.exists():
            existing = pd.read_parquet(destination)
            if "fuente_demo" not in existing or not existing["fuente_demo"].eq(True).all():
                raise ValueError(
                    f"Refusing to overwrite non-demo data at {destination}. "
                    "Use a separate AUTO_RELIABILITY_ROOT for the demonstration."
                )
    catalog = make_demo_catalog()
    gold = catalog.loc[catalog["cohorte_completa"]].copy()
    gold.to_parquet(paths.gold_path, index=False)
    catalog.to_parquet(paths.inference_catalog_path, index=False)
    return gold, catalog
