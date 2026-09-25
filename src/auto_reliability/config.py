"""Rutas y constantes centrales sin efectos secundarios."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def repository_root() -> Path:
    """Devuelve la raíz del repositorio en instalaciones editables o desde el código fuente."""

    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Contrato de archivos compartido por ingesta, entrenamiento e interfaz.
    ensure_runtime_directories solo crea directorios de escritura ausentes; nunca vacía
    fuentes, procesados, Gold, documentación ni código.
    """

    root: Path

    @classmethod
    def discover(cls) -> ProjectPaths:
        return cls(Path(os.environ.get("AUTO_RELIABILITY_ROOT", repository_root())).resolve())

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def temp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def gold_path(self) -> Path:
        return self.gold_dir / "gold_us_car_reliability.parquet"

    @property
    def inference_catalog_path(self) -> Path:
        return self.gold_dir / "us_car_inference_catalog.parquet"

    @property
    def audit_path(self) -> Path:
        return self.processed_dir / "audit_fuzzy_matches.csv"

    @property
    def sqlite_path(self) -> Path:
        return self.processed_dir / "auto_reliability.sqlite"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "reliability_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "model_metrics.json"

    def ensure_runtime_directories(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.gold_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


CURRENT_YEAR = datetime.now(timezone.utc).year
MIN_MODEL_YEAR = 1995
OBSERVATION_WINDOW_YEARS = 3

# La documentación establece este contrato exacto de severidad.
RECALL_SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 3.0,
    "moderate": 1.5,
    "low": 1.0,
}

RECALL_SEVERITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "critical": ("engine", "transmission", "brake", "steering", "fire"),
    "moderate": ("suspension", "airbag", "electrical", "fuel"),
    "low": ("infotainment", "trim", "paint", "seat", "visibility"),
}
