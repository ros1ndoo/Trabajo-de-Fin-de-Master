"""Backend facade consumed by the Streamlit dashboard.

The service keeps inference, fallback logic and display context in one place so
the UI never reads a vehicle's own recall-derived target as a prediction
feature. It can bootstrap an explicitly labelled synthetic demo when a fresh
clone has no data artefacts yet.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .contracts import PREDICTION_FEATURES, canonical_text, dataset_fingerprint, vehicle_id
from .demo_data import write_demo_catalog
from .explainability import explain_prediction, summarize_attributions
from .modeling import (
    BrandSegmentMeanBaseline,
    ReliabilityModelArtifact,
    load_model_artifact,
    predict_with_artifact,
    train_and_select_model,
)
from .narrative import build_grounded_narrative
from .reporting import build_prediction_report_pdf


class VehicleNotFoundError(LookupError):
    """Raised when a requested make/model/year does not exist in the catalogue."""


class InsufficientEvidenceError(ValueError):
    """No trained model or mature reference exists; do not invent a score."""


@dataclass(frozen=True)
class ServiceStatus:
    """A small, UI-friendly description of the loaded data and model state."""

    demo_mode: bool
    catalog_rows: int
    gold_rows: int
    model_available: bool
    message: str
    technical_year_min: int | None = None
    technical_year_max: int | None = None
    ingestion_warnings: tuple[str, ...] = ()
    inventory_alignment: dict[str, Any] | None = None


class ReliabilityService:
    """Load project artefacts and make leakage-safe vehicle predictions.

    Parameters
    ----------
    paths:
        Optional custom project paths, useful for isolated tests/deployments.
    auto_bootstrap_demo:
        When explicitly true, a clone without outputs receives a clearly
        labelled synthetic demo so ``streamlit run app.py`` is immediately
        usable. Set ``AUTO_RELIABILITY_AUTO_DEMO=0`` or pass ``False`` to
        require real artefacts instead.
    """

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        *,
        auto_bootstrap_demo: bool | None = None,
    ) -> None:
        self.paths = paths or ProjectPaths.discover()
        self.project_paths = self.paths
        if auto_bootstrap_demo is None:
            auto_bootstrap_demo = os.getenv("AUTO_RELIABILITY_AUTO_DEMO", "0") == "1"
        self.auto_bootstrap_demo = bool(auto_bootstrap_demo)
        self._catalog: pd.DataFrame | None = None
        self._gold: pd.DataFrame | None = None
        self._artifact: ReliabilityModelArtifact | None = None
        self._demo_mode = False
        self._artifact_error = ""
        self._artifact_checked = False
        self._runtime_revision: tuple | None = None

    def data_revision(self) -> tuple:
        """A cheap revision token for invalidating a live session after ingestion."""
        return tuple((str(path), path.stat().st_mtime_ns if path.exists() else None)
                     for path in (self.paths.gold_path, self.paths.inference_catalog_path, self.paths.model_path))

    def status(self) -> ServiceStatus:
        """Return current service status without exposing internal exceptions."""

        catalog = self.load_catalog()
        gold = self._load_gold()
        artifact = self._load_artifact()
        if catalog.empty:
            message = "No hay fichas técnicas cargadas. La consulta directa de recalls sigue disponible."
        elif self._demo_mode:
            message = "Modo demostración: los datos son sintéticos y no representan evidencia NHTSA real."
        elif artifact is None:
            message = "Catálogo cargado; el dashboard utilizará baseline hasta entrenar un modelo."
        else:
            message = "Capa Gold y modelo entrenado cargados."
        if not catalog.empty:
            first, last = int(catalog.ano_fabricacion.min()), int(catalog.ano_fabricacion.max())
            message += f" Cobertura técnica: {first}–{last} ({len(catalog)} modelos-año)."
            if not self._demo_mode and last < datetime.now(timezone.utc).year:
                message += " No incluye especificaciones de lanzamientos actuales."
        if self._artifact_error:
            message += " " + self._artifact_error
        manifest_path = self.paths.processed_dir / "pipeline_manifest.json"
        warnings: tuple[str, ...] = ()
        if manifest_path.exists() and not self._demo_mode:
            try:
                warnings = tuple(json.loads(manifest_path.read_text(encoding="utf-8")).get("warnings", []))
            except (OSError, ValueError):
                pass
        return ServiceStatus(
            demo_mode=self._demo_mode,
            catalog_rows=len(catalog),
            gold_rows=len(gold),
            model_available=artifact is not None,
            message=message,
            technical_year_min=int(catalog.ano_fabricacion.min()) if not catalog.empty else None,
            technical_year_max=int(catalog.ano_fabricacion.max()) if not catalog.empty else None,
            ingestion_warnings=warnings,
            inventory_alignment=self.inventory_alignment_status() if not self._demo_mode else None,
        )

    def inventory_alignment_status(self) -> dict[str, Any] | None:
        """Read optional independent-inventory progress, never prediction inputs."""
        path = self.paths.processed_dir / "inventory_resolution" / "latest.json"
        if not path.exists():
            return None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            for key in ("variants", "source_names", "linked_source_names"):
                if type(report[key]) is not int or report[key] < 0:
                    raise ValueError("Invalid inventory count")
            if not report["linked_source_names"] <= report["source_names"] <= report["variants"]:
                raise ValueError("Inconsistent inventory counts")
            return {key: report[key] for key in ("variants", "source_names", "linked_source_names",
                                                "epa_snapshot_date", "policy_version")}
        except (OSError, ValueError, KeyError, TypeError):
            return {"error": "No se pudo validar el informe del inventario complementario."}

    def load_catalog(self) -> pd.DataFrame:
        """Return one row per vehicle-year, including recent inference cohorts."""

        self._ensure_artifacts()
        if self._catalog is not None:
            return self._catalog.copy()
        gold = self._load_gold()
        inference = self._read_parquet_if_exists(self.paths.inference_catalog_path)
        if inference.empty:
            catalog = gold.copy()
        elif gold.empty:
            catalog = inference.copy()
        else:
            # Gold is authoritative for completed labelled cohorts. The
            # inference table contributes only rows absent from Gold.
            gold_ids = set(gold.get("id_vehiculo_ano", pd.Series(dtype=str)).astype(str))
            only_inference = inference.loc[
                ~inference.get("id_vehiculo_ano", pd.Series(dtype=str)).astype(str).isin(gold_ids)
            ].copy()
            catalog = pd.concat([gold, only_inference], ignore_index=True, sort=False)
        if catalog.empty:
            self._catalog = catalog
            return catalog.copy()
        catalog = catalog.drop_duplicates("id_vehiculo_ano", keep="first") if "id_vehiculo_ano" in catalog else catalog
        catalog["marca"] = catalog["marca"].astype(str).str.strip().str.lower()
        catalog["modelo"] = catalog["modelo"].astype(str).str.strip().str.lower()
        catalog["ano_fabricacion"] = pd.to_numeric(catalog["ano_fabricacion"], errors="coerce").astype("Int64")
        catalog = catalog.dropna(subset=["marca", "modelo", "ano_fabricacion"])
        catalog["ano_fabricacion"] = catalog["ano_fabricacion"].astype(int)
        self._catalog = catalog.sort_values(["marca", "modelo", "ano_fabricacion"]).reset_index(drop=True)
        return self._catalog.copy()

    def predict(self, marca: str, modelo: str, ano_fabricacion: int, *, identity_verified: bool = False) -> dict[str, Any]:
        """Predict one catalogued vehicle using only launch-time features."""

        catalog = self.load_catalog()
        row = self._lookup(catalog, marca, modelo, ano_fabricacion)
        artifact = self._load_artifact()
        context = self._context(catalog, row)
        if artifact is None:
            return self._baseline_prediction(row, context, reason=self._artifact_error or "No hay modelo entrenado disponible.")
        if not identity_verified and "coincidencia_recall_aceptada" in row and not _as_bool(row["coincidencia_recall_aceptada"]):
            return self._baseline_prediction(row, context, reason="Vehículo sin cruce NHTSA aceptado.")

        feature_row = pd.DataFrame([{feature: row.get(feature) for feature in PREDICTION_FEATURES}])
        # ``predict_with_artifact`` deliberately only receives PREDICTION_FEATURES;
        # neither score_recalls_bruto nor indice_fiabilidad_100 can leak in.
        output = predict_with_artifact(artifact, feature_row, include_explanations=False).iloc[0].to_dict()
        factors = explain_prediction(artifact, feature_row, top_n=4)
        result: dict[str, Any] = {
            "id_vehiculo_ano": row.get("id_vehiculo_ano"),
            "marca": str(row["marca"]),
            "modelo": str(row["modelo"]),
            "ano_fabricacion": int(row["ano_fabricacion"]),
            "categoria_vehiculo": row.get("categoria_vehiculo", "sin_categoria"),
            "prediccion_indice_100": float(output["prediccion_indice_100"]),
            "mae": _finite_or_none(output.get("mae")),
            "modelo_usado": _human_model_name(artifact.model_name),
            "es_baseline": artifact.model_name == "baseline" or isinstance(artifact.estimator, BrandSegmentMeanBaseline),
            "fallback": False,
            "fuente_demo": self._demo_mode or _as_bool(row.get("fuente_demo", False)),
            "explicacion_factores": summarize_attributions(factors),
            "factores": factors,
            "fecha_ejecucion": output.get("fecha_ejecucion")
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **context,
        }
        if result["es_baseline"]:
            result["mensaje"] = "El proceso de selección conservó el baseline según los criterios de error y complejidad registrados en la evaluación."
        if not _as_bool(row.get("es_cohorte_completa", row.get("cohorte_completa", True))):
            result["mensaje"] = (
                "Cohorte con ventana de observación incompleta: se muestra una predicción basada solo en "
                "especificaciones y el historial previo de la marca; sus propios recalls no se usan."
            )
        available_year = artifact.metadata.get("prediction_available_from_year")
        if available_year is None and artifact.train_year_max is not None:
            available_year = artifact.train_year_max + 3
        result["evaluacion_retrospectiva"] = bool(available_year is not None and
                                                  int(row["ano_fabricacion"]) < int(available_year))
        if result["evaluacion_retrospectiva"]:
            result["mensaje"] = (result.get("mensaje", "") + " Consulta retrospectiva: el modelo se entrenó con "
                "cohortes posteriores o coincidentes. No representa una predicción hecha en la fecha de lanzamiento.").strip()
        result["mae_descripcion"] = "MAE de evaluación temporal; no es un intervalo de confianza individual."
        narrative = build_grounded_narrative(factors, result["explicacion_factores"])
        result["explicacion_factores"] = narrative["text"]
        result["tipo_explicacion"] = narrative["note"]
        result["modo_explicacion"] = narrative["mode"]
        return result

    def comparison_context(self, marca: str, modelo: str, ano_fabricacion: int) -> dict[str, Any]:
        """Expose non-predictive display context for integrations beyond Streamlit."""

        catalog = self.load_catalog()
        return self._context(catalog, self._lookup(catalog, marca, modelo, ano_fabricacion))

    def predict_on_demand(self, marca: str, modelo: str, ano_fabricacion: int) -> dict[str, Any]:
        """Acquire official evidence before answering, without leaking outcomes.

        Current recalls are returned as separate provenance only. Prediction
        features and the frozen training artifact are never enriched with the
        selected vehicle's own outcomes. Historical features are prepared by
        the complete offline pipeline, not trained inside a web request.
        """
        catalog = self.load_catalog()
        row = self._lookup(catalog, marca, modelo, ano_fabricacion)
        if self._demo_mode:
            return self.predict(marca, modelo, ano_fabricacion)
        from .bulk_recalls import BulkRecallStore
        from .data_sources import DataSourceError
        from .matching import accepted_matches, match_technical_to_recalls

        artifact = self._load_artifact()
        baseline_supported = bool(
            artifact is not None and isinstance(artifact.estimator, BrandSegmentMeanBaseline)
            and canonical_text(row["marca"]) in artifact.estimator.brand_means_
        )
        if (artifact is not None and isinstance(artifact.estimator, BrandSegmentMeanBaseline)
                and not baseline_supported):
            raise InsufficientEvidenceError(
                "La marca no está representada en el baseline entrenado. No se sustituye por una media global; "
                "puede consultar sus recalls oficiales."
            )
        store = BulkRecallStore(self.project_paths)
        official = store.catalog() if store.path.exists() else pd.DataFrame()
        if official.empty and not baseline_supported:
            raise InsufficientEvidenceError("Falta el catálogo oficial. Descargue los datos NHTSA antes de predecir.")
        matches = (accepted_matches(match_technical_to_recalls(pd.DataFrame([row]), official, require_catalog_verified=True))
                   if not official.empty else pd.DataFrame())
        if matches.empty and not baseline_supported:
            raise InsufficientEvidenceError("Identidad NHTSA no verificada o cruce ambiguo. No se generará una puntuación.")
        if row.get("origen_hist_fiabilidad_marca") != "marca_cohortes_completadas_previas" and not baseline_supported:
            raise InsufficientEvidenceError("Sin cohortes previas suficientes de esta marca para una predicción individual. Consulte los recalls oficiales.")
        # An observed own label is not a prerequisite for a trained prediction.
        # The selected baseline only uses brand/segment, not recent-history or
        # technical numeric features. Its exact fitted policy and MAE stay fixed.
        query_make = str(matches.iloc[0]["marca_nhtsa"]) if not matches.empty else str(row["marca"])
        query_model = str(matches.iloc[0]["modelo_nhtsa"]) if not matches.empty else str(row["modelo"])
        try:
            evidence = self.real_recalls(query_make, query_model, ano_fabricacion)
            provenance = {key: evidence[key] for key in ("count", "fetched_at", "from_cache", "source")}
            provenance["identity_status"] = "matched_nhtsa" if not matches.empty else "technical_name_unverified_in_nhtsa"
            provenance["status"] = "queried"
        except (DataSourceError, OSError) as exc:
            if not baseline_supported:
                raise
            provenance = {"count": None, "fetched_at": None, "from_cache": False, "source": None,
                          "status": "unavailable", "error_type": type(exc).__name__,
                          "notice": "Consulta oficial no disponible. No significa cero recalls; puede reintentarse."}
        # Identity comes from the validated technical catalog when recall name
        # alignment is unavailable. No current recall enters predictor features.
        result = self.predict(marca, modelo, ano_fabricacion, identity_verified=True)
        result["evidencia_oficial"] = provenance
        result["mensaje"] = (str(result.get("mensaje", "")) +
            " Consulta oficial intentada por separado; los recalls propios no se utilizan como variables predictoras.").strip()
        if baseline_supported:
            _, sources = artifact.estimator.predict_with_sources(pd.DataFrame([row]))
            result["nivel_estimacion"] = sources[0]
            result["tipo_resultado"] = "baseline_entrenado_de_grupo"
            result["mensaje"] += (
                " Estimación del baseline entrenado para esta marca y, cuando hay soporte, segmento; "
                "no distingue modelos dentro del mismo grupo ni usa la potencia para ajustar el índice. "
                "El MAE corresponde a la evaluación global del artefacto, no a una garantía para este coche."
            )
            if provenance["status"] == "unavailable":
                result["mensaje"] += " La API no está disponible; no se ha supuesto ausencia de recalls."
            elif matches.empty:
                result["mensaje"] += " La denominación técnica no tiene equivalencia NHTSA confirmada; revise los registros por VIN."
        return result

    def build_prediction_report_pdf(self, prediction: Mapping[str, Any]) -> bytes:
        """Dashboard hook for a controlled, local PDF summary."""

        return build_prediction_report_pdf(prediction)

    def _ensure_artifacts(self) -> None:
        revision = self.data_revision()
        if revision != self._runtime_revision:
            self._runtime_revision = revision
            self._catalog = self._gold = self._artifact = None
            self._artifact_checked = False
            self._artifact_error = ""
        if self.paths.gold_path.exists():
            self._demo_mode = self._detect_demo_mode()
            return
        if not self.auto_bootstrap_demo:
            return
        if self.paths.inference_catalog_path.exists():
            return  # A partial real pipeline must never be overwritten by a demo.
        self.paths = ProjectPaths(self.project_paths.data_dir / "demo")
        if self.paths.gold_path.exists():
            self._demo_mode = True
            return
        self.paths.ensure_runtime_directories()
        gold, _ = write_demo_catalog(self.paths)
        train_and_select_model(gold, artifact_dir=self.paths.artifacts_dir, persist=True)
        self._demo_mode = True

    def _detect_demo_mode(self) -> bool:
        gold = self._read_parquet_if_exists(self.paths.gold_path)
        return bool("fuente_demo" in gold.columns and gold["fuente_demo"].fillna(False).astype(bool).any())

    def _load_gold(self) -> pd.DataFrame:
        self._ensure_artifacts()
        if self._gold is None:
            self._gold = self._read_parquet_if_exists(self.paths.gold_path)
        return self._gold.copy()

    def _load_artifact(self) -> ReliabilityModelArtifact | None:
        self._ensure_artifacts()
        if self._artifact_checked:
            return self._artifact
        self._artifact_checked = True
        try:
            self._artifact = load_model_artifact(self.paths.model_path)
        except (FileNotFoundError, TypeError, OSError, ValueError, EOFError):
            return None
        signature = self._artifact.metadata.get("dataset_fingerprint")
        if not signature or signature != dataset_fingerprint(self._load_gold()):
            self._artifact = None
            self._artifact_error = "Modelo desactualizado o de otra fuente: ejecute auto-reliability train."
        return self._artifact

    @staticmethod
    def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _lookup(catalog: pd.DataFrame, marca: str, modelo: str, ano_fabricacion: int) -> pd.Series:
        year = int(ano_fabricacion)
        if catalog.empty:
            raise VehicleNotFoundError("No hay un catálogo técnico cargado.")
        mask = (
            catalog["marca"].map(canonical_text).eq(canonical_text(marca))
            & catalog["modelo"].map(canonical_text).eq(canonical_text(modelo))
            & catalog["ano_fabricacion"].eq(year)
        )
        found = catalog.loc[mask]
        if found.empty:
            raise VehicleNotFoundError(
                f"No hay datos de catálogo para {marca} {modelo} ({year}) en el mercado estadounidense."
            )
        return found.iloc[0]

    def _context(self, catalog: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
        year = int(row["ano_fabricacion"])
        category = str(row.get("categoria_vehiculo", "sin_categoria"))
        older_segment = catalog.loc[
            (catalog["categoria_vehiculo"].map(canonical_text) == canonical_text(category))
            & (catalog["ano_fabricacion"] < year)
        ].copy()
        feature_labels = {
            "Cilindros": "mediana_cilindros",
            "Potencia (CV)": "mediana_cv",
            "Historial previo disponible": "hist_fiabilidad_marca",
        }
        vehicle_features: dict[str, float] = {}
        segment_features: dict[str, float] = {}
        feature_ranges: dict[str, tuple[float, float]] = {}
        for label, column in feature_labels.items():
            vehicle_value = _finite_or_none(row.get(column))
            segment = pd.to_numeric(older_segment.get(column), errors="coerce").dropna()
            if vehicle_value is not None:
                vehicle_features[label] = vehicle_value
            if not segment.empty:
                segment_features[label] = float(segment.mean())
                lower = min(float(segment.min()), vehicle_value if vehicle_value is not None else float(segment.min()))
                upper = max(float(segment.max()), vehicle_value if vehicle_value is not None else float(segment.max()))
                feature_ranges[label] = (lower, upper)
        historical = catalog.loc[
            (catalog["marca"].map(canonical_text) == canonical_text(row["marca"]))
            & (catalog["ano_fabricacion"] + 3 <= year)
            & pd.to_numeric(catalog.get("indice_fiabilidad_100", pd.Series(np.nan, index=catalog.index)), errors="coerce").notna()
        ].copy()
        if self._artifact is not None and self._artifact.metadata.get("target_normalizer") and not historical.empty:
            from .modeling import TemporalTargetNormalizer
            historical["indice_fiabilidad_100"] = TemporalTargetNormalizer.from_dict(
                self._artifact.metadata["target_normalizer"]).transform(historical)
        history: list[dict[str, Any]] = []
        if not historical.empty:
            series = historical.groupby("ano_fabricacion", as_index=False)["indice_fiabilidad_100"].mean()
            history = [
                {
                    "ano_fabricacion": int(item["ano_fabricacion"]),
                    "indice_fiabilidad_100": float(item["indice_fiabilidad_100"]),
                }
                for _, item in series.iterrows()
            ]
        return {
            "hist_fiabilidad_marca": _finite_or_none(row.get("hist_fiabilidad_marca")),
            "origen_hist_fiabilidad_marca": row.get("origen_hist_fiabilidad_marca"),
            "vehicle_features": vehicle_features,
            "segment_features": segment_features,
            "feature_ranges": feature_ranges,
            "brand_history": history,
        }

    def _baseline_prediction(
        self, row: pd.Series, context: dict[str, Any], *, reason: str
    ) -> dict[str, Any]:
        gold = self._load_gold()
        year = int(row["ano_fabricacion"])
        if gold.empty:
            raise InsufficientEvidenceError("Sin datos suficientes para calcular un índice. Puede consultar recalls oficiales en la pestaña NHTSA.")
        historical = gold.loc[gold["ano_fabricacion"] + 3 <= year].copy()
        if not historical.empty:
            from .transform import fit_reliability_normalizer
            historical = fit_reliability_normalizer(historical, train_end_year=year - 3).transform(historical)
        same_brand = historical.loc[
            historical["marca"].map(canonical_text).eq(canonical_text(row["marca"]))
        ]
        same_segment = same_brand.loc[
            same_brand["categoria_vehiculo"].map(canonical_text).eq(canonical_text(row.get("categoria_vehiculo")))
        ]
        pool = next((pool for pool in (same_segment, same_brand, historical) if not pool.empty), pd.DataFrame())
        if pool.empty:
            raise InsufficientEvidenceError("No hay cohortes maduras anteriores al lanzamiento para un baseline defendible.")
        else:
            values = pd.to_numeric(pool["indice_fiabilidad_100"], errors="coerce").dropna()
            if values.empty:
                raise InsufficientEvidenceError("Las cohortes anteriores no contienen etiquetas válidas.")
            score = float(values.mean())
            source = (
                "historial de marca y segmento" if pool is same_segment
                else "historial de la marca" if pool is same_brand
                else "historial del mercado disponible, no específico de esta marca,"
            )
        return {
            "id_vehiculo_ano": row.get("id_vehiculo_ano"),
            "marca": str(row["marca"]),
            "modelo": str(row["modelo"]),
            "ano_fabricacion": year,
            "categoria_vehiculo": row.get("categoria_vehiculo", "sin_categoria"),
            "prediccion_indice_100": float(np.clip(score, 0, 100)),
            "mae": None,  # Dispersion around a fitted mean is not held-out MAE.
            "modelo_usado": "Baseline histórico",
            "es_baseline": True,
            "fallback": True,
            "fuente_demo": self._demo_mode,
            "mensaje": f"{reason} Se usa {source} anterior al lanzamiento.",
            "explicacion_factores": "La referencia utiliza únicamente cohortes anteriores y no incorpora recalls del vehículo consultado.",
            "factores": [
                {
                    "feature": "marca_y_categoria",
                    "contribution": 0.0,
                    "direction": "media_historica",
                }
            ],
            "fecha_ejecucion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **context,
        }

    def real_recalls_catalog(self) -> pd.DataFrame:
        """Load the full official index, not only vehicles previously queried."""
        from .bulk_recalls import BulkRecallStore
        store = BulkRecallStore(self.project_paths)
        if store.path.exists():
            return store.catalog()
        # Compatibility for deployments that have not yet downloaded the bulk
        # source. The UI labels this cache as incomplete, not a full catalogue.
        rows = []
        for path in (self.project_paths.processed_dir / "live_recalls").glob("*.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return pd.DataFrame(rows, columns=["marca", "modelo", "ano_fabricacion"])

    def real_recalls(self, marca: str, modelo: str, year: int) -> dict[str, Any]:
        """Fetch official evidence on demand, isolated from all ML/demo labels.

        Daily snapshot folders preserve raw evidence and allow later refreshes
        without overwriting an earlier response. This is not a VIN lookup.
        """
        from .data_sources import DataSourceError, NHTSARecallClient, normalise_nhtsa_response
        from .storage import atomic_json
        year = int(year)
        if not marca.strip() or not modelo.strip() or max(len(marca), len(modelo)) > 120:
            raise ValueError("Indique marca y modelo válidos (máximo 120 caracteres).")
        if not 1995 <= year <= datetime.now(timezone.utc).year + 1:
            raise ValueError("El año debe estar entre 1995 y el próximo año-modelo.")
        day = datetime.now(timezone.utc).date().isoformat()
        client = NHTSARecallClient(self.project_paths.raw_dir / "nhtsa" / day,
                                  max_retries=1, timeout_seconds=20)
        source = "https://api.nhtsa.gov/recalls/recallsByVehicle"
        fallback_notice = ""
        try:
            result = client.fetch_vehicle(marca.strip(), modelo.strip(), year)
            payload = result.payload
            fetched_at = datetime.fromtimestamp(result.cache_path.stat().st_mtime, tz=timezone.utc).isoformat()
            from_cache = result.from_cache
        except DataSourceError:
            from .bulk_recalls import BASE_URL, BulkRecallStore
            store = BulkRecallStore(self.project_paths)
            payload = store.payload(marca, modelo, year) if store.path.exists() else None
            if payload is None:
                raise
            fetched_at = store.metadata()["snapshot_date"]
            from_cache = True
            source = BASE_URL
            fallback_notice = " API no disponible: se muestran registros del snapshot masivo oficial, no una respuesta actualizada de la API."
        identifier = vehicle_id(marca, modelo, year)
        records = normalise_nhtsa_response(payload, make=marca.strip(), model=modelo.strip(),
                                           year=year, nhtsa_vehicle_id=identifier)
        records = records.drop_duplicates("campana_nhtsa")
        safe_records = json.loads(records.to_json(orient="records", date_format="iso"))
        response = {"marca": marca.strip(), "modelo": modelo.strip(), "ano_fabricacion": year,
                    "records": safe_records, "count": len(safe_records), "fetched_at": fetched_at,
                    "from_cache": from_cache, "source": source,
                    "fuente_demo": False,
                    "notice": "Consulta por modelo-año, no por VIN. Cero resultados no acredita ausencia de recalls: compruebe la denominación y consulte su VIN en NHTSA." + fallback_notice}
        directory = self.project_paths.processed_dir / "live_recalls"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identifier}_{day}.json"
        # Derived views can be rebuilt after a parser correction. Raw evidence
        # above is immutable and remains the authoritative original response.
        atomic_json(path, response)
        return response


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    """Interpret nullable dataframe flags without triggering pandas NA errors."""

    if value is None or pd.isna(value):
        return False
    return bool(value)


def _finite_or_default(value: Any, default: float) -> float:
    return _finite_or_none(value) if _finite_or_none(value) is not None else default


def _human_model_name(value: str) -> str:
    return {
        "ridge": "Regresión Ridge",
        "random_forest": "Random Forest",
        "baseline": "Baseline histórico",
    }.get(value, str(value))


__all__ = ["InsufficientEvidenceError", "ReliabilityService", "ServiceStatus", "VehicleNotFoundError"]
