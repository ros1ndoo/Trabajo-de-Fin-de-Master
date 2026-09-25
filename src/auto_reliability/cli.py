"""Command-line entry points for reproducible pipeline operations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .acquisition import download_cooperunion
from .config import ProjectPaths
from .demo_data import write_demo_catalog
from .eda import eda_summary, generate_eda_figures
from .modeling import train_and_select_model
from .pipeline import refresh_inference_catalog, run_pipeline
from .service import ReliabilityService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-reliability",
        description="Pipeline y predictor de propensión a recalls NHTSA.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("download-data", help="Descargar el CSV original CooperUnion desde Kaggle y registrar procedencia.")
    subcommands.add_parser("download-inventory", help="Descargar inventario EPA/DOE independiente; no modifica Gold ni el modelo.")
    resolution = subcommands.add_parser("resolve-inventory", help="Auditar nombres EPA/NHTSA sin crear etiquetas ni predicciones.")
    resolution.add_argument("--snapshot-date", required=True, help="Fecha ISO del snapshot EPA local.")
    test_cars = subcommands.add_parser("download-test-cars", help="Adquirir evidencia de potencia EPA; no modifica el predictor.")
    test_cars.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2027)))
    subcommands.add_parser("download-recalls", help="Descargar e indexar ambos ficheros masivos oficiales NHTSA, sin recorte de marcas.")
    catalog = subcommands.add_parser("refresh-catalog", help="Cargar todas las marcas y modelos técnicos, independientemente de la muestra NHTSA.")
    catalog.add_argument("--technical-csv", help="CSV técnico; por defecto el original descargado en data/raw.")

    demo = subcommands.add_parser("bootstrap-demo", help="Crear datos sintéticos claramente etiquetados y entrenar el demo.")
    demo.add_argument("--no-train", action="store_true", help="Crear solo los parquet de demostración.")

    pipeline = subcommands.add_parser("pipeline", help="Ejecutar Raw -> Processed -> Gold con CSV CooperUnion y NHTSA.")
    pipeline.add_argument("--technical-csv", required=True, help="Ruta al CSV Car Features and MSRP de CooperUnion.")
    pipeline.add_argument("--as-of-date", help="Fecha de corte ISO (YYYY-MM-DD) para cohortes completas.")
    pipeline.add_argument("--train-end-year", type=int, default=2018)
    pipeline.add_argument("--max-vehicles", type=int, help="Límite seguro para una prueba corta; omitir para ejecución completa.")
    pipeline.add_argument("--manual-decisions-csv", help="CSV opcional de decisiones de fuzzy matching.")
    pipeline.add_argument("--raw-filename", default="cooperunion_car_features.csv")
    pipeline.add_argument("--request-delay", type=float, default=0.20)
    pipeline.add_argument("--makes", nargs="+", help="Marcas para una ingesta acotada (ej. ford toyota honda).")
    pipeline.add_argument("--model-years", nargs="+", type=int, help="Años-modelo específicos; omitir para todos los años del CSV.")
    pipeline.add_argument("--snapshot-date", help="Carpeta de snapshot NHTSA ISO; por defecto la fecha UTC actual.")
    pipeline.add_argument("--recall-source", choices=("bulk", "api"), default="bulk",
                          help="Fuente de recalls; bulk utiliza el índice de download-recalls sin miles de peticiones.")

    recalls = subcommands.add_parser("recalls", help="Consultar recalls oficiales sin mezclar datos de entrenamiento.")
    recalls.add_argument("--make", required=True)
    recalls.add_argument("--model", required=True)
    recalls.add_argument("--year", type=int, required=True)

    train = subcommands.add_parser("train", help="Entrenar y evaluar el modelo con Gold existente.")
    train.add_argument("--gold-path", help="Parquet Gold alternativo.")
    train.add_argument("--train-end-year", type=int, default=2018)
    train.add_argument("--validation-end-year", type=int, default=2021)

    eda = subcommands.add_parser("eda", help="Generar resumen y figuras EDA desde Gold.")
    eda.add_argument("--gold-path", help="Parquet Gold alternativo.")
    eda.add_argument("--output-dir", help="Carpeta destino; por defecto reports/eda.")

    subcommands.add_parser("status", help="Mostrar el estado de artefactos sin crear ni entrenar datos.")
    subcommands.add_parser("audit-science", help="Auditar sensibilidad del proxy y selección sin cambiar el predictor.")
    exclusions = subcommands.add_parser("review-exclusions", help="Contrastar excluidos con EPA y consultar casos sin crear etiquetas.")
    exclusions.add_argument("--snapshot-date", required=True)
    exclusions.add_argument("--query-limit", type=int, default=0, help="Consultas pendientes por ejecución; 0 solo audita identidades.")
    subcommands.add_parser("publish-release", help="Validar y activar un paquete inmutable para nuevas sesiones web.")
    subcommands.add_parser("request-status", help="Mostrar fallos NHTSA pendientes; no ejecuta reintentos.")
    subcommands.add_parser("evaluate-temporal", help="Ejecutar ventanas retrospectivas aisladas; no publica modelos en la web.")
    audit = subcommands.add_parser("audit-model", help="Auditar errores por grupos sin reentrenar ni cambiar la web.")
    audit.add_argument("--minimum-group-size", type=int, default=30)
    subcommands.add_parser("dashboard", help="Iniciar Streamlit con el dashboard.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.discover()
    if args.command == "review-exclusions":
        from .exclusion_review import review_exclusions
        print(review_exclusions(paths, snapshot_date=args.snapshot_date, query_limit=args.query_limit))
        return 0
    if args.command == "audit-science":
        from .scientific_audit import run_scientific_audit
        print(run_scientific_audit(paths))
        return 0
    if args.command == "publish-release":
        from .releases import publish_release
        print(publish_release(paths))
        return 0
    if args.command == "evaluate-temporal":
        from .temporal_experiments import run_temporal_experiments
        print(run_temporal_experiments(paths))
        return 0
    if args.command == "request-status":
        from .request_state import RequestState
        state_path = paths.raw_dir / "nhtsa" / ".nhtsa-operations.sqlite"
        print(json.dumps(RequestState(state_path).failures() if state_path.exists() else [], ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-model":
        from .model_audit import publish_model_audit
        print(publish_model_audit(paths, minimum_group_size=args.minimum_group_size))
        return 0
    if args.command == "download-test-cars":
        from .epa_test_cars import acquire_test_cars
        print(json.dumps(acquire_test_cars(paths, args.years), ensure_ascii=False, indent=2))
        return 0
    if args.command == "resolve-inventory":
        from .inventory_resolution import publish_resolution
        summary = publish_resolution(paths, args.snapshot_date)
        print(json.dumps({k: v for k, v in summary.items() if k not in {"by_make", "by_year"}}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "download-inventory":
        from .fuel_economy import acquire_inventory
        print(json.dumps(acquire_inventory(paths), ensure_ascii=False, indent=2))
        return 0
    if args.command == "download-recalls":
        from .bulk_recalls import build_bulk_index, download_snapshot
        print(json.dumps(build_bulk_index(paths, download_snapshot(paths)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "download-data":
        print(download_cooperunion(paths.raw_dir))
        return 0
    if args.command == "refresh-catalog":
        catalog = refresh_inference_catalog(args.technical_csv or paths.raw_dir / "cooperunion_car_features.csv", paths=paths)
        print(f"Catálogo completo: {len(catalog)} modelos-año, {catalog.marca.nunique()} marcas.")
        return 0
    if args.command == "recalls":
        response = ReliabilityService(paths, auto_bootstrap_demo=False).real_recalls(args.make, args.model, args.year)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    if args.command == "bootstrap-demo":
        gold, catalog = write_demo_catalog(paths)
        if not args.no_train:
            training = train_and_select_model(gold, artifact_dir=paths.artifacts_dir, persist=True)
            print(f"Demo listo: {len(gold)} cohortes Gold, {len(catalog)} vehículos de catálogo.")
            print(f"Modelo seleccionado: {training.artifact.model_name}")
        else:
            print(f"Catálogo demo creado: {len(gold)} cohortes Gold, {len(catalog)} vehículos.")
        print("AVISO: el demo es sintético y no debe interpretarse como evidencia NHTSA real.")
        return 0

    if args.command == "pipeline":
        result = run_pipeline(
            args.technical_csv,
            paths=paths,
            as_of_date=args.as_of_date,
            train_end_year=args.train_end_year,
            max_vehicles=args.max_vehicles,
            manual_decisions_csv=args.manual_decisions_csv,
            raw_filename=args.raw_filename,
            request_delay_seconds=args.request_delay,
            makes=args.makes,
            model_years=args.model_years,
            snapshot_date=args.snapshot_date,
            recall_source=args.recall_source,
        )
        print(f"Gold creado: {result.paths.gold_path} ({len(result.gold)} filas)")
        print(f"Catálogo de inferencia: {result.paths.inference_catalog_path} ({len(result.inference_catalog)} filas)")
        print(f"Auditoría fuzzy: {result.paths.audit_path}")
        print(f"Manifest: {result.manifest_path}")
        for warning in result.warnings:
            print(f"AVISO: {warning}")
        return 0

    if args.command == "train":
        gold_path = Path(args.gold_path) if args.gold_path else paths.gold_path
        if not gold_path.is_file():
            raise FileNotFoundError(f"No existe Gold: {gold_path}. Ejecute pipeline o bootstrap-demo primero.")
        result = train_and_select_model(
            pd.read_parquet(gold_path),
            artifact_dir=paths.artifacts_dir,
            persist=True,
            train_end_year=args.train_end_year,
            validation_end_year=args.validation_end_year,
        )
        print(f"Modelo seleccionado: {result.artifact.model_name}")
        print(json.dumps(result.metrics["final_test"], ensure_ascii=False))
        return 0

    if args.command == "eda":
        gold_path = Path(args.gold_path) if args.gold_path else paths.gold_path
        if not gold_path.is_file():
            raise FileNotFoundError(f"No existe Gold: {gold_path}.")
        gold = pd.read_parquet(gold_path)
        output_dir = Path(args.output_dir) if args.output_dir else paths.reports_dir / "eda"
        generated = generate_eda_figures(gold, output_dir)
        print(json.dumps(eda_summary(gold), ensure_ascii=False, indent=2))
        for figure in generated:
            print(figure)
        return 0

    if args.command == "status":
        state = ReliabilityService(paths, auto_bootstrap_demo=False).status()
        print(json.dumps(state.__dict__, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dashboard":
        app_path = paths.root / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.address=127.0.0.1", "--browser.gatherUsageStats=false"])
    raise AssertionError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
