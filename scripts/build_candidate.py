"""Rebuild and evaluate a local candidate without touching the serving release.

Run from the repository root: python scripts/build_candidate.py --destination output/candidate-20260925
Only existing official snapshot data are used; no aliases or zero labels invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from auto_reliability.config import ProjectPaths
from auto_reliability.modeling import train_and_select_model
from auto_reliability.pipeline import run_pipeline
from auto_reliability.releases import serving_paths
from auto_reliability.storage import atomic_json
from auto_reliability.temporal_experiments import run_temporal_experiments


def build(destination: Path) -> Path:
    """Create a fresh isolated workspace and record before/after coverage."""
    source = ProjectPaths.discover()
    active = serving_paths(source)
    destination = destination.resolve()
    output = (source.root / "output").resolve()
    if output not in destination.parents or destination.exists():
        raise ValueError("Destination must be a NEW directory beneath output/")
    candidate = ProjectPaths(destination)
    candidate.ensure_runtime_directories()
    database = source.processed_dir / "nhtsa_bulk.sqlite"
    original_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    shutil.copy2(database, candidate.processed_dir / database.name)
    if hashlib.sha256((candidate.processed_dir / database.name).read_bytes()).hexdigest() != original_hash:
        raise ValueError("Snapshot copy mismatch")
    metrics = json.loads((active.artifacts_dir / "model_metrics.json").read_text(encoding="utf-8"))
    split = metrics["split"]
    protocol = {"snapshot_sha256": original_hash, "as_of_date": "2026-09-18",
                "train_end": split["configured_train_end_year"],
                "validation_end": split["configured_validation_end_year"],
                "interpretation": "Previously examined population; exploratory comparison, not untouched test.",
                "manual_equivalences_applied": 0, "invented_zero_labels": 0}
    atomic_json(destination / "candidate_protocol.json", protocol, immutable=True)
    result = run_pipeline(source.raw_dir / "cooperunion_car_features.csv", paths=candidate,
                          as_of_date=protocol["as_of_date"], train_end_year=protocol["train_end"],
                          recall_source="bulk", request_delay_seconds=0)
    trained = train_and_select_model(result.gold, artifact_dir=candidate.artifacts_dir,
                                     train_end_year=protocol["train_end"],
                                     validation_end_year=protocol["validation_end"], persist=True)
    previous = pd.read_parquet(active.gold_path)
    added = sorted(set(result.gold.id_vehiculo_ano) - set(previous.id_vehiculo_ano))
    removed = sorted(set(previous.id_vehiculo_ano) - set(result.gold.id_vehiculo_ano))
    report = {"protocol": protocol, "previous_gold_rows": len(previous), "candidate_gold_rows": len(result.gold),
              "added_ids": added, "removed_ids": removed, "selected_model": trained.artifact.model_name,
              "previous_metrics": metrics,
              "candidate_metrics": json.loads((candidate.artifacts_dir / "model_metrics.json").read_text(encoding="utf-8")),
              "remaining_exclusions": result.matches.match_status.value_counts().to_dict(),
              "temporal_experiments": str(run_temporal_experiments(candidate)),
              "published": False}
    atomic_json(destination / "candidate_comparison.json", report, immutable=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    print(build(parser.parse_args().destination))
