"""Integrate a validated local candidate into the main project, retaining backups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from filelock import FileLock

from auto_reliability.config import ProjectPaths
from auto_reliability.releases import FILES, publish_release, serving_paths
from auto_reliability.storage import atomic_destination, atomic_json

DERIVED = (
    "data/processed/auto_reliability.sqlite", "data/processed/pipeline_manifest.json",
    "data/processed/target_normalizer.json", "data/processed/audit_fuzzy_matches.csv",
    "data/processed/nhtsa_vehicle_catalog.parquet",
    "artifacts/test_predictions.csv", "artifacts/validation_predictions.csv",
    "artifacts/feature_attributions.json",
)


def promote(root: Path, candidate: Path) -> str:
    """Activate last, rolling back loose files if validation fails.

    Input is a trusted locally generated candidate, never an external pickle.
    README, docs, raw data and source files are outside this allowlist.
    """
    root, candidate = root.resolve(), candidate.resolve()
    if (root / "output") not in candidate.parents:
        raise ValueError("Candidate must be inside this project's output directory")
    paths = ProjectPaths(root)
    identifier = publish_release(ProjectPaths(candidate))
    validated = serving_paths(ProjectPaths(candidate))
    names = [*FILES, *(name for name in DERIVED if (candidate / name).is_file())]
    contents = {name: ((validated.root if name in FILES else candidate) / name).read_bytes() for name in names}
    with FileLock(str(root / "releases/.promotion.lock"), timeout=30):
        previous = {name: (root / name).read_bytes() if (root / name).exists() else None for name in names}
        digest = hashlib.sha256(b"".join(value or b"" for value in previous.values())).hexdigest()
        backup = root / "reports/promotion_backups" / digest
        for name, value in previous.items():
            if value is not None:
                with atomic_destination(backup / name, immutable=True) as temporary:
                    temporary.write_bytes(value)
        try:
            for name, value in contents.items():
                with atomic_destination(root / name) as temporary:
                    temporary.write_bytes(value)
            result = publish_release(paths)
        except Exception:
            for name, value in previous.items():
                if value is None:
                    (root / name).unlink(missing_ok=True)  # Only this operation's new allowlisted files.
                else:
                    with atomic_destination(root / name) as temporary:
                        temporary.write_bytes(value)
            raise
        atomic_json(root / "reports/promotion_latest.json", {
            "candidate_release": identifier, "active_release": result, "backup": str(backup),
            "files": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"active_release": promote(ProjectPaths.discover().root, args.candidate)}))
