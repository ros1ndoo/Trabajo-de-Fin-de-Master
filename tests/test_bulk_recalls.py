"""Contratos locales de la fuente masiva oficial; no inventar ceros ante ausencias."""
from __future__ import annotations

import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from auto_reliability.bulk_recalls import (
    ARCHIVES,
    FIELDS,
    BulkRecallStore,
    build_bulk_index,
    read_archive,
)
from auto_reliability.config import ProjectPaths
from auto_reliability.data_sources import DataSourceError
from auto_reliability.storage import atomic_json


def _snapshot(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, name in enumerate(ARCHIVES):
        row = dict.fromkeys(FIELDS, "")
        row.update(record_id=str(i), campaign=f"{i}0V001", make="AUDI", model="A3", year="2017",
                   recall_type="V", component="ENGINE", summary="Engine test", received_date="20180131")
        path = directory / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name.replace(".zip", ".txt"), "\t".join(row.values()) + "\n")
        entries.append({"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    atomic_json(directory / "manifest.json", {"snapshot_date": "2026-09-18", "files": entries})
    return directory


def test_complete_snapshot_catalog_and_exact_evidence(tmp_path):
    paths = ProjectPaths(tmp_path)
    meta = build_bulk_index(paths, _snapshot(tmp_path / "raw"))
    assert meta["source_rows"] == 2
    store = BulkRecallStore(paths)
    assert store.catalog().marca.tolist() == ["AUDI"]
    payload = store.payload("audi", "a3", 2017)
    assert payload["Count"] == 2
    assert payload["results"][0]["ReportReceivedDate"] == "2018-01-31"
    assert store.payload("Audi", "Nonexistent", 2017) is None
    # La publicación también debe funcionar en Windows; todos los lectores deben cerrarse.
    build_bulk_index(paths, tmp_path / "raw")
    assert store.metadata()["vehicles"] == 1


def test_missing_archive_and_raw_tampering_fail_closed(tmp_path):
    source = _snapshot(tmp_path / "raw")
    path = source / ARCHIVES[0]
    path.write_bytes(b"changed")
    with pytest.raises(DataSourceError, match="hash mismatch"):
        build_bulk_index(ProjectPaths(tmp_path), source)
    assert not (tmp_path / "data/processed/nhtsa_bulk.sqlite").exists()


def test_schema_drift_does_not_silently_skip_records(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.txt", "one\ttwo\n")
    with pytest.raises(DataSourceError, match="expected 29 fields"):
        read_archive(archive)


def test_atomic_immutable_concurrent_publication_is_complete(tmp_path):
    destination = tmp_path / "raw.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: atomic_json(destination, {"n": n, "data": "x" * 10000}, immutable=True), range(8)))
    content = json.loads(destination.read_text(encoding="utf-8"))
    assert content["n"] in range(8)
    assert content["data"] == "x" * 10000
    assert len(list(tmp_path.iterdir())) == 1
