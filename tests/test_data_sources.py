from __future__ import annotations

import json

import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.data_sources import NHTSARecallClient
from auto_reliability.pipeline import stage_technical_csv


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"Count": 1, "results": [{"NHTSACampaignNumber": "24V001"}]}


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return _Response()


def test_nhtsa_client_writes_raw_json_once_and_then_uses_cache(tmp_path) -> None:
    session = _Session()
    client = NHTSARecallClient(tmp_path, session=session, sleeper=lambda _: None)
    first = client.fetch_vehicle("Ford", "Explorer", 2020)
    second = client.fetch_vehicle("Ford", "Explorer", 2020)
    assert first.from_cache is False
    assert second.from_cache is True
    assert session.calls == 1
    assert json.loads(first.cache_path.read_text(encoding="utf-8"))["Count"] == 1


def test_staging_refuses_to_overwrite_different_raw_evidence(tmp_path) -> None:
    paths = ProjectPaths(tmp_path / "project")
    source_one = tmp_path / "one.csv"
    source_two = tmp_path / "two.csv"
    source_one.write_text("Make,Model,Year\nFord,Explorer,2020\n", encoding="utf-8")
    source_two.write_text("Make,Model,Year\nToyota,RAV4,2020\n", encoding="utf-8")
    staged = stage_technical_csv(source_one, paths)
    assert staged.exists()
    with pytest.raises(FileExistsError, match="immutable Raw"):
        stage_technical_csv(source_two, paths)
