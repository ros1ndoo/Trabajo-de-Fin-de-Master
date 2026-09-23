"""Shared cooldown survives client/snapshot changes without network retries."""

import sqlite3
from contextlib import closing

import pytest
from test_api_resilience import Response, Session

from auto_reliability.data_sources import DataSourceError, NHTSARecallClient, NHTSARequestError
from auto_reliability.request_state import RequestState


def test_cooldown_shared_across_queries_and_daily_snapshots(tmp_path):
    first = NHTSARecallClient(tmp_path / "2026-09-19", session=Session([Response(429, {"Retry-After": "120"})]),
                              clock=lambda: 1000, request_delay_seconds=0)
    with pytest.raises(NHTSARequestError):
        first.fetch_vehicle("ford", "focus", 2017)
    session = Session([Response()])
    second = NHTSARecallClient(tmp_path / "2026-09-20", session=session, clock=lambda: 1050, request_delay_seconds=0)
    with pytest.raises(NHTSARequestError) as caught:
        second.fetch_vehicle("audi", "a3", 2017)
    assert caught.value.attempts == 0
    assert caught.value.retry_after == 70
    assert session.calls == 0
    second.clock = lambda: 1120
    second.fetch_vehicle("ford", "focus", 2017)
    assert session.calls == 1
    assert second.request_state.failures() == []
    with closing(sqlite3.connect(second.request_state.path)) as db:
        assert db.execute("SELECT attempts FROM requests").fetchone()[0] == 2


def test_cached_evidence_remains_available_during_cooldown(tmp_path):
    session = Session([Response(), Response(503, {"Retry-After": "90"})])
    client = NHTSARecallClient(tmp_path / "day", session=session, clock=lambda: 1000, request_delay_seconds=0)
    client.fetch_vehicle("ford", "focus", 2017)
    with pytest.raises(NHTSARequestError):
        client.fetch_vehicle("audi", "a3", 2017)
    assert client.fetch_vehicle("ford", "focus", 2017).from_cache
    assert session.calls == 2


def test_permanent_failure_does_not_cooldown_other_vehicles(tmp_path):
    client = NHTSARecallClient(tmp_path / "day", session=Session([Response(400), Response()]), request_delay_seconds=0)
    with pytest.raises(NHTSARequestError):
        client.fetch_vehicle("ford", "unknown", 2017)
    client.fetch_vehicle("audi", "a3", 2017)
    assert client.request_state.failures()[0]["state"] == "permanent_failure"


def test_corrupt_operational_database_fails_closed(tmp_path):
    # Directory instead of a database simulates inaccessible operational storage.
    (tmp_path / ".nhtsa-operations.sqlite").mkdir()
    session = Session([])
    client = NHTSARecallClient(tmp_path / "2026-09-19", session=session, request_delay_seconds=0)
    with pytest.raises(DataSourceError, match="coordination"):
        client.fetch_vehicle("ford", "focus", 2017)
    assert session.calls == 0


def test_hosts_are_independent_and_cooldowns_never_shorten(tmp_path):
    state = RequestState(tmp_path / "state.sqlite")
    for delay in [120, 20]:
        state.record("https://api.nhtsa.gov/x", {}, now=1000, state="retryable", status=429, attempts=1, delay=delay)
    assert state.remaining("https://api.nhtsa.gov/y", 1050) == 70
    assert state.remaining("https://vpic.nhtsa.dot.gov/x", 1050) == 0
