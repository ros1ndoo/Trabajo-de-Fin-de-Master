"""Prueba errores HTTP, Retry-After y concurrencia de caché con datos deterministas."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from auto_reliability.data_sources import NHTSARecallClient, NHTSARequestError, retry_after_seconds


class Response:
    def __init__(self, status=200, headers=None):
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("fixture error", response=self)

    def json(self):
        return {"Count": 0, "results": []}


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return next(self.responses)


def test_400_success_message_is_not_a_zero_and_actual_attempts_are_reported(tmp_path):
    session = Session([Response(400)])
    client = NHTSARecallClient(tmp_path, session=session, max_retries=3)
    with pytest.raises(NHTSARequestError) as caught:
        client.fetch_vehicle("Ford", "Explorer", 2017)
    assert caught.value.status == 400
    assert caught.value.attempts == 1
    assert not caught.value.transient
    assert not list(tmp_path.glob("*.json"))


def test_429_respects_retry_after_and_then_succeeds(tmp_path):
    delays = []
    client = NHTSARecallClient(tmp_path, session=Session([Response(429, {"Retry-After": "3"}), Response()]),
                               max_retries=1, request_delay_seconds=0, sleeper=delays.append)
    client.fetch_vehicle("Audi", "A3", 2017)
    assert len(delays) == 1 and 3 <= delays[0] <= 3.5


def test_long_retry_after_fails_recoverably_instead_of_retrying_early(tmp_path):
    delays = []
    client = NHTSARecallClient(tmp_path, session=Session([Response(429, {"Retry-After": "120"})]),
                               sleeper=delays.append, request_delay_seconds=0)
    with pytest.raises(NHTSARequestError) as caught:
        client.fetch_vehicle("Audi", "A3", 2017)
    assert caught.value.retry_after == 120
    assert caught.value.transient
    assert not delays


@pytest.mark.parametrize("value, expected", [("4", 4), ("-1", 0), ("NaN", None), ("bad", None), (None, None)])
def test_retry_after_parser(value, expected):
    assert retry_after_seconds(value) == expected


def test_simultaneous_sessions_fetch_once(tmp_path):
    session = Session([Response()])
    clients = [NHTSARecallClient(tmp_path, session=session, request_delay_seconds=0) for _ in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda client: client.fetch_vehicle("Audi", "A3", 2017), clients))
    assert session.calls == 1
    assert sum(not result.from_cache for result in results) == 1


def test_503_recovers_without_persisting_error_as_evidence(tmp_path):
    session = Session([Response(503), Response()])
    client = NHTSARecallClient(tmp_path, session=session, request_delay_seconds=0, sleeper=lambda _: None)
    result = client.fetch_vehicle("Audi", "A3", 2017)
    assert session.calls == 2
    assert result.payload["Count"] == 0


def test_timeout_is_typed_and_never_cached(tmp_path):
    class OfflineSession:
        def get(self, *_args, **_kwargs):
            raise requests.Timeout("fixture timeout")
    client = NHTSARecallClient(tmp_path, session=OfflineSession(), request_delay_seconds=0,
                               max_retries=1, sleeper=lambda _: None)
    with pytest.raises(NHTSARequestError) as caught:
        client.fetch_vehicle("Audi", "A3", 2017)
    assert caught.value.transient
    assert caught.value.attempts == 2
    assert not list(tmp_path.glob("*.json"))
