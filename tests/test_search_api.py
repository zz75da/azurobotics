"""Search API with fake search and embedding clients: no Azure, no model download."""
import json
import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("azure.search.documents")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from search.api import create_app  # noqa: E402

CHUNKS = [
    {"note_id": "hdf-01", "title": "Cooling system inspection", "failure_type": "HDF", "section": "When to use", "content": "8.6 K", "source": "a.md"},
    {"note_id": "hdf-01", "title": "Cooling system inspection", "failure_type": "HDF", "section": "Procedure", "content": "fans", "source": "a.md"},
    {"note_id": "hdf-02", "title": "Cleaning the heat exchanger", "failure_type": "HDF", "section": "Procedure", "content": "fins", "source": "b.md"},
    {"note_id": "twf-02", "title": "Replacing a worn cutting tool", "failure_type": "TWF", "section": "Procedure", "content": "tool", "source": "c.md"},
    {"note_id": "pwf-01", "title": "Checking the spindle drive power supply", "failure_type": "PWF", "section": "Procedure", "content": "power", "source": "d.md"},
]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]


class FakeSearch:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        wanted = kwargs["filter"].split("'")[1] if kwargs["filter"] else None
        return [dict(c, **{"@search.score": 1.0}) for c in CHUNKS if wanted in (None, c["failure_type"])]


@pytest.fixture
def api():
    fake = FakeSearch()
    return TestClient(create_app(fake, FakeEmbedder())), fake


def test_search_groups_chunks_by_note_and_logs_latency(api, caplog):
    client, fake = api
    with caplog.at_level(logging.INFO, logger="robotics-search"):
        response = client.get("/search", params={"q": "cooling fans", "k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["note_id"] for r in results] == ["hdf-01", "hdf-02"]
    assert len(results[0]["passages"]) == 2
    assert fake.calls[0]["vector_queries"][0].fields == "vector"
    record = json.loads(caplog.records[-1].getMessage())
    assert record["event"] == "search_query" and record["top_k"] == 2 and record["latency_ms"] >= 0
    assert "cooling" not in caplog.records[-1].getMessage()  # the query text is never logged


def test_search_filters_by_failure_type(api):
    client, fake = api
    response = client.get("/search", params={"q": "power", "failure_type": "PWF"})
    assert fake.calls[0]["filter"] == "failure_type eq 'PWF'"
    assert [r["failure_type"] for r in response.json()["results"]] == ["PWF"]


def test_search_rejects_unknown_failure_type_and_short_query(api):
    client, _ = api
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.get("/search", params={"q": "power", "failure_type": "ABC"}).status_code == 422


def test_procedures_for_a_heat_dissipation_reading(api):
    client, fake = api
    reading = {"product_type": "M", "air_temperature_k": 300.0, "process_temperature_k": 308.4,
               "rotational_speed_rpm": 1350, "torque_nm": 45.0, "tool_wear_min": 50}
    body = client.post("/procedures", json=reading).json()
    assert body["failure_types"] == ["HDF"]
    assert body["matches"][0]["kind"] == "documented failure condition met"
    assert {p["failure_type"] for p in body["matches"][0]["procedures"]} == {"HDF"}


def test_tool_wear_is_reported_as_a_risk_window_not_a_failure(api):
    client, _ = api
    reading = {"product_type": "H", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
               "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 215}
    body = client.post("/procedures", json=reading).json()
    assert body["failure_types"] == ["TWF"]
    assert body["matches"][0]["label"] == "tool-wear risk window (200-240 min)"
    assert body["matches"][0]["kind"] == "risk window"


def test_procedures_when_no_condition_is_met(api):
    client, _ = api
    reading = {"product_type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
               "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 50}
    body = client.post("/procedures", json=reading).json()
    assert body["failure_types"] == [] and body["matches"] == [] and body["message"]


def test_procedures_validates_the_reading(api):
    client, _ = api
    bad = {"product_type": "X", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
           "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 50}
    assert client.post("/procedures", json=bad).status_code == 422
