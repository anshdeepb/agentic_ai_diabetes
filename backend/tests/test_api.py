from fastapi.testclient import TestClient

from main import app
from pipeline.schemas import Intent


def test_healthz():
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_post_message_returns_typed_result(fake_runner):
    fake_runner.queue(Intent.risk_only, "0")
    with TestClient(app) as client:
        res = client.post("/v1/sessions/api-test-session/messages", json={"query": "patient 0 risk?"})
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == "api-test-session"
    assert body["result"]["type"] == "risk_assessment"


def test_rejects_empty_and_oversized_queries():
    with TestClient(app) as client:
        assert client.post("/v1/sessions/api-test-session/messages", json={"query": ""}).status_code == 422
        too_long = "x" * 5000
        assert client.post("/v1/sessions/api-test-session/messages", json={"query": too_long}).status_code == 422


def test_rejects_malformed_session_id():
    with TestClient(app) as client:
        res = client.post("/v1/sessions/bad id!/messages", json={"query": "hi"})
    assert res.status_code == 422


def test_delete_session(fake_runner):
    with TestClient(app) as client:
        assert client.delete("/v1/sessions/api-test-session").status_code == 204
