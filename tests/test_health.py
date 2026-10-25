"""Health endpoint and service-metadata contract tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_returns_documented_shape(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "service", "version", "environment"}
    assert body["status"] == "ok"
    assert body["service"] == "drcip-rag-api"
    assert body["version"] == "1.0"
    assert body["environment"] == "test"
    assert response.headers["content-type"].startswith("application/json")


def test_health_needs_no_authentication(client: TestClient) -> None:
    """Uptime pingers and Render's health check cannot send X-API-Key."""
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers={"X-API-Key": "definitely-wrong"}).status_code == 200


def test_health_issues_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["X-Request-ID"].startswith("req_")
    assert len(response.headers["X-Request-ID"]) == 20
    assert response.headers["X-Response-Time-Ms"].isdigit()


def test_health_echoes_caller_supplied_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "client-trace-0001"})

    assert response.headers["X-Request-ID"] == "client-trace-0001"


def test_health_replaces_malformed_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id with spaces and $ymbols"})

    request_id = response.headers["X-Request-ID"]
    assert request_id != "bad id with spaces and $ymbols"
    assert request_id.startswith("req_")


def test_security_headers_are_present(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_docs_are_served_when_enabled(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/health" in openapi.json()["paths"]


def test_docs_can_be_disabled(settings: Settings) -> None:
    hidden = settings.model_copy(update={"docs_enabled": False})
    app = create_app(hidden)
    with TestClient(app) as test_client:
        docs = test_client.get("/docs")
        assert docs.status_code == 404
        assert test_client.get("/openapi.json").status_code == 404
        assert docs.json()["error"]["code"] == "not_found"


def test_root_redirects_to_docs_when_enabled(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/docs"


def test_root_returns_metadata_when_docs_disabled(settings: Settings) -> None:
    hidden = settings.model_copy(update={"docs_enabled": False})
    with TestClient(create_app(hidden)) as test_client:
        response = test_client.get("/")

    assert response.status_code == 200
    assert response.json()["endpoints"]["health"] == "/health"
