"""Error-envelope tests: every failure must share one JSON shape."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def assert_envelope(response, expected_code: str) -> dict:
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == expected_code
    assert body["error"]["message"]
    assert body["error"]["request_id"].startswith("req_")
    assert response.headers["X-Request-ID"] == body["error"]["request_id"]
    return body


def test_unknown_route_returns_envelope(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert_envelope(response, "not_found")


def test_wrong_method_returns_envelope(client: TestClient) -> None:
    response = client.post("/health")

    assert response.status_code == 405
    assert_envelope(response, "method_not_allowed")


def test_validation_errors_are_normalised(settings: Settings) -> None:
    app = create_app(settings)

    @app.get("/_test/echo")
    async def echo(n: int) -> dict[str, int]:  # pragma: no cover - route under test
        return {"n": n}

    with TestClient(app) as test_client:
        response = test_client.get("/_test/echo", params={"n": "not-a-number"})

    assert response.status_code == 422
    body = assert_envelope(response, "validation_error")
    assert "n" in body["error"]["message"]
    assert "not-a-number" not in body["error"]["message"]


def test_unhandled_exception_returns_generic_envelope(settings: Settings) -> None:
    app = create_app(settings)

    @app.get("/_test/boom")
    async def boom() -> None:  # pragma: no cover - route under test
        raise RuntimeError("internal detail that must not leak")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/_test/boom")

    assert response.status_code == 500
    body = assert_envelope(response, "internal_error")
    assert "internal detail" not in body["error"]["message"]
    assert body["error"]["message"] == "An unexpected error occurred."


def test_cache_control_is_set_for_versioned_endpoints(settings: Settings) -> None:
    app = create_app(settings)

    @app.get("/v1/_test/ping")
    async def ping() -> dict[str, bool]:  # pragma: no cover - route under test
        return {"ok": True}

    with TestClient(app) as test_client:
        response = test_client.get("/v1/_test/ping")

    assert response.headers["Cache-Control"] == "no-store"


def test_openapi_documents_the_error_schema(settings: Settings) -> None:
    with TestClient(create_app(settings)) as test_client:
        schema = test_client.get("/openapi.json").json()

    components = schema["components"]["schemas"]
    assert "ErrorResponse" in components
    assert set(components["ErrorDetail"]["properties"]) == {"code", "message", "request_id"}


@pytest.mark.parametrize("path", ["/health", "/does-not-exist"])
def test_request_id_is_always_returned(client: TestClient, path: str) -> None:
    response = client.get(path, headers={"X-Request-ID": "trace-abc-12345"})

    assert response.headers["X-Request-ID"] == "trace-abc-12345"
