"""Observability helpers: request correlation ids, access logging, security headers."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-Ms"

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

access_logger = logging.getLogger("drcip.access")


def configure_logging(level: str = "INFO") -> None:
    """Configure plain, single-line logs (Render captures stdout/stderr)."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or new_request_id()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id, emit an access log line and set safety headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request.state.request_id = (
            incoming if _REQUEST_ID_PATTERN.match(incoming) else new_request_id()
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - started) * 1000)
            access_logger.exception(
                "request_id=%s method=%s path=%s status=500 duration_ms=%d",
                request.state.request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        response.headers[RESPONSE_TIME_HEADER] = str(duration_ms)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/v1/"):
            response.headers.setdefault("Cache-Control", "no-store")

        access_logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%d",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
