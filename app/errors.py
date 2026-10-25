"""Single, predictable error envelope for every failure mode.

Every error response looks like::

    {"error": {"code": "unauthorized", "message": "...", "request_id": "req_..."}}

FastAPI's default 422 body (a bare list) is normalised into the same shape so the
integrating developer only ever parses one structure.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger("drcip.errors")

CODE_BY_STATUS: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "bad_request",
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_403_FORBIDDEN: "forbidden",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_413_CONTENT_TOO_LARGE: "payload_too_large",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "validation_error",
    status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "internal_error",
    status.HTTP_502_BAD_GATEWAY: "upstream_error",
    status.HTTP_504_GATEWAY_TIMEOUT: "upstream_timeout",
}

DEFAULT_MESSAGES: dict[str, str] = {
    "bad_request": "The request could not be understood.",
    "unauthorized": "Missing or invalid API key.",
    "forbidden": "This action is not permitted.",
    "not_found": "The requested resource does not exist.",
    "method_not_allowed": "This HTTP method is not supported for this endpoint.",
    "payload_too_large": "The request body is too large.",
    "validation_error": "The request did not pass validation.",
    "rate_limited": "Too many requests. Please retry later.",
    "internal_error": "An unexpected error occurred.",
    "upstream_error": "A downstream service failed. Please retry shortly.",
    "upstream_timeout": "A downstream service timed out. Please retry shortly.",
}


class ApiError(Exception):
    """Raised by application code to produce a controlled error response."""

    def __init__(
        self,
        status_code: int,
        code: str | None = None,
        message: str | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code or CODE_BY_STATUS.get(status_code, "error")
        self.message = message or DEFAULT_MESSAGES.get(self.code, "Request failed.")
        self.headers = headers or {}
        super().__init__(self.message)


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    merged = {REQUEST_ID_HEADER: request_id}
    if headers:
        merged.update(headers)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers=merged,
    )


def _format_validation_error(exc: RequestValidationError) -> str:
    """Compact, value-free description of the first validation problem."""
    details: list[Any] = exc.errors() or []
    if not details:
        return DEFAULT_MESSAGES["validation_error"]
    first = details[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    reason = first.get("msg", "invalid value")
    parts = [f"{location}: {reason}" if location else reason]
    if len(details) > 1:
        parts.append(f"(and {len(details) - 1} more)")
    return " ".join(parts)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(
            exc.status_code,
            exc.code,
            exc.message,
            get_request_id(request),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            _format_validation_error(exc),
            get_request_id(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = CODE_BY_STATUS.get(exc.status_code, "error")
        detail = exc.detail if isinstance(exc.detail, str) else None
        message = detail or DEFAULT_MESSAGES.get(code, "Request failed.")
        return error_response(
            exc.status_code,
            code,
            message,
            get_request_id(request),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = get_request_id(request)
        logger.exception(
            "request_id=%s unhandled_error=%s path=%s",
            request_id,
            type(exc).__name__,
            request.url.path,
        )
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            DEFAULT_MESSAGES["internal_error"],
            request_id,
        )
