"""Middleware de contexto de requisição: request-id, tempo e access log.

O log registra apenas metadados (método, rota, status, duração e id).
O texto dos laudos nunca é logado — são dados pessoais sensíveis (LGPD).
"""

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

logger = logging.getLogger("triage.access")

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

CallNext = Callable[[Request], Awaitable[Response]]


def resolve_request_id(incoming: str | None) -> str:
    """Reaproveita um id válido recebido do cliente ou gera um novo."""
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


async def request_context(request: Request, call_next: CallNext) -> Response:
    """Propaga o request-id, mede a duração e registra o access log."""
    request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.3f}"
    logger.info(
        "%s %s %s %.3fms request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response
