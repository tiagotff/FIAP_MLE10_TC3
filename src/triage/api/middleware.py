"""Middleware de contexto de requisição: request-id, tempo, métricas e log.

O log registra apenas metadados (método, rota, status, duração e id).
O texto dos laudos nunca é logado — são dados pessoais sensíveis (LGPD).

As métricas HTTP usam o *template* da rota (`/predict`), nunca a URL bruta,
para manter a cardinalidade sob controle; caminhos inexistentes viram
`unmatched`. O próprio `/metrics` fica de fora para a coleta do Prometheus
não poluir as séries de tráfego.
"""

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from triage.monitoring.metrics import ApiMetrics

logger = logging.getLogger("triage.access")

REQUEST_ID_HEADER = "X-Request-ID"
UNMATCHED_ROUTE = "unmatched"
UNTRACKED_ROUTES = frozenset({"/metrics"})
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

CallNext = Callable[[Request], Awaitable[Response]]


def resolve_request_id(incoming: str | None) -> str:
    """Reaproveita um id válido recebido do cliente ou gera um novo."""
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


def route_template(request: Request) -> str:
    """Template da rota atendida (ex.: `/predict`) ou `unmatched`."""
    route = request.scope.get("route")
    return getattr(route, "path", UNMATCHED_ROUTE)


def tracked_metrics(request: Request) -> ApiMetrics | None:
    """Métricas da aplicação, ou None se a rota não deve ser contabilizada."""
    if request.url.path in UNTRACKED_ROUTES:
        return None
    return getattr(request.app.state, "metrics", None)


async def request_context(request: Request, call_next: CallNext) -> Response:
    """Propaga o request-id, mede a duração, registra métricas e access log."""
    request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    metrics = tracked_metrics(request)
    if metrics is not None:
        metrics.in_progress.inc()
    start = time.perf_counter()
    status = 500  # se call_next lançar exceção, o handler responde com 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        if metrics is not None:
            metrics.in_progress.dec()
            metrics.observe_request(
                request.method, route_template(request), status, elapsed
            )
    elapsed_ms = elapsed * 1000
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
