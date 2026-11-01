"""Tratamento centralizado de erros.

Nenhuma resposta de erro ecoa o texto do laudo: dados de saúde são
sensíveis (LGPD) e respostas de erro costumam acabar em logs de proxies
e clientes.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from triage.api.middleware import REQUEST_ID_HEADER

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Erro interno ao processar a requisição."
VALIDATION_ERROR_MESSAGE = "Requisição inválida."


def _error_response(
    request: Request, status_code: int, content: dict[str, Any]
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content={**content, "request_id": request_id},
        headers=headers,
    )


def sanitize_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mantém apenas local, tipo e mensagem dos erros (sem o valor enviado)."""
    return [
        {
            "loc": list(err.get("loc", [])),
            "type": err.get("type"),
            "msg": err.get("msg"),
        }
        for err in errors
    ]


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Responde 422 com os erros de validação, sem ecoar o conteúdo enviado."""
    content = {
        "detail": VALIDATION_ERROR_MESSAGE,
        "errors": sanitize_errors(exc.errors()),
    }
    return _error_response(request, 422, content)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Responde 500 genérico sem expor detalhes internos."""
    logger.error(
        "Erro não tratado em %s %s request_id=%s: %s",
        request.method,
        request.url.path,
        getattr(request.state, "request_id", None),
        type(exc).__name__,
    )
    return _error_response(request, 500, {"detail": INTERNAL_ERROR_MESSAGE})


def register_error_handlers(app: FastAPI) -> None:
    """Registra os handlers de erro na aplicação."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
