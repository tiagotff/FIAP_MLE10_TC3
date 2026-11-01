"""Aplicação FastAPI de triagem de laudos.

Execução local:
    uvicorn triage.api.main:app --port 8000
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from triage import __version__
from triage.api.routes import router
from triage.config.settings import get_settings
from triage.inference.loader import load_predictor
from triage.inference.predictor import UrgencyPredictor

logger = logging.getLogger(__name__)

PredictorFactory = Callable[[], UrgencyPredictor]


def default_predictor_factory() -> UrgencyPredictor:
    """Carrega o preditor definido pelas configurações de ambiente."""
    settings = get_settings()
    return load_predictor(settings, settings.inference_backend)


async def add_process_time_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Middleware que expõe o tempo total de processamento no header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
    return response


def create_app(
    predictor_factory: PredictorFactory = default_predictor_factory,
) -> FastAPI:
    """Cria a aplicação; o preditor é injetável para facilitar testes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor_factory()
        version = getattr(app.state.predictor, "model_version", "nenhum")
        logger.info("Modelo carregado: %s", version)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Triagem de Laudos Médicos",
        description="Classifica a urgência de laudos em normal, atencao ou urgente.",
        version=__version__,
        lifespan=lifespan,
    )
    app.middleware("http")(add_process_time_header)
    app.include_router(router)
    return app


app = create_app()
