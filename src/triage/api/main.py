"""Aplicação FastAPI de triagem de laudos.

Execução local:
    uvicorn triage.api.main:app --port 8000

O modelo é carregado na inicialização (lifespan). Se o artefato não puder
ser carregado, a aplicação falha ao subir (fail-fast) em vez de aceitar
tráfego sem modelo.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from triage import __version__
from triage.api.errors import register_error_handlers
from triage.api.logging_config import configure_logging
from triage.api.middleware import request_context
from triage.api.routes import router
from triage.config.settings import get_settings
from triage.inference.loader import load_predictor
from triage.inference.predictor import UrgencyPredictor

logger = logging.getLogger(__name__)

PredictorFactory = Callable[[], UrgencyPredictor | None]


def default_predictor_factory() -> UrgencyPredictor:
    """Carrega o preditor definido pelas configurações de ambiente."""
    settings = get_settings()
    return load_predictor(settings, settings.inference_backend)


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

    configure_logging(get_settings().log_level)
    app = FastAPI(
        title="Triagem de Laudos Médicos",
        description="Classifica a urgência de laudos em normal, atencao ou urgente.",
        version=__version__,
        lifespan=lifespan,
    )
    app.middleware("http")(request_context)
    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
