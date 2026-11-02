# Imagem do serviço de inferência (FastAPI + modelo TF-IDF/LogReg).
#
# Multi-stage:
#   1. poetry-base -> Poetry + manifestos de dependências (camada cacheável)
#   2. trainer     -> baixa o corpus (com checksum), treina o modelo e o
#                     exporta para ONNX (com verificação de paridade)
#   3. api-deps    -> venv só com o necessário para servir (sem pandas)
#   4. runtime     -> imagem final enxuta, usuário não-root, sem Poetry
#
# O modelo é treinado no build para que `docker build` sozinho produza
# uma imagem funcional e reprodutível (seed e dados fixados). Um modelo
# retreinado fora do build pode ser montado em /app/models via volume.

ARG PYTHON_VERSION=3.12

# ---- Stage 1: base com Poetry ----
FROM python:${PYTHON_VERSION}-slim AS poetry-base

ENV POETRY_VERSION=2.4.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "poetry==${POETRY_VERSION}"
WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./

# ---- Stage 2: treino do modelo ----
FROM poetry-base AS trainer

RUN poetry install --only main,training --no-root
COPY configs ./configs
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"
RUN python -m triage.data.download \
    && python -m triage.training.train \
    && python -m triage.optimization.export_onnx

# ---- Stage 3: dependências da API ----
FROM poetry-base AS api-deps

RUN poetry install --only main,api --no-root

# ---- Stage 4: runtime ----
FROM python:${PYTHON_VERSION}-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app
COPY --from=api-deps /app/.venv ./.venv
COPY --from=trainer --chown=app:app /app/models ./models
COPY configs ./configs
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INFERENCE_BACKEND=onnx \
    PORT=8000

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/ready', timeout=2)"

# INFERENCE_BACKEND=onnx serve o modelo otimizado (ONNX Runtime); use
# INFERENCE_BACKEND=sklearn para o modelo original. $PORT segue a convenção
# do Cloud Run (padrão local: 8000). O access log
# do uvicorn é desligado porque a aplicação registra o próprio, com
# request-id e sem conteúdo dos laudos.
CMD ["sh", "-c", "exec uvicorn triage.api.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
