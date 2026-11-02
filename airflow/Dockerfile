# Imagem do Airflow com o ambiente de ML do projeto isolado.
#
# As constraints oficiais do Airflow fixam versões de scikit-learn/pandas
# diferentes das do projeto. Para não treinar com versões diferentes das
# que a API usa para carregar o modelo, as tasks de ML rodam num venv
# próprio (/opt/triage/.venv), instalado a partir do MESMO poetry.lock da
# API, via `@task.external_python` (ver dags/triage_retraining.py).
#
# Build (a partir da raiz do repositório):
#   docker compose -f airflow/docker-compose.yml build

ARG AIRFLOW_VERSION=3.3.2
ARG PYTHON_VERSION=3.12

# ---- Stage 1: venv do projeto (mesmo interpretador da imagem final) ----
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION} AS triage-env

USER root
RUN mkdir -p /opt/triage && chown airflow:0 /opt/triage
USER airflow

# A imagem do Airflow define PIP_USER=true; dentro de venvs isso precisa
# estar desligado.
ENV POETRY_VERSION=2.4.3 \
    POETRY_NO_INTERACTION=1 \
    PIP_USER=false \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /home/airflow/poetry \
    && /home/airflow/poetry/bin/pip install "poetry==${POETRY_VERSION}"

WORKDIR /opt/triage
COPY --chown=airflow:0 pyproject.toml poetry.lock README.md ./
COPY --chown=airflow:0 src ./src

# O venv é criado explicitamente e ativado (VIRTUAL_ENV) para o Poetry
# instalar nele. Sem isso, o Poetry detecta o ambiente Python do próprio
# Airflow como "ativo", instala as dependências lá e o /opt/triage/.venv
# nunca é criado. A última linha falha o build se o venv não funcionar.
RUN python -m venv /opt/triage/.venv \
    && VIRTUAL_ENV=/opt/triage/.venv PATH="/opt/triage/.venv/bin:${PATH}" \
       /home/airflow/poetry/bin/poetry install --only main,training \
    && /opt/triage/.venv/bin/python -c "import sklearn, triage.pipeline.steps; print('venv ok, scikit-learn', sklearn.__version__)"

# ---- Stage 2: Airflow + venv do projeto, sem Poetry ----
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

COPY --from=triage-env --chown=airflow:0 /opt/triage /opt/triage
COPY --chown=airflow:0 airflow/dags /opt/airflow/dags

ENV TRIAGE_PYTHON=/opt/triage/.venv/bin/python

# Garante que o venv copiado funciona na imagem final (falha o build se não).
RUN ${TRIAGE_PYTHON} -c "import sklearn, triage.pipeline.steps"
