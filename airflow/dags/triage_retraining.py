"""Retreino do classificador de urgência de laudos.

Fluxo: download -> preparação (deduplicação + divisão 70/15/15) ->
calibração do limiar (validação cruzada) -> treino -> exportação para ONNX
(com verificação de paridade) -> avaliação no teste -> quality gate ->
publicação do modelo (joblib + ONNX).

As tasks de ML rodam com `@task.external_python` num ambiente virtual
próprio (`TRIAGE_PYTHON`), instalado a partir do mesmo `poetry.lock` da
API. Assim o modelo é treinado com exatamente as mesmas versões de
scikit-learn/numpy que vão carregá-lo em produção, sem conflitar com as
dependências do Airflow. Cada task chama uma etapa de
`triage.pipeline.steps`, e os artefatos trafegam por arquivos em
`artifacts/runs/<run_id>/`; pelo XCom passam apenas resumos.

Se o quality gate reprovar o modelo, a publicação não acontece e o modelo
em produção continua o mesmo.

Observação: o Airflow extrai o código-fonte de cada função `external_python`
para executá-lo no outro interpretador. Por isso os imports ficam dentro das
funções e o decorador é sempre escrito por extenso (`@task.external_python`).
"""

import os
from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task

TRIAGE_PYTHON = os.environ.get("TRIAGE_PYTHON", "/opt/triage/.venv/bin/python")

# O venv do projeto não tem Airflow nem pendulum: as funções não usam o
# contexto da task, apenas os argumentos explícitos.
EXTERNAL_OPTIONS = {"expect_airflow": False, "expect_pendulum": False}

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="triage_retraining",
    description="Retreino semanal do classificador de urgência de laudos",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["triage", "ml", "retreino"],
    doc_md=__doc__,
)
def triage_retraining() -> None:
    """Orquestra o pipeline de retreino, etapa por etapa."""

    @task
    def run_identifier() -> str:
        """Nome da execução (run_id do Airflow), usado para isolar os artefatos."""
        return get_current_context()["run_id"]

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def download_data() -> dict:
        """Baixa o corpus com verificação de SHA-256 (idempotente)."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import download_step

        logging.basicConfig(level="INFO")
        return download_step(get_settings())

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def prepare_data(run_name: str) -> dict:
        """Deduplica e divide o corpus em treino/validação/teste."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import prepare_step

        logging.basicConfig(level="INFO")
        return prepare_step(get_settings(), run_name)

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def calibrate_threshold(run_name: str) -> dict:
        """Calibra o limiar de `urgente` com predições out-of-fold."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import calibrate_step

        logging.basicConfig(level="INFO")
        return calibrate_step(get_settings(), run_name)

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def train_model(run_name: str) -> str:
        """Treina o modelo final em treino + validação."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import train_step

        logging.basicConfig(level="INFO")
        return train_step(get_settings(), run_name)

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def export_onnx(run_name: str) -> dict:
        """Exporta para ONNX e falha se o modelo não reproduzir o original."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import export_onnx_step

        logging.basicConfig(level="INFO")
        return export_onnx_step(get_settings(), run_name)

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def evaluate_model(run_name: str) -> dict:
        """Avalia o modelo no conjunto de teste."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import evaluate_step

        logging.basicConfig(level="INFO")
        return evaluate_step(get_settings(), run_name)

    # Sem retry: o gate é determinístico (mesmas métricas, mesmo resultado).
    # Repetir só atrasaria a falha de um modelo reprovado.
    @task.external_python(python=TRIAGE_PYTHON, retries=0, **EXTERNAL_OPTIONS)
    def quality_gate(run_name: str) -> dict:
        """Reprova (falha a task) se alguma métrica ficar abaixo do mínimo."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import quality_gate_step

        logging.basicConfig(level="INFO")
        return quality_gate_step(get_settings(), run_name)

    @task.external_python(python=TRIAGE_PYTHON, **EXTERNAL_OPTIONS)
    def publish_model(run_name: str) -> str:
        """Arquiva a versão e substitui o modelo em uso de forma atômica."""
        import logging

        from triage.config.settings import get_settings
        from triage.pipeline.steps import publish_step

        logging.basicConfig(level="INFO")
        return publish_step(get_settings(), run_name)

    run_name = run_identifier()
    prepared = prepare_data(run_name)
    download_data() >> prepared
    calibrated = calibrate_threshold(run_name)
    trained = train_model(run_name)
    exported = export_onnx(run_name)
    evaluated = evaluate_model(run_name)
    approved = quality_gate(run_name)
    published = publish_model(run_name)
    prepared >> calibrated >> trained >> exported >> evaluated
    evaluated >> approved >> published


triage_retraining()
