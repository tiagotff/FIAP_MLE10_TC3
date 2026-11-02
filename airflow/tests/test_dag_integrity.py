"""Integridade da DAG de retreino (requer Airflow instalado).

Executado no CI em um ambiente só com Airflow:
    pytest airflow/tests
"""

from pathlib import Path

import pytest

# `airflow.sdk` (e não `airflow`): a pasta airflow/ do repositório também é
# importável como namespace package e não indica que o Airflow está instalado.
pytest.importorskip("airflow.sdk")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_FOLDER = Path(__file__).resolve().parents[1] / "dags"
DAG_ID = "triage_retraining"
ML_TASKS = [
    "download_data",
    "prepare_data",
    "calibrate_threshold",
    "train_model",
    "evaluate_model",
    "quality_gate",
    "publish_model",
]


@pytest.fixture(scope="module")
def dag():
    # Lê as DAGs apenas do disco (sem banco de metadados), como no parse
    # feito pelo dag-processor.
    bag = DagBag(dag_folder=str(DAGS_FOLDER))
    assert bag.import_errors == {}, bag.import_errors
    return bag.dags[DAG_ID]


def test_dag_is_loaded(dag):
    assert dag is not None
    assert dag.catchup is False
    assert dag.max_active_runs == 1


def test_dag_has_expected_tasks(dag):
    assert set(dag.task_ids) == {"run_identifier", *ML_TASKS}


def test_tasks_run_in_pipeline_order(dag):
    chain = ML_TASKS[1:]
    for upstream, downstream in zip(chain, chain[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids
    assert "prepare_data" in dag.get_task("download_data").downstream_task_ids


def test_publish_depends_on_quality_gate(dag):
    upstream = dag.get_task("publish_model").get_flat_relative_ids(upstream=True)
    assert {"quality_gate", "evaluate_model", "train_model"} <= set(upstream)


def test_ml_tasks_use_project_environment(dag):
    for task_id in ML_TASKS:
        task = dag.get_task(task_id)
        assert type(task).__name__ == "_PythonExternalDecoratedOperator"
        assert task.retries == 1
