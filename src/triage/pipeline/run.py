"""Executa o pipeline de retreino completo, sem Airflow.

Uso:
    python -m triage.pipeline.run [--run-id NOME]

Roda as mesmas etapas, na mesma ordem, que a DAG `triage_retraining` do
Airflow — útil para desenvolvimento e para ambientes sem orquestrador.
"""

import argparse
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from triage.config.settings import Settings, get_settings
from triage.pipeline import steps

logger = logging.getLogger(__name__)

RunStep = Callable[[Settings, str], Any]

PIPELINE: tuple[tuple[str, RunStep], ...] = (
    ("prepare", steps.prepare_step),
    ("calibrate", steps.calibrate_step),
    ("train", steps.train_step),
    ("evaluate", steps.evaluate_step),
    ("quality_gate", steps.quality_gate_step),
    ("publish", steps.publish_step),
)


def default_run_id(now: datetime | None = None) -> str:
    """Identificador baseado em timestamp UTC (ex.: manual__20261101T201500)."""
    moment = now or datetime.now(UTC)
    return f"manual__{moment.strftime('%Y%m%dT%H%M%S')}"


def run_pipeline(settings: Settings, run_id: str) -> dict[str, Any]:
    """Executa todas as etapas em sequência e devolve o resumo de cada uma."""
    results: dict[str, Any] = {"download": steps.download_step(settings)}
    for name, step in PIPELINE:
        logger.info("Etapa %s (execução %s)", name, run_id)
        results[name] = step(settings, run_id)
    return results


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None, help="Nome da execução")
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    results = run_pipeline(settings, args.run_id or default_run_id())
    logger.info(
        "Modelo publicado: %s | teste: %s", results["publish"], results["evaluate"]
    )


if __name__ == "__main__":
    main()
