"""Busca de hiperparâmetros com Optuna.

Uso:
    python -m triage.training.tune

Cada trial treina no conjunto de treino e é avaliado na validação (o teste
nunca é usado aqui). A melhor configuração é gravada em
`configs/model.yaml`, que o treino consome — a busca é um passo offline
e o treino permanece determinístico a partir da configuração versionada.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml
from pydantic import BaseModel
from sklearn.metrics import f1_score

from triage.config.loader import load_yaml
from triage.config.settings import Settings, get_settings
from triage.data.dataset import TextDataset
from triage.data.schema import DataConfig
from triage.models.factory import ClassifierConfig, ModelConfig
from triage.training.artifacts import save_json
from triage.training.config import TrainingConfig
from triage.training.train import fit, prepare_datasets, set_seeds

logger = logging.getLogger(__name__)

ParamSpec = dict[str, Any] | list[Any]


class SearchSpace(BaseModel):
    """Hiperparâmetros fixos e hiperparâmetros a explorar."""

    fixed: dict[str, Any] = {}
    search: dict[str, ParamSpec] = {}


class TuningConfig(BaseModel):
    """Configuração da busca (configs/tuning.yaml)."""

    n_trials: int
    timeout_seconds: int | None = None
    metric: Literal["f1_macro"] = "f1_macro"
    vectorizer: SearchSpace
    classifiers: dict[str, SearchSpace]

    @classmethod
    def from_yaml(cls, path: Path) -> "TuningConfig":
        """Cria a configuração a partir de um arquivo YAML."""
        return cls.model_validate(load_yaml(path))


def suggest(trial: optuna.Trial, name: str, spec: ParamSpec) -> Any:  # noqa: ANN401
    """Sugere um valor conforme a especificação declarada no YAML."""
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    low, high = spec["low"], spec["high"]
    if isinstance(low, int) and isinstance(high, int) and not spec.get("log"):
        return trial.suggest_int(name, low, high, step=spec.get("step", 1))
    return trial.suggest_float(
        name, float(low), float(high), log=spec.get("log", False), step=spec.get("step")
    )


def sample_space(
    trial: optuna.Trial, space: SearchSpace, prefix: str
) -> dict[str, Any]:
    """Combina os parâmetros fixos com os sugeridos pelo trial."""
    sampled = {k: suggest(trial, f"{prefix}{k}", v) for k, v in space.search.items()}
    return {**space.fixed, **sampled}


def sample_model_config(trial: optuna.Trial, config: TuningConfig) -> ModelConfig:
    """Sorteia uma configuração completa de pipeline para o trial."""
    name = trial.suggest_categorical("classifier", list(config.classifiers))
    vectorizer = sample_space(trial, config.vectorizer, "tfidf__")
    params = sample_space(trial, config.classifiers[name], f"{name}__")
    return ModelConfig(
        vectorizer=vectorizer,
        classifier=ClassifierConfig(name=name, params={name: params}),
    )


def make_objective(
    train: TextDataset, validation: TextDataset, config: TuningConfig, seed: int
) -> Callable[[optuna.Trial], float]:
    """Cria a função objetivo: F1 macro na validação."""

    def objective(trial: optuna.Trial) -> float:
        model_config = sample_model_config(trial, config)
        trial.set_user_attr("model_config", model_config.model_dump())
        predicted = fit(train, model_config, seed).predict(validation.texts)
        return float(f1_score(validation.labels, predicted, average="macro"))

    return objective


def write_model_config(config: ModelConfig, path: Path, study: optuna.Study) -> Path:
    """Grava a melhor configuração como YAML versionável, com cabeçalho."""
    header = (
        "# Gerado por `python -m triage.training.tune` — não editar à mão.\n"
        f"# Melhor F1 macro na validação: {study.best_value:.4f} "
        f"(trial {study.best_trial.number} de {len(study.trials)}).\n"
    )
    body = yaml.safe_dump(config.model_dump(), sort_keys=False, allow_unicode=True)
    path.write_text(header + body, "utf-8")
    return path


def save_tuning_report(study: optuna.Study, out_dir: Path) -> None:
    """Salva o histórico de trials (CSV) e o resumo da busca (JSON)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.to_csv(out_dir / "trials.csv", index=False)
    summary = {
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metric": "f1_macro (validação)",
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": round(study.best_value, 6),
        "best_params": study.best_params,
    }
    save_json(summary, out_dir / "summary.json")


def load_tuning_data(settings: Settings) -> tuple[TextDataset, TextDataset]:
    """Reproduz a mesma divisão do treino e devolve treino e validação.

    O teste nunca é usado na busca.
    """
    data_config = DataConfig.from_yaml(settings.data_config_path)
    training_config = TrainingConfig.from_yaml(settings.training_config_path)
    splits, _ = prepare_datasets(settings, data_config, training_config)
    return splits.train, splits.validation


def run_tuning(settings: Settings) -> optuna.Study:
    """Executa a busca e persiste a melhor configuração e o relatório."""
    set_seeds(settings.random_seed)
    config = TuningConfig.from_yaml(settings.tuning_config_path)
    train, validation = load_tuning_data(settings)
    study = optuna.create_study(
        study_name="triage-tfidf",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
    )
    objective = make_objective(train, validation, config, settings.random_seed)
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)
    best = ModelConfig.model_validate(study.best_trial.user_attrs["model_config"])
    write_model_config(best, settings.model_config_path, study)
    save_tuning_report(study, settings.reports_dir / "tuning")
    return study


def main() -> None:
    """Ponto de entrada de linha de comando."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = run_tuning(settings)
    logger.info(
        "Melhor F1 macro (validação): %.4f | parâmetros: %s",
        study.best_value,
        study.best_params,
    )


if __name__ == "__main__":
    main()
