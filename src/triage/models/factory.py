"""Factory de pipelines de classificação de texto (TF-IDF + classificador).

Novos classificadores são adicionados registrando um construtor em
`CLASSIFIER_REGISTRY`, sem alterar o restante do código (Open/Closed).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from triage.config.loader import load_yaml

ClassifierBuilder = Callable[[dict[str, Any], int], ClassifierMixin]


def _logistic_regression(params: dict[str, Any], seed: int) -> ClassifierMixin:
    return LogisticRegression(random_state=seed, **params)


def _random_forest(params: dict[str, Any], seed: int) -> ClassifierMixin:
    return RandomForestClassifier(random_state=seed, **params)


def _complement_nb(params: dict[str, Any], seed: int) -> ClassifierMixin:
    # Naive Bayes é determinístico: a semente não se aplica.
    return ComplementNB(**params)


CLASSIFIER_REGISTRY: dict[str, ClassifierBuilder] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "complement_nb": _complement_nb,
}


class ClassifierConfig(BaseModel):
    """Classificador ativo e hiperparâmetros por classificador."""

    name: str
    params: dict[str, dict[str, Any]] = {}


class ModelConfig(BaseModel):
    """Configuração completa do pipeline (configs/model.yaml)."""

    vectorizer: dict[str, Any]
    classifier: ClassifierConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        """Cria a configuração a partir de um arquivo YAML."""
        return cls.model_validate(load_yaml(path))


def build_classifier(config: ClassifierConfig, seed: int) -> ClassifierMixin:
    """Instancia o classificador registrado sob `config.name`.

    Raises:
        ValueError: Se o classificador não estiver registrado.
    """
    builder = CLASSIFIER_REGISTRY.get(config.name)
    if builder is None:
        available = ", ".join(sorted(CLASSIFIER_REGISTRY))
        raise ValueError(
            f"Classificador '{config.name}' desconhecido. Opções: {available}"
        )
    return builder(config.params.get(config.name, {}), seed)


def build_pipeline(config: ModelConfig, seed: int) -> Pipeline:
    """Monta o pipeline scikit-learn TF-IDF -> classificador."""
    vectorizer_params = dict(config.vectorizer)
    if "ngram_range" in vectorizer_params:
        vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", build_classifier(config.classifier, seed)),
        ]
    )
