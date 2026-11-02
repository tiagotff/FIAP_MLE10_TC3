"""Exporta o pipeline scikit-learn para ONNX e verifica a paridade.

Uso:
    python -m triage.optimization.export_onnx

Lê `models/model.joblib`, grava `models/model.onnx` e registra o resultado
da verificação em `models/metadata.json` (chave `onnx`). A exportação só é
gravada se o modelo ONNX reproduzir o original em todo o corpus: mesmos
rótulos e diferença de probabilidade abaixo da tolerância.

Decisões para a conversão ser exata e portátil:
  - o `lowercase` sai do grafo e é feito em Python pelo backend ONNX (o
    operador `StringNormalizer` depende dos locales do sistema e falha em
    imagens Docker enxutas);
  - o tokenizador recebe uma expressão RE2 equivalente ao `token_pattern`
    do scikit-learn, com letras acentuadas (o padrão do conversor só aceita
    ASCII e quebraria "torácica" em dois tokens);
  - só se aceitam unigramas e o analisador padrão de palavras, que é o
    espaço de busca do Optuna (`configs/tuning.yaml`).
"""

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from triage.config.settings import Settings, get_settings
from triage.data.dataset import load_corpus
from triage.data.schema import DataConfig
from triage.inference.onnx_predictor import (
    CLASSES_KEY,
    INPUT_NAME,
    TEXT_PROFILE_KEY,
    OnnxPredictor,
    TextProfile,
    create_session,
)
from triage.training.artifacts import load_metadata, save_json

logger = logging.getLogger(__name__)

SKLEARN_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
# RE2 (usado pelo ONNX Runtime): sequências de 2+ letras, dígitos ou "_",
# equivalente ao `token_pattern` padrão do scikit-learn.
ONNX_TOKEN_PATTERN = r"[\p{L}\p{N}_]{2,}"
DEFAULT_TOLERANCE = 1e-4
# O TfIdfVectorizer do ONNX produz uma matriz densa (lote x vocabulário):
# a verificação roda em lotes para não alocar gigabytes de uma vez.
PARITY_BATCH_SIZE = 256


class OnnxExportError(Exception):
    """O modelo não pode ser exportado ou não reproduz o original."""


@dataclass(frozen=True)
class ParityReport:
    """Comparação entre o modelo original e o exportado."""

    samples: int
    max_abs_diff: float
    label_agreement: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Mesmos rótulos e probabilidades dentro da tolerância."""
        return self.label_agreement == 1.0 and self.max_abs_diff <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        """Serializa o relatório (com o resultado)."""
        return {**asdict(self), "passed": self.passed}


def _vectorizer(pipeline: Pipeline) -> TfidfVectorizer:
    vectorizer = pipeline.steps[0][1]
    if not isinstance(vectorizer, TfidfVectorizer):
        raise OnnxExportError(
            "A primeira etapa do pipeline deve ser um TfidfVectorizer."
        )
    return vectorizer


def check_exportable(pipeline: Pipeline) -> None:
    """Garante que o vetorizador usa só opções com conversão exata.

    Raises:
        OnnxExportError: Se alguma opção não for suportada.
    """
    vectorizer = _vectorizer(pipeline)
    unsupported = {
        "analyzer": vectorizer.analyzer != "word",
        "ngram_range": tuple(vectorizer.ngram_range) != (1, 1),
        "token_pattern": vectorizer.token_pattern != SKLEARN_TOKEN_PATTERN,
        "tokenizer": vectorizer.tokenizer is not None,
        "preprocessor": vectorizer.preprocessor is not None,
        "strip_accents": vectorizer.strip_accents is not None,
        "lowercase": not vectorizer.lowercase,
    }
    invalid = [name for name, bad in unsupported.items() if bad]
    if invalid:
        raise OnnxExportError(f"Opções do TF-IDF sem conversão exata: {invalid}")


def text_profile(vectorizer: TfidfVectorizer) -> TextProfile:
    """Perfil de texto do vetorizador ajustado (usado pelo backend ONNX)."""
    stop_words = vectorizer.get_stop_words() or frozenset()
    return TextProfile(
        token_pattern=vectorizer.token_pattern,
        stop_words=frozenset(stop_words),
        vocabulary=frozenset(vectorizer.vocabulary_),
    )


def convert(pipeline: Pipeline) -> bytes:
    """Converte o pipeline para ONNX (bytes do modelo serializado).

    Raises:
        OnnxExportError: Se o pipeline usar opções sem conversão exata.
    """
    check_exportable(pipeline)
    graph_pipeline = copy.deepcopy(pipeline)
    vectorizer = _vectorizer(graph_pipeline)
    vectorizer.lowercase = False  # feito em Python pelo OnnxPredictor
    classifier = graph_pipeline.steps[-1][1]
    model = convert_sklearn(
        graph_pipeline,
        name="triage_urgency_classifier",
        initial_types=[(INPUT_NAME, StringTensorType([None, 1]))],
        options={
            id(classifier): {"zipmap": False},
            id(vectorizer): {"tokenexp": ONNX_TOKEN_PATTERN},
        },
    )
    metadata = {
        CLASSES_KEY: json.dumps([str(c) for c in pipeline.classes_]),
        TEXT_PROFILE_KEY: text_profile(_vectorizer(pipeline)).to_json(),
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    return model.SerializeToString()


def check_parity(
    pipeline: Pipeline,
    predictor: OnnxPredictor,
    texts: list[str],
    tolerance: float = DEFAULT_TOLERANCE,
) -> ParityReport:
    """Compara probabilidades e rótulos do original e do exportado."""
    expected = pipeline.predict_proba(texts)
    actual = np.vstack(
        [
            predictor.predict_proba(texts[start : start + PARITY_BATCH_SIZE])
            for start in range(0, len(texts), PARITY_BATCH_SIZE)
        ]
    )
    return ParityReport(
        samples=len(texts),
        max_abs_diff=float(np.abs(expected - actual).max()),
        label_agreement=float((expected.argmax(1) == actual.argmax(1)).mean()),
        tolerance=tolerance,
    )


def export_onnx(
    pipeline: Pipeline,
    path: Path,
    texts: list[str],
    tolerance: float = DEFAULT_TOLERANCE,
) -> ParityReport:
    """Converte, verifica a paridade e só então grava o arquivo (atômico).

    Raises:
        OnnxExportError: Se a conversão falhar ou a paridade não for atingida.
    """
    content = convert(pipeline)
    report = check_parity(
        pipeline, OnnxPredictor(create_session(content)), texts, tolerance
    )
    if not report.passed:
        raise OnnxExportError(f"Modelo ONNX diverge do original: {report.to_dict()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return report


def onnx_metadata(report: ParityReport, path: Path) -> dict[str, Any]:
    """Bloco `onnx` dos metadados do modelo."""
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "parity": report.to_dict(),
    }


def run_export(settings: Settings) -> dict[str, Any]:
    """Exporta o modelo publicado e registra a paridade nos metadados.

    A paridade é verificada em todo o corpus (treino e teste originais).
    """
    pipeline = joblib.load(settings.model_path)
    config = DataConfig.from_yaml(settings.data_config_path)
    texts = load_corpus(settings.data_raw_dir, config).texts
    report = export_onnx(pipeline, settings.onnx_path, texts)
    block = onnx_metadata(report, settings.onnx_path)
    for path in (
        settings.metadata_path,
        settings.reports_dir / "model" / "metrics.json",
    ):
        if path.exists():
            save_json({**load_metadata(path), "onnx": block}, path)
    return block


def main() -> None:
    """Ponto de entrada de linha de comando."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    block = run_export(settings)
    parity = block["parity"]
    logger.info(
        "ONNX exportado em %s (%d bytes) | paridade em %d textos: "
        "rótulos %.2f%%, diferença máxima %.2e",
        settings.onnx_path,
        block["size_bytes"],
        parity["samples"],
        parity["label_agreement"] * 100,
        parity["max_abs_diff"],
    )


if __name__ == "__main__":
    main()
