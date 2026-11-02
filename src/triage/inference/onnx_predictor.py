"""Backend de inferência com ONNX Runtime.

O modelo exportado (`triage.optimization.export_onnx`) recebe o texto já em
minúsculas: o `lower()` do Python é o mesmo usado pelo scikit-learn e evita
o operador `StringNormalizer` do ONNX, que depende dos locales instalados no
sistema operacional (e falha, por exemplo, em imagens Docker enxutas).

Os metadados do arquivo `.onnx` guardam a ordem das classes e o perfil de
texto (padrão de tokens, stop words e vocabulário). Assim o backend também
mede a cobertura de vocabulário — o sinal de drift do monitoramento — sem
carregar o modelo scikit-learn.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from triage.inference.policy import ArgmaxPolicy, DecisionPolicy
from triage.inference.predictor import (
    Prediction,
    column_order,
    to_predictions,
    token_coverage,
)

INPUT_NAME = "text"
PROBABILITIES_OUTPUT = "probabilities"
CLASSES_KEY = "triage.classes"
TEXT_PROFILE_KEY = "triage.text_profile"


@dataclass(frozen=True)
class TextProfile:
    """Pré-processamento do TF-IDF necessário fora do grafo ONNX.

    Reproduz o analisador do scikit-learn para unigramas: minúsculas,
    tokens pelo `token_pattern` e remoção de stop words.
    """

    token_pattern: str
    stop_words: frozenset[str]
    vocabulary: frozenset[str]

    def tokens(self, text: str) -> list[str]:
        """Tokens do texto, como o `build_analyzer()` do scikit-learn."""
        found = re.findall(self.token_pattern, text.lower())
        return [token for token in found if token not in self.stop_words]

    def coverage(self, text: str) -> float:
        """Fração dos tokens do texto presentes no vocabulário."""
        return token_coverage(self.tokens(text), self.vocabulary)

    def to_json(self) -> str:
        """Serializa o perfil para os metadados do modelo ONNX."""
        return json.dumps(
            {
                "token_pattern": self.token_pattern,
                "stop_words": sorted(self.stop_words),
                "vocabulary": sorted(self.vocabulary),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, content: str) -> "TextProfile":
        """Reconstrói o perfil gravado por `to_json`."""
        data = json.loads(content)
        return cls(
            token_pattern=data["token_pattern"],
            stop_words=frozenset(data["stop_words"]),
            vocabulary=frozenset(data["vocabulary"]),
        )


def create_session(model: bytes | Path) -> ort.InferenceSession:
    """Sessão do ONNX Runtime ajustada para requisições pequenas.

    Um laudo por chamada não se beneficia de paralelismo dentro do operador;
    uma thread por sessão evita disputa de CPU entre requisições
    concorrentes, o que reduz a cauda de latência (p99).
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    source = str(model) if isinstance(model, Path) else model
    return ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])


class OnnxPredictor:
    """Backend que executa o modelo exportado com o ONNX Runtime."""

    backend = "onnx"

    def __init__(
        self,
        session: ort.InferenceSession,
        policy: DecisionPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata or {}
        self.model_version = str(self.metadata.get("model_version", "unknown"))
        custom = session.get_modelmeta().custom_metadata_map
        classes = json.loads(custom[CLASSES_KEY])
        self._labels = list(self.metadata.get("labels") or classes)
        self._order = column_order(classes, self._labels)
        self._profile = TextProfile.from_json(custom[TEXT_PROFILE_KEY])
        self._session = session
        self._policy = policy or ArgmaxPolicy()

    @classmethod
    def from_path(
        cls,
        path: Path,
        policy: DecisionPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "OnnxPredictor":
        """Carrega o modelo ONNX do disco.

        Raises:
            FileNotFoundError: Se o arquivo `.onnx` não existir.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX não encontrado em {path}. "
                "Execute: python -m triage.optimization.export_onnx"
            )
        return cls(create_session(path), policy, metadata)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilidades na ordem das classes do modelo (sem reordenar)."""
        batch = np.array([text.lower() for text in texts], dtype=object)
        inputs = {INPUT_NAME: batch.reshape(-1, 1)}
        [probas] = self._session.run([PROBABILITIES_OUTPUT], inputs)
        return probas

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica uma lista de textos (probabilidades na ordem canônica)."""
        probas = self.predict_proba(texts)[:, self._order]
        chosen = self._policy.decide(probas, self._labels)
        return to_predictions(probas, self._labels, chosen)

    def vocabulary_coverage(self, texts: list[str]) -> list[float]:
        """Fração dos termos de cada texto que o modelo reconhece."""
        return [self._profile.coverage(text) for text in texts]
