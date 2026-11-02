"""Estatísticas de latência reutilizáveis (API e modelo isolado).

O p99 é a métrica de referência do projeto: cobre 99 de cada 100
requisições e expõe a cauda de latência que a média esconde.
"""

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class LatencySummary:
    """Resumo estatístico de uma série de medições, em milissegundos."""

    n: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    throughput_rps: float

    def to_dict(self) -> dict[str, float]:
        """Serializa o resumo com valores arredondados."""
        return {k: round(v, 3) for k, v in asdict(self).items()}

    def headline(self) -> str:
        """Resumo em uma linha, com o p99 em destaque."""
        return (
            f"p99={self.p99_ms:.2f} ms (p50={self.p50_ms:.2f} ms, "
            f"média={self.mean_ms:.2f} ms, n={self.n})"
        )


def summarize(samples_ms: Sequence[float]) -> LatencySummary:
    """Calcula média, desvio, percentis e vazão sequencial.

    Raises:
        ValueError: Se a lista de medições estiver vazia.
    """
    if not samples_ms:
        raise ValueError("É necessário ao menos uma medição de latência.")
    values = np.asarray(samples_ms, dtype=float)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return LatencySummary(
        n=len(values),
        mean_ms=float(values.mean()),
        std_ms=statistics.pstdev(values.tolist()),
        min_ms=float(values.min()),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        max_ms=float(values.max()),
        throughput_rps=1000.0 / float(values.mean()),
    )


def measure(call: Callable[[T], object], texts: Sequence[T]) -> list[float]:
    """Executa `call` para cada entrada (texto ou lote) e retorna as latências em ms."""
    samples: list[float] = []
    for text in texts:
        start = time.perf_counter()
        call(text)
        samples.append((time.perf_counter() - start) * 1000)
    return samples
