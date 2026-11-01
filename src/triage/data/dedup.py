"""Remoção de textos duplicados com resolução conservadora de rótulos.

No corpus original, um mesmo resumo aparece uma vez para cada condição
associada a ele (inclusive em splits diferentes). Mantê-lo duplicado gera
rótulos contraditórios e vazamento entre treino e teste. Cada texto passa
a aparecer uma única vez, com o rótulo de MAIOR urgência entre os seus —
regra conservadora de triagem: na dúvida, prioriza-se o paciente.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from triage.data.dataset import TextDataset


@dataclass(frozen=True)
class DeduplicationReport:
    """Resumo do que a deduplicação removeu e resolveu."""

    input_rows: int
    unique_texts: int
    duplicates_removed: int
    conflicting_texts: int

    def to_dict(self) -> dict[str, int]:
        """Serializa o relatório."""
        return asdict(self)


def deduplicate(
    dataset: TextDataset, severity: Sequence[str]
) -> tuple[TextDataset, DeduplicationReport]:
    """Mantém um exemplar por texto, com o rótulo mais urgente.

    Args:
        dataset: Corpus possivelmente com textos repetidos.
        severity: Rótulos em ordem crescente de urgência.

    Raises:
        ValueError: Se houver rótulo fora da escala de urgência.
    """
    rank = {label: position for position, label in enumerate(severity)}
    unknown = set(dataset.labels) - set(rank)
    if unknown:
        raise ValueError(f"Rótulos fora da escala de urgência: {sorted(unknown)}")
    labels_by_text: dict[str, set[str]] = {}
    for text, label in zip(dataset.texts, dataset.labels, strict=True):
        labels_by_text.setdefault(text, set()).add(label)
    resolved = {t: max(ls, key=rank.__getitem__) for t, ls in labels_by_text.items()}
    report = DeduplicationReport(
        input_rows=len(dataset),
        unique_texts=len(resolved),
        duplicates_removed=len(dataset) - len(resolved),
        conflicting_texts=sum(len(ls) > 1 for ls in labels_by_text.values()),
    )
    return TextDataset(list(resolved), list(resolved.values())), report
