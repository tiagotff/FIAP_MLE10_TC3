"""Divisões estratificadas e reprodutíveis do corpus."""

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from triage.data.dataset import TextDataset


@dataclass(frozen=True)
class DatasetSplits:
    """Treino, validação e teste."""

    train: TextDataset
    validation: TextDataset
    test: TextDataset

    def sizes(self) -> dict[str, int]:
        """Quantidade de amostras por conjunto."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def _split_by_count(
    dataset: TextDataset, count: int, seed: int
) -> tuple[TextDataset, TextDataset]:
    x_rest, x_part, y_rest, y_part = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=count,
        random_state=seed,
        stratify=dataset.labels,
    )
    rest = TextDataset(list(x_rest), list(y_rest))
    return rest, TextDataset(list(x_part), list(y_part))


def stratified_split(
    dataset: TextDataset, size: float, seed: int
) -> tuple[TextDataset, TextDataset]:
    """Separa uma fração estratificada do dataset.

    A fração é convertida em contagem absoluta (arredondada), evitando
    erros de ponto flutuante no tamanho de cada parte.

    Args:
        dataset: Dataset a ser dividido.
        size: Fração destinada à segunda parte (0 < size < 1).
        seed: Semente para tornar a divisão reprodutível.

    Returns:
        Tupla (restante, fração separada).

    Raises:
        ValueError: Se `size` estiver fora do intervalo (0, 1).
    """
    if not 0.0 < size < 1.0:
        raise ValueError(f"Fração da divisão deve estar entre 0 e 1: {size}")
    return _split_by_count(dataset, max(1, round(size * len(dataset))), seed)


def train_validation_test_split(
    dataset: TextDataset, validation: float, test: float, seed: int
) -> DatasetSplits:
    """Divide em treino, validação e teste com frações sobre o total.

    Ex.: validation=0.15 e test=0.15 resultam em 70/15/15.

    Raises:
        ValueError: Se as frações não deixarem espaço para o treino.
    """
    if validation + test >= 1.0:
        raise ValueError(
            f"Validação ({validation}) + teste ({test}) devem somar menos que 1."
        )
    total = len(dataset)
    remainder, test_set = _split_by_count(dataset, round(test * total), seed)
    train, validation_set = _split_by_count(remainder, round(validation * total), seed)
    return DatasetSplits(train=train, validation=validation_set, test=test_set)
