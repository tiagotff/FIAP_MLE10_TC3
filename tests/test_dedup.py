import pytest

from triage.data.dataset import TextDataset
from triage.data.dedup import deduplicate

SEVERITY = ["normal", "atencao", "urgente"]


def test_deduplicate_keeps_most_urgent_label():
    dataset = TextDataset(
        ["a", "b", "a", "c", "b", "a"],
        ["normal", "atencao", "urgente", "normal", "atencao", "atencao"],
    )
    unique, report = deduplicate(dataset, SEVERITY)
    assert dict(zip(unique.texts, unique.labels, strict=True)) == {
        "a": "urgente",
        "b": "atencao",
        "c": "normal",
    }
    assert report.to_dict() == {
        "input_rows": 6,
        "unique_texts": 3,
        "duplicates_removed": 3,
        "conflicting_texts": 1,
    }


def test_deduplicate_preserves_first_seen_order():
    dataset = TextDataset(["z", "y", "z", "x"], ["normal"] * 4)
    unique, _ = deduplicate(dataset, SEVERITY)
    assert unique.texts == ["z", "y", "x"]


def test_deduplicate_without_duplicates_is_identity():
    dataset = TextDataset(["a", "b"], ["normal", "urgente"])
    unique, report = deduplicate(dataset, SEVERITY)
    assert unique == dataset
    assert report.duplicates_removed == 0


def test_deduplicate_rejects_unknown_label():
    with pytest.raises(ValueError, match="fora da escala"):
        deduplicate(TextDataset(["a"], ["critico"]), SEVERITY)
