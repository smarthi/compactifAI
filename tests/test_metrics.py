from __future__ import annotations

from quantum_tensors.metrics import exact_match, token_f1


def test_exact_match_normalizes_articles_and_case() -> None:
    assert exact_match("The Plan", "plan") == 1.0


def test_token_f1_partial_overlap() -> None:
    assert 0.0 < token_f1("approved plan", "the plan was rejected") < 1.0

