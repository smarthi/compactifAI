from __future__ import annotations

import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": token_f1(prediction, reference), "rouge2": 0.0, "rougeL": token_f1(prediction, reference)}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {name: score.fmeasure for name, score in scores.items()}


def aggregate_metric_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in numeric.items() if values}

