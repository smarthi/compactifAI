from __future__ import annotations

import os
import re


JUDGE_RUBRIC = """Score from 1 to 10.
10: The answer is fully correct and complete.
7-9: The answer is mostly correct with minor omissions.
4-6: The answer is partially correct but misses important facts.
2-3: The answer has little overlap with the reference.
1: The answer is irrelevant or contradicts the reference.
Return only a JSON object with keys score and rationale."""


def score_with_openai_judge(
    question: str,
    reference: str,
    prediction: str,
    model: str,
) -> dict[str, object]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI judge scoring.")

    from openai import OpenAI

    client = OpenAI()
    prompt = (
        f"{JUDGE_RUBRIC}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Candidate answer:\n{prediction}\n"
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a strict meeting QA evaluator."},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.output_text
    score_match = re.search(r"score[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    score = float(score_match.group(1)) if score_match else None
    return {"judge_score": score, "judge_raw": text}

