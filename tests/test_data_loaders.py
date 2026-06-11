from __future__ import annotations

import json
from pathlib import Path

from quantum_tensors.benchmarks.data import load_elitr, load_qmsum


def test_load_qmsum_original_shape(tmp_path: Path) -> None:
    """Verify the QMSum loader normalizes the original meeting/query structure."""
    data_dir = tmp_path / "data" / "ALL"
    data_dir.mkdir(parents=True)
    payload = [
        {
            "meeting_id": "m1",
            "meeting_transcripts": [{"speaker": "A", "content": "We approved the plan."}],
            "general_query_list": [{"query": "Summarize the whole meeting.", "answer": "The plan was approved."}],
            "specific_query_list": [{"query": "What was approved?", "answer": "The plan."}],
        }
    ]
    (data_dir / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    examples = load_qmsum(tmp_path, split="test")
    assert len(examples) == 2
    assert examples[0].transcript == "A: We approved the plan."


def test_load_elitr_inline_shape(tmp_path: Path) -> None:
    """Verify the ELITR loader accepts records with inline transcript and QA data."""
    payload = [
        {
            "meeting_id": "e1",
            "transcript": "A: The demo is on Friday.",
            "questions": [{"question": "When is the demo?", "answer": "Friday.", "type": "When"}],
        }
    ]
    (tmp_path / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    examples = load_elitr(tmp_path, split="test", setting="qa")
    assert len(examples) == 1
    assert examples[0].question == "When is the demo?"
    assert examples[0].reference == "Friday."
