from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class QMSumExample:
    """Normalized query-focused summarization example.

    QMSum source files contain meeting transcripts plus general and specific
    query lists. This dataclass gives benchmark and healing code one stable
    shape to consume regardless of the original file layout.
    """

    example_id: str
    meeting_id: str
    split: str
    query: str
    reference: str
    transcript: str
    query_type: str
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the example into a JSON-serializable dictionary.

        Prediction writers need to preserve inputs, references, and metadata next
        to model outputs. Use this before appending per-example benchmark rows.
        """
        return asdict(self)


@dataclass
class ELITRExample:
    """Normalized ELITR-Bench meeting question-answering example.

    ELITR-Bench mixes meeting context, QA, and conversation-style settings. This
    dataclass provides the common fields needed by single-turn and multi-turn
    runners.
    """

    example_id: str
    meeting_id: str
    split: str
    question: str
    reference: str
    transcript: str
    setting: str
    question_index: int = 0
    question_type: str | None = None
    answer_position: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the example into a JSON-serializable dictionary.

        Benchmark output rows need the original question, reference, split, and
        meeting identifiers for later slicing. Use this when writing predictions
        or judge traces.
        """
        return asdict(self)


def _read_json_any(path: Path) -> Any:
    """Read either JSON or JSONL data from disk.

    Dataset releases and local conversions often differ in container format, so
    loaders need one reader that handles both common cases. Use it at the edge of
    dataset ingestion before normalizing records.
    """
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def _flatten_records(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield meeting-like dictionaries from several possible JSON layouts.

    Public benchmark files can be arrays, dictionaries keyed by meeting id, or a
    single meeting object. Use this helper so downstream parsing can operate on
    one record at a time.
    """
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        if any(key in payload for key in ["meeting_transcripts", "transcript", "questions", "qa_pairs"]):
            yield payload
        else:
            for key, value in payload.items():
                if isinstance(value, dict):
                    record = dict(value)
                    record.setdefault("meeting_id", key)
                    yield record
                elif isinstance(value, list):
                    yield {"meeting_id": key, "questions": value}


def _find_split_files(root: Path, split: str, preferred_domain: str | None = None) -> list[Path]:
    """Find candidate JSON/JSONL files for a split and optional domain.

    QMSum and ELITR checkouts may place split names in filenames or directories,
    so a tolerant search avoids hard-coding one release layout. Use this before
    parsing a dataset split from a local checkout.
    """
    split_tokens = {
        "validation": ["validation", "valid", "val", "dev"],
        "valid": ["validation", "valid", "val", "dev"],
        "val": ["validation", "valid", "val", "dev"],
        "dev": ["validation", "valid", "val", "dev"],
        "test": ["test"],
        "train": ["train"],
    }.get(split, [split])
    candidates = []
    for suffix in ["*.jsonl", "*.json"]:
        for path in root.rglob(suffix):
            lowered = str(path).lower()
            if preferred_domain and preferred_domain.lower() not in lowered:
                continue
            if any(token in path.stem.lower() or f"/{token}/" in lowered for token in split_tokens):
                candidates.append(path)
    return sorted(set(candidates))


def _meeting_transcript_to_text(transcript: Any) -> str:
    """Normalize transcript objects into speaker-prefixed plain text.

    Generation prompts need a single string, while datasets may store transcript
    turns as strings or dictionaries. Use this in every loader before truncating
    transcript text for a model context window.
    """
    if isinstance(transcript, str):
        return transcript
    if isinstance(transcript, list):
        lines = []
        for turn in transcript:
            if isinstance(turn, str):
                lines.append(turn)
            elif isinstance(turn, dict):
                speaker = turn.get("speaker") or turn.get("speaker_name") or turn.get("role") or "Speaker"
                content = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
                if content:
                    lines.append(f"{speaker}: {content}")
        return "\n".join(lines)
    return ""


def load_qmsum(
    qmsum_path: str | Path,
    split: str = "test",
    domain: str = "ALL",
    include_general: bool = True,
    include_specific: bool = True,
) -> list[QMSumExample]:
    """Load and normalize QMSum examples from a local checkout.

    The QMSum benchmark evaluates query-focused meeting summarization, and this
    loader turns both general and specific query lists into independent examples.
    Use it for benchmarking, healing data construction, or smoke tests with a
    manually downloaded QMSum repository.
    """
    root = Path(qmsum_path)
    if not root.exists():
        raise FileNotFoundError(f"QMSum path does not exist: {root}")

    files = _find_split_files(root, split=split, preferred_domain=domain)
    if not files:
        files = _find_split_files(root, split=split)
    if not files:
        raise FileNotFoundError(f"Could not find QMSum split files for split={split!r} under {root}.")

    examples: list[QMSumExample] = []
    for file_path in files:
        for meeting_index, record in enumerate(_flatten_records(_read_json_any(file_path))):
            transcript = _meeting_transcript_to_text(record.get("meeting_transcripts") or record.get("transcript"))
            meeting_id = str(record.get("meeting_id") or record.get("id") or f"{file_path.stem}-{meeting_index}")
            file_domain = next((part for part in file_path.parts if part in {"ALL", "Academic", "Product", "Committee"}), None)

            if include_general:
                for query_index, query_record in enumerate(record.get("general_query_list", []) or []):
                    query = query_record.get("query") or query_record.get("question")
                    answer = query_record.get("answer") or query_record.get("summary")
                    if query and answer:
                        examples.append(
                            QMSumExample(
                                example_id=f"{meeting_id}:general:{query_index}",
                                meeting_id=meeting_id,
                                split=split,
                                query=query,
                                reference=answer,
                                transcript=transcript,
                                query_type="general",
                                domain=file_domain,
                            )
                        )

            if include_specific:
                for query_index, query_record in enumerate(record.get("specific_query_list", []) or []):
                    query = query_record.get("query") or query_record.get("question")
                    answer = query_record.get("answer") or query_record.get("summary")
                    if query and answer:
                        examples.append(
                            QMSumExample(
                                example_id=f"{meeting_id}:specific:{query_index}",
                                meeting_id=meeting_id,
                                split=split,
                                query=query,
                                reference=answer,
                                transcript=transcript,
                                query_type="specific",
                                domain=file_domain,
                            )
                        )
    return examples


def _collect_transcript_files(root: Path) -> dict[str, str]:
    """Collect standalone transcript text files keyed by filename stem.

    Some ELITR-Bench layouts keep transcripts outside the QA JSON records. Use
    this lookup as a fallback when a meeting record references context indirectly.
    """
    transcripts: dict[str, str] = {}
    for suffix in ["*.txt", "*.md", "*.transcript"]:
        for path in root.rglob(suffix):
            if "readme" in path.name.lower():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            transcripts[path.stem] = text
    return transcripts


def _questions_from_record(record: dict[str, Any], setting: str) -> list[dict[str, Any]]:
    """Extract QA or conversation question records from a meeting record.

    ELITR-Bench variants use several key names for question lists. Use this
    helper to select the requested setting while keeping the main loader compact.
    """
    keys = [
        f"{setting}_questions",
        f"{setting}_qa",
        setting,
        "questions",
        "qa_pairs",
        "qas",
    ]
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if record.get("question") and (record.get("answer") or record.get("reference")):
        return [record]
    return []


def _transcript_from_elitr_record(record: dict[str, Any], transcripts: dict[str, str]) -> str:
    """Resolve transcript text embedded in or referenced by an ELITR record.

    Benchmark prompts must include meeting context even when the data is split
    across files. Use this after loading auxiliary transcript files and before
    constructing ``ELITRExample`` instances.
    """
    transcript = _meeting_transcript_to_text(
        record.get("transcript")
        or record.get("meeting_transcript")
        or record.get("context")
        or record.get("document")
    )
    if transcript:
        return transcript
    meeting_id = str(record.get("meeting_id") or record.get("id") or "")
    return transcripts.get(meeting_id, "")


def load_elitr(
    elitr_path: str | Path,
    split: str = "test",
    setting: str = "qa",
) -> list[ELITRExample]:
    """Load and normalize ELITR-Bench examples from a local checkout.

    The runner supports both single-turn QA and conversation-style settings, so
    this loader maps the chosen setting into a shared example format. Use it
    before running ELITR benchmarks or building local inspection datasets.
    """
    root = Path(elitr_path)
    if not root.exists():
        raise FileNotFoundError(f"ELITR-Bench path does not exist: {root}")

    transcripts = _collect_transcript_files(root)
    files = _find_split_files(root, split=split)
    if not files:
        files = sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Could not find ELITR-Bench JSON files under {root}.")

    wanted_setting = "conv" if setting.lower().startswith("conv") else "qa"
    examples: list[ELITRExample] = []
    for file_path in files:
        lowered = str(file_path).lower()
        if split.lower() not in lowered and split.lower() not in {"valid", "validation", "val", "dev"}:
            continue
        for record_index, record in enumerate(_flatten_records(_read_json_any(file_path))):
            meeting_id = str(record.get("meeting_id") or record.get("id") or file_path.stem)
            transcript = _transcript_from_elitr_record(record, transcripts)
            questions = _questions_from_record(record, wanted_setting)
            for question_index, qa in enumerate(questions):
                question = (
                    qa.get("question")
                    or qa.get("query")
                    or qa.get(f"{wanted_setting}_question")
                    or qa.get("prompt")
                )
                answer = qa.get("answer") or qa.get("reference") or qa.get("ground_truth") or qa.get("gold")
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                if not question or not answer:
                    continue
                examples.append(
                    ELITRExample(
                        example_id=str(qa.get("id") or f"{meeting_id}:{wanted_setting}:{question_index}"),
                        meeting_id=meeting_id,
                        split=split,
                        question=str(question),
                        reference=str(answer),
                        transcript=transcript,
                        setting=wanted_setting,
                        question_index=int(qa.get("question_index") or question_index),
                        question_type=qa.get("question_type") or qa.get("type"),
                        answer_position=qa.get("answer_position") or qa.get("position"),
                    )
                )
    return sorted(examples, key=lambda item: (item.meeting_id, item.question_index, item.example_id))
