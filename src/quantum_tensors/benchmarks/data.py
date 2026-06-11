from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class QMSumExample:
    example_id: str
    meeting_id: str
    split: str
    query: str
    reference: str
    transcript: str
    query_type: str
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ELITRExample:
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
        return asdict(self)


def _read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def _flatten_records(payload: Any) -> Iterable[dict[str, Any]]:
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

