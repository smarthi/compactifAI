from __future__ import annotations

from quantum_tensors.training.heal import LABEL_IGNORE_INDEX, _build_sft_example


class _FakeTokenizer:
    """Minimal tokenizer with a deterministic chat template."""

    chat_template = "fake"

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
        max_length=None,
    ):
        ids: list[int] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            ids.append(hash(role) % 1000)
            ids.extend(ord(ch) for ch in content)
            ids.append(0)
        if add_generation_prompt:
            ids.append(hash("assistant") % 1000)
        return ids if tokenize else "".join(map(str, ids))

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=True):
        return {"input_ids": [ord(ch) for ch in text]}


def test_build_sft_example_masks_prompt_tokens() -> None:
    """Prompt tokens become -100; only completion tokens contribute to the loss."""
    tokenizer = _FakeTokenizer()
    record = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    input_ids, labels = _build_sft_example(tokenizer, record, max_seq_length=128)

    assert len(input_ids) == len(labels)
    masked = [label == LABEL_IGNORE_INDEX for label in labels]
    assert any(masked), "prompt tokens should be masked"
    assert not all(masked), "completion tokens must remain unmasked"
    # The unmasked tail should equal the corresponding input ids.
    for input_id, label in zip(input_ids, labels):
        assert label in (LABEL_IGNORE_INDEX, input_id)


def test_build_sft_example_text_record_uses_full_loss() -> None:
    """Records with only ``text`` train on the full sequence (no masking)."""
    tokenizer = _FakeTokenizer()
    input_ids, labels = _build_sft_example(tokenizer, {"text": "hello"}, max_seq_length=128)
    assert input_ids == labels
    assert LABEL_IGNORE_INDEX not in labels
