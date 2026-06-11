# Quantum Tensors

Quantum Tensors is a research implementation of the CompactifAI idea from
`CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired
Tensor Networks`, adapted for `openai/gpt-oss-20b` and meeting benchmarks.

The project contains:

- MPO/TT-matrix tensorization of `torch.nn.Linear` weights with sequential SVD.
- Hugging Face model conversion that stores tensorized layers as a lightweight adapter.
- Healing fine-tuning for compressed weights.
- QMSum query-based meeting summarization benchmarking.
- ELITR-Bench meeting assistant QA benchmarking in single-turn and multi-turn modes.
- Optional LLM-as-judge scoring for ELITR-Bench.

## Why this shape

The paper replaces self-attention and MLP weight matrices with Matrix Product
Operators (MPOs), controls compression through the MPO bond dimension, then runs a
short healing phase. It also recommends layer sensitivity profiling because early
layers and the final MLP output projection can be more sensitive to compression.

This repo follows that recipe while making the target model configurable. For
`gpt-oss-20b`, the default is to tensorize large linear modules in transformer
blocks, skip embeddings, the LM head, router-like modules, and optionally skip MLP
output projections.

## Sources

- CompactifAI paper: `/Users/suneel.marti/Downloads/CompactifAI.pdf`
- OpenAI gpt-oss release: https://openai.com/index/introducing-gpt-oss/
- gpt-oss-20b Hugging Face model: https://huggingface.co/openai/gpt-oss-20b
- QMSum dataset repo: https://github.com/Yale-LILY/QMSum
- QMSum paper: https://arxiv.org/abs/2104.05938
- ELITR-Bench paper and data link: https://arxiv.org/abs/2403.20262
- ELITR-Bench data repo path: https://github.com/utter-project/UTTER-MS9-meetingdata/tree/master/ELITR-Bench

## Install

```bash
cd /Users/suneel.marti/opensourceprojects/quantum-tensors
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval,quant]"
```

For gpt-oss inference, install the latest Transformers stack supported by your
CUDA or Apple Silicon environment. OpenAI's model card also supports vLLM,
Ollama, and SGLang for base-model serving.

## Data

QMSum:

```bash
python scripts/download_qmsum.py data/qmsum
```

ELITR-Bench:

```bash
python scripts/download_elitr.py data/elitr
```

The loaders also accept manually downloaded copies. They intentionally do not
vendor the datasets into this repository.

## Compress gpt-oss-20b

```bash
quantum-tensors compress \
  --model-id openai/gpt-oss-20b \
  --output-dir checkpoints/gpt-oss-20b-mpo-r16 \
  --max-rank 16 \
  --order 4 \
  --layer-start 8 \
  --skip-mlp-output
```

The command saves:

- `tensorized_config.json`: module names, MPO shapes, rank choices, and base model id.
- `tensorized_model.safetensors`: MPO cores and tensorized biases.
- `conversion_report.json`: parameter counts and compression ratios.
- tokenizer files, when available.

## Healing

Healing fine-tunes the tensorized adapter after truncation. You can train from
QMSum examples or a JSONL instruction file with `messages`, `prompt/completion`,
or `text` fields.

```bash
quantum-tensors heal \
  --checkpoint-dir checkpoints/gpt-oss-20b-mpo-r16 \
  --dataset-jsonl data/healing.jsonl \
  --output-dir checkpoints/gpt-oss-20b-mpo-r16-healed \
  --max-steps 200 \
  --learning-rate 1e-5
```

For QMSum-based healing:

```bash
quantum-tensors heal \
  --checkpoint-dir checkpoints/gpt-oss-20b-mpo-r16 \
  --qmsum-path data/qmsum/QMSum \
  --qmsum-split train \
  --output-dir checkpoints/gpt-oss-20b-mpo-r16-healed
```

## Benchmarks

QMSum summarization:

```bash
quantum-tensors benchmark-qmsum \
  --model-id openai/gpt-oss-20b \
  --checkpoint-dir checkpoints/gpt-oss-20b-mpo-r16-healed \
  --qmsum-path data/qmsum/QMSum \
  --split test \
  --output-dir outputs/qmsum-mpo-r16
```

ELITR-Bench QA:

```bash
quantum-tensors benchmark-elitr \
  --model-id openai/gpt-oss-20b \
  --checkpoint-dir checkpoints/gpt-oss-20b-mpo-r16-healed \
  --elitr-path data/elitr/UTTER-MS9-meetingdata/ELITR-Bench \
  --split test \
  --mode single-turn-qa \
  --output-dir outputs/elitr-single-mpo-r16
```

Multi-turn conversation mode:

```bash
quantum-tensors benchmark-elitr \
  --model-id openai/gpt-oss-20b \
  --checkpoint-dir checkpoints/gpt-oss-20b-mpo-r16-healed \
  --elitr-path data/elitr/UTTER-MS9-meetingdata/ELITR-Bench \
  --split test \
  --mode multi-turn-conv \
  --output-dir outputs/elitr-conv-mpo-r16
```

If `OPENAI_API_KEY` is set, add `--judge-model gpt-4.1-mini` to score
ELITR-Bench answers with a rubric-style judge. Without a judge, the runner
reports ROUGE-L, token F1, and exact match proxies.

## Layer profiling

```bash
quantum-tensors profile \
  --model-id openai/gpt-oss-20b \
  --output-file outputs/layer_profile.json \
  --max-rank 16 \
  --order 4 \
  --max-modules 64
```

This is a fast reconstruction-error profile. It is not a replacement for the
paper's task-metric sensitivity profile, but it is useful for deciding where to
start before expensive benchmark-driven profiling.

## Notes

- Tensorized checkpoints are adapter-style: load the base model, replace the
  selected linear layers with MPO modules, then load the saved MPO cores.
- For very large matrices, sequential SVD is expensive. Run conversion on a
  GPU machine with enough memory, and start with later layers and modest ranks.
- The default generation path uses `tokenizer.apply_chat_template`, which is
  required for gpt-oss harmony formatting in Transformers.

