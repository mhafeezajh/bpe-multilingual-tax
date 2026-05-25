# bpe-multilingual-tax

> A minimal byte-level BPE tokenizer in ~120 lines of Python, with a measured experiment that quantifies the **multilingual tokenization tax** — the architectural reason your LLM bill costs 3–5× more in any non-English language.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## What this is

A from-scratch implementation of byte-level Byte Pair Encoding — the algorithm family used by GPT-2, GPT-4 (`cl100k_base`, `o200k_base`), Llama, Mistral, and Qwen. Pure Python, standard library only, no `tokenizers` / `tiktoken` / `sentencepiece` dependencies.

Plus a self-contained experiment that demonstrates the multilingual tokenization tax with measured numbers — using Arabic as the worked example, but the same mechanism applies to Hindi, Chinese, Japanese, Korean, Thai, Russian, and every other non-Latin-heavy script.

## The headline result

Training two BPE tokenizers on equal-sized corpora (~100 KB each) and encoding the same Arabic text with both — chosen as the worked example, but the pattern is universal across non-English languages:

| Tokenizer trained on | Tokens for 1 KB of Arabic | Bytes / token |
|---|---:|---:|
| English Shakespeare | **1,838** | 1.00 |
| Arabic Wikipedia | **382** | 4.82 |
| **Ratio** | **4.81×** | |

That ratio is what every English-trained commercial tokenizer (`cl100k_base`, the Llama tokenizer, etc.) does to non-English text — in proportion to how under-represented the language was in the training corpus. Your invoice reflects it.

## How the tax varies by language

| Script family | Approximate token ratio vs English | UTF-8 bytes/char |
|---|---|---|
| Latin with accents (Spanish, French, German) | 1.3–2× | 1–2 |
| Cyrillic, Greek (Russian, Bulgarian, Greek) | 2–3× | 2 |
| Arabic, Hebrew | 3–5× | 2 |
| CJK (Chinese, Japanese, Korean) | 3–6× | 3 |
| Devanagari (Hindi, Marathi, Sanskrit) | 4–8× | 3 |
| Emoji-heavy text | 4–10× | 4 |

Exact numbers vary by content and target tokenizer. Run the experiment on your specific data — that's what this repo is for.

## Quickstart

```bash
git clone https://github.com/<your-user>/bpe-multilingual-tax
cd bpe-multilingual-tax

# Run the test suite (9 tests, all pure Python)
python test_bpe.py

# Inspect the 256-byte base vocabulary
python inspect_vocab.py

# Download Tiny Shakespeare for the English experiment
curl -L -o shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train on English and see the first merges learned
python train_english.py

# Run the multilingual experiment (headline)
python arabic_experiment.py
```

No dependencies to install. Python 3.11+ recommended.

## Adapting the experiment to your own language

The repo ships with an Arabic sample in `arabic_sample.txt`. To measure the tax for Hindi, Chinese, Japanese, Korean, or any other language:

1. Replace `arabic_sample.txt` with a sample in your target language (any UTF-8 text, ~5–10 KB is enough).
2. Re-run `python arabic_experiment.py` (or rename the script — it's language-agnostic).
3. The ratio you see is the tax your real workloads pay.

The script doesn't know what language it's tokenizing — it just measures bytes-per-token compression. The pattern works for any language under-represented in English-heavy training corpora.

## The implementation

`bpe.py` exposes three functions:

```python
from bpe import train, encode, decode

# Train a tokenizer
merges, vocab = train("your training text", vocab_size=512)

# Encode new text
ids = encode("hello world", merges)
# -> [some list of integer token IDs]

# Decode back
text = decode(ids, vocab)
# -> "hello world"
```

The full implementation is ~120 lines including docstrings. Read [`bpe.py`](./bpe.py) — every line is commented.

## Read the deep-dive article

[**`DEEP_DIVE.md`**](./DEEP_DIVE.md) — full architectural walkthrough, ~5,000 words.

Covers:
- What BPE actually is and how byte-level differs from earlier variants
- Why the 256-byte base vocabulary is non-negotiable (with concrete examples of what breaks without it)
- The frequency auction that makes English win every merge slot
- What the frozen tokenizer means at inference time
- Step-by-step: what happens when you hit `send` on a prompt
- Architectural implications for non-English deployments worldwide

For a condensed LinkedIn version see [`LINKEDIN_POST.md`](./LINKEDIN_POST.md).

## Reproducing the headline number

```
$ python arabic_experiment.py
Training English-only tokenizer (Shakespeare)...
  Done. 256 merges learned.
  First 5 merges: [b'e ', b'th', b't ', b's ', b', ']

Training Arabic-only tokenizer (Wikipedia)...
  Done. 256 merges learned.
  First 5 merges: [b'\xd8\xa7', b'\xd9\x84', b'\xd8\xa7\xd9\x84', b'\xd9\x8a', b'\xd9\x85']

======================================================================
THE MULTILINGUAL TOKENIZATION TAX
======================================================================

Arabic sample: 1000 chars, 1842 bytes

  via English-trained tokenizer:   1838 tokens  (1.00 bytes/token)
  via Arabic-trained tokenizer:     382 tokens  (4.82 bytes/token)

  RATIO: 4.81× more tokens via English tokenizer.

That ratio IS the multilingual tax.
Multiply by your $/1M tokens — that's your bill difference.
```

The exact ratio will vary by ±0.5 depending on your sample and target language, but the pattern is remarkably stable across runs and content types.

## What this is not

- **Not a production tokenizer.** Use `tiktoken`, `tokenizers`, or `sentencepiece` for that. This is teaching code.
- **Not a full GPT-4 reproduction.** No regex pre-tokenization step (which `cl100k_base` uses), no parallelism, no fancy optimizations.
- **Not benchmarked against `tiktoken`.** The point is the *algorithm* and the *multilingual experiment*, not throughput.

## Why this exists

Most engineers know BPE conceptually but have never built it. After building it from scratch and running the multilingual experiment, three things become viscerally true:

1. Tokens are not characters. They're greedy compressions of byte sequences.
2. The tokenizer is frozen at model release time and cannot be retroactively rebalanced.
3. The cost of an LLM API call is decided at the moment of encoding — before the model does any work.

Those three facts drive every meaningful architectural decision in enterprise GenAI deployment in any non-English market: model selection, FinOps, context-window planning, prompt-caching strategy, and language-specific cost forecasting.

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgements

This implementation is a teaching adaptation. The algorithm is byte-level BPE as introduced by GPT-2 (Radford et al., 2019), with the cleanest reference implementation being Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).

---

*By [Mohammed Hafeez Ali Khan](https://linkedin.com/in/hafeezaj/) — Senior AI Solutions Architect.*
