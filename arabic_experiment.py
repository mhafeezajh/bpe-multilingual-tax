"""The headline experiment: measure the multilingual tokenization tax.

Trains two BPE tokenizers on equal-sized corpora — one English, one Arabic —
then encodes the same Arabic text with both, demonstrating the 3-5× ratio
that is the architectural fact behind every "Arabic costs more" conversation.

Run with: python arabic_experiment.py

If shakespeare.txt is missing, download it with:
  curl -L -o shakespeare.txt \\
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

arabic_sample.txt ships with the repo — you can also paste your own Arabic
content into that file for different experiments.
"""

import os
import sys

from bpe import train, encode

ENGLISH_CORPUS = "shakespeare.txt"
ARABIC_CORPUS = "arabic_sample.txt"
CORPUS_SIZE_BYTES = 100_000
VOCAB_SIZE = 512


def main():
    for path in (ENGLISH_CORPUS, ARABIC_CORPUS):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            if path == ENGLISH_CORPUS:
                print("Download with: curl -L -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
            sys.exit(1)

    # ===== Tokenizer A: English-trained =====
    with open(ENGLISH_CORPUS) as f:
        english = f.read()[:CORPUS_SIZE_BYTES]

    print("Training English-only tokenizer (Shakespeare)...")
    en_merges, en_vocab = train(english, vocab_size=VOCAB_SIZE, verbose=False)
    print(f"  Done. {len(en_merges)} merges learned.")
    print(f"  First 5 merges: {[en_vocab[256 + i] for i in range(5)]}\n")

    # ===== Tokenizer B: Arabic-trained =====
    with open(ARABIC_CORPUS, encoding="utf-8") as f:
        arabic = f.read()

    # Pad up to match English corpus size for a fair comparison
    while len(arabic.encode("utf-8")) < CORPUS_SIZE_BYTES:
        arabic = arabic + arabic
    arabic = arabic.encode("utf-8")[:CORPUS_SIZE_BYTES].decode("utf-8", errors="ignore")

    print("Training Arabic-only tokenizer (Wikipedia)...")
    ar_merges, ar_vocab = train(arabic, vocab_size=VOCAB_SIZE, verbose=False)
    print(f"  Done. {len(ar_merges)} merges learned.")
    print(f"  First 5 merges: {[ar_vocab[256 + i] for i in range(5)]}\n")

    # ===== The experiment =====
    sample = arabic[:1000]
    n_chars = len(sample)
    n_bytes = len(sample.encode("utf-8"))

    ids_en = encode(sample, en_merges)
    ids_ar = encode(sample, ar_merges)

    print("=" * 70)
    print("THE MULTILINGUAL TOKENIZATION TAX")
    print("=" * 70)
    print(f"\nArabic sample: {n_chars} chars, {n_bytes} bytes\n")
    print(
        f"  via English-trained tokenizer:  {len(ids_en):>5} tokens  "
        f"({n_bytes / len(ids_en):.2f} bytes/token)"
    )
    print(
        f"  via Arabic-trained tokenizer:   {len(ids_ar):>5} tokens  "
        f"({n_bytes / len(ids_ar):.2f} bytes/token)"
    )
    print(f"\n  RATIO: {len(ids_en) / len(ids_ar):.2f}× more tokens via English tokenizer.")
    print("\nThat ratio IS the multilingual tax.")
    print("Multiply by your $/1M tokens — that's your bill difference.\n")


if __name__ == "__main__":
    main()
