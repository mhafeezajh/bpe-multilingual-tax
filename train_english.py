"""Train BPE on Shakespeare and inspect the first merges learned.

Run with: python train_english.py

If shakespeare.txt is missing, download it first:
    curl -L -o shakespeare.txt \\
      https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import os
import sys

from bpe import train, encode

CORPUS_PATH = "shakespeare.txt"
CORPUS_SIZE_BYTES = 100_000
VOCAB_SIZE = 512


def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: {CORPUS_PATH} not found.")
        print("Download it first with:")
        print("  curl -L -o shakespeare.txt \\")
        print("    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        sys.exit(1)

    with open(CORPUS_PATH) as f:
        text = f.read()[:CORPUS_SIZE_BYTES]

    print(f"Training BPE on {len(text):,} chars of Shakespeare...")
    merges, vocab = train(text, vocab_size=VOCAB_SIZE, verbose=False)
    print(f"Learned {len(merges)} merges (vocab size = {VOCAB_SIZE}).\n")

    print("First 20 merges (most frequent English byte pairs in this corpus):\n")
    print(f"  {'#':>3s}  {'left':>14s}  {'right':<14s}  {'->  result':<20s}")
    print(f"  {'-' * 3}  {'-' * 14}  {'-' * 14}  {'-' * 20}")
    for i, (pair, new_id) in enumerate(list(merges.items())[:20]):
        a, b = pair
        print(f"  {i + 1:3d}  {vocab[a]!r:>14s}  {vocab[b]!r:<14s}  ->  {vocab[new_id]!r}")

    print()
    sample = text[:1000]
    encoded = encode(sample, merges)
    n_bytes = len(sample.encode("utf-8"))
    print(f"Compression on 1KB of English text:")
    print(f"  {n_bytes} bytes  ->  {len(encoded)} tokens")
    print(f"  = {n_bytes / len(encoded):.2f} bytes/token")
    print()
    print(f"For comparison, GPT-4's cl100k_base achieves ~3.5–4 bytes/token on English,")
    print(f"but it has a vocabulary of 100,000 trained on terabytes of internet text.")
    print(f"Your 512-vocab tokenizer trained on 100KB doing ~2 bytes/token is the right ballpark.")


if __name__ == "__main__":
    main()
