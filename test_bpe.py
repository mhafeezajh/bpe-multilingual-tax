"""Tests for bpe.py — run with: python test_bpe.py"""

from bpe import get_stats, merge, train, encode, decode


def test_get_stats():
    assert get_stats([]) == {}
    assert get_stats([1]) == {}
    assert get_stats([1, 2]) == {(1, 2): 1}
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([], (1, 2), 99) == []
    assert merge([1, 2], (1, 2), 99) == [99]
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    # Non-overlapping: aaaa -> (aa)(aa)
    assert merge([97, 97, 97, 97], (97, 97), 256) == [256, 256]
    # No match
    assert merge([1, 2, 3], (5, 6), 99) == [1, 2, 3]


def test_train_canonical():
    """Karpathy's canonical BPE example."""
    text = "aaabdaaabac"
    merges, vocab = train(text, vocab_size=259)

    expected_merges = {(97, 97): 256, (256, 97): 257, (257, 98): 258}
    assert merges == expected_merges, f"Expected {expected_merges}, got {merges}"

    assert vocab[256] == b"aa"
    assert vocab[257] == b"aaa"
    assert vocab[258] == b"aaab"


def test_encode_canonical():
    text = "aaabdaaabac"
    merges, vocab = train(text, vocab_size=259)

    encoded = encode(text, merges)
    assert encoded == [258, 100, 258, 97, 99], f"Got {encoded}"


def test_round_trip_multi_script():
    """Even an under-trained tokenizer must round-trip all UTF-8 strings."""
    merges, vocab = train("aaabdaaabac", vocab_size=259)

    samples = [
        "hello world",
        "aaabdaaabac",
        "السلام عليكم",      # Arabic
        "नमस्ते",              # Hindi (Devanagari)
        "你好,世界",          # Chinese
        "🚀 emoji works too",
        "",                   # empty string
        "a",                  # single char
    ]
    for s in samples:
        encoded = encode(s, merges)
        decoded = decode(encoded, vocab)
        assert decoded == s, f"Round-trip failed for: {s!r} (got {decoded!r})"


def test_vocab_size_assertion():
    """Calling train with vocab_size < 256 must raise."""
    try:
        train("hello", vocab_size=100)
    except AssertionError:
        return
    raise AssertionError("Should have raised AssertionError for vocab_size < 256")


def test_empty_input():
    merges, vocab = train("hello", vocab_size=300)
    assert encode("", merges) == []
    assert decode([], vocab) == ""


def test_english_compression():
    """A reasonably-sized corpus should produce meaningful compression."""
    text = "the quick brown fox jumps over the lazy dog. " * 100
    merges, vocab = train(text, vocab_size=400)

    sample = "the quick brown fox"
    encoded = encode(sample, merges)
    n_bytes = len(sample.encode("utf-8"))
    # Should compress at least 1.5x on this repetitive corpus
    assert n_bytes / len(encoded) > 1.5, (
        f"Compression too low: {n_bytes} bytes / {len(encoded)} tokens"
    )


def test_minimum_vocab_size():
    """vocab_size = 256 means zero merges (legal, useful as baseline)."""
    merges, vocab = train("hello world", vocab_size=256)
    assert merges == {}
    assert len(vocab) == 256
    # Should still round-trip
    encoded = encode("hello world", merges)
    decoded = decode(encoded, vocab)
    assert decoded == "hello world"
    # And should be exactly one token per byte
    assert len(encoded) == len("hello world".encode("utf-8"))


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests...\n")
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            raise
    print(f"\nAll {len(tests)} tests passed.")
