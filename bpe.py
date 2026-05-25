"""
bpe.py — A minimal byte-level BPE tokenizer in pure Python.

Implements training, encoding, and decoding for byte-level Byte Pair Encoding
(the same algorithm family as GPT-2, GPT-4 cl100k_base, Llama, Mistral, Qwen).

No external dependencies. ~120 lines including docstrings.

Usage:
    from bpe import train, encode, decode

    merges, vocab = train("your training text", vocab_size=512)
    ids = encode("new text", merges)
    out = decode(ids, vocab)

Companion repo to the deep-dive article: see DEEP_DIVE.md.
"""

from typing import Dict, List, Tuple


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Count consecutive pair frequencies in a list of token IDs.

    Args:
        ids: A list of integer token IDs.

    Returns:
        A dict mapping each adjacent (a, b) pair to its count.

    Example:
        >>> get_stats([1, 2, 3, 1, 2])
        {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts: Dict[Tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every occurrence of `pair` in `ids` with the single token `new_id`.

    Pairs are matched left-to-right, non-overlapping.

    Args:
        ids: The current list of token IDs.
        pair: The (a, b) pair to replace.
        new_id: The token ID to replace it with.

    Returns:
        A new list of token IDs with the pair replaced.

    Example:
        >>> merge([1, 2, 3, 1, 2], (1, 2), 99)
        [99, 3, 99]
    """
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(
    text: str,
    vocab_size: int,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, bytes]]:
    """Train a byte-level BPE tokenizer on `text`.

    Args:
        text: The training corpus as a UTF-8 string.
        vocab_size: Target vocabulary size. Must be >= 256.
        verbose: If True, print each merge as it is learned.

    Returns:
        A tuple (merges, vocab) where:
        - merges: dict mapping (a, b) -> new_id, in the order learned.
        - vocab: dict mapping every token ID (0 to vocab_size-1) to its bytes.

    Raises:
        AssertionError: If vocab_size < 256.

    Note:
        The first 256 token IDs are always the raw byte values 0-255.
        Merges start at ID 256. This is what guarantees byte-level BPE can
        encode any UTF-8 string (the zero-OOV property).
    """
    assert vocab_size >= 256, "vocab_size must be >= 256 (the byte-level base vocab)"
    num_merges = vocab_size - 256

    # Encode text as UTF-8, convert to list of ints in [0, 256)
    ids = list(text.encode("utf-8"))

    # Initialize: every byte value gets a token; merges start empty
    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break  # Corpus too short to learn any more merges

        # argmax: ties broken by dict insertion order (Python 3.7+)
        pair = max(stats, key=stats.get)
        new_id = 256 + i

        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
            print(
                f"  merge {i + 1:4d}/{num_merges}: "
                f"{pair} -> {new_id} "
                f"({vocab[new_id]!r}), count={stats[pair]}"
            )

    return merges, vocab


def encode(text: str, merges: Dict[Tuple[int, int], int]) -> List[int]:
    """Encode `text` using the trained merges.

    Applies merges in the order they were learned during training:
    earliest-learned merges (highest frequency at train time) have highest priority.

    Args:
        text: The text to encode.
        merges: The merges dict returned by train().

    Returns:
        A list of token IDs.
    """
    ids = list(text.encode("utf-8"))

    while len(ids) >= 2:
        stats = get_stats(ids)
        # Smallest merge index = earliest learned = highest priority.
        # Unlearned pairs get infinity priority so they are never picked.
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # No remaining pair has been learned
        ids = merge(ids, pair, merges[pair])

    return ids


def decode(ids: List[int], vocab: Dict[int, bytes]) -> str:
    """Decode a list of token IDs back to a string.

    Args:
        ids: The list of token IDs to decode.
        vocab: The vocab dict returned by train().

    Returns:
        The decoded string. Invalid UTF-8 sequences are replaced with U+FFFD
        (the replacement character). Switch to errors='strict' in production
        if you want decoding errors to surface as exceptions.
    """
    raw_bytes = b"".join(vocab[i] for i in ids)
    return raw_bytes.decode("utf-8", errors="replace")
