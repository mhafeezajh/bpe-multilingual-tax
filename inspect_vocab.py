"""Inspect the 256-byte base vocabulary of byte-level BPE.

Shows what every initial BPE token represents — BEFORE any merge training.
These 256 tokens are always present; learned merges start at ID 256.

Run with: python inspect_vocab.py
"""


def display_byte(i: int) -> str:
    """Return a readable description of what byte value `i` represents."""
    b = bytes([i])
    try:
        ch = b.decode("utf-8")
        if ch.isprintable():
            return f"{ch!r}"
        return "(control)"
    except UnicodeDecodeError:
        return "(UTF-8 fragment)"


def main():
    print("=" * 70)
    print("The 256-byte base vocabulary of byte-level BPE")
    print("=" * 70)
    print(
        "\nEvery byte-level BPE tokenizer starts with these 256 tokens — one per\n"
        "possible byte value (0 to 255). Learned merges add new tokens starting\n"
        "at ID 256. This base is what gives byte-level BPE its zero-OOV guarantee:\n"
        "every conceivable UTF-8 string is encodable because every byte has a token.\n"
    )

    sections = [
        (0, 32,    "Control characters (non-printable: newline, tab, etc.)"),
        (32, 127,  "Printable ASCII (space, digits, letters, punctuation)"),
        (127, 128, "DEL (delete control character)"),
        (128, 192, "UTF-8 continuation bytes — never appear alone in valid UTF-8"),
        (192, 224, "UTF-8 lead bytes for 2-byte chars (Arabic, Hebrew, Cyrillic, Greek)"),
        (224, 240, "UTF-8 lead bytes for 3-byte chars (Hindi, CJK, Thai, Korean)"),
        (240, 248, "UTF-8 lead bytes for 4-byte chars (emoji, rare CJK, math symbols)"),
        (248, 256, "Reserved/illegal in modern UTF-8 (token still exists)"),
    ]

    for start, end, label in sections:
        print(f"\nBytes {start}–{end - 1}: {label}")
        print("-" * 70)
        # Show first 8 in each section, then a summary
        for i in range(start, min(start + 8, end)):
            b = bytes([i])
            print(f"  token {i:3d}  byte {b!r:6s}  ->  {display_byte(i)}")
        if end - start > 8:
            print(f"  ... ({end - start - 8} more in this range)")

    print("\n" + "=" * 70)
    print("Key takeaway: every UTF-8 string is encodable, but multi-byte characters")
    print("cost multiple tokens BEFORE any merge training. Arabic chars are 2 tokens.")
    print("Hindi/CJK chars are 3 tokens. Emoji are 4 tokens. Training only adds")
    print("compression for byte sequences common in the training corpus.")
    print("=" * 70)


if __name__ == "__main__":
    main()
