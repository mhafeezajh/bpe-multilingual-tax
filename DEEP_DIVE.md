# Understanding BPE: A Deep Dive Into the Multilingual Tokenization Tax

*A walkthrough of byte-level Byte Pair Encoding — what it is, how it works, why the same algorithm makes your LLM bill 3–5× larger in any non-English language, and what to do about it architecturally.*

*Approx. 20-minute read · Companion code: [bpe-multilingual-tax](https://github.com/) (link to your repo)*

---

## The puzzle

Here's a question that comes up every time someone deploys a GenAI product outside the Anglosphere — whether it's an Arabic-language chatbot in Dubai, a Hindi customer-support agent in Bangalore, a Mandarin product assistant in Shenzhen, or a Japanese contract reviewer in Tokyo:

> *Why does the same product cost 3–5× more in our language than it does in English?*

The natural guesses are all wrong. It's not that the language is harder for the model. It's not that the model is worse at the language. It's not that response generation takes longer. None of those things are even substantially true.

The answer lives one layer below the model — in the **tokenizer**. And once you understand it, you see the same effect everywhere: Arabic costs more, Hindi costs more, Chinese costs more, Japanese costs more, Korean costs more, Cyrillic costs more, emoji-heavy text costs more. Every language that isn't English-dominant pays a tax measured in tokens, every API call, forever.

This article walks through the whole story end-to-end: what BPE is, why it has the 256-byte foundation that's so important, why English-trained tokenizers can never learn the merges for other languages, what happens at inference time when you actually hit `send`, and what to do about it architecturally.

I've implemented the algorithm from scratch — about 120 lines of pure Python, no dependencies — and you can run the experiment yourself with the companion repo. I used Arabic as the worked example because it's the language I had the cleanest comparison corpora for; the measured ratio on my run was **4.81× more tokens** via an English-trained tokenizer encoding Arabic, compared to an Arabic-trained tokenizer encoding the same text. That ratio varies from ~1.5× (Spanish, French) to ~8× (Hindi) to ~10× (heavy emoji) depending on your target language. The mechanism is the same; only the magnitude differs.

---

## Part 1 — What BPE actually is

Byte Pair Encoding has a curiously layered history. It was invented in 1994 by Philip Gage as a simple **data compression** technique for replacing the most common pair of consecutive bytes in a file with a byte that doesn't appear in it. It then lived a quiet life until 2016, when Rico Sennrich, Barry Haddow, and Alexandra Birch repurposed it for neural machine translation in their paper "Neural Machine Translation of Rare Words with Subword Units."

The 2016 version operated on Unicode characters. It worked, but it had ugly failure modes for emoji and rare scripts — characters the algorithm hadn't seen in training could fail to tokenize cleanly.

In 2019, OpenAI's GPT-2 paper introduced **byte-level BPE**: the same algorithm, but running on raw UTF-8 bytes instead of Unicode characters. The base vocabulary became a fixed 256 (one token per possible byte value), and every conceivable UTF-8 string became encodable — no exceptions. This is the variant used by virtually every production decoder-only model since: GPT-3, GPT-4, the Llama family, Mistral, Qwen, DeepSeek, and so on.

### The training algorithm in plain English

Take a corpus of training text. Encode it as UTF-8, then view the bytes as a list of integers in `[0, 256)`. Now repeat the following until you've learned as many merges as you want:

1. Count how often each adjacent pair of integers appears in the current list.
2. Find the most frequent pair.
3. Assign it a new token ID, starting at 256 and going up.
4. Replace every occurrence of that pair in the list with the new ID.
5. Record the merge: "pair (a, b) → new ID."

That's the entire algorithm. About thirty lines of Python.

Here's what the Python looks like, with annotation:

```python
def train(text, vocab_size):
    num_merges = vocab_size - 256
    ids = list(text.encode("utf-8"))
    
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}   # the base byte vocab
    
    for i in range(num_merges):
        stats = get_stats(ids)                    # count adjacent pairs
        if not stats: break
        pair = max(stats, key=stats.get)          # most frequent pair
        new_id = 256 + i
        ids = merge(ids, pair, new_id)            # replace in the corpus
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    
    return merges, vocab
```

### The encoding algorithm

Given trained `merges`, to encode new text: UTF-8-encode it, then repeatedly apply the earliest-learned merge whose pair appears in the current sequence. Stop when no learned merge applies.

```python
def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges: break
        ids = merge(ids, pair, merges[pair])
    return ids
```

The subtle bit is "earliest-learned merge." Each merge has an index (its ID, since IDs start at 256 and increment by one per merge). The earliest-learned merge has the lowest index. We pick that one first because it's the merge that was most frequent in training — and applying merges in training order is how we reproduce the same tokenization the model saw.

### The decoding algorithm

Trivially simple. Look up each token ID in `vocab`, get its bytes, concatenate, decode as UTF-8:

```python
def decode(ids, vocab):
    raw_bytes = b"".join(vocab[i] for i in ids)
    return raw_bytes.decode("utf-8", errors="replace")
```

`errors="replace"` is for development; in production you'd usually want `errors="strict"` so decoding errors surface as exceptions rather than silent corruption.

---

## Part 2 — The 256-byte foundation

The single most important design choice in byte-level BPE is the initial vocabulary: **always exactly 256 tokens, one per possible byte value**.

This isn't an arbitrary number. It's the contract that makes byte-level BPE work the way it does. The guarantee:

> Every conceivable UTF-8 string is encodable as some sequence of tokens, because every byte is already a token. Learned merges only ever add **compression**; they never enable encoding.

This property — formally called **zero-OOV (zero out-of-vocabulary)** — is what distinguishes byte-level BPE from character-level BPE (the original 2016 variant) and from word-level tokenization. Earlier algorithms could fail catastrophically on inputs they hadn't seen in training. Byte-level BPE cannot, because there's no such thing as "an unknown byte" — every byte is in `[0, 255]` by definition.

### What's actually in each of the 256 base tokens

The 256-byte vocabulary partitions cleanly into regions based on UTF-8 mechanics:

| Byte range | What's there |
|---|---|
| **0–31** | Control characters (newline `\n`, tab `\t`, carriage return, escape, NUL, etc.) |
| **32–126** | Printable ASCII (space, digits, punctuation, A–Z, a–z, `~`) |
| **127** | DEL (a control character at the edge of ASCII) |
| **128–191** | **UTF-8 continuation bytes** — never appear alone in valid UTF-8; always come *after* a lead byte |
| **192–223** | **UTF-8 lead bytes for 2-byte chars** — Arabic, Hebrew, Cyrillic, Greek, Latin extensions |
| **224–239** | **UTF-8 lead bytes for 3-byte chars** — Hindi, CJK (Chinese / Japanese / Korean), Thai, Korean Hangul, Bengali |
| **240–247** | **UTF-8 lead bytes for 4-byte chars** — emoji, mathematical symbols, ancient scripts |
| **248–255** | Reserved/illegal in modern UTF-8 (token still exists for completeness) |

So the lowercase `a` is token 97 (its ASCII value). Space is token 32. A newline is token 10. But what's token 216? It's the first byte of every Arabic character — the lead byte `0xD8` — which, on its own, has no Unicode meaning. It only completes a character when followed by a continuation byte from the 128–191 range.

Run `python inspect_vocab.py` in the companion repo to see all 256 with their byte values and (where applicable) printable representations.

### The cost of multi-byte characters

Here's where the multilingual tax begins to take shape. UTF-8 encodes characters in 1 to 4 bytes:

| Character | UTF-8 encoding | Base tokens used |
|---|---|---|
| `a` | `0x61` (1 byte) | **1 token** |
| `é` | `0xC3 0xA9` (2 bytes) | **2 tokens** |
| `ا` (Arabic alif) | `0xD8 0xA7` (2 bytes) | **2 tokens** |
| `中` (Chinese) | `0xE4 0xB8 0xAD` (3 bytes) | **3 tokens** |
| `नम` (Hindi nama) | `0xE0 0xA4 ...` (3 bytes each) | **3 tokens each** |
| `🚀` (rocket) | `0xF0 0x9F 0x9A 0x80` (4 bytes) | **4 tokens** |

Before BPE training has done anything, every Arabic character is already 2 tokens. Every Chinese / Japanese / Korean / Hindi character is already 3. Every emoji is already 4. These costs are inherent to UTF-8 — the tokenizer just reflects them.

Training *can* compress this. If the algorithm learns the merge `(0xD8, 0xA7) → some new ID`, then Arabic alif becomes 1 token instead of 2. But that compression only happens if the byte pair was frequent enough in the training corpus to win the frequency competition. Which brings us to the central problem.

---

## Part 3 — What breaks if you cut the foundation short

Before going there, let's understand why the 256-byte base is non-negotiable by looking at what happens if you tried to skimp on it.

Imagine someone designed a "compact" BPE variant where the base vocab is only 200 tokens. The reasoning sounds plausible: "bytes 200–255 are rare anyway, why waste 56 token slots?"

Here's what actually breaks.

### Example 1: Pure English — appears to work fine

```python
broken_encode("hello world", merges)
# Every byte is < 200. No crash. Tests pass.
```

This is the dangerous part. **The bug is invisible until non-ASCII data hits production.**

### Example 2: An emoji — crash

The rocket emoji `🚀` is `0xF0 0x9F 0x9A 0x80` — four bytes, all `>= 200`. The encoder produces token IDs that don't exist in the vocab:

```python
broken_encode("I love 🚀 rockets")
# Returns IDs including 240, 159, 154, 128 — not in our 200-vocab

broken_decode([..., 240, 159, 154, 128, ...], vocab)
# KeyError: 240
```

A loud, obvious error. Bad, but at least visible.

### Example 3: Any non-ASCII language — silent corruption

Every Arabic character starts with a lead byte in the range 216–217. Every Hindi character starts with a lead byte in the 224–239 range. Every Chinese character does too. Every Cyrillic character starts at 208–215. All of these are `>= 200`. The "compact" tokenizer crashes on every non-English input.

Now imagine someone "fixes" this by silently dropping unknown IDs:

```python
def broken_decode_lenient(ids, vocab):
    return b"".join(vocab[i] for i in ids if i in vocab).decode("utf-8", errors="replace")
```

The user types "السلام" (Arabic for peace) or "नमस्ते" (Hindi for hello) or "你好" (Chinese for hello). The decoder drops all the lead bytes (which are >= 200), leaves only the continuation bytes (128–191, which are in vocab), and tries to decode them as UTF-8. They're not valid UTF-8 on their own, so the user receives `������` — replacement characters. **No exception. No log. No alert.**

Three weeks later, a support ticket arrives asking why all the non-English chat history has turned into question marks.

This is the failure mode the 256-byte base prevents. It's the exact bug that broke pre-2019 tokenizers and is the reason GPT-2 specifically switched to byte-level. The base vocabulary isn't a number someone picked because it was round. It's a contract that says *we can encode anything in any language, no exceptions.*

---

## Part 4 — The frequency auction

Now we get to the heart of why the multilingual tax exists.

BPE training is a **greedy, frequency-driven competition**. At every step, the algorithm asks: "Of all the byte pairs in my training data, which one appears most often right now?" That pair wins the round and becomes a new token. Then the algorithm moves on to the next round. And the next. And the next.

Only one pair wins per round. The competition is **winner-take-all**, every round, for the entire training run.

So when we say "English-heavy training data," what we really mean is: **all 100,000+ rounds of this competition were won by English byte pairs**, because they outnumbered everything else.

### A concrete numerical example

Imagine your training corpus is 1 GB of text, 95% English and the remaining 5% split across all other languages combined — roughly representative of the open web.

- The English byte pair `(101, 32)` — that's `'e '` (lowercase e followed by space) — appears maybe **30 million times** in the corpus.
- The most common Arabic byte pair (say `(216, 167)` — the encoding of the Arabic letter ا "alif") might appear **200,000 times**.
- The most common Hindi byte pair, even rarer in the corpus, might appear **50,000 times**.
- The most common Chinese byte pair might appear **100,000 times**.

When the training loop calls `max(stats, key=stats.get)`, the English pair with 30 million wins. Every non-English pair loses by factors of 100–600×.

So merge #1 is some English bigram. Merge #2 is another English bigram. Merge #3, the same. By merge #1,000, your tokenizer has learned 1,000 merges, **all of them English**. The Arabic alif, the Hindi devanagari, the Chinese ideographs have *never* won a round. They're still encoded as their raw bytes — 2, 3, 3 tokens per character respectively, exactly the same as before training started.

### "But surely it'd eventually get there?"

This is the natural follow-up — and the answer is *no, it almost certainly won't*. Here's why.

As English merges accumulate, English compound tokens get *created*. The merge `('t', 'h') → 'th'` becomes a single token. Then `(th, 'e') → 'the'`. Then `(' ', 'the') → ' the'`. Each of these compound tokens then participates in **new pair competitions** at the next iteration. New pairs with millions of occurrences keep appearing on the English side, just from the algorithm's own work.

Non-English pairs, capped by the corpus's 5% non-English content, cannot keep up. **The English merges literally generate more competition for themselves and keep winning.** This is the runaway dynamic that makes the imbalance worse over time, not better.

By the time you've reached merge 50,000 (the vocab size of GPT-2) or 100,000 (`cl100k_base`) or 200,000 (`o200k_base`), you might *finally* see a handful of non-English merges — usually only the most common Arabic prefix patterns, the most common Hindi conjuncts, the most common CJK ideographs. But the *vast majority* of the non-English byte pairs are still uncompressed. The tokens spent encoding non-English text remain dominated by raw-byte costs.

This isn't hypothetical. Go look at the actual `cl100k_base` vocabulary, which is public. You'll find tens of thousands of English word fragments, common HTML/code tokens, Reddit slang, even Stack Overflow answer patterns. You'll find a small fraction allocated to non-English content, and the distribution is heavily skewed by training data prevalence — more for European languages and Chinese (well-represented on the open web), less for Arabic and Indic scripts (more under-represented).

### Why this can't be retroactively fixed

Once the tokenizer is trained and shipped, **it never updates**.

This is a critical architectural fact most people don't realize. GPT-4's `cl100k_base` was trained once, around 2022. GPT-4 itself has been retrained, fine-tuned, and updated many times since — but the *tokenizer* stays the same. Why?

Because changing the tokenizer would change every token ID. Which would invalidate every learned embedding in the model. Which would mean retraining the entire model from scratch. The tokenizer sits underneath every layer of the model and is the foundational layer of how token IDs map to learned representations.

So even if OpenAI decided tomorrow "we should balance our tokenizer for non-English content," they can't — not without throwing away GPT-4 and starting over. The English-heavy auction results from 2022 are baked in for the lifetime of every model that was trained on top of them.

**This is why the multilingual tax is a *structural* property, not a fixable bug.** It was decided at tokenizer training time, years before any customer ever paid a bill.

### Why no one rebalanced it on purpose

You might fairly ask: didn't anyone notice this when designing the tokenizer? Yes — and some teams deliberately chose to address it. Cohere's Command-R+, the Aya-23 model family, Qwen models from Alibaba, and BLOOM are all explicitly multilingually-trained. Their tokenizers see balanced training corpora across many languages, so their merges distribute across scripts more evenly.

OpenAI, Anthropic, and Meta have all prioritized English performance in their tokenizer training. That's a defensible choice — English is the dominant language of the open web and most enterprise customers — but the cost lands on every non-English customer downstream. The cost is the multilingual tokenization tax.

---

## Part 5 — How the tax varies by language

The mechanism is the same for every non-English language, but the magnitude of the tax differs significantly. Two factors compound:

1. **UTF-8 byte cost per character.** Latin-with-accents is 1–2 bytes; Cyrillic/Arabic/Hebrew is 2 bytes; CJK and Devanagari are 3 bytes; emoji are 4 bytes.
2. **Training-corpus representation.** European languages with Latin alphabets are heavily represented on the open web. CJK is well-represented. Arabic and Indic scripts are under-represented. Emoji are essentially absent except as one-offs.

Combining the two:

| Language family | Typical ratio vs English | Why |
|---|---|---|
| Spanish, French, German, Portuguese | **1.3–2×** | Latin-with-accents = 1–2 bytes; well-represented in training |
| Russian, Bulgarian, Greek | **2–3×** | Cyrillic/Greek = 2 bytes; moderately represented |
| Arabic, Hebrew, Persian | **3–5×** | 2 bytes + significantly under-represented |
| Chinese, Japanese, Korean | **3–6×** | 3 bytes but well-represented; the byte cost dominates |
| Hindi, Bengali, Tamil, other Indic | **4–8×** | 3 bytes + heavily under-represented in training |
| Emoji-heavy text, ancient scripts | **4–10×+** | 4 bytes + essentially absent from training |

So a Spanish customer-support deployment might see a 1.5× cost premium versus English. A Hindi deployment might see 6×. A Chinese deployment might see 4×. An emoji-heavy creative-writing application might see 8×.

These ratios are not invariant — they depend on the specific tokenizer (cl100k_base vs Llama vs Qwen) and the specific content (formal text tokenizes differently from informal). But the *direction* is consistent, the *floor* is roughly 1.5×, and the *ceiling* is shockingly high for under-represented scripts.

Take any user-facing prompt in your target language. Tokenize it through your candidate model's tokenizer. Compare the count to the same content's English translation. The ratio is your tax for that language, on that workload, today.

---

## Part 6 — What happens when you send a prompt

So the tokenizer is frozen. The auction results are baked in. Now what actually happens when you send a prompt to GPT-4 or Claude?

This is the part most engineers don't internalize: **no training happens at inference time. Ever.** The tokenizer is a static lookup table — `merges.txt` + `vocab.json`, totaling maybe 5–10 MB on disk — that's loaded into memory when the server starts. From then on, every encoding call is just a fast lookup against frozen state.

Here's the full timeline of a typical 100-word prompt asking for a 500-word answer (in any language):

```
T+0ms        Client sends request
T+50ms       Server receives, deserializes the JSON
T+52ms       Tokenize input: 100 words → N tokens          (~2ms — negligible)
T+55ms       Embedding layer: N token IDs → N vectors
T+200ms      Prefill phase: process all N tokens through the transformer.
             KV cache populated. First output logit produced.
T+250ms      First output token sampled, decoded, streamed to client   ← TTFT
T+275ms      Next decode step: one token through transformer (~25ms each)
             Decode the new token (vocab lookup) → stream to client
T+300ms      ...and another...
...
T+18s        ~500th output token emitted, EOS detected
T+18s        Server closes stream
```

The crucial difference for non-English deployments: **N is much larger than the English equivalent**. A 100-word English prompt might be ~130 tokens. A 100-word Hindi or Arabic prompt of the same semantic content might be ~500–800 tokens. The model has to prefill more, decode more on output, and you pay for every one.

Two important observations:

**First, tokenization itself is fast and not the bottleneck.** Encoding even 800 tokens takes about 5 ms — still single-digit milliseconds on a single CPU core. The model's forward passes (prefill at hundreds of ms, then ~25 ms per output token) dominate the timeline by two orders of magnitude. Tokenization is in the noise.

**Second, the cost is decided before any model work begins.** The moment encoding finishes, you know exactly how many input tokens you'll be charged for. The output tokens you don't know in advance, but the input tokens — the typically larger and more controllable half — are fully determined at encoding time. This makes tokenization a powerful **FinOps lever**: count tokens client-side, predict costs before sending, and reject or restructure prompts that exceed budget thresholds.

### The pre-tokenization regex

One detail worth knowing: modern production tokenizers don't run BPE directly on the raw byte stream. They first split the text on a regex pattern so that merges cannot cross certain boundaries (like spaces, contractions, or digit/letter boundaries).

GPT-4's `cl100k_base` uses (simplified):

```
'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
```

That splits text into chunks like `["Explain", " how", " attention", " works"]`, and BPE merges only happen within each chunk. This is why you see space-prefixed tokens like ` the` and ` of` in real tokenizers — the regex keeps a leading space attached to most words.

The repo in this article implements the *core* BPE without the regex pre-split, for clarity. Production tokenizers add the pre-split layer for ~10-20% better compression on natural text.

---

## Part 7 — The experiment

To make the tax concrete, I trained two BPE tokenizers, each on 100 KB of corpus with vocab size 512 (256 base + 256 learned merges).

- **Tokenizer A** on English Shakespeare.
- **Tokenizer B** on Arabic Wikipedia.

Then I encoded the same 1 KB of Arabic text through both and counted tokens. (Arabic was chosen as the worked example because I had clean parallel corpora; the same experiment works for any language — replace `arabic_sample.txt` with Hindi, Chinese, Japanese, etc.)

Results from my run:

```
Arabic sample: 1000 chars, 1842 bytes

  via English-trained tokenizer:  1838 tokens  (1.00 bytes/token)
  via Arabic-trained tokenizer:    382 tokens  (4.82 bytes/token)

  RATIO: 4.81× more tokens via English tokenizer.
```

The English-trained tokenizer barely compresses Arabic at all (1.00 bytes/token means essentially one token per byte — almost no merges applied). The Arabic-trained tokenizer compresses to 4.82 bytes/token, comparable to what an English-trained tokenizer achieves on English (~3.5–4 bytes/token on much larger production tokenizers).

**Multiply that 4.81 ratio by your $/1M token rate and you have the invoice difference.** For an enterprise customer running a customer-support chatbot, the difference between a $1,000/month bill and a $4,810/month bill is precisely this ratio.

The exact number will vary by language and corpus, but the pattern is remarkably stable. Run the experiment with content from your target language using the [companion repo](https://github.com/) — it takes about 30 seconds. Compare your number against the table in Part 5.

---

## Part 8 — Architectural implications

Knowing the mechanism, what should you actually *do*? Here are the practical levers, ordered by impact.

### 1. Model selection is the biggest lever

If you have flexibility on model choice, this is where the architectural decision pays off. Models trained with multilingual tokenizers — **Cohere Command-R+**, **Aya-23**, **Qwen 2.5**, the **Gemma** family (uses SentencePiece), and **BLOOM** — tokenize non-English text 2–4× more efficiently than English-heavy tokenizers like `cl100k_base`.

For an Arabic-heavy or Hindi-heavy workload, you can sometimes get equal-or-better task performance at one-third the per-character cost just by switching models. Measure this in your specific domain before committing.

### 2. Measure before you architect

The official tokenizer endpoints — OpenAI's tokenizer playground, Anthropic's token counter, Hugging Face's `AutoTokenizer` — all let you count tokens client-side without making a billable API call. Take a representative 10 KB sample of your real customer data in your real target language and tokenize it through each candidate model. The numbers are deterministic and reproducible. Pick on numbers, not vendor marketing.

### 3. Context windows shrink — plan accordingly

A "128K context window" gives you 128K *tokens*, not characters. In English, 128K tokens is roughly 250K characters or 500 pages. In Hindi or Arabic, the same 128K tokens might be only 30–60K characters or 60–120 pages of equivalent content. In Chinese, somewhere in between. Every heuristic you tuned for English RAG (chunk size, retrieval k, max input length) needs to be re-evaluated for your target language.

### 4. Throughput drops proportionally

Decode is sequential — the model emits one token per forward pass. If your language takes 3× more tokens for the same semantic content, your perceived tokens-per-second drops by 3× for the same model on the same hardware. SLO budgets need to reflect this.

### 5. Prompt caching becomes mandatory

Anthropic's prompt caching and OpenAI's automatic prefix caching offer flat-percentage discounts on cached input tokens (90% off cached reads at Anthropic; ~50% off at OpenAI). Because the absolute cost is higher in non-English, the absolute savings from caching are also higher. Cache system prompts, RAG context, and few-shot examples aggressively. In non-English deployments, caching shifts from a nice-to-have to a financial necessity.

### 6. Sovereign-cloud and edge inference may flip the math

For deployments where data residency matters (UAE, KSA, Qatar, India, China), running self-hosted inference on multilingual-tokenizer open models (Aya-23, Qwen, BLOOM) on regional GPUs converts the per-token charge into a per-GPU-hour charge. For high-volume non-English workloads, the math often favors self-hosting outright — especially when you also factor in regulatory compliance.

### 7. Communicate in the right vocabulary to your stakeholders

If you're an engineer pitching a model swap to a CFO, "we'll reduce token cost by 4×" is worse than "this will reduce our per-user inference cost from $X to $Y per month." Always re-denominate technical gains into the units your audience uses for decisions.

---

## The takeaway

The multilingual tokenization tax is not a model bug. It is not a fixable problem in the model layer. It is **a structural property of how byte-level BPE tokenizers are trained**, specifically: a greedy frequency-driven auction that was won, every round, by English byte pairs because English was overrepresented in the training corpus.

This decision was made years ago, when the tokenizer was trained. It was frozen at model release. It propagates into every API call, every invoice, every cost forecast, every context window budget, and every latency SLO for every non-English deployment in every market.

The fix is not "wait for OpenAI to update the tokenizer" — they can't without retraining the entire model. The fix is **architectural awareness**: pick models with appropriate tokenizers for your workload, measure before committing, plan context windows in the right units, lean hard on prompt caching, and consider sovereign-cloud / self-hosted inference when the math justifies it.

If you're building anything in Arabic, Hindi, Persian, Chinese, Japanese, Korean, Hebrew, Greek, Thai, Russian, or any non-English-heavy language, this article is the math you need on your invoice line. Run the experiment with your own data. Measure your number. Make the decisions on the basis of that number, not on the basis of English benchmarks projected onto a different language.

---

## Code and further reading

- The implementation: [bpe-multilingual-tax](https://github.com/) — 120 lines, no dependencies. Clone it, run it.
- For the canonical reference implementation: [Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe) plus his [tokenizer video](https://www.youtube.com/watch?v=zduSFxRajkE).
- For the original byte-level BPE paper: GPT-2 (Radford et al., 2019).
- For the 2016 NMT-era BPE: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016).
- For an alternative algorithm with different multilingual properties: SentencePiece (Kudo & Richardson, 2018) — used by T5, mT5, Gemma.

---

*By Mohammed Hafeez Ali Khan — Senior AI Solutions Architect.*
*If you found this useful, follow on [LinkedIn](https://linkedin.com/in/hafeezaj/) for more deep-dives on enterprise GenAI architecture, RAG, MCP, and inference internals.*
