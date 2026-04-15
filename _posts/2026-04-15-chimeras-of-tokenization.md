---
layout: post
title: "The Chimeras of Tokenization"
author: "Pradipta Mitra"
date: 2026-04-15
---

The Byte Pair Encoding (BPE) method is odd!

## Tokenization in a nutshell

[Karpathy's video](https://youtube.com/watch?v=zduSFxRajkE) on the topic is marvelous.

BPE is an algorithm to perform *tokenization* for natural language processing. Tokenization
is the process of building the atomic units (**tokens**) the language model would work with.
You could simply do no work and use characters *or* words as tokens, but practitioners
have found it useful to construct an optimized "alphabet" for language models, that sits
somewhere between a character and a word (a word turns out to be about 2-3 tokens). GPT-2,
for example, has an "alphabet" of 50,257 tokens.

BPE (byte pair encoding), which is one such algorithm, builds
its token vocabulary by starting from an alphabet of atomic symbols and
repeatedly merging the most frequent adjacent pair. You keep merging until you
hit a target vocabulary size. 

Assume you run BPE on top of ASCII characters, and as a concrete example on the string `aaabdaaabac`:

```
Start:  a a a b d a a a b a c       vocabulary: {a, b, c, d}

Most frequent pair: (a, a) — appears 4 times → merge into Z
        Z a b d Z a b a c           vocabulary: {a, b, c, d, Z}

Most frequent pair: (Z, a) — appears 2 times → merge into Y
        Y b d Y b a c               vocabulary: {a, b, c, d, Z, Y}

Most frequent pair: (Y, b) — appears 2 times → merge into X
        X d X a c                   vocabulary: {a, b, c, d, Z, Y, X}
```

Each merged token has a history — a binary tree of merges that produced it:

```
  a   a   a   b
   \ /    |   |
   Z(aa)  a   b
     \   /    |
     Y(aaa)   b
        \    /
        X(aaab)
```

After all merges, the vocabulary
contains every node in every merge tree (this is useful to note -- 'a', 'Z' etc are retained
as tokens).

As it happens, BPE doesn't start from characters. It starts one level lower:
**bytes** (thus "Byte" pair encoding).

## Bytes vs. characters

The most common unicode encoding, UTF-8, is a variable length encoding. Characters occupy between 1 and 4 bytes:

| Character | Script | UTF-8 bytes | Length |
|-----------|--------|-------------|--------|
| `a` | Latin | `61` | 1 |
| `é` | Latin extended | `c3 a9` | 2 |
| `م` | Arabic | `d9 85` | 2 |
| `न` | Devanagari (Hindi) | `e0 a4 a8` | 3 |
| `น` | Thai | `e0 b8 99` | 3 |
| `你` | CJK (Chinese) | `e4 bd a0` | 3 |
| `🙏` | Emoji | `f0 9f 99 8f` | 4 |


The starting 127 are the common ASCII characters, followed by the expanded character set.
To avoid decoding ambiguity UTF-8 has a prefix structure. The first 4 bits encoding a character
provides its length, so you know how many bytes to decode before a new character starts.
 

While not strictly necessary for en/decoding, it also tends to be the case that characters from the same script share leading bytes:


```
Every Thai character:         e0 b8 ··
Every Bengali character:      e0 a6 ··  or  e0 a7 ··
Every emoji in U+1F000–1FFFF: f0 9f ·· ··
```



## BPE on bytes

Thus BPE starts with 256 tokens, e.g. all possible byte values. For English text,
this is equivalent to starting from characters — the byte `74` *is* the letter
`t`. The merge tree for the word "the" is straightforward:

```
  74(t)  68(h)  65(e)
    \   /        /
   [74 68](th)  /
       \       /
    [74 68 65](the)
```

But for a Chinese character like 你, BPE starts from three bytes that are
individually meaningless:

```
  e4   bd     a0    three bytes encoding 你 (U+4F60)
   \ /       / 
 [e4 bd]    /          ← still invalid UTF-8
     \     /
  [e4 bd a0]            ← valid: the character 你
```


There are a couple of things to note here. 

**First**, in spite of forming the valid character 你 eventually,
the invalid intermediaries remain as tokens. This opens the possibility of nonsensical
output at the generative phase -- the LLM *can* generate [e4 bd] followed by [e0 a6] (prefix of a Bengali character). If it does so, it can't be decoded, and we simply deal with it with hardcoded rules (e.g. dropping it, replacing with a filler character, etc). But one hopes
that this would be (very) rare, since such sequences are never present in the training data.

**Second**, it's possible that we don't form the [e4 bd a0] token at all. BPE stops after
a certain number of merges, so relatively infrequent characters may have part of their characters formed into prefixes. 

Let's assume, for illustration, that tokenization stopped after the [e4 bd] merge. This is
not the end of the world, since generation could still do the right thing by simply outputting [e4 bd] [a0] when appropriate. With enough training data, this is likely, but then again, the premise we started out with is that [e4 bd a0] token wasn't formed because it is *relatively* infrequent.

## Chimeras at construction time
Our focus in this article is not loose prefixes or suffixes like [e4 bd] or [a0], nor the generation of decodable output as such. Rather, we will concern ourselves with the *possibility* of more complex invalid constructions during tokenization.

For example, could it be the case that BPE merges bytes
that cross a character boundary — the tail of one character with the head of
the next?

Picture the byte stream for a Hindi word followed by an emoji:

```
Character:  ...    न              🪔          ...
UTF-8:      ... e0 a4 a8     f0 9f aa 94     ...
                        ↑    ↑
                        tail head
```

If `a8` (the final byte of न) merges with `f0` (the first byte of 🪔), the
result is `[a8 f0]` — the tail of a Hindi consonant welded onto the beginning
of a Diwali lamp. A **chimera**: one token, two unrelated characters. While you could still imagine generation "fixing" it by 
generating meaningful prefixes and suffixes, it becomes just a bit harder, because generation has to do more to get it right. 

<p align="center">
  <img src="/assets/images/chimera.png" alt="The Chimera" width="80%">
</p>


This sounds alarming. But these chimeras are in fact very hard to make. One of the values of
the following exercise has been to convince myself that BPE is actually rather robust in most situations.

## Why chimeras are hard to make

**Shared prefixes mean characters merge naturally.** Every Thai character
starts with `e0 b8`. In any corpus with Thai text, the pair `(e0, b8)` has
frequency equal to the *total count of all Thai characters* — which would naturally be very high in a corpus where the language even has modest representation. It merges early. After that, individual
Thai characters complete naturally: `([e0 b8], xx)` merges whenever a specific
Thai character appears often enough. Prefix sharing accelerates the "good"
merges.

**Tail bytes from different characters can't be adjacent.** Could BPE fuse the
final bytes of two successive multi-byte characters? No — there's always a
prefix between them:

```
Character A         Character B
[e0 a4] [a8]      [e0 a4] [ae]
                   ^^^^
            prefix of B sits here
```

The tail bytes `a8` and `ae` are never neighbors in valid text -- these so called "continuation" bytes will never also be a prefix. An end-plus-end chimera is thus impossible.

**Within-character merges outrun cross-boundary merges.** 
Let's go back the posited Hindi-Emoji chimera [a8  f0]. 

For a cross-boundary
merge `(a8, f0)` to happen, it must beat `([e0 a4], a8)` — which completes
the character न. But trivially `([e0 a4], a8)` has at least as high a frequency --
the number of न in any corpus is necessarily larger (or at least equal to)
the number of न🪔 in that corpus. The completion merge always wins the race.

Empirically, GPT-2 confirms this. Of 50,257 tokens, 344 are invalid UTF-8. But
128 of those are just the single-byte tokens for byte values `0x80`–`0xFF` —
they're "invalid" only trivially, because every byte gets a token. Most others are
prefixes or suffixes (mostly the former). Claude assures me that there are no more than 28
chimera-like constructions.


## Let's form a chimera
Our goal is not to analyze those tokens in GPT-2. We want to a) define the requirement for chimera formation and b) be able to come up with *plausible* scenarios where this may occur which
is semantically satisfactory.


To build out this construction I need two sets:

1. **P** -- a set of multi-byte characters that share their last byte (but not their prefixes). Let's call this common last byte β.
2. **S** -- a set of multi-byte characters that share their all-but-last prefix (but not their last byte). Let's call this common prefix ρ.

Now we need the frequency of P x S pairs, and thus the frequency of the βρ sequence to
be larger than the frequency of any *single* character in P and S.

If this is the case, the βρ chimera will be formed. To see that this is mathematically possible,
let's reiterate that the chimera's frequency is a sum over the 
**cartesian product** P × S. Each
competing merge draws from just one character. If the co-occurrence is real, but the individual
characters are rare, the chimera wins.

This is *still* a tall order. But let's make it happen!

## A recipe: the visarga and the sacred lamp

I'll construct a plausible chimera from a specific Indic diacritic and a
specific cluster of emoji. The ingredients:

- **The suffix**: the **visarga** (ः / ঃ / ਃ) — a breath mark common to many Indic languages.
- **The prefix**: the emoji block containing 🪔 (diya lamp) and 🪷 (lotus)

### Ingredient 1: The loose byte

The visarga is a small mark that appears at the end of Indic words. 
It represents a voiceless breath — the "-ah" in *namah* (as in *Om Namah
Shivaya*) or the "-ih" in *shantih* (as in *Om Shanti Shanti Shantih*). It's
formal, devotional, and rare outside of mantras.

The Unicode Consortium placed each Indic script's visarga at the **same
offset** within its block — position 0x03. The result:

| Script | Visarga | UTF-8 | Loose byte |
|--------|---------|-------|------------|
| Devanagari (Hindi) | ः | `e0 a4 83` | **`83`** |
| Bengali | ঃ | `e0 a6 83` | **`83`** |
| Gujarati | ઃ | `e0 aa 83` | **`83`** |
| Tamil | ஃ | `e0 ae 83` | **`83`** |
| Telugu | ః | `e0 b0 83` | **`83`** |
| Kannada | ಃ | `e0 b2 83` | **`83`** |
| Malayalam | ഃ | `e0 b4 83` | **`83`** |


Now, take any of these languages, say Bengali. The [e0 a6] prefix will *obviously* form a token, since
it is common across all Bengali characters. But visarga being an infrequent character, it
is plausible that it won't form one (indeed that is what happens in GPT-2). So, now we have
`83` as a loose byte across all these languages.


### Ingredient 2: The liturgical context
Now consider the fact that visarga is more common in Sanskrit derived words, and most frequently in words from hymns and chants, such as the famous refrain from Devi Mahatmyam:

> नमस्तस्यै नमस्तस्यै नमस्तस्यै नमो नमः
>
> *namastasyai namastasyai namastasyai namo namaḥ*
>
> "I bow to Her, I bow to Her, I bow to Her, salutations and salutations"

Consider now, a specific corpus, perhaps a twitter-like social media where people tend to post such hymns, followed usually by a devotional emoji.


### Ingredient 3: The fragmented emoji prefix

The emoji in the range U+1FA80–U+1FABF all encode as `f0 9f aa ··`:

| Emoji | Name | UTF-8 | Tail byte |
|-------|------|-------|-----------|
| 🪔 | diya lamp | `f0 9f aa 94` | `94` |
| 🪷 | lotus | `f0 9f aa b7` | `b7` |
| 🪈 | flute | `f0 9f aa 88` | `88` |
| 🪘 | drum | `f0 9f aa 98` | `98` |
| | *(~50 more objects)* | `f0 9f aa ··` | *(various)* |


These are relatively obscure emoji unlikely to be widely used and yet they contain at least two with a strong liturgical correlation with those Hindu chants (the lotus and the diya lamp -- and perhaps the flute and the drum too).


Now we have the *plausible* construction where continuations such as नमः🪔, when aggregated
across all Indic languages and all plausible emoji, exceed in count either the visarga or the
🪔.


## A parting shot -- by Cupid
Having leaned heavily on high-Sanskrit words, I'd be remiss as a Bengali if I didn't invoke যাঃ / আঃ (jah/ah) -- Bengali exclamations that end with visarga, used to express coy irritation or a breathy sigh in romantic context. Does our emoji cluster contain anything that could be a reasonable continuation? Yeah, I think the folding hand fan 🪭 captures the essence of these utterances rather well. 

The end.

😁
