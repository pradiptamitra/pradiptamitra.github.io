---
layout: post
title: "Speculative Decoding for Fun and Profit II"
author: "Pradipta Mitra"
date: 2026-06-25
series: specdecode
part: 2
---

In [Part I]({% post_url 2026-06-17-speculative-decoding-for-fun-and-profit-i %}) of this series of posts, we started exploring if the draft model in a speculative decoding setup can be leveraged for downstream tasks. Specifically, we proposed we could do this by appending a suffix prompt, only to the prompt for the draft model. Consider the specific case where the main task is document summarization, and the downstream task is to generate hashtags from that summary. Then the target model would receive the usual "summarize this document" prompt while the draft model would receive the prompt "summarize this document. then produce some hashtags".

In Part I we found that this compound prompt measurably lowered its acceptance rate — a crucial performance metric in a speculative decoding setup. So this doesn't work, or at least not well enough to accept as is: the presence of the "produce hashtags" suffix changes the draft's distribution over summaries for the worse.

We now proceed to fix this issue.

The solution that presents itself is relatively simple, although it requires a bit of model surgery:

*While generating the draft summary, do not attend to the suffix.*

In slightly more detail:

1. During prefill (which includes the "hashtag suffix"), generate the token representation as usual using a causal mask.
2. During summary generation, the summary tokens would attend to all previous tokens (including the prefill) in the usual fashion, except that they would not attend to the suffix.
3. Once the summary generation is complete, we unmask the suffix and hashtag generation proceeds normally.

![Masked attention during summary generation](/assets/images/spec-decode-masked-attention.svg)
{: #masked-attention}

*During summary generation the summary tokens attend to the document and instruction as usual, but skip the masked suffix. The suffix tokens themselves still attend backward — they are ordinary prefill tokens — they are merely hidden from the summary.*

For the minor cost of a bit of model surgery, this seems perfect. One can nitpick with bullet point 3 (e.g., what if the draft model continues summarizing instead of producing the hashtags straight-away), but we haven't gone that far yet. For now, we are concerned with the acceptance rate vis-a-vis the target model, and it appears that we have hit on an airtight solution.

Except that there is a wrinkle.

## The wrinkle: positional encodings
All modern transformers have some way of modeling the absolute position of a token, which becomes part of the token representation. The most popular way these days is to use [RoPE](https://arxiv.org/abs/2104.09864) — rotary position embeddings. This involves applying a rotation matrix to token representations — but the actual process is not particularly important. What is important is that we need to deal with the fact that the hashtag suffix, although hidden via masking, is still affecting the positions of the tokens, at least if we don't intervene.

Let's introduce a bit of notation:

- **D** — the document length
- **s** — the summary instruction (`"Summarize."`)
- **n** — the hashtag suffix (`"then produce some hashtags"`), about 25 tokens for our real suffix
- **S** — the generated summary length


The **target** always runs the bare prompt, so its summary tokens live at positions $D+s+1,\ D+s+2,\ \dots$. The **draft**, with the masked suffix sitting in between, puts its summary tokens at $D+s+n+1,\ D+s+n+2,\ \dots$ — shifted by $n$. And yet it attends to tokens sitting at locations $1, 2,\ \dots, D + s$, with the positions for the suffix just missing. How does this affect the acceptance rate?

Before we examine that, let's ponder what the solution to this "problem", such as it is, would be. There are several options. First, one could "rebase" the positions. In effect, start the summary at $D+s+1$, thus overlapping the suffix. For the purposes of summary generation, this really is exactly like normal speculative decoding. Whether or not this has a deleterious effect on hashtag generation is an open question, but also one we are not tackling right now. Our concern right now is that this is getting more complex, and we'd rather avoid it if we can.

The other solution is to simply postpone the suffix prefill until the summary is done. Once summary generation finishes, we prefill the suffix in exactly the slot where it belongs, and continue generation. At first blush, one may think that this extra prefill means an extra sweep of loading the model weights from memory, but in fact it can be folded into the first hashtag token's generation pass. The real concern is two-fold:

1. One needs to be careful about what the suffix says. We can hardly say "After summarizing, generate hashtags" anymore (we could try, but it is positionally ghastly — we are already past the summary). So we'd now have to write something like "And here are some hashtags:". This is odd, and one wonders whether it will work well. After all, the model may be puzzled: why did I start generating hashtags when none were asked for?
2. Even more model surgery — and this time the kind that might throw off inference schedulers: a sudden bout of prefill in the middle of generation.

So let's see if we can survive without any change!

## Results: masking does not degrade acceptance rate

First, three terms (carried over from Part I). The **acceptance rate** $\alpha$ is how often the target accepts a token the draft proposed — equivalently, the overlap $\sum_v \min(p_v, q_v)$ of the two models' next-token distributions. **Baseline** is ordinary speculative decoding — draft and target both get the bare summarize prompt, no suffix — and it is the ceiling we are trying to match. **Masked** is the fix from above: the suffix is present in the draft's context but masked during the summary, so the summary tokens never attend to it, but do land $n$ positions later.

All the numbers in this post are reproducible with the [`speculative-sidecar`](https://github.com/pradiptamitra/speculative-sidecar) repo — see its README for the exact commands.

Over 100 CNN/DailyMail articles, with the real $n \approx 25$ suffix:

| pair | α(baseline) | α(masked) | position cost (baseline − masked) | relative |
|---|---|---|---|---|
| small 1.5B/0.5B | 0.6493 | 0.6447 | **+0.0045** | **+0.7%** |
| big 7B/1.5B | 0.6361 | 0.6391 | **−0.0030** | **−0.5%** |

(The *relative* column is the position cost as a fraction of that pair's baseline α.)

Masking recovers the baseline at both scales. For the small pair the position shift costs well under a percent of acceptance (+0.7%); for the big pair it is *negative* — the shift is marginally **beneficial** for 7B/1.5B. Either way, nothing to write home about.

But $n=25$ is a short suffix, and the offset *is* $n$ — so the one thing that could overturn this is a longer suffix. So we swept over larger $n$ (the relative columns are again as a fraction of baseline α):

| n | small (baseline − masked) | small (rel) | big (baseline − masked) | big (rel) |
|---|---|---|---|---|
| 25 | +0.0045 | +0.7% | −0.0030 | −0.5% |
| 50 | +0.0068 | +1.0% | −0.0037 | −0.6% |
| 100 | +0.0091 | +1.4% | −0.0081 | −1.3% |
| 200 | +0.0123 | +1.9% | −0.0099 | −1.6% |
| 400 | +0.0150 | +2.3% | −0.0093 | −1.5% |

At $n=400$ — roughly 80% of the mean prompt length ($P \approx 511$) — the small pair has given up only about 2.3% of its acceptance (1.5 points), and the big pair has actually *improved*.

So the simplest fix wins: plain masking holds acceptance at baseline across every suffix length we'd use, and the rebasing and deferred-prefill surgeries we worried about buy nothing here. With the summary phase settled, Part 3 turns to the downstream task proper — actually producing the hashtags, and seeing whether they're any good.
