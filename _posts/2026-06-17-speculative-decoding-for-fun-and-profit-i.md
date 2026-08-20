---
layout: post
title: "Speculative Decoding for Fun and Profit I"
author: "Pradipta Mitra"
date: 2026-06-17
series: specdecode
part: 1
---

Take a prompt summarizing a document:

"Document: [Document content]. Please summarize".

Now take a second prompt:
"Document: [Document content]. Please summarize. Now generate several hashtags based on this summary".

Is it the case that the second prompt generates the summary from the same distribution as the first one? It should, right?

Allow me to back up and tell you why I am interested in this question.

In Generative AI applications, there often is a set of downstream ML tasks after the "main" work is done. Our prompt pair above points to a simple example -- hashtag generation following summarization. There can be many others. In production environments, one may not be able to use the "main" LLM for this second task. It may be deemed too costly for the downstream task. There might be organizational reasons.

This can result in a sub-optimal situation for the folks responsible for hash-tags. They may have to spin up a bunch of servers to take the summary (transferred over network) and process it to produce hashtags. The utilization may be low -- perhaps hash-tags are needed only for a sub-set of summaries, resulting in low inference time batching. This may result in them being able to only deploy a very small model, and now they are on hook to improve quality by fine-tuning (which they may lack expertise in).  And yet, the problem may be easily solvable using in-context learning had they had access to a "large enough" model.

Now suppose the main summarizer is served with speculative decoding — as large models increasingly are. This n-part (n ~ 3) post will explore whether such setup can be leveraged for downstream tasks.

[Speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) is a technique to improve generation performance for LLMs. We have a target LLM we want to generate from (in our story, this would be the "main" model responsible for summarization). Speculative decoding adds to this a smaller "draft" model. Both models are given the *same* prompt and now the draft model generates a number of tokens -- cheaply. The target model validates these tokens in a single parallel forward pass (which is cheaper than autoregressive generation) --  accepts a prefix of them, then corrects the first mismatch. Assuming the draft is good, the accepted prefix will be long and the overall process would be more performant. The validation process guarantees that this process is identical to sampling from the target distribution.

We will describe the validation loop more precisely in a moment. But for now, notice: in the draft model, we have potentially found "the large enough" model that can perform downstream tasks, sitting there for "free" (or is it?).

How could we leverage this model? Well, we could simply change the draft prompt (but not the target one) to:

"Document: [Document content]. Please summarize. Now generate several hashtags based on this summary".

and then work the speculative decoding loop as usual.

Immediate questions:
1. Is this correct? Will the guarantees of speculative decoding hold if the prompts are different?
2. Will this be as performant in terms of "acceptance rate" (the probability that a draft token will be accepted by the target model)?


## Correctness
Turns out correctness is independent of the draft prompt. You could put "Yankee Doodle" in your draft prompt and the guarantee would hold.

Let's go through how acceptance works.

At a given context, let $p$ be the target's next-token distribution and $q$ the
draft's. The draft proposes a token $t \sim q$. We draw $u \sim
\mathrm{Uniform}(0,1)$ and **accept** $t$ iff

$$u \le \min\!\left(1,\ \frac{p(t)}{q(t)}\right).$$

- If $q(t) \le p(t)$ (the draft was no more confident than the target), $t$ is
  accepted with probability 1.
- If $q(t) > p(t)$ (the draft over-proposed $t$), it is accepted with
  probability $p(t)/q(t) < 1$.

On **rejection**, we emit a token resampled from the *corrected* distribution —
the part of the target's mass the draft failed to cover $\propto \max(0,\,p-q)$,
renormalized. (We'll skip the nitty-gritty details).

With the appropriate normalization, it is mostly algebra to show that the distribution of the output is identical to sampling from the target directly.
The intuition is easy to grok: if the draft proposes something the target finds plausible, keep it; otherwise fall back to the target's own (corrected) choice.

Now what is the effect of the draft prompt in this calculation? Well it changes $q(t)$. But the proof works for any $q(t)$. Correctness *regardless of draft prompt* is by construction.

## Acceptance
The problem, of course, is that a $q(t)$ widely divergent from $p(t)$ would result in draft tokens almost never being accepted. The sampled token would come from the high $q(t)$ regime, and if $p(t)$ is not correspondingly high, you'd end up not accepting the token. At that point, you might as well simply sample from the target.

So we focus on 
**acceptance rate** $\alpha$: how often a draft-proposed token survives,

$$\alpha \;=\; \mathbb{E}_{t\sim q}\!\left[\min\!\left(1,\ \tfrac{p(t)}{q(t)}\right)\right]
\;=\; \sum_{v} \min\big(p(v),\, q(v)\big).$$


To achieve high $\alpha$, we need a $q(t)$ concordant with $p(t)$  . This depends on a) the draft model being a reasonable proxy for the target but also b) the prompts being the same. This is why "Yankee Doodle" doesn't suffice in practice.

But what about  the specific modification that we have proposed, i.e. the hashtag generating suffix?

Semantically, we *can* argue that $q(t)$ for the summary part of the generation should remain unchanged. That is:

$$q(t \mid \text{original prompt}) = q(t \mid \text{original prompt} + \text{hashtag suffix})$$ as long as we are generating the summary. 

Is it though, in practice? Only experiments can tell:


## Experiments
To reproduce, follow [this repo](https://github.com/pradiptamitra/speculative-sidecar).


We take two target/draft pairs, both runnable locally on a Mac Studio:

1. **small** — Qwen2.5-1.5B-Instruct (target) / Qwen2.5-0.5B-Instruct (draft)
2. **big** — Qwen2.5-7B-Instruct (target) / Qwen2.5-1.5B-Instruct (draft)

The task is summarizing CNN/DailyMail articles, over 100 documents. For each pair
we compute α two ways and compare:

1. **identical prompts** — draft and target both get the plain summarize prompt
   (ordinary speculative decoding).
2. **compound draft prompt** — the draft gets "summarize … now output hashtags,"
   the target still gets the plain prompt.

### How to compute α
Going in, I expected to run the usual speculative decoding pipeline (draft proposes, <del>god</del> target disposes) and gather acceptance stats. Claude, however, insisted on the following:

1. Generate the summary once with the *target* (greedily); keep its per-token distributions $p$.
2. **Teacher-force the draft** over that same summary — a single parallel pass, fed the target's tokens — to get its distributions $q$.
3. Average the overlap over the summary tokens (and documents): $\alpha = \operatorname{mean}_t \sum_v \min(p, q)$.

This role-inversion initially discombulated me, but two things make this valid. First: the real loop's output is, by construction, a sample from the target, so the summaries that actually occur are target-generated — and that's what we do. Second, *why a single pass per model suffices*: the per-summary acceptance is the symmetric, closed-form overlap $\sum_v \min(p, q)$, so we just read both distributions and take the minimum.

This is also more robust: there's no Monte-Carlo variance (it's the exact expectation), and both prompt conditions are scored against the *same* target summary, so the only variable is the draft prompt.

### Results

| pair | α (identical prompts) | α (compound draft prompt) | relative drop |
|---|---|---|---|
| small 1.5B/0.5B | 0.649 | 0.616 | **−5.1%** |
| big 7B/1.5B | 0.636 | 0.627 | **−1.4%** |


This is a meaningful drop in acceptance rate. It is reassuring that the penalty shrinks for the larger pair, but even ~1.4% isn't obviously negligible. I find this interesting in its own right. Why is this reasonable "generation-prefix invariance" violated in practice[^1]?

[^1]: I wonder whether this kind of invariance has ever been used as an explicit training objective. 


But for our specific idea, we would have to park this thought and simply move on to find better ways of accomodating the downstream task within the draft model.

To be continued...

