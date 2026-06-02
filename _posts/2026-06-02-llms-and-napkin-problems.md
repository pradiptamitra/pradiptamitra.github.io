---
layout: post
title: "LLMs and Napkin Problems"
author: "Pradipta Mitra"
date: 2026-06-02
---

On [May 20th](https://openai.com/index/model-disproves-discrete-geometry-conjecture/), [Tim Gowers](https://en.wikipedia.org/wiki/Timothy_Gowers) advised fellow mathematicians to sit down before reading [the tweet that was to follow](https://x.com/wtgowers/status/2057175727271800912). In it, he declared that AI had solved [Erdős's unit-distance problem](https://openai.com/index/model-disproves-discrete-geometry-conjecture/), a celebrated problem in [discrete geometry](https://en.wikipedia.org/wiki/Discrete_geometry) first posed by [Paul Erdős](https://en.wikipedia.org/wiki/Paul_Erd%C5%91s).

The [OpenAI announcement](https://openai.com/index/model-disproves-discrete-geometry-conjecture/), the [proof write-up](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf), the [companion remarks](https://arxiv.org/abs/2605.20695), and even a [rewritten chain of thought](https://cdn.openai.com/pdf/1625eff6-5ac1-40d8-b1db-5d5cf925de8b/unit-distance-cot.pdf) are available online.

Encouraged by this, I tried GPT-5.5 with extra-high reasoning on a couple of open problems from my own old papers on network algorithms. I had tried this sort of thing before, with no real success, but after the Erdős result it seemed worth another attempt. The model could not solve them.

For context, here is a GPT-rephrasing of the two questions I threw at it (I did write up precise conjectures for the model prompt):

> **GPT-rephrased, not a formal theorem statement.**
>
> 1. In [Towards Tight Bounds for Local Broadcasting](https://arxiv.org/abs/1207.1836), can we really pin down the right complexity of [local broadcast](https://arxiv.org/abs/1207.1836) in the physical [SINR model](https://en.wikipedia.org/wiki/Signal-to-interference-plus-noise_ratio), especially whether a stubborn `log^2 n` term is inherent rather than an artifact of the algorithm or proof?
> 2. In [Wireless Connectivity and Capacity](https://arxiv.org/abs/1110.0938), is the paper's `O(log n)` algorithm for strong connectivity in the SINR model actually tight, or is there a faster topology/scheduling construction hiding somewhere?

This failure is interesting because although [OpenAI](https://openai.com/) apparently used an internal model for the Erdős result, it is hard to believe that, even after accounting for that, my problems are anywhere near as deep or difficult as the [unit-distance problem](https://openai.com/index/model-disproves-discrete-geometry-conjecture/). So what explains the difference? 

Prompt engineering skills are unlikely to be the difference, since the [Erdős prompt](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf) is available and is as clear as day -- it is hardly more complex than "Solve this problem please". In any case, I asked my GPT to look at [their prompt](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf) and rewrite the prompt to match its style and content. 

Ok, so what is it then?

I asked ChatGPT. In cringey LLM-ese, it suggested that the Erdős problem was "culturally saturated" while mine wasn't. While looking down my nose at this phrase, I rather dejectedly realized that this may simply be a euphemism for "[Noga Alon](https://en.wikipedia.org/wiki/Noga_Alon) cares about the Unit Distance problem and doesn't give two hoots about yours". As far as I am concerned, ChatGPT is not sycophantic enough.

Jilted ego aside, there's something in what is said -- as ChatGPT expounded -- "Model difficulty is about whether the right idea lies in a high-probability path through the model’s learned space of arguments."

This seems plausible. The deep jump to [algebraic number theory](https://en.wikipedia.org/wiki/Algebraic_number_theory) that has impressed many mathematicians may not have been an alien jump in the training distribution. Examples, analogies, and partial results may already sit near one another in the model’s learned space. And my problem being "shallower" may precisely be the issue -- the eventual solution might involve basic mathematics: a clever observation, a careful construction, some high-school calculus, and college-level discrete math. But that also means there may not be a long trail of deep analogies and field-crossing proofs for the model to exploit.

Perhaps this is of consolation to those who are concerned about the imminent death of that great intellectual endeavor?

Those mathematicians who have sat down, perhaps would like stand up ala Slim Shady, and help solve some of our more humble problems.
