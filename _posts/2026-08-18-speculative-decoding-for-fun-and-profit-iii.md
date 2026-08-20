---
layout: post
title: "Speculative Decoding for Fun and Profit III"
author: "Pradipta Mitra"
date: 2026-08-18
series: specdecode
part: 3
---

Recap: We are exploring the idea of leveraging a speculative decoding setup for extra work. In speculative decoding, we speed up decoding from a target model by hitching a draft model to it. The way this works is: we sample a bunch of tokens from the draft model, and validate them using the target model. It has been [shown](https://arxiv.org/abs/2211.17192) that this allows us to sample tokens more cheaply compared to sampling from the target model directly. 

Now assume that we have a downstream task that needs to be performed after our main generative task is complete. The running example we have been using is:

a) Main task -- summarizing a document.
b) Downstream task -- generating hashtags from the summary.

Our proposition is this: After generating the summary, the draft model has the whole thing in its KV cache. Can we not let it run a bit more and generate a few hashtags? We'd do so by updating the prompt, instead of *"[the document] … Summarize the document"*, our prompt would be *"[the document] … Summarize the document. Then output 3–5 short hashtags for the summary"*.

In the first two parts, we focused on the effect of this prompt modification on the ability of the draft model to sample tokens which are likely to match the target model distribution.
It turned out that parity with a pure draft model was achievable, but after applying a couple of tricks.

Now, in this last post, we focus on the quality of the generated hashtags.

Let's recall the draft model architecture we landed with. We provide the draft model with the prompt: *"[the document] … Summarize the document. Then output 3–5 short hashtags for the summary"*. We observed that this last bit tended to reduce the efficiency of the draft model in summary generation. To fix that, we masked the hashtag portion of the prompt when generating the summary. [This image in Part II]({% post_url 2026-06-25-speculative-decoding-for-fun-and-profit-ii %}#masked-attention) captures it pretty well. 

Now let's move onward with hashtag generation. Once the summary is completed, we simply unmask the hashtag portion of prompt and keep the draft model going.

Does this work? The obvious setup to compare this to is a "standalone" version of the same Qwen2.5-1.5B-Instruct model -- this would be a separate deployment provided with just the summary and a prompt to generate hashtags. In what follows, we will call the speculative decoding setup a "sidecar" to be contrasted with the "standalone" setup.

# Eyeball

The corpus we are using is 100 news articles from the [CNN/DailyMail dataset](https://huggingface.co/datasets/abisee/cnn_dailymail) (the test split). We ignore the dataset's own reference summaries — the summaries here are produced by our 7B target model (Qwen2.5-7B-Instruct), with the 1.5B model playing the role of draft model and hashtag generator.

As an example, here are the tags from both systems ('sidecar' and 'standalone') on a story.

> **Document (excerpt):** *London (CNN) A 19-year-old man was charged Wednesday with terror offenses after he was arrested as he returned to Britain from Turkey, London's Metropolitan Police said. Yahya Rashid, a UK national from northwest London, was detained at Luton airport on Tuesday after he arrived on a flight from Istanbul…*
>
> **Summary (the 7B target's, which both taggers see):**
> - A 19-year-old UK man was arrested at Luton airport upon returning from Turkey.
> - Charged with preparing terrorist acts and aiding others to commit such acts.
> - Arrest related to activities between November 1 and March 31.
>
> **Sidecar:** `terror-arrest`, `uk-police`, `uk-terrorism`  
> **Standalone:** `uk_police`, `terrorist_activity`, `uk_police_arrest`

Here's another:

> **Document (excerpt):** *(CNN) Never mind cats having nine lives. A stray pooch in Washington State has used up at least three of her own after being hit by a car, apparently whacked on the head with a hammer in a misguided mercy killing and then buried in a field — only to survive…*
>
> **Summary:**
> - A stray dog named Theia survived being hit by a car, hit with a hammer, and buried.
> - Theia, a young bully breed mix, suffered multiple injuries but was cared for at no cost.
> - Surviving pets inspire donations; funds will help Theia and others in need.
>
> **Sidecar:** `surviving-pets`, `dog-survival`, `animal-rescue`  
> **Standalone:** `surviving_pet`, `pet_care`, `free_pets`

I think most of us would agree that it's a wash, at the very least no side is clearly superior.

We scaled this process up a bit with a LLM judge and the results are:

> **Sidecar vs. standalone: 86 ties. The other 14 split 9–5.**

The small number of decided cases seem to be *factual* slips by the underlying 1.5B: e.g. tagging Duke as `#douglas-university`, calling Avril Lavigne `#laurie-lavigne` etc. 


# Why attend the document?
Now, in our sidecar setup, we still attend the document. This is just a natural implication of how the thing is set up. The whole prompt, including the document is in the attention set by virtue of how transformers work.

![The document dominates the draft's KV cache, yet the hashtag tokens only need the summary](/assets/images/ppm_doc_kv_cache.svg)

According to our LLM judge, this evidently has no quality implication, but it might have a performance implication, no? The document (and thus its KV cache) is much larger than the summary, and the standalone version does not attend (or indeed have access to) the document. Quantitatively, the KV cache is about **18 MB** per request in the sidecar setup, compared to about **3 MB** in the standalone case. We could mask the document out of attention during the tagging step. Seems like an easy (and important) win.

Except that this is dwarfed by the model size. On a single decode step the model reads all **3.09 GB** of its weights. Against that, an 18 MB cache is **0.6%** — trivial. So that doesn't pan out (yet). 

But this exercise made us look at the tags again carefully -- is it possible that attending the document can cause incorrect tags to be generated?

And we did find interesting cases of failures:

> **Document (excerpt):** *(CNN) Blue Bell ice cream has temporarily shut down one of its manufacturing plants over the discovery of listeria contamination in a serving of ice cream. Public health officials warned consumers Friday not to eat any Blue Bell-branded products made at the company's **Broken Arrow, Oklahoma**, plant…*
>
> **Summary:**
> - Blue Bell closed its Oklahoma plant due to listeria contamination, recalling specific ice cream products.
> - The company is investigating potential links to previous outbreaks in Kansas and Texas.
> - Blue Bell CEO expresses concern over the situation and commitment to quality improvement.
>
> **Sidecar:** `bluebellicecream`, `listeriacontamination`, `brokenarrowplant`  
> **Standalone:** `bluebellicecream`, `listeriacontamination`, `qualityimprovement`

Gotcha! The sidecar emitted `brokenarrowplant`, leveraging a fact that can be found in the document but nowhere in the summary — the summary only says "its Oklahoma plant." Is this a fluke? To test it we forced the same summary and generated the tags **60 times**, under two conditions:
1. Generate tags with the sidecar as usual (document attended).
2. Generate tags with the sidecar with the document masked.

In a quarter of the runs, the attended sidecar tagged the plant `brokenarrowplant`. With the document masked, that number was **zero**; instead the tagger fell back to `oklahoma`, the summary's own word.

Now that we have identified a quality gain, let's revisit the performance question. We already saw that for a single request there is virtually none. But as the inference batch size increases, the model weight streaming is amortized and a gain may in fact show up.
A back-of-the-envelope on an A100 (2 TB/s of bandwidth, our 3.09 GB of weights, 18 MB vs. 3.7 MB of cache per request) gives us the following:

| Batch | Full sidecar / step | Doc-masked / step | Decode speedup |
|---|---|---|---|
| 1 | 3.11 GB | 3.09 GB | ~0% |
| 16 | 3.38 GB | 3.15 GB | 7% |
| 32 | 3.67 GB | 3.21 GB | 14% |
| 64 | 4.24 GB | 3.33 GB | 28% |
| 128 | 5.39 GB | 3.56 GB | 51% |

The speed-up is already non-negligible at the moderate batch size of 32. So there's no reason not to approve this minor model surgery. The insurance companies will agree, one hopes.


# So, is it worth it?

This document attending trick, one needs to understand, makes the sidecar behavior more or less identical to that of standalone. Quality wise, the deployments are similar, which is a good thing, but not a differentiator.

But then why are we doing this? As we discussed in Part I, one major reason is organizational. A separate deployment may simply have overheads that are not captured in performance and latency numbers. Keeping the deployment up, running it efficiently, etc. may be a drag on a potentially much smaller, less well-resourced "hashtag team". 

From a latency/utilization standpoint, there may be gains in certain regimes. First, the thing we clearly avoid when using the sidecar is a) the network cost of shipping the summary to another device and b) prefilling the summary. Both of these are modest effects -- within a data center, latency is low, and prefill is famously much faster than decode. Nevertheless, an about 5-10% trim in the latency of the tagging steps seems possible.

There is also a potential utilization win if the hashtag generation has lower or burstier traffic than the summary task (this is entirely plausible in real scenarios, downstream tasks may only be applicable to a fraction of traffic). In this case, the hashtag task can enjoy better weight amortization by simply being co-occurrent with the main task, whereas a standalone draft model may have poorer utilization.


Finally, looping back to the organizational issue, as a practical matter, the hash-tag team might only have access to slower GPUs in practice, which are often also more expensive per token (a T4 has ~6× less bandwidth than an A100 but is only ~5× cheaper to rent, so ~25% more per generated token), so the actual gains from a sidecar may be higher still, including a significant latency win. 

Overall, if a well-packaged API is provided for adding such sidecars to generative tasks, there might be cases where its adoption will provide modest resource and significant operational gains.


