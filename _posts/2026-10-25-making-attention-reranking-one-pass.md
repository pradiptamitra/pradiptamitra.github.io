---
layout: post
title: "Making Attention Reranking One Pass"
author: "Pradipta Mitra"
date: 2026-10-25
excerpt: "Attention-based reranking calibrates against a second prefill with a dummy query. We try to fold both into one pass, and find that a single masked forward pass keeps two-pass quality at twice the speed, without the extra memory of a KV cache."
---

[Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers](https://arxiv.org/abs/2410.02642) uses the internals of a large language model in an interesting way to approach the re-ranking problem. The setting is information retrieval -- to answer a query, one has fetched $N$ (say, 30) documents using a retriever (traditional IR, vector based, whatever). Now the task is to rerank them to pick a final $n$ (say, 5). This two-stage system allows a cheap initial retrieval followed by a more discriminating, and more expensive, final selection.

How would one use an LLM for reranking? A natural way to do this would be to ask the model:

```text
Given the query: How do I blah?
Identify the top 5 relevant documents and output their IDs in rank order.

Document 1: Foo...
Document 2: Bar...
```

This could work, but involves auto-regressive generation, which is (in)famously expensive. It may also not produce a valid output. Instead, the paper gives the model a question-answering or information-extraction prompt:

```text
Consider the following documents:
Document 1: Foo...
Document 2: Bar...

Now, use these documents to answer the Query: [How do I blah?]
```

They pass this prompt through the model to compute the prefill, and stop there (no generation). Now they look at the attention scores from the query to each document individually, and sum them up to obtain relevance scores. The hypothesis is that a document that the query paid a lot of attention to is an important one and deserves to be ranked higher.

![Query tokens attend back to the document tokens; summing that attention gives each document a relevance score, with no generation.](/assets/images/icr-attention-aggregation.svg)
{: #icr-attention-aggregation}

This is reasonable, but as the paper points out, attention to a token may have a "prior" (for example, perhaps articles like "the" or "a" attract inordinate attention from all queries).

The authors therefore perform a second prefill with a modified prompt:

```text
Consider the following documents:
Document 1: Foo...
Document 2: Bar...

Now, use these documents to answer the Query: N/A
```

By swapping the actual query with the "empty" query, they aim to measure the baseline attention accorded to those tokens. They subtract this baseline from the original, thereby arriving at a residual attention attributable to the query itself.

They call their technique in-context reranking, or ICR. This evidently works nicely. Here's an excerpt from one of their reported results ([Table 1](https://arxiv.org/html/2410.02642v1#S4.T1) in the paper):

| Method | Backbone | SciFact nDCG@10 |
|---|---|---:|
| BM25 | — | 67.9 |
| RankGPT | Mistral 7B | 65.5 |
| **ICR** | **Mistral 7B** | **72.4** |


SciFact ([Wadden et al., 2020](https://aclanthology.org/2020.emnlp-main.609)) pairs scientific claims with the research abstracts that support or refute them. The paper uses it as one of the retrieval tasks in the [BEIR](https://arxiv.org/abs/2104.08663) benchmark.

## No refills for prefills
The pain point with this calibration scheme is doing the prefill twice. Can we avoid it? Why not simply do:


```text
Consider the following documents:
Document 1: Foo...
Document 2: Bar...

Now, use these documents to answer the Query: N/A [How do I blah?]
```

Simply append the two queries, and read off the attention from the appropriate part. The worry is that the actual query can now attend to `N/A`. It has also been shifted right, changing its positions relative to the documents. But who knows, maybe it'll still work.

To check, we picked the SciFact dataset and used Qwen2.5-7B-Instruct, on a Mac Studio with 32 GB of unified memory, to rerank 20 BM25 candidates per query, with each passage truncated to 60 words. We use the paper's SciFact-style information-extraction prompt: an instruction before the numbered paragraphs, followed by `Query: <query>` (a detail: the SciFact "queries" are actually claims to be validated, e.g. "0-dimensional biomaterials show inductive properties."). 

Our setup differs from the paper's (a different model, a 20-candidate pool, truncated passages), and we report nDCG on a 0 to 1 scale, so our numbers are not comparable to theirs. For reference, BM25's own ordering of our 20 candidates scores 0.6519 nDCG@10.

Here is what happened over all 300 test queries:

| Metric | Two-pass | Naive one pass |
|---|---:|---:|
| Mean nDCG@10 | 0.6712 | 0.6675 |
| Mean Kendall tau versus two-pass | 1.000 | 0.816 |

nDCG rewards putting relevant documents near the top. Kendall tau measures agreement between rankings; 1 means they agree completely.

The nDCG drop is small, but the rankings move around substantially. The naive version is not computing the same thing.

## KV caching

As it happens, the paper's authors have already addressed this dual prefill problem. They note ([Section 3.3](https://arxiv.org/html/2410.02642v1#S3.SS3) of the paper) that the two queries can share the documents' KV cache. With causal attention, the documents cannot see the query that follows them. Their representations are therefore the same for either query. We can compute the prefix once, save each layer's keys and values, and use them for both continuations:


```text
prefill documents → saved prefix KV cache
                       ├── process real-query suffix
                       └── process N/A suffix
```

Rather neat! But can we do even better?


## Fork the attention without caching

A cleaner one-pass version would fork the attention appropriately. We'll call this the **branched** version. In effect, we do prefill on:

```text
shared document prefix Query: N/A Query: [How do I blah?]
```

But pass in a custom attention matrix that masks the attention from the real query to the empty one.

![One prefill over the shared documents followed by both queries. Each query attends to the documents, attention between the two queries is masked, and both start at position P.](/assets/images/icr-branched-attention.svg)
{: #icr-branched-attention}


There is just one tensor and one forward pass. An attention mask prevents the branches from seeing each other. Each branch can read the shared prefix and its own preceding tokens. We also give the "real query" branch the positions it would have had immediately after the prefix, instead of the shifted positions implied by its location in the packed tensor.

The actual function computed in all methods should be identical (two-pass, KV cache and branched).

Before looking at performance, here is the final quality comparison over all
300 SciFact queries:

| Metric | Two-pass | KV cache | Branched |
|---|---:|---:|---:|
| Mean nDCG@10 | 0.6712 | 0.6701 | 0.6709 |
| Mean Kendall tau versus two-pass | 1.000 | 0.984 | 0.991 |

Both are more or less identical to the two-pass version, as expected. The small gap between KV cache and branched is noise.

## Why could this be better (or not)?
The possible benefit of using the branched version is memory and performance. Let's start with memory. In the KV cache version you have to save the entire KV cache for the shared prefix. You can forgo this in the branched version. 

Since the overall attention value we use for re-ranking is a simple aggregation of per layer attention, after a layer has been computed all attention values for that layer can be dropped. Thus, we only really need to pay for per layer memory. The 2-pass version can do this also, naturally. In fact,
the KV cache version can do this also with respect to the activations and attention values, but it still has to build up an all-layer KV cache for the prefix first.


On the other side of the scale, we have this custom attention mask whose effect on computation is non-obvious. The mask is quadratic in the sequence length (it can be expressed linearly, but somehow it seemed hard to accomplish this on a Mac). This may be fine because attention is quadratic in the sequence length anyway. But we've got to measure it.

Memory first.

### Memory

We measured memory in two ways. 

First we used `footprint -p <pid>` to measure the physical footprint of the program:


| Method | Peak footprint during startup | Peak footprint during scoring |
|---|---:|---:|
| Two-pass | approximately 16 GiB | approximately 17 GiB |
| KV cache | approximately 16 GiB | approximately 20 GiB |
| Branched | approximately 16 GiB | approximately 17 GiB |

In other words, the extra footprint associated with the algorithm itself is about 4 GiB for KV cache, versus about 1 GiB for the other two. A clear win for branched.

We also did an in-process measurement over 30 queries, sampling the driver allocation (i.e. `torch.mps.driver_allocated_memory()`) with similar results:

| Method | Mean extra driver allocation | Worst query |
|---|---:|---:|
| Two-pass | 1.01 GiB | 1.01 GiB |
| KV cache | 1.93 GiB | 4.04 GiB |
| Branched | 1.01 GiB | 1.02 GiB |

The KV cache's extra memory is also unpredictable. It ranged from about 1 GiB to about 4 GiB depending on the query, with the longest prompts the worst. Two-pass and branched took the same amount on every query.

Interestingly, the saved prefix cache itself is only about 115 MiB, far too small to explain the gap. Most likely the cached computation carries extra overhead that the single branched pass avoids.



## Speed

Next up, speed.



| Metric | Two-pass | KV cache | Branched |
|---|---:|---:|---:|
| Mean seconds per query | 6.847 | 3.717 | 3.447 |

Both ways of sharing the prefix are substantially faster than recomputing it. KV cache is 1.84x faster than two-pass, while branched is 1.99x faster. Branched is about 7% quicker than cached here. 


This is very neat! While the memory benefit seems rather predictable, the speed one wasn't. In hindsight it makes sense. The KV cache version makes three model calls instead of one, writing the prefix keys and values into the cache and reading them back for each suffix. The branched version does all of that inside a single forward pass.

Overall, using branched computation seems conceptually simpler, while providing a neat bundle of speed and memory improvements.
