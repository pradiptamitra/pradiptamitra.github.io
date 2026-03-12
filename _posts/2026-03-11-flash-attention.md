---
layout: post
title: "Flash Attention in a Jiffy"
author: "Pradipta Mitra"
date: 2026-03-11
---

To celebrate the release of [Flash Attention 4](https://arxiv.org/abs/2603.05681), I
think it will be fun to work through the *basic* idea in Flash Attention (and only
for the forward pass).

## Toy Attention
In transformers, tokens (i.e. word-fragements) are represented by vectors. Let's model
this out minimally and say we have $N$ token, represented by vectors $x_1, \ldots, x_N$,
which we can arrange as a matrix $X \in \mathbb{R}^{N \times d}$ (here $d$ is the
"embedding" dimension of the token vectors).

Now the idea of attention is that you compute the affinities between the tokens, and
then use these affinities to compute a weighted sum of the token vectors as a *deeper*
representation of the tokens (thus "deep learning").

Since we are talking vectors, the natural way to compute affinities is via the dot
product. So we can say that the affinity between token $i$ and token $j$ is given
by $\lambda_{ij} = x_i \cdot x_j$. And then, the representation of token $i$ at the next layer is given by

$$x_i' = \sum_{j=1}^N \lambda_{ij} x_j$$

Or in matrix form, writing $$\Lambda = XX^\top$$:

$$X' = \Lambda X$$

All rather neat.

## Attention Proper

In reality, we have three matrices, queries $Q$, keys $K$, and values $V$, all (simplifying slightly $\in \mathbb{R}^{N \times d}$). The affinities are computed between $Q$ and $K$, and the result is used to weight the values $V$. And the affinities are passed through, in the time tested manner of the dark arts of deep learning, the softmax function.

Thus, we have

$$
S = Q \dot K^\top
P = softmax(S)
O = P \dot V
$$

where softmax is applied row-wise and is defined as

$$
softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
So every entry is scaled up by the exponent and then normalized by the sum of the exponents. This means that the output $P$ is a probability distribution, i.e. all entries are between $0$ and $1$ and they sum to $1$.

## Efficiency Concerns
The problem is memory bandwidth, which is a bottleneck for GPUs. In particular in a typical setting $N$ may be quite large, say $N=10000$ (or longer, if we have monster context windows). Thus $P$ is a montrosity of $N \times N$. Now, there is not way
to avoid computing the entries of $P$, but the main problem from the memory bandwidth perspective is the naive implementation would "materialize" (aka "write") $P$ to the GPU's main memory (HBM) and then read it back to compute $O = P \dot V$. 

Flash Attention is an elegant solution to avoid this.

## Dijkstra's Ghost
In his "A parable" (and elsewhere), Dijkstra (one of the immortals), tells us to create encapsulations (APIs) to hide the reduce complexity. But we destined not to have nice things, because optimization often requires us to break the encapsulation and peek under the hood. 

And so it is with Flash Attention. One could for example, think about something like tiling to reduce the cost of $Q \dot K^\top$ but that would have hardly made a dent (the materialization would have remained).

