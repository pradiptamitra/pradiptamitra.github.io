---
layout: post
title: "Flash Attention in a Jiffy"
author: "Pradipta Mitra"
date: 2026-03-11
---

To celebrate the release of [Flash Attention 4](https://arxiv.org/abs/2603.05681), I
think it will be fun to work through the basic idea of Flash Attention. To keep things simple, we will focus on
the forward pass and only on memory *write* traffic.

## Toy Attention
In transformers, tokens (i.e. word-fragments) are represented by vectors. Let's model
this out minimally and say we have $N$ tokens, represented by vectors $x_1, \ldots, x_N$,
which we can arrange as a matrix $X \in \mathbb{R}^{N \times d}$ (here $d$ is the
"embedding" dimension of the token vectors).

Now the idea of attention is that you:

1. compute the affinities between the tokens
2. use these affinities to compute a weighted sum of the token vectors as a *deeper*
representation of the tokens (thus "deep learning").

Since we are talking vectors, the natural way to compute affinities is via the dot
product. So we can say that the affinity between token $i$ and token $j$ is given
by $\lambda_{ij} = x_i \cdot x_j$. And then, the representation of token $i$ at the next layer is given by

$$x_i' = \sum_{j=1}^N \lambda_{ij} x_j$$

Or in matrix form, writing $\Lambda = XX^\top$:

$$X' = \Lambda X$$

All rather neat.

## Attention Proper

In reality, we have three matrices, queries $Q$, keys $K$, and values $V$, all  $\in \mathbb{R}^{N \times d}$ (simplifying somewhat). The affinities are computed between $Q$ and $K$, and then used to compute the weighted sum of $V$. Before the weighted sum, one passes the "affinity matrix" through the softmax function (this being a ritual incantation of deep learning).

Thus, we have:

$$
\begin{aligned}
S &= Q K^\top / \sqrt{d} \\
P &= \operatorname{softmax}(S) \\
O &= P V
\end{aligned}
$$

(We will forgo the $1/\sqrt{d}$ scaling in what follows for sake of simplicity.) Here softmax is applied row-wise and is defined as:

$$
softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

So every entry is scaled up by the exponent and then normalized by the sum of the exponents. 

## Efficiency Concerns
One problem with this calculation is the $N \times N$ intermediate matrices $S$ and $P$. In modern LLMs, $N$ (which is the sequence length) can be quite large (e.g. $10000$ quite easily). "Materializing" (writing out) these matrices to GPU's main memory (HBM) is expensive.

To see just how expensive, note that $d$ is likely to be much smaller than $N$ (e.g. $128$). So the output $O \in \mathbb{R}^{N \times d}$ costs $Nd$ writes, while $S$ and $P$ together cost $2N^2$ writes. The ratio is:

$$\frac{2N^2}{Nd} = \frac{2N}{d}$$

With $N = 10000$ and $d = 128$, that's roughly $156\times$ more data written for $S$ and $P$ than for the output $O$ we actually care about.

Flash Attention solves this by redesigning the computation so that the intermediate state is small enough to live in SRAM at all times. The $N \times N$ matrices $S$ and $P$ are never written to HBM; they simply cease to exist as materialized objects.


## Dijkstra's Ghost
The immortal Edsger Dijkstra implores us to encapsulate logic to hide complexity and avoid over-optimization (in [The Humble Programmer](https://www.cs.utexas.edu/~EWD/transcriptions/EWD03xx/EWD340.html), [A Parable](https://www.cs.utexas.edu/~EWD/transcriptions/EWD05xx/EWD594.html) and elsewhere). But we are destined not to have nice things -- once in a while life requires us to break the encapsulation and peek under the hood. 

And so it is with Flash Attention. The trick is to optimize through the entire attention mechanism, instead of "respecting" the "API" of matrix multiplication. If we were to do the latter, we could use [tiling](https://www.linkedin.com/pulse/tiling-matrix-multiplication-heuristic-view-pradipta-mitra-mutpe/) to reduce the cost of $Q K^\top$ but that would have hardly made a dent (because the materialization would have remained).

## Flash attention

**Observation 1:** Token computations are independent. The computation can be parallelized across the rows of $Q$ (on separate threads/cores). So let's focus, for now, on a single row $i$:

$$
\begin{aligned}
S_i &= Q_i K^\top \quad &(1 \times d) \cdot (d \times N) \Rightarrow (1 \times N) \\
P_i &= \operatorname{softmax}(S_i) \quad &(1 \times N) \\
O_i &= P_i V \quad &(1 \times N) \cdot (N \times d) \Rightarrow (1 \times d)
\end{aligned}
$$

With this limited single row view, our goal is to avoid the materialization of the "large" ($1 \times N$) vectors (which translates, and over all rows, to the dreaded $N \times N$ matrices).

Now the softmax function is:

$$P_{ij} = \frac{e^{S_{ij}}}{\sum_j e^{S_{ij}}}$$

The only thing that prevents this being computable independently per $j$ is the normalization term. Let's defer the normalization term and see what happens.

For a single key index $j$, compute the score:

$$s_{ij} = Q_i \cdot K_j^\top$$

This is a scalar — the dot product of two $d$-dimensional vectors. Now exponentiate:

$$r_{ij} = \exp(s_{ij})$$

Notice that $O_i = P_i V$ is a linear combination of the rows of $V$, with $P_{ij}$ as the coefficients. Since $r_{ij}$ is proportional to $P_{ij}$ (missing only the global normalizer), we can accumulate the unnormalized contributions as we go. Initialize $\operatorname{Norm}_i = 0$ and $O_i = \mathbf{0}$ (a $d$-dimensional zero vector), then for each $j = 1, \ldots, N$:

$$\operatorname{Norm}_i \mathrel{+}= r_{ij}, \qquad O_i \mathrel{+}= r_{ij} \cdot V_j$$

After all $j$ have been processed, $\operatorname{Norm}_i$ holds the correct global normalizer, so we finalize:

$$O_i \leftarrow \frac{O_i}{\operatorname{Norm}_i}$$

**Observation 2:** At no point did we need to store the full vectors $S_i$ or $P_i$. The only state we maintain is $O_i$ ($d$ floats) and $\operatorname{Norm}_i$ (1 float).

Putting it all together:

$$
\boxed{
\begin{aligned}
&\quad \operatorname{Norm}_i \leftarrow 0, \quad O_i \leftarrow \mathbf{0} \\
&\quad \textbf{for } j = 1, \ldots, N: \\
&\quad\quad s_{ij} \leftarrow Q_i \cdot K_j^\top \\
&\quad\quad r_{ij} \leftarrow \exp(s_{ij}) \\
&\quad\quad \operatorname{Norm}_i \mathrel{+}= r_{ij} \\
&\quad\quad O_i \mathrel{+}= r_{ij} \cdot V_j \\
&\quad O_i \leftarrow O_i / \operatorname{Norm}_i
\end{aligned}
}
$$

It's worth pointing out that $O_i$ is *repeatedly* updated. Because $O_i$ is "small" and can be kept in SRAM/cache, this cost can be ignored.


## Tiling

The algorithm above processes one row of $Q$ at a time, streaming over all of $K$ and $V$. This is already enough to avoid materializing $S$ and $P$. However, it fails to leverage the full power of the cache and of compute parallelism (threads).

In practice, you want to process a *block* of $Q$ rows together, while simultaneously loading a block of $K$ and $V$ rows into SRAM. The reason these two ideas go together is that thread parallelism and cache efficiency point to the same structure: a block of threads working on a tile of $Q$ rows can all reuse the same block of $K$ and $V$ that was loaded into SRAM, amortizing that load across the whole thread block.

Tiling also improves the memory *reads*: a block of $K$ and $V$ loaded into SRAM is reused across all $Q$ rows in the tile, reducing round-trips to HBM.

The encapsulation break pays off here too: instead of isolated tiling of (say) $QK^\top$, the tile spans the entire fused operation, keeping both thread occupancy and SRAM usage in mind simultaneously.

## Demo: Seeing the Memory Benefit on CPU

The algorithm above is designed for GPUs, but the memory benefit can be observed on CPU too — [this companion repo](https://github.com/pradiptamitra/flashattn_demo) has a plain C++20 implementation of both baseline and flash-style attention. CUDA kernels are powerful but hard to read; C++ on CPU makes the mechanism easy to read by mere mortals (in this case, perhaps even Dijkstra included).

On CPU the algorithms are typically compute-bound, so runtime comparisons are not very meaningful. But the memory story is clear: at `seq_len=2048`, peak RSS drops from 38 MB (baseline) to 6 MB (flash) — the $\sim 6\times$ reduction you'd expect from avoiding the $N \times N$ score matrix.

However, we do make a stab at making the runtime story a bit more exciting. The repo includes a memory bandwidth hog that runs in parallel and hammers shared memory. On Apple Silicon, CPU and GPU share unified memory, so this creates something resembling GPU-like memory pressure. With the hog running, baseline slows down sharply while flash-style stays comparatively steady — you can see it below.

<iframe width="560" height="315" src="https://www.youtube.com/embed/lO83TVSMgZg" frameborder="0" allowfullscreen></iframe>


Thanks for reading.
