---
layout: post
title: "Clustering Graphs: Spectral vs HAC"
author: "Pradipta Mitra"
date: 2026-04-02
---

<p align="center">
  <img src="/assets/images/threshold_fine_grid_pout_007_quantile10.svg" alt="Quantile HAC in the noisier hard-band regime" width="60%">
</p>
You are looking at a plot comparing two clustering algorithms on parameterized graph
model. As the parameter changes, the graph transitions from one regime to another, the preferred algorithm changes (as evidenced by the y-axis: higher is better). The rest of the post will build
up to this plot.

---

All code is available [here](https://github.com/pradiptamitra/clustering_threshold).

## Preface
Having worked on spectral clustering and hierarchical agglomerative clustering, I've had this idea of comparing them on a natural probabilistic model for quite some time. But having a day job, this hasn't quite happened. Now thanks to LLM coding agents,
an empirical study was relatively easy to set up. The agent du jour was codex,
who did a ton of autonomous work. I was happy to report though that some crucial guidance from meat computer was still required.

---

## Two ways to cluster a graph

First the algorithms themselves. For simplicity, let's restrict ourselves to undirected graphs.

The raw object in front of us is the adjacency matrix $A$.

<table>
  <tr>
    <td align="center">
      <img src="/assets/images/my_post_small_graph.svg" alt="A small six-node graph" width="280">
      <br>
      <em>A small graph.</em>
    </td>
    <td align="center">
      <img src="/assets/images/my_post_small_graph_adj.svg" alt="Adjacency matrix of the same graph" width="220">
      <br>
      <em>Its adjacency matrix.</em>
    </td>
  </tr>
</table>

*Figure 1. Graph on the left, its adjacency matrix on the right.*

### Spectral clustering

Spectral clustering comes in several variants, but the idea is standard: take a low-dimensional eigenspace of the adjacency matrix, and then run a simple clustering algorithm in that embedding space.

So one workflow is:

1. compute the top $k$ eigenvectors (for a chosen $k$) of the adjacency matrix,
2. project the nodes onto the space spanned by those vectors as an embedding into $\mathbb{R}^k$,
3. run something simple such as $k$-means in that space.

That is the spirit of standard implementations such as scikit-learn's [SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html).

It's worth pointing out that spectral clustering is a **global** algorithm. It looks at the matrix as a whole and tries to extract the dominant large-scale structure encoded in its leading eigenvectors.

### HAC

Hierarchical agglomerative clustering (HAC) is in contrast a bottom-up algorithm.

You start with each node as its own cluster. Then, at every step, you merge the two clusters that are closest according to the distance between them.

So you need:
- a distance metric defined on nodes, and
- a rule for aggregating node distances to define the distance between clusters.

For graphs, a natural node-level distance is Jaccard distance between neighborhoods:

$$d_J(i,j) = 1 - \frac{|N(i) \cap N(j)|}{|N(i) \cup N(j)|}$$

where $N(i)$ is the set of neighbors of node $i$. That is, two nodes are close if they connect to many of the same other nodes.

> **Note**: This is one thing codex struggled with. It started out by proposing trivial or otherwise unhelpful distance metrics.

Now the cluster level distance has many variations also, but let's start with the standard **average linkage**:

$$d_{\text{avg}}(C, C') = \text{average of } d(i,j) \text{ over } i \in C, j \in C'$$

## Goal

We set out to define a parameterized probabilistic model that will favor one algorithm over the other as the parameter changes. And the parameter should capture a transition between two regimes of graphs that are somewhat natural.

## The spectral side: random graphs and SBMs

The natural starting point for the spectral (haunted?) side is the *stochastic block model* (SBM).

Let's start though with the venerable Erdős-Rényi random graph $G(n,p)$: every edge appears independently with probability $p$.

[A famous fact](https://web.math.princeton.edu/~nalon/PDFS/spectcol4.pdf) about $G(n,p)$'s spectral structure is that:

- the leading adjacency eigenvalue is on the order of $np$, which is also the expected degree,
- the remaining eigenvalues are much smaller, on the order of $\sqrt{np}$.

The stochastic block model (SBM) adds planted communities to this structure -- suppose the nodes are now partitioned into $k$ latent clusters, with within-cluster edge probability $p_{\text{in}}$ and between-cluster edge probability $p_{\text{out}}$, where $p_{\text{in}} > p_{\text{out}}$.

As an example, a graph with 2 clusters, each of size 4, $p_{\text{in}} = 0.5$ and $p_{\text{out}} = 0.1$ might look like as follows:

<table>
  <tr>
    <td align="center">
      <img src="/assets/images/sbm_two_block_example.svg" alt="SBM graph instance with two clusters" width="280">
      <br>
      <em>A graph instance of the SBM model.</em>
    </td>
    <td align="center">
      <img src="/assets/images/sbm_two_block_example_expected_adj.svg" alt="Expected adjacency matrix of the SBM model" width="220">
      <br>
      <em>Expected adjacency matrix of the SBM model.</em>
    </td>
  </tr>
</table>

On the right hand side, we have $\mathbb{E}(A)$, the expected adjacency matrix of the graph. This is clearly a rank-2 matrix. If we extend the spectral gap intuition to this case,
we'd expect the top 2 eigenvalues to be large, and the rest to be small -- and by implication, spectral clustering by projecting to the top 2 eigenvectors to work well.

For a classical statement in this direction, McSherry's [2001 paper](https://ieeexplore.ieee.org/document/959929) is a good reference.

## What sort of graph should make spectral methods struggle?

Ok. Now I'd like to parameterize this model further to introduce a regime where the spectral model will start failing.

Let's start with theory. Spectral clustering relies on the spectral gap of the adjacency matrix -- let's find a matrix where this spectral gap is small(er).

One obvious candidate is a graph with strong **local** structure: a graph where node $i$ mostly connects to nearby nodes in some underlying order, and much less to faraway ones.

That kind of graph has an adjacency matrix that looks banded rather than block-constant. For example, consider a graph with ordered nodes, where each node $i$ connects to all $j$ with $|i-j| \leq 2$.

<table>
  <tr>
    <td align="center">
      <img src="/assets/images/local_ordered_n8_k2.svg" alt="Banded graph instance" width="280">
      <br>
      <em>A graph instance of the local ordered model.</em>
    </td>
    <td align="center">
      <img src="/assets/images/local_ordered_n8_k2_adj.svg" alt="Expected adjacency matrix of the local ordered model" width="220">
      <br>
      <em>Expected adjacency matrix of the local ordered model.</em>
    </td>
  </tr>
</table>

Let's look at the spectral properties of such a graph in a matched-degree comparison: on the left an Erdős-Rényi graph with $n = 128$ and expected degree about $10$, and on the right a cycle band graph on $128$ nodes where each node connects to its $5$ nearest neighbors on each side.

![Spectral comparison between a random graph and a band graph](/assets/images/my_post_spectral_examples.svg)

*Figure 2. At matched average degree, the random graph has one dominant adjacency eigenvalue, while the band graph has a flatter top spectrum with several large modes.*

## What does this model, err, model?

Happily, this graph does capture -- in toy form -- a notion of a locally evolving graph.

Consider that the nodes are news stories about a long presidential campaign. Early on, stories are about the primaries; then the nominees; then conventions, debates, polling, election day, the result, and the postmortem. If node index roughly tracks time, and edges encode a crude notion of similarity, then the adjacency matrix should not look like a uniform block. It should look more like a noisy band: each story is most similar to nearby stories in the same phase, not equally similar to everything in the same broad topic.

## The question

> Can we write down a probabilistic graph model that interpolates between an SBM-like regime and a local banded regime, and then see the preferred clustering algorithm switch from spectral clustering to HAC as the parameter moves?

## A simple parameterized model

Consider the following simple model:

Assume the nodes are partitioned into $k$ latent clusters. Assume the nodes have been ordered such that the nodes in each cluster form a contiguous block. The inter-cluster
probability is, exactly like SBM, $p_{\text{out}}$. Within a cluster, the probability of
edge formation is as follows:

$$P_\alpha(A_{ij} = 1) = (1 - \alpha)\, p_{\text{in}} + \alpha \cdot \mathbf{1}[|i-j| \leq w]$$

where $w = \lfloor p_{\text{in}} (m-1) / 2 \rfloor$.

Note that:

- At $\alpha = 0$, this is exactly the stochastic block model.
- At $\alpha = 1$, within-block edges become a deterministic neighbor graph on each block with average degree $2w$ (except at the boundaries).

## A first experiment

Consider the following setup:

- $n = 96$
- $k = 4$
- block size $m = 24$ (so $p_{\text{in}} = 2w / (m-1) = 8/23 \approx 0.348$)
- band width $w = 4$
- $p_{\text{out}} = 0.03$

While $p_{\text{out}}$ is small, there are 3 times as many possible inter-cluster edges as intra-cluster edges. Nevertheless, this is an "easy" setup so we expect both methods to perform well.

Here and below, the score is **ARI**, the adjusted Rand index: $1$ means perfect recovery of the planted clustering, while values near $0$ mean little agreement beyond chance.

On this slice, we get the behavior I had hoped for:
- at $\alpha = 0.50$: spectral $0.986$, HAC-Jaccard $0.952$
- at $\alpha = 0.75$: spectral $0.972$, HAC-Jaccard $0.986$
- at $\alpha = 1.00$: spectral $0.298$, HAC-Jaccard $0.836$

So in the more exchangeable, SBM-like part of the model, spectral clustering is better. But once the local geometry becomes dominant, HAC-Jaccard overtakes it.

<p align="center">
  <img src="/assets/images/threshold_fine_grid.svg" alt="Successful threshold experiment" width="60%">
</p>

*Figure 3. Spectral is better for smaller $\alpha$, then HAC-Jaccard overtakes it as the graph becomes more local.*

## Let's make it a little harder

Keep the same small-scale setup as above, but increase the noise level to $p_{\text{out}} = 0.07$.

Now the problem is visibly less forgiving for both methods.

On this harder slice:
- at $\alpha = 0.50$: spectral $0.835$, HAC-Jaccard $0.732$
- at $\alpha = 0.75$: spectral $0.530$, HAC-Jaccard $0.724$
- at $\alpha = 1.00$: spectral $0.179$, HAC-Jaccard $0.310$

Spectral still has the advantage on the more SBM-like side, HAC-Jaccard still wins once the local geometry takes over.

<p align="center">
  <img src="/assets/images/threshold_fine_grid_pout_007.svg" alt="Harder small-scale threshold experiment" width="60%">
</p>

*Figure 4. The same small-scale hard-band experiment with larger $p_{\text{out}}$. The crossover survives, but both methods are now noticeably less accurate throughout the sweep.*

## Soft-band

The hard band is conceptually clean, but too rigid. Surely, a real-world graph would not have such a sharp cutoff. So the next question was whether the same story survives if the locality preference is present but softened.

What I tried was a simple variation -- keep the same block setup, but replace the hard cutoff inside each block by a linearly decaying probability.

More precisely, if $i$ and $j$ lie in the same planted block and $r = |i-j|$, then

$$P_\alpha(A_{ij} = 1) = (1 - \alpha)\, p_{\text{in}} + \alpha \cdot \max(1 - r / R,\; 0)$$

with $R = p_{\text{in}} (m-1)$.

So at $\alpha = 0$ this is again the SBM baseline with within-block probability $p_{\text{in}}$, while at $\alpha = 1$ the local endpoint is no longer a hard band. Instead, nearby vertices are much more likely to connect, and that probability then decreases linearly with distance until it reaches $0$.

*Note:* This was another place where codex required hand-holding. It could not come up with good softened versions, and could not validate good ideas because it was experimenting on graphs too small to show the effect.

In a large run below we used:

- $n = 384$
- $k = 4$
- $p_{\text{in}} = 0.2$, so $R = 19$
- $p_{\text{out}} = 0.06$

This gives:

<p align="center">
  <img src="/assets/images/threshold_fine_grid_band_linear_derived_p02_p006_t8.svg" alt="Softened threshold experiment at large scale" width="60%">
</p>

*Figure 6. With the softened local connectivity the crossover to HAC-Jaccard still survives.*

## The curious case of the large $\alpha$ regime

These experiments were satisfying, except where they were not. It was surprising to me that at the very high $\alpha$ regime ($\alpha = 1.00$, specifically), HAC wasn't doing as well as I had hoped. My a priori assumption was that HAC would be basically perfect in that regime.

While codex showed stochastic-parrot-like incuriosity about this phenomenon, it jumped on the bandwagon once I alerted it and provided data that pointed out the following.

Consider the local-ordered adjacency matrix from before. As you merge, you may naturally land in a situation where a planted cluster gets split into the contiguous pieces $1$--$4$ and $5$--$8$. Now the average distance between them is actually rather large and may be larger than the distance between nodes in different clusters (with the right $p_{\text{out}}$ and some luck).

<p align="center">
  <img src="/assets/images/local_ordered_n8_k2_adj_circled.svg" alt="Local ordered adjacency matrix with nodes 1-4 and nodes 5-8 highlighted, connected by an average-distance line" width="45%">
</p>

This suggested a natural variation: use a low quantile of those distances instead of the average.

And this works in our small experiments:

<table>
  <tr>
    <td align="center">
      <img src="/assets/images/threshold_fine_grid_pout_007_avg_vs_quantile10.svg" alt="Hard-band regime with spectral, average-linkage HAC, and quantile-10% HAC" width="95%">
      <br>
      <em>Hard band.</em>
    </td>
    <td align="center">
      <img src="/assets/images/soft_k3_p008_spectral_avg_quantile10_t4_5pt.svg" alt="Softened k=3 regime with spectral, average-linkage HAC, and quantile-10% HAC" width="95%">
      <br>
      <em>Softened model.</em>
    </td>
  </tr>
</table>

*Figure 7. In both the hard-band and softened regimes, quantile-HAC retains strong performance in the high-$\alpha$ local regime, even as average-linkage HAC falters.*

The problem with quantile-HAC is that it is expensive; the straightforward implementation requires computing all pairwise distances between clusters at each step, which is $O(n^3)$.

Exploring ways to approximate it or replace it with a faster alternative would be an interesting problem to take a crack at in a future post! For now, we stop here.

*Coda*: Codex couldn't come up with any interesting theory. Evidently, this problem is harder than some of Erdős'! 😛
