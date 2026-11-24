# EUTERPE — Model Comparison & Architecture Benchmark Plan
**Version:** 1.0.0  
**Phase:** 0 — Scientific Specification and Research Foundation  

---

## 1. Candidate Generation & Model Taxonomy

EUTERPE benchmarks six core model configurations across three paradigms: **Classical Heuristic Baselines**, **Collaborative Matrix Factorization**, and **Dual-Encoder Neural Retrieval**, followed by **Discovery Reranking**.

```
[Level 1: Candidate Generation / First-Stage Retrieval (Top M = 100)]
  ├── Baseline 1: Global Popularity (Static frequency ranking)
  ├── Baseline 2: Recency-Weighted Popularity (Exponential half-life decay)
  ├── Classical 1: Implicit ALS (Hu et al., confidence-weighted matrix factorization)
  ├── Classical 2: Bayesian Personalized Ranking (Rendle et al., pairwise BPR-MF)
  └── Neural: Two-Tower Dual-Encoder (Yi et al., FAISS approximate vector search)
         │
         ▼
[Level 2: Second-Stage Discovery Reranker (Top K = 20)]
  ├── Identity Pass-through (Unmodified Top-K)
  ├── Maximal Marginal Relevance (MMR with latent cosine distance)
  └── Discovery-Constrained Pareto Ranking (Novelty floor optimization)
```

---

## 2. Mathematical Specifications of Model Architectures

### 2.1 Baseline 1: Global Popularity
Non-personalized upper-bound benchmark for aggregate catalog visibility:
$$\text{Score}(u, i) = \sum_{v \in \mathcal{U}} \log(1 + r_{vi})$$
Tracks already in user $u$'s training history are masked during inference.

### 2.2 Baseline 2: Recency-Weighted Popularity
Time-decayed aggregate popularity favoring recently emergent tracks:
$$\text{Score}_{\text{recency}}(u, i) = \sum_{v \in \mathcal{U}} \sum_{t \in T_{vi}} 2^{-(t_{\text{ref}} - t) / \tau_{\text{half}}}$$
where $t_{\text{ref}}$ is the latest timestamp in the training split, and $\tau_{\text{half}} = 90\text{ days}$.

### 2.3 Classical Model 1: Implicit Alternating Least Squares (ALS)
* **Objective:** Minimize $\mathcal{L}_{\text{ALS}} = \sum_{u, i} c_{ui} (p_{ui} - \mathbf{x}_u^T \mathbf{y}_i)^2 + \lambda \left(\sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2\right)$
* **Confidence scaling:** $c_{ui} = 1 + \alpha \cdot \log(1 + r_{ui})$ with $\alpha = 40.0$.
* **Hyperparameters:** Latent dimension $d = 64$, regularization $\lambda = 0.05$, iterations $= 25$.

### 2.4 Classical Model 2: Bayesian Personalized Ranking (BPR)
* **Objective:** Maximize pairwise posterior probability:
  $$\ln p(\Theta | >_u) = \sum_{(u, i, j) \in D_S} \ln \sigma(\mathbf{x}_u^T \mathbf{y}_i - \mathbf{x}_u^T \mathbf{y}_j) - \frac{\lambda}{2} \|\Theta\|^2$$
* **Sampling:** 5 uniform negative items per positive observation.
* **Hyperparameters:** Latent dimension $d = 64$, learning rate $\eta = 0.05$, regularization $\lambda = 0.005$, epochs $= 50$.

### 2.5 Neural Intelligence: Dual-Encoder Two-Tower System
* **Architecture:**
  * **User Tower $\psi(u)$:** 
    $$\mathbf{h}_u = \text{LayerNorm}\left(\mathbf{W}_2 \text{GELU}(\mathbf{W}_1 \mathbf{e}_u + \mathbf{b}_1) + \mathbf{b}_2\right)$$
    normalized to unit hypersphere $\|\mathbf{u}\|_2 = 1$.
  * **Track Tower $\phi(i)$:**
    $$\mathbf{h}_i = \text{LayerNorm}\left(\mathbf{V}_2 \text{GELU}(\mathbf{V}_1 [\mathbf{e}_{\text{track}} \,\|\, \mathbf{e}_{\text{artist}}] + \mathbf{c}_1) + \mathbf{c}_2\right)$$
    normalized to unit hypersphere $\|\mathbf{v}\|_2 = 1$.
* **Loss Function (Popularity-Corrected In-Batch Softmax):**
  $$\mathcal{L} = -\frac{1}{B} \sum_{k=1}^B \log \frac{\exp\left((\mathbf{u}_k^T \mathbf{v}_k - \log \hat{P}_k)/\tau\right)}{\sum_{j=1}^B \exp\left((\mathbf{u}_k^T \mathbf{v}_j - \log \hat{P}_j)/\tau\right)}$$
  where $\hat{P}_j$ is the training-set marginal item frequency and temperature $\tau = 0.07$.
* **Inference / Indexing:** All catalog track embeddings $\mathbf{v}_i \in \mathbb{R}^{64}$ indexed into FAISS (`IndexFlatIP` for exact cosine similarity, and `IndexIVFFlat` for approximate retrieval).

### 2.6 Discovery Reranking Modules
* **MMR Reranker:**
  Iteratively selects track $i^* \in \mathcal{C} \setminus S$ maximizing:
  $$\text{Score}_{\text{MMR}}(i) = \lambda \cdot \hat{s}(u, i) - (1 - \lambda) \max_{j \in S} \cos(\mathbf{v}_i, \mathbf{v}_j)$$
  Evaluated across $\lambda \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$.
* **Discovery-Constrained Optimizer:**
  Ranks items by predicted relevance $\hat{s}(u, i)$ under a strict novelty floor:
  $$\max \sum_{i \in S} \hat{s}(u, i) \quad \text{subject to} \quad \frac{1}{|S|}\sum_{i \in S} -\log_2 P(i) \ge \tau_{\text{novelty}}$$

---

## 3. Controlled Experimental Matrix

| Experiment ID | Candidate Model | Reranking Policy | Target Question ($RQ$) | Key Evaluation Trade-off |
|---|---|---|---|---|
| `EXP-01` | Global Popularity | None (Top-20) | $RQ_2$ (Popularity baseline) | Non-personalized ceiling |
| `EXP-02` | Recency Popularity | None (Top-20) | $RQ_2$ (Temporal baseline) | Freshness vs popularity |
| `EXP-03` | Implicit ALS | None (Top-20) | $RQ_1, RQ_2$ (Primary baseline) | Unconstrained CF accuracy |
| `EXP-04` | Implicit ALS | MMR ($\lambda=0.6$) | $RQ_4$ (Classical diversity) | Accuracy vs diversity trade-off |
| `EXP-05` | BPR-MF | None (Top-20) | $RQ_1, RQ_2$ (Pairwise baseline) | Personalized ranking quality |
| `EXP-06` | Two-Tower Neural | None (Top-20) | $RQ_3$ (Neural retrieval) | Semantic candidate breadth |
| `EXP-07` | Two-Tower Neural | MMR ($\lambda=0.6$) | $RQ_4$ (Neural diversity) | Neural Pareto frontier |
| `EXP-08` | Two-Tower Neural | Discovery-Constrained | $RQ_4, RQ_5$ (Constrained discovery) | Guaranteed long-tail exposure |

---

## 4. Frozen Evaluation Safeguards

1. **Frozen Baseline Checkpoints:** ALS and BPR baseline checkpoints are trained and evaluated first. Their hyperparameter selections and metrics are permanently frozen before evaluating neural or reranked outputs.
2. **Identical Candidate Universe:** All models generate recommendations from the exact same pre-filtered active catalog ($|\mathcal{I}| \ge 5$ scrobbles).
3. **No Target Leakage:** Training items from user histories are masked with score $-\infty$ across all models during candidate selection.
