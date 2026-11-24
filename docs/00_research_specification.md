# EUTERPE — Research Specification & Theoretical Foundation
**Version:** 1.0.0  
**Phase:** 0 — Scientific Specification and Research Foundation  
**System Identity:** EUTERPE — Discovery Intelligence for Music  
**Tagline:** *Beyond what you listen to. Into what you'll love.*  

---

## 1. Executive Research Problem Statement

Contemporary music recommendation platforms are dominated by collaborative filtering algorithms tuned to optimize short-term predictive engagement (e.g., immediate click-through rate, completion rate, or next-track play count). While effective at surfacing familiar, high-probability items, this paradigm leads to significant structural failures:

1. **The Matthew Effect (Popularity Reinforcement):** Popular tracks accumulate disproportional feedback, driving matrix factorization and neural embedding models to continually over-recommend top-decile head items at the expense of relevant long-tail music.
2. **Filter Bubble Homogenization:** Recommend-what-was-listened-to loops narrow user exploration, collapsing listening variety into tight acoustic or genre clusters.
3. **The False Dichotomy of Relevance vs. Discovery:** Conventional systems treat novelty and serendipity as trade-offs that inevitably degrade predictive accuracy. 

### Core Scientific Thesis
**EUTERPE** is built upon the premise that **personalized recommendation and long-tail discovery are not mutually exclusive**. By coupling dual-encoder neural retrieval with constraint-aware and diversity-based reranking (Maximal Marginal Relevance and Discovery-Constrained Optimization), an intelligent system can surface items from the musical long tail while maintaining competitive predictive relevance against classical matrix factorization baselines.

---

## 2. Formal Research Questions

The research agenda of EUTERPE is organized around five interconnected questions:

* **Primary Question ($RQ_1$):**  
  *Can a personalized recommendation system improve long-tail music discovery (measured via Novelty, Intra-List Diversity, and Catalog Coverage) while maintaining competitive predictive relevance ($\text{Recall}@20$ and $\text{NDCG}@20$) compared to optimized classical baselines?*

* **Supporting Question 1 ($RQ_2$ — Baseline Popularity Skew):**  
  *To what quantitative degree do industry-standard collaborative filtering models (Implicit ALS and Bayesian Personalized Ranking) concentrate recommendation exposure on the top $5\%$ most popular tracks in implicit listening logs?*

* **Supporting Question 2 ($RQ_3$ — Neural Retrieval Capacity):**  
  *Does a dual-encoder Two-Tower neural architecture with popularity-aware negative sampling and Approximate Nearest Neighbor (ANN/FAISS) retrieval produce higher candidate diversity and long-tail recall than linear latent factor models?*

* **Supporting Question 3 ($RQ_4$ — Reranking & The Pareto Frontier):**  
  *How does the empirical Pareto frontier between predictive relevance ($\text{NDCG}@20$) and discovery quality ($\text{Novelty}@20$, $\text{ILD}$) shift when transitioning from unconstrained top-$K$ scoring to Maximal Marginal Relevance (MMR) and Discovery-Constrained Reranking?*

* **Supporting Question 4 ($RQ_5$ — Listener Heterogeneity):**  
  *Which distinct user cohorts (mainstream listeners, genre specialists, niche enthusiasts, exploratory listeners) derive the greatest utility gain from discovery intelligence, and where is the relevance degradation most acute?*

---

## 3. Hypotheses & Validation Gates

| ID | Hypothesis / Target | Mathematical Target Criteria | Statistical Validation Method |
|:---:|:---|:---|:---|
| **$H_1$** | **Popularity Concentration in Classical CF** | 1. **Decile Concentration Target:** Unconstrained Implicit ALS and BPR allocate $> 70\%$ of top-20 recommendation slots to the top $10\%$ most frequent items.<br>2. **Inequality Target:** Recommendation exposure exhibits Gini coefficient $G > 0.75$ (measured separately from decile concentration). | Empirical bootstrap confidence intervals ($B = 1000, \text{CI} = 95\%$) against uniform distribution baseline ($G = 0$). |
| **$H_2$** | **Neural Retrieval Target** | Two-Tower neural retrieval with in-batch negatives will target competitive predictive relevance ($\text{NDCG}@20 \ge 0.95 \times \text{NDCG}_{\text{ALS}}$) alongside a targeted catalog coverage expansion ($\Delta \text{Coverage} \ge 25\%$). | Predefined minimum detectable effect size ($\delta = 0.02$ NDCG), paired two-tailed bootstrap test ($B = 1000, \alpha = 0.01$). |
| **$H_3$** | **Discovery Pareto Frontier Shift** | Discovery-aware reranking (MMR parameterized by $\lambda \in [0.4, 0.7]$) yields a statistically significant increase in $\text{Novelty}@20$ and Intra-List Diversity ($p < 0.01$) with relevance degradation bounded by $\le 15\%$ drop in $\text{NDCG}@20$. | Non-parametric Wilcoxon signed-rank test across user evaluation partitions. |
| **$H_4$** | **Segment-Specific Discovery Sensitivity** | Niche and exploratory listeners will demonstrate higher novelty tolerance (significantly lower fractional $\Delta \text{NDCG} / \Delta \text{Novelty}$ loss) than mainstream listeners. | Cohort-level two-way ANOVA (segment $\times$ reranking policy interaction). |

---

## 4. Theoretical Foundations & Literature Matrix

EUTERPE's methodology is grounded directly in peer-reviewed recommendation and information retrieval literature:

```
[Collaborative Filtering]
  ├── Matrix Factorization (Koren et al., 2009)
  ├── Implicit Feedback & Confidence Weighting (Hu, Koren, Volinsky, 2008)
  └── Pairwise Ranking: BPR (Rendle, Freudenthaler, Gantner, Schmidt-Thieme, 2009)
         │
[Deep Learning & Neural Retrieval]
  ├── Neural Collaborative Filtering (He et al., 2017)
  └── Two-Tower Scalable Retrieval with Sampled Negatives (Yi et al., Google, 2019)
         │
[Discovery Science & Multi-Objective Ranking]
  ├── Beyond Accuracy: Coverage & Serendipity (Herlocker et al., 2004; Celma, 2010)
  └── Maximal Marginal Relevance / Diversity Reranking (Carbonell & Goldstein, 1998)
```

### 4.1 Classical Implicit Collaborative Filtering
In music consumption, explicit negative ratings ("1 star") are rare; listeners express preference through implicit scrobbles (listening events). Hu, Koren, and Volinsky (2008) formulated implicit matrix factorization by decoupling binary preference $p_{ui}$ from observation confidence $c_{ui}$:
$$p_{ui} = \begin{cases} 1 & \text{if } r_{ui} > 0 \\ 0 & \text{if } r_{ui} = 0 \end{cases}, \quad c_{ui} = 1 + \alpha r_{ui}$$
where $r_{ui}$ is the scrobble count and $\alpha$ is the confidence rate hyperparameter. The optimization minimizes:
$$\mathcal{L}_{\text{ALS}} = \sum_{u, i} c_{ui} (p_{ui} - \mathbf{x}_u^T \mathbf{y}_i)^2 + \lambda \left(\sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2\right)$$

### 4.2 Bayesian Personalized Ranking (BPR)
Rendle et al. (2009) demonstrated that pointwise regression is suboptimal for item ranking. BPR directly optimizes the pairwise ranking criterion:
$$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in \mathcal{D}_S} \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \lambda_\Theta \|\Theta\|_2^2$$
where track $i$ was consumed by user $u$, track $j$ is an unobserved negative sample, and $\hat{x}_{ui} = \mathbf{w}_u^T \mathbf{h}_i$.

### 4.3 Neural Two-Tower Retrieval
Following Yi et al. (2019), the dual-encoder architecture projects users and tracks into a shared metric embedding space $\mathbb{R}^d$:
$$\mathbf{u} = \psi(x_u; \theta_U), \quad \mathbf{v} = \phi(y_i; \theta_V)$$
Scoring is defined via inner product or cosine similarity $s(u, i) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$. Logits are trained under sampled softmax with temperature $\tau$ and popularity bias correction:
$$\mathcal{L}_{\text{TwoTower}} = -\sum_{i \in \mathcal{B}} \log \frac{\exp((s(u_i, v_i) - \log P(v_i)) / \tau)}{\sum_{j \in \mathcal{B}} \exp((s(u_i, v_j) - \log P(v_j)) / \tau)}$$

### 4.4 Discovery & Reranking Formulation
Following Carbonell & Goldstein (1998) and Celma (2010), discovery reranking operates on top-$M$ candidate items retrieved from the first-stage model ($M \gg K$). The Maximal Marginal Relevance (MMR) objective iteratively selects item $i^*$ for recommendation set $S$:
$$i^* = \arg\max_{i \in \mathcal{C} \setminus S} \left[ \lambda \cdot \text{Rel}(u, i) - (1 - \lambda) \max_{j \in S} \text{Sim}(i, j) \right]$$
where $\text{Rel}(u, i)$ is the normalized model relevance score, $\text{Sim}(i, j)$ is the cosine similarity between item representation vectors in embedding space, and $\lambda \in [0, 1]$ controls the relevance-diversity trade-off.

---

## 5. Scope Boundaries

### In Scope (Phases 0–6 MVP):
* Data pipeline processing Last.fm 1K scrobbles with canonical track resolution and strict global chronological splitting.
* 4 classical baselines: Global Popularity, Recency-Weighted Popularity, Implicit ALS, and BPR.
* Dual-encoder PyTorch Two-Tower neural retrieval system with FAISS ANN index.
* Discovery reranking engine implementing MMR and discovery-constrained optimization.
* Offline evaluation module reporting predictive metrics ($\text{Recall}@K, \text{NDCG}@K$) alongside discovery metrics ($\text{Novelty}@K, \text{ILD}, \text{Catalog Coverage}, \text{Gini}$).
* Full-stack interactive system: FastAPI backend service + React frontend dashboard with 5 core discovery experiences.

### Out of Scope (Reserved for Phases 7–9):
* Complex sequential Transformers (SASRec, BERT4Rec) and multi-session RNNs.
* Real-time streaming ingestion (Kafka) and distributed training (Ray / Kubernetes).
* Online A/B live traffic experimentation (Last.fm 1K provides historical offline logs).
