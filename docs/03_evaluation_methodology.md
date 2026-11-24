# EUTERPE — Evaluation Methodology & Discovery Metrics Specification
**Version:** 1.0.0  
**Phase:** 0 — Scientific Specification and Research Foundation  

---

## 1. Temporal Splitting Protocol

In offline evaluation of recommendation engines, **random user or interaction splitting introduces fatal data leakage**: future tracks appear in the training split, artificially inflating predictive scores and disguising temporal drift.

### 1.1 Global Chronological Partitioning
EUTERPE establishes strict **global chronological cutoffs** across the unified scrobble timeline:

```
Timeline: [2005-02-14] =============================> [2009-05-05]
│◀────────────── Train Split (80%) ──────────────▶│◀── Val (10%) ──▶│◀── Test (10%) ──▶│
│ t < T_train                                      │ T_train <= t < T_val │ t >= T_val         │
```

* **Cutoff Determination:**
  * $T_{\text{train}}$: Timestamp corresponding to the $80^{\text{th}}$ percentile of all interaction events.
  * $T_{\text{val}}$: Timestamp corresponding to the $90^{\text{th}}$ percentile of all interaction events.
  * $T_{\text{test}}$: All remaining interactions ($90^{\text{th}}$ to $100^{\text{th}}$ percentile).
* **Leakage Verification Invariants:**
  1. $\max(t_{\text{train}}) < T_{\text{train}} \le \min(t_{\text{val}})$.
  2. $\max(t_{\text{val}}) < T_{\text{val}} \le \min(t_{\text{test}})$.
  3. No track popularity counts, artist statistics, or negative samples computed on validation/test sets can be accessed during training.

### 1.2 User and Item Eligibility Protocol (Zero Future Leakage)
> [!IMPORTANT]
> **Strict Pre-Condition on Eligibility:**
> User and item eligibility thresholds MUST be computed exclusively from interactions occurring before the training cutoff ($t < T_{\text{train}}$). Future interactions occurring in the validation or test windows ($t \ge T_{\text{train}}$) are NEVER accessible for filtering.
> 
> * **Eligible Training Users ($\mathcal{U}_{\text{train}}$):** Users with $\ge 50$ valid scrobbles in the training window ($t < T_{\text{train}}$).
> * **Eligible Training Tracks ($\mathcal{I}_{\text{train}}$):** Tracks with $\ge 5$ valid scrobbles in the training window ($t < T_{\text{train}}$).
> * **Validation & Test Invariant:** Only interactions from users in $\mathcal{U}_{\text{train}}$ and items in $\mathcal{I}_{\text{train}}$ form the warm-start test sets. Cold-start interactions (new users or items appearing exclusively after $T_{\text{train}}$) are partitioned and reported in separate cold-start diagnostic tables.

---

## 2. Predictive Relevance Metrics (Rank Cutoff $K = 20$)

Let $\mathcal{T}_u$ denote the set of ground-truth tracks consumed by user $u$ in the test window, and $\hat{R}_u = [i_1, i_2, \dots, i_K]$ be the ordered list of top-$K$ recommendations.

### 2.1 Conventional Recall@K (Primary Benchmark Metric)
Fraction of ground-truth test items successfully retrieved in the top-$K$, aligned with established recommender literature (Herlocker et al., Rendle et al.):
$$\text{Recall}@K(u) = \frac{|\hat{R}_u \cap \mathcal{T}_u|}{|\mathcal{T}_u|}, \quad \text{Recall}@K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{Recall}@K(u)$$

### 2.2 Capped Recall@K (Auxiliary Diagnostic Metric)
To assess hit-rate saturation when $|\mathcal{T}_u| > K$:
$$\text{CappedRecall}@K(u) = \frac{|\hat{R}_u \cap \mathcal{T}_u|}{\min(K, |\mathcal{T}_u|)}$$

### 2.2 Normalized Discounted Cumulative Gain (NDCG@K)
Evaluates ranking position of relevant tracks with logarithmic discount:
$$\text{DCG}@K(u) = \sum_{j=1}^K \frac{2^{\mathbb{I}(i_j \in \mathcal{T}_u)} - 1}{\log_2(j + 1)}, \quad \text{IDCG}@K(u) = \sum_{j=1}^{\min(K, |\mathcal{T}_u|)} \frac{1}{\log_2(j + 1)}$$
$$\text{NDCG}@K(u) = \frac{\text{DCG}@K(u)}{\text{IDCG}@K(u)}, \quad \text{NDCG}@K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{NDCG}@K(u)$$

### 2.3 Mean Reciprocal Rank (MRR@K)
Speed of reaching the first relevant recommendation:
$$\text{MRR}@K(u) = \begin{cases} \frac{1}{\min \{ j \le K \mid i_j \in \mathcal{T}_u \}} & \text{if } \hat{R}_u \cap \mathcal{T}_u \ne \emptyset \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Discovery Intelligence Metrics

Traditional accuracy metrics penalize systems that suggest long-tail items simply because unobserved items in historical logs cannot be distinguished from disliked items. Discovery metrics quantify catalog breadth, unexpectedness, and distributional equity.

### 3.1 Catalog & Artist Coverage
Proportion of unique items and artists surfaced across the entire evaluation cohort:
$$\text{CatalogCoverage}@K = \frac{|\bigcup_{u \in \mathcal{U}} \hat{R}_u|}{|\mathcal{I}|}, \quad \text{ArtistCoverage}@K = \frac{|\bigcup_{u \in \mathcal{U}} \text{Artists}(\hat{R}_u)|}{|\mathcal{A}|}$$

### 3.2 Novelty@K (Self-Information)
Quantifies long-tail exposure by measuring the average negative log probability (surprisal) of recommended items relative to catalog popularity:
$$\text{Novelty}@K(u) = -\frac{1}{K} \sum_{i \in \hat{R}_u} \log_2 P(i), \quad P(i) = \frac{\sum_{u} r_{ui}}{\sum_{v \in \mathcal{U}} \sum_{j \in \mathcal{I}} r_{vj}}$$
Higher values indicate recommendations concentrated in the long tail.

### 3.3 Intra-List Diversity (ILD)
Average pairwise semantic distance between recommended tracks in embedding space $\mathbb{R}^d$:
$$\text{ILD}@K(u) = \frac{2}{K(K-1)} \sum_{j=1}^K \sum_{l=j+1}^K \left( 1 - \frac{\mathbf{v}_j^T \mathbf{v}_l}{\|\mathbf{v}_j\|_2 \|\mathbf{v}_l\|_2} \right)$$

### 3.4 Gini Coefficient of Recommendation Exposure
Quantifies the inequality of recommendation exposure across all catalog items:
$$G = \frac{\sum_{i=1}^{|\mathcal{I}|} (2i - |\mathcal{I}| - 1) \, c_{(i)}}{|\mathcal{I}| \sum_{i=1}^{|\mathcal{I}|} c_{(i)}}$$
where $c_{(i)}$ is the sorted ascending count of times item $i$ appears in any user's top-$K$ recommendations.
* $G = 1.0$: Absolute monopoly (a few popular tracks receive 100% of recommendations).
* $G = 0.0$: Perfectly uniform catalog exposure.

### 3.5 Serendipity@K
Quantifies items that are both **relevant** (present in test set $\mathcal{T}_u$) and **unexpected** relative to the user's historical profile $\mathcal{H}_u$:
$$\text{Serendipity}@K(u) = \frac{1}{K} \sum_{i \in \hat{R}_u} \mathbb{I}(i \in \mathcal{T}_u) \cdot \min_{j \in \mathcal{H}_u} \left( 1 - \frac{\mathbf{v}_i^T \mathbf{v}_j}{\|\mathbf{v}_i\|_2 \|\mathbf{v}_j\|_2} \right)$$

---

## 4. User Segmentation Framework

Listeners do not consume music uniformly. EUTERPE evaluates performance across four distinct user segments:

| Cohort | Defining Criterion | Hypothesis Under Discovery Reranking |
|---|---|---|
| **Mainstream Listeners** | $> 70\%$ of scrobbles in top $10\%$ popular artists | High sensitivity to novelty; steep NDCG degradation if reranked aggressively. |
| **Genre Specialists** | High concentration ($> 60\%$) in a single acoustic/genre cluster | High intra-list cohesion required; MMR within cluster preserves relevance. |
| **Niche Enthusiasts** | $> 50\%$ of scrobbles in the long tail ($P(i) < 0.01$) | Receptive to high novelty; minimal NDCG penalty. |
| **Exploratory Listeners** | High historical ILD ($> 0.70$) across diverse genres | Highest serendipity utility gain from Two-Tower + discovery reranker. |

---

## 5. Statistical Rigor & Uncertainty Estimation

To ensure scientific defensibility:
* **Bootstrap Confidence Intervals:** Every reported metric is accompanied by a $95\%$ confidence interval computed via $B = 1,000$ paired user bootstrap iterations.
* **Paired Hypothesis Testing:** Wilcoxon signed-rank tests are executed between baseline and neural/discovery models across identical user evaluation sets. Statistical significance is declared only when $p < 0.01$.
