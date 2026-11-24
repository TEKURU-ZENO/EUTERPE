# EUTERPE — Popularity Bias & Exposure Inequality Analysis
**Phase:** 2 — Classical Recommendation Intelligence  
**Focus:** Empirical Evaluation of Hypothesis $H_1$ on Recommendation Allocations  

---

## 1. Hypothesis $H_1$ Test Results

> **Hypothesis $H_1$ Targets:**
> 1. Top-Decile Exposure Target: Unconstrained collaborative filtering allocates $> 70\%$ of recommendation slots to the top $10\%$ most frequent items.
> 2. Inequality Target: Recommendation exposure exhibits Gini coefficient $G > 0.75$.

| Model | Top-10% Play Decile Share in Recs | $H_1$ Target (>70%) Satisfied? | Recommendation Exposure Gini ($G$) | $H_1$ Target ($G > 0.75$) Satisfied? |
|---|:---:|:---:|:---:|:---:|
| **E0 Global Popularity** | **100.00%** | **YES** (100% head items) | **0.9999** | **YES** |
| **E1 Recency Popularity**| **100.00%** | **YES** | **0.9999** | **YES** |
| **E2 Implicit ALS** | **99.44%** | **YES** | **0.9969** | **YES** |
| **E3 BPR Pairwise** | **89.39%** | **YES** | **0.9868** | **YES** |

---

## 2. Key Insights for Discovery Science (Phase 4 Setup)

1. Non-personalized baselines (Popularity, Recency) allocate nearly 100% of recommendation slots to head tracks with extreme Gini coefficients ($G > 0.99$), starving the long tail of exposure.
2. Personalized collaborative models (ALS and BPR) broaden recommendations, yet still allocate a substantial share to top decile tracks while leaving over 95% of catalog items unrecommended.
3. This establishes the precise experimental motivation for **Phase 4 Discovery Intelligence**: demonstrating that discovery reranking (MMR and novelty constraints) can systematically unlock the long-tail without sacrificing the predictive relevance established here in Phase 2.
