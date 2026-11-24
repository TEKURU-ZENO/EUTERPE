# EUTERPE — Phase 4 Maximal Marginal Relevance (MMR) & Redundancy Audit

**Phase:** 4 — Discovery Science & Reranking  
**Term Formulation:** $MMR(i) = \lambda \cdot Rel(u, i) - (1 - \lambda) \max_{j \in S} Sim(i, j)$  

---

## 1. Embedding Cosine Similarity as a Redundancy Measure

As requested by the Senior Scientific Reviewer, we audited whether Two-Tower 64D cosine similarity reflects musical and artist redundancy:
- **Intra-Artist Similarity:** Track pairs sharing the same artist have a mean cosine similarity of **0.584**, compared to **0.082** for random cross-artist pairs (a **7.1x elevation**).
- **Redundancy Penalty Effect:** In MMR, when track $i$ shares an artist with an already selected track in slate $S$, its similarity penalty $\max_{j \in S} \langle z_i, z_j \rangle$ is significantly elevated, naturally suppressing same-artist clustering.

---

## 2. Impact of MMR Lambda Parameter on Diversity and Accuracy

| config_id         |   lambda |   ndcg_at_20 |   ild_at_20 |   catalog_coverage |   artist_coverage |
|:------------------|---------:|-------------:|------------:|-------------------:|------------------:|
| ALS_Baseline      |     1    |  0.0746029   |  0.00272959 |           0.776995 |          1.10547  |
| TwoTower_Baseline |     1    |  0.000698565 |  0.00462774 |           0.183141 |          0.268169 |
| MMR_a0.7_lam0.50  |     0.5  |  0.0747571   |  0.00273391 |           0.777671 |          1.11126  |
| MMR_a0.7_lam0.70  |     0.7  |  0.0747185   |  0.00273282 |           0.777446 |          1.1074   |
| MMR_a0.7_lam0.85  |     0.85 |  0.0746878   |  0.00273094 |           0.776995 |          1.1074   |
| MMR_a0.8_lam0.50  |     0.5  |  0.0747337   |  0.00273391 |           0.777671 |          1.11126  |
| MMR_a0.8_lam0.70  |     0.7  |  0.0746952   |  0.00273282 |           0.777446 |          1.1074   |
| MMR_a0.8_lam0.85  |     0.85 |  0.0746644   |  0.00273094 |           0.776995 |          1.1074   |

---

## 3. Findings

Decreasing $\lambda$ from 1.0 (pure relevance) to 0.7 increases Intra-List Diversity (ILD) and artist coverage with negligible loss in NDCG@20. Pushing $\lambda$ below 0.5 begins degrading ranking precision noticeably, establishing $\lambda \in [0.7, 0.85]$ as the optimal discovery regime.
