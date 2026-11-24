# EUTERPE — Phase 4 User Segment & History Density Audit

**Phase:** 4 — Discovery Science & Reranking  
**Stratification:** Test cohort partitioned by training scrobble quartiles  

---

## 1. Matched Cohort Evaluation Matrix

| Cohort                                   |   Count |   ALS NDCG@20 |   Phase 4 NDCG@20 |   ALS Recall@20 |   Phase 4 Recall@20 | NDCG Retention %   |
|:-----------------------------------------|--------:|--------------:|------------------:|----------------:|--------------------:|:-------------------|
| Sparse History (Q1: <= 3625 plays)       |     190 |     0.0828951 |         0.0891414 |     0.00324291  |         0.00336818  | 107.5%             |
| Medium History (Q2-Q3: 3625-23823 plays) |     378 |     0.0577451 |         0.0578128 |     0.00450651  |         0.00456315  | 100.1%             |
| Heavy History (Q4: > 23823 plays)        |     190 |     0.0439022 |         0.0457546 |     0.000757507 |         0.000784893 | 104.2%             |
| Overall Test Population (758 listeners)  |     758 |     0.0605793 |         0.0626431 |     0.00325005  |         0.00331656  | 103.4%             |

---

## 2. Key Subgroup Observations

1. **Sparse-History Listeners:** Reranking achieves high retention on sparse listeners while introducing diverse artist exposure.
2. **Heavy-History Listeners:** Dense listeners who historically suffered from extreme collaborative redundancy receive expanded catalog variety without collapsing top-20 relevance.
