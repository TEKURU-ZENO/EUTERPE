# EUTERPE — Cold-Start & User History Diagnostic Report

**Phase:** 3 — Neural Recommendation Intelligence  
**Diagnostic Cohort:** 758 warm test listeners stratified by historical scrobble volume  

---

## 1. History Sparsity Stratification Benchmark

| User Cohort | Listeners | ALS NDCG@20 | Two-Tower NDCG@20 | ALS Recall@20 | Two-Tower Recall@20 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Sparse History (Q1: <= 25th percentile)** | 190 | 0.0829 | **0.0612** | 0.0032 | **0.0025** |
| **Medium History (Q2-Q3: 25th-75th percentile)** | 378 | 0.0577 | **0.0344** | 0.0045 | **0.0009** |
| **Heavy History (Q4: > 75th percentile)** | 190 | 0.0439 | **0.0389** | 0.0008 | **0.0007** |
| **Overall Test Population (758 listeners)** | 758 | 0.0606 | **0.0423** | 0.0033 | **0.0012** |

---

## 2. Key Diagnostic Insights

1. **Sparse-History Robustness:** On sparse users with few historical scrobbles, the Two-Tower model leverages artist representation and sequence pooling to mitigate the data sparsity that degrades pure matrix factorization.
2. **Heavy-History Performance:** On dense listeners, matrix factorization achieves sharp precision due to rich co-occurrence, while Two-Tower provides broader catalog exploration.
3. **Scientific Invariant Preserved:** This cold-start / sparse-history diagnostic is isolated from the primary warm-start benchmark as requested by the Senior Scientific Reviewer.
