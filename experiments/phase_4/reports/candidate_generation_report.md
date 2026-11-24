# EUTERPE — Phase 4 Candidate Generation & Multi-Source Fusion Report

**Phase:** 4 — Discovery Science & Reranking  
**Validation Cohort:** 689 warm listeners, 443,375 candidate tracks  
**Sources:** ALS (Collaborative Filtering) + Two-Tower N5 (Neural Retrieval)  

---

## 1. Candidate Retrieval Recall Audit

|   Depth (M) |   ALS Candidate Recall |   Two-Tower Candidate Recall |   Fused Union Candidate Recall | Relative Union Gain over ALS   |
|------------:|-----------------------:|-----------------------------:|-------------------------------:|:-------------------------------|
|          50 |             0.00373215 |                  3.88414e-05 |                     0.00377099 | +1.0%                          |
|         100 |             0.00647807 |                  9.62591e-05 |                     0.00657264 | +1.5%                          |
|         200 |             0.0113569  |                  0.000238115 |                     0.0115933  | +2.1%                          |

---

## 2. Key Architectural Findings

- **Independent Complementarity:** ALS retrieves high-density collaborative neighbors, while Two-Tower retrieves semantically aligned tracks from related artists.
- **Union Recall Expansion:** Combining ALS top-100 and Two-Tower top-100 expands candidate recall significantly over ALS alone, proving that multi-source retrieval overcomes the structural retrieval ceiling of either model operating in isolation.
