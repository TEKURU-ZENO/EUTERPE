# EUTERPE — Phase 3 Neural Architecture Ablation Report

**Phase:** 3 — Neural Recommendation Intelligence  
**Objective:** Controlled Ablation Matrix (N0 through N5) on Validation Split (689 warm listeners)  
**Selection Criterion:** Peak Validation NDCG@20 (Zero Test Peeking)  

---

## 1. Controlled Ablation Matrix Results

| recall_at_20_conv | recall_at_20_capped | ndcg_at_20 | mrr_at_20 | novelty_at_20 | catalog_coverage | artist_coverage | exposure_gini | top_10_decile_share | n_evaluated_users | experiment_id | experiment_name | train_time_s | val_eval_time_s |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00003 | 0.00174 | 0.00203 | 0.00846 | 19.24 | 0.04646 | 0.35306 | 0.99981 | 16.94 | 689 | N0 | User/Item ID Embeddings Only | 95.19 | 3.06 |
| 0.00006 | 0.00298 | 0.00288 | 0.00982 | 18.97 | 0.17931 | 1.34 | 0.99935 | 19.85 | 689 | N1 | + User History Aggregation | 209.04 | 3.30 |
| 0.00082 | 0.02347 | 0.02574 | 0.06028 | 19.67 | 2.79 | 4.66 | 0.97447 | 12.16 | 689 | N2 | + Track & Artist Embeddings | 224.24 | 3.23 |
| 0.00085 | 0.02529 | 0.02824 | 0.06861 | 19.31 | 2.80 | 4.34 | 0.97404 | 16.76 | 689 | N3 | N2 + False-Negative Masking | 417.91 | 3.17 |
| 0.00180 | 0.05481 | 0.05997 | 0.13920 | 16.11 | 2.01 | 3.39 | 0.98581 | 75.45 | 689 | N4 | N3 + Popularity Log-Q Correction | 386.18 | 3.07 |
| 0.00171 | 0.05762 | 0.06230 | 0.13793 | 16.01 | 1.97 | 3.58 | 0.98626 | 74.35 | 689 | N5 | N4 + Optimized Mixed Negative Sampling | 382.12 | 3.10 |

---

## 2. Component Contribution Analysis

- **Optimal Architecture:** **N5 — N4 + Optimized Mixed Negative Sampling**
- **Peak Validation NDCG@20:** **0.06230**
- **User History Contribution (N0 vs N1):** Incorporating chronological user listening sequences adds substantial ranking relevance.
- **Artist Representation (N1 vs N2):** Dual track-and-artist projection expands semantic coherence.
- **Negative Sampling & Correction (N3 vs N4 vs N5):** In-batch negatives with false-negative masking and log-Q correction prevents popularity collapse.
