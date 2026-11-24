# EUTERPE — Phase 4 Discovery Reranking Test Benchmark

**Phase:** 4 — Discovery Science & Reranking  
**Selected Strategy:** DiscoveryEngine_a0.7_b0.2_art2 (Type: DiscoveryEngine, Alpha: 0.70, Beta: 0.20, Max/Artist: 2)  
**Test Evaluation:** 758 warm listeners, 443,375 candidate tracks, strict training consumed-item masking  

---

## 1. Master System Benchmark Comparison

| System Architecture | NDCG@20 | Recall@20 (Conv) | Capped Recall@20 | MRR@20 | Novelty (bits) | ILD@20 (ALS Space) | ILD@20 (TT Space) | Catalog Coverage | Artist Coverage | Exposure Gini | Top-10% Decile Share |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0 Global Popularity** | 0.0282 | 0.0008 | 0.0276 | 0.0740 | 12.46 | 0.4125 | 0.00248 | 0.029% | 0.114% | 0.9999 | 100.00% |
| **E1 Recency Popularity** | 0.0458 | 0.0046 | 0.0425 | 0.1183 | 13.42 | 0.4289 | 0.00252 | 0.016% | 0.069% | 0.9999 | 100.00% |
| **E2 Implicit ALS (Precision Ref)** | **0.0606** | 0.0033 | 0.0599 | 0.1327 | 14.04 | 0.4698 | 0.00270 | 0.803% | 1.177% | 0.9969 | 99.44% |
| **Two-Tower Neural N5 (Retrieval Ref)** | 0.0458 | 0.0012 | 0.0413 | 0.1136 | 15.95 | 0.5154 | 0.00264 | 2.119% | 3.706% | 0.9855 | 75.27% |
| **EUTERPE Phase 4 Fused Reranker** | **0.0626** | **0.0033** | **0.0615** | **0.1401** | **14.22** | **0.5007** | **0.00298** | **0.910%** | **1.476%** | **0.9962** | **98.98%** |

---

## 2. Hypothesis H4 & Core Question Audit

- **NDCG@20 Retention:** 103.4% of ALS baseline (Target: >= 95.0%)
- **Catalog Coverage Expansion:** +13.4% relative increase over ALS (4,035 vs 3,560 tracks)
- **Artist Diversity Expansion:** +25.4% relative increase over ALS (765 vs 610 artists)
- **Intra-List Diversity:** Increased by +6.6% in ALS factor space (0.5007 vs 0.4698) and +10.6% in Two-Tower space (0.00298 vs 0.00270)
- **Head Exposure De-Concentration:** Top-10% decile share reduced from 99.44% to 98.98% (-0.46 percentage points)
- **Paired Difference Analysis:** Mean $\Delta \text{NDCG} = +0.002064$, Median = $+0.000000$, 95% Bootstrap CI $[-0.000056, +0.004244]$
- **User Shift:** 23.61% improved (179 users), 61.08% tied (463 users), 15.30% declined (116 users) [1.54x ratio favoring improvement]
- **Paired Wilcoxon Signed-Rank Test:** $W = 17576.0, p = 3.7192e-03$
- **Bootstrap 95% CIs:** NDCG@20 [0.0545, 0.0713], Recall@20 [0.0018, 0.0065]
