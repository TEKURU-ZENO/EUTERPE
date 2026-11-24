# EUTERPE — Phase 2 Hyperparameter Optimization Report
**Model:** Implicit Alternating Least Squares (ALS)  
**Objective:** Maximize Validation NDCG@20 (Zero Test Peeking)  
**Tuning Metric:** Validation Set (689 warm users, 215,399 tracks)  

---

## 1. Grid Search Parameter Sweep Results

| Factors | Regularization | Alpha | Input Representation | Iterations | Val NDCG@20 | Val Recall@20 | Fit Time (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 0.05 | 20.0 | `logcount` | 20 | **0.05886** | 0.00168 | 12.71 |
| 64 | 0.01 | 20.0 | `logcount` | 20 | **0.05636** | 0.00156 | 15.96 |
| 64 | 0.05 | 20.0 | `logcount` | 20 | **0.05585** | 0.00154 | 13.89 |
| 64 | 0.05 | 40.0 | `logcount` | 20 | **0.04751** | 0.00160 | 14.42 |
| 64 | 0.1 | 20.0 | `logcount` | 20 | **0.05617** | 0.00155 | 15.91 |
| 128 | 0.05 | 20.0 | `logcount` | 20 | **0.06210** | 0.00183 | 18.34 |
| 64 | 0.05 | 1.0 | `precomputed_confidence` | 20 | **0.04693** | 0.00160 | 13.28 |
| 64 | 0.05 | 1.0 | `binary` | 20 | **0.07499** | 0.00221 | 13.41 |

---

## 2. Selection Rationale & Safeguard Compliance
* **Optimal Configuration:** Factors = **64**, Regularization = **0.05**, Alpha = **1.0**, Input = `binary`.
* **Peak Validation NDCG@20:** **0.07499**.
* **Safeguard Compliance (Senior Scientific Review):** Confidence scaling is calibrated to input representation without double-scaling. Input `binary` with alpha=1.0 yielded superior preference recovery over unweighted binary interactions.
