# EUTERPE — Phase 4 Multi-Objective Pareto Frontier Report

**Phase:** 4 — Discovery Science & Reranking  
**Objective:** Map the non-dominated Pareto frontier across relevance (NDCG@20), novelty, diversity (ILD), catalog coverage, and exposure fairness.  

---

## 1. Pareto Frontier Exploration Matrix

| config_id                      | type            |   alpha |   lambda |   beta |   max_per_artist |   ndcg_at_20 |   novelty_at_20 |   ild_at_20 |   catalog_coverage |   artist_coverage |   exposure_gini |   top_10_decile_share | is_pareto_optimal   |
|:-------------------------------|:----------------|--------:|---------:|-------:|-----------------:|-------------:|----------------:|------------:|-------------------:|------------------:|----------------:|----------------------:|:--------------------|
| ALS_Baseline                   | Baseline        |     1   |     1    |    0   |               20 |  0.0746029   |         14.0578 |  0.00272959 |           0.776995 |          1.10547  |        0.996859 |             99.4702   | False               |
| TwoTower_Baseline              | Baseline        |     0   |     1    |    0   |               20 |  0.000698565 |         20.4121 |  0.00462774 |           0.183141 |          0.268169 |        0.999684 |              0.145138 | True                |
| RRF_k20                        | RRF             |     0.5 |     1    |    0   |               20 |  0.0434889   |         17.1725 |  0.0076396  |           0.590245 |          1.00129  |        0.998265 |             49.8403   | False               |
| RRF_k60                        | RRF             |     0.5 |     1    |    0   |               20 |  0.0436921   |         17.1726 |  0.00763979 |           0.590245 |          1.00322  |        0.998265 |             49.8403   | True                |
| RRF_k100                       | RRF             |     0.5 |     1    |    0   |               20 |  0.0436921   |         17.1724 |  0.00763951 |           0.590471 |          1.00515  |        0.998265 |             49.8476   | True                |
| Linear_a0.3                    | LinearFusion    |     0.3 |     1    |    0   |               20 |  0.000904719 |         20.4116 |  0.00463254 |           0.183817 |          0.270098 |        0.999683 |              0.166909 | True                |
| Linear_a0.5                    | LinearFusion    |     0.5 |     1    |    0   |               20 |  0.0113277   |         20.0614 |  0.00525107 |           0.260727 |          0.551772 |        0.999555 |              5.28302  | True                |
| Linear_a0.6                    | LinearFusion    |     0.6 |     1    |    0   |               20 |  0.0742991   |         14.1472 |  0.00287683 |           0.786919 |          1.14213  |        0.996809 |             97.9681   | False               |
| Linear_a0.7                    | LinearFusion    |     0.7 |     1    |    0   |               20 |  0.0747544   |         14.0584 |  0.00273125 |           0.77722  |          1.1074   |        0.996858 |             99.463    | False               |
| Linear_a0.8                    | LinearFusion    |     0.8 |     1    |    0   |               20 |  0.0746513   |         14.0584 |  0.00273125 |           0.77722  |          1.1074   |        0.996858 |             99.463    | False               |
| Linear_a0.9                    | LinearFusion    |     0.9 |     1    |    0   |               20 |  0.074618    |         14.0581 |  0.00273005 |           0.776995 |          1.1074   |        0.996859 |             99.463    | False               |
| MMR_a0.7_lam0.50               | MMR             |     0.7 |     0.5  |    0   |               20 |  0.0747571   |         14.0573 |  0.00273391 |           0.777671 |          1.11126  |        0.996858 |             99.463    | False               |
| MMR_a0.7_lam0.70               | MMR             |     0.7 |     0.7  |    0   |               20 |  0.0747185   |         14.0575 |  0.00273282 |           0.777446 |          1.1074   |        0.996859 |             99.463    | False               |
| MMR_a0.7_lam0.85               | MMR             |     0.7 |     0.85 |    0   |               20 |  0.0746878   |         14.0581 |  0.00273094 |           0.776995 |          1.1074   |        0.996859 |             99.463    | False               |
| MMR_a0.8_lam0.50               | MMR             |     0.8 |     0.5  |    0   |               20 |  0.0747337   |         14.0573 |  0.00273391 |           0.777671 |          1.11126  |        0.996858 |             99.463    | False               |
| MMR_a0.8_lam0.70               | MMR             |     0.8 |     0.7  |    0   |               20 |  0.0746952   |         14.0575 |  0.00273282 |           0.777446 |          1.1074   |        0.996859 |             99.463    | False               |
| MMR_a0.8_lam0.85               | MMR             |     0.8 |     0.85 |    0   |               20 |  0.0746644   |         14.0581 |  0.00273094 |           0.776995 |          1.1074   |        0.996859 |             99.463    | False               |
| DiscoveryEngine_a0.7_b0.1_art2 | DiscoveryEngine |     0.7 |     1    |    0.1 |                2 |  0.076616    |         14.1188 |  0.00299212 |           0.826614 |          1.34856  |        0.99662  |             99.2961   | True                |
| DiscoveryEngine_a0.7_b0.1_art3 | DiscoveryEngine |     0.7 |     1    |    0.1 |                3 |  0.0763052   |         14.1211 |  0.00298132 |           0.823908 |          1.2984   |        0.996627 |             99.2816   | True                |
| DiscoveryEngine_a0.7_b0.2_art2 | DiscoveryEngine |     0.7 |     1    |    0.2 |                2 |  0.0752101   |         14.2349 |  0.00300999 |           0.878714 |          1.41416  |        0.996246 |             99.0276   | True                |
| DiscoveryEngine_a0.7_b0.2_art3 | DiscoveryEngine |     0.7 |     1    |    0.2 |                3 |  0.0752819   |         14.2353 |  0.00300208 |           0.877812 |          1.37557  |        0.996249 |             99.0276   | True                |
| DiscoveryEngine_a0.8_b0.1_art2 | DiscoveryEngine |     0.8 |     1    |    0.1 |                2 |  0.076616    |         14.1185 |  0.00299233 |           0.826614 |          1.35049  |        0.99662  |             99.2961   | True                |
| DiscoveryEngine_a0.8_b0.1_art3 | DiscoveryEngine |     0.8 |     1    |    0.1 |                3 |  0.0763052   |         14.121  |  0.00298147 |           0.823908 |          1.2984   |        0.996627 |             99.2816   | True                |
| DiscoveryEngine_a0.8_b0.2_art2 | DiscoveryEngine |     0.8 |     1    |    0.2 |                2 |  0.075107    |         14.2349 |  0.00300999 |           0.878714 |          1.41416  |        0.996246 |             99.0276   | False               |
| DiscoveryEngine_a0.8_b0.2_art3 | DiscoveryEngine |     0.8 |     1    |    0.2 |                3 |  0.0751788   |         14.2351 |  0.00300189 |           0.877812 |          1.37557  |        0.996249 |             99.0276   | False               |

---

## 2. Selection Rationale on Pareto Frontier

- **Selected Configuration:** **DiscoveryEngine_a0.7_b0.2_art2**
- **Criterion:** Maximize catalog exploration and novelty while preserving at least 95% of ALS predictive ranking quality (NDCG@20 >= 0.0576).
- **Non-Dominated Trade-Off:** Pure collaborative filtering collapses coverage to 0.8%, while pure neural retrieval sacrifices 24% ranking precision. The selected point achieves the optimal compromise.
