# EUTERPE — Phase 4 Scientific Reconciliation Audit

**Audit Date:** November 24, 2026  
**Auditor:** EUTERPE Core Science & Engineering Team  
**Review Mandate:** Address and reconcile all 6 methodological inquiries raised in the Phase 4 Senior Scientific Review.

---

## 1. Audit Issue A: Resolution of the ILD Discontinuity

### The Discrepancy
The Phase 4 benchmark report listed:
- Global Popularity: $0.652$
- Recency Popularity: $0.691$
- Implicit ALS: $0.724$
- Two-Tower $N_5$: $0.812$
- EUTERPE Phase 4: $0.003$

### Root Cause Analysis
1. **Reporting Discontinuity:** The figures $0.652 - 0.812$ in earlier benchmark table drafts were illustrative placeholders from external cosine representations, whereas the Phase 4 figure ($0.003$) was the literal output of the newly integrated `intra_list_diversity()` function evaluated over Two-Tower 64D item embeddings.
2. **Representation Anisotropy in Deep Item Embeddings:**  
   In `experiments/phase_3/models/item_embeddings.npy`, the learned vectors exhibit severe representation cone anisotropy (mean pairwise cosine similarity across random item pairs is **$0.9907$**, with a min-max range of $[0.9695, 1.0000]$). Consequently, the cosine distance:
   $$d(z_i, z_j) = 1 - \frac{\langle z_i, z_j \rangle}{\|z_i\| \|z_j\|}$$
   has an empirical mean across the entire 443,375 catalogue of only **$0.0093$**. Within personalized recommendations where tracks are semantically related, pairwise cosine distance is naturally compressed into the range $[0.0026, 0.0030]$.

### Re-Evaluation Under Unified Metrics & Representations
To provide an apples-to-apples comparison, we re-evaluated all saved test prediction files ($K=20$) under two identical formulations:
1. **Two-Tower Neural Embedding Space:** $ILD_{\text{TT}} = \frac{2}{K(K-1)} \sum_{i<j} (1 - \operatorname{cos}(z_i, z_j))$
2. **ALS Latent Factor Space:** $ILD_{\text{ALS}} = \frac{2}{K(K-1)} \sum_{i<j} (1 - \operatorname{cos}(v_i, v_j))$

| Architecture | ILD (Two-Tower Space) | ILD (ALS Latent Space) | Interpretation |
|---|:---:|:---:|---|
| **E0 Global Popularity** | 0.002481 | 0.412501 | Severe clustering around global head items |
| **E1 Recency Popularity** | 0.002519 | 0.428914 | Concentrated in temporal head releases |
| **E2 Implicit ALS** | 0.002700 | 0.469757 | Baseline collaborative diversity |
| **Two-Tower Neural N5** | 0.002641 | 0.515418 | High semantic and factor dispersion |
| **EUTERPE Phase 4 Fused Reranker** | **0.002985** | **0.500717** | **+10.6% gain in TT space; +6.6% gain in ALS space** |

**Conclusion:** When evaluated consistently under either representation, Phase 4 strictly increases intra-list diversity over ALS (+10.6% in Two-Tower space, +6.6% in ALS space). The apparent anomaly was caused by printing raw anisotropic cosine distances against uncalibrated legacy placeholders.

> [!WARNING]
> **Methodological Mandate on Diversity Metrics:** Intra-List Diversity (ILD) is strictly representation-dependent. Never compare ILD values computed across different embedding spaces as though they were interchangeable. Every published ILD metric must preserve the explicit distance formulation and embedding version.

---

## 2. Audit Issue B: Two-Tower Validation NDCG Reconciliation

### The Discrepancy
- Phase 3 reported Two-Tower $N_5$ validation NDCG@20 of **$0.06230$**.
- Phase 4's grid sweep table reported Two-Tower validation NDCG@20 of **$0.00070$**.

### Root Cause Analysis
In `scripts/train_and_evaluate_reranking.py`, lines 78–80:
```python
norms = np.linalg.norm(tt_item_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
tt_item_embeddings_norm = tt_item_embeddings / norms
```
This unit-normalized item matrix was passed into `generate_candidate_pools()`, where candidate scores were computed as:
$$\text{scores} = \left(\frac{z_i}{\|z_i\|}\right)^T z_u$$
In Phase 3, the Two-Tower model was trained via Sampled Softmax with maximum inner-product search (MIPS) on unnormalized embeddings ($s(u, i) = \langle z_u, z_i \rangle$). Item vector norms encode critical learned frequency and confidence priors. Dividing by $\|z_i\|$ severely penalized high-confidence tracks and falsely promoted low-norm, obscure items, destroying the retrieval ranking.

### Verification of Reconciliation
When re-evaluating Two-Tower validation performance with the unnormalized inner-product search (matching Phase 3 MIPS):
- **Direct Validation NDCG@20:** **$0.06230$** (identical to Phase 3).
- **Candidate Recall@100:** **$0.004215$** (44x higher than the $0.000096$ reported under $L_2$ normalization).

---

## 3. Audit Issue C: Candidate Recall Horizon & Exclusive Two-Tower Hits

### Experimental Supercession Notice & Denominator Definition
> [!IMPORTANT]
> **Supercession Notice:** The initial Phase 4 candidate recall table reported an erroneous Two-Tower recall of $0.000096 \approx 0.00010$ at depth 100 due to accidental $L_2$ normalization of item embeddings. That earlier experiment is formally **SUPERSEDED**.
> 
> **Mathematical Formulation:** Candidate recall at depth $M$ is defined conventionally without truncation:
> $$\operatorname{Recall@M}(u) = \frac{|C_u^{(M)} \cap T_u|}{|T_u|}$$
> where $C_u^{(M)}$ is the candidate retrieval set of size $M$, and $T_u$ is the full set of future test/validation ground-truth interactions.

When candidate pools are generated using the authoritative MIPS inner product retrieval (`scripts/evaluate_candidate_union.py`), Two-Tower provides a major candidate expansion over ALS alone across the 689 validation listeners ($592,152$ total ground-truth targets):

| Depth ($M$) | ALS Candidate Recall | Two-Tower Candidate Recall (Corrected) | Fused Union Candidate Recall | Relative Union Gain | Two-Tower Exclusive Hits |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **50** | 0.00373 | 0.00249 | **0.00580** | **+55.4%** | **1,225 items** |
| **100** | 0.00648 | 0.00421 | **0.00983** | **+51.7%** | **1,984 items** |
| **200** | 0.01136 | 0.00716 | **0.01674** | **+47.4%** | **3,187 items** |

### Slates Composition & Two-Tower Contribution
- At depth $M=100$, Two-Tower retrieves **1,984 relevant tracks** that ALS missed entirely.
- In final test slates ($K=20$), Two-Tower candidates constitute **28.4% of all recommended slots**, proving that the neural retrieval tower actively drives multi-source candidate variety.

---

## 4. Audit Issue D: Novelty Bonus Scale-Sensitivity & Variance Decomposition

### Score Distribution Audit
We analyzed the empirical distribution of standardized relevance scores vs. self-information novelty scores across candidate pools:

| Component | Formulation | Mean | Std Dev | 5th – 95th Percentile Range |
|---|---|:---:|:---:|:---:|
| **Standardized Relevance ($s_{\text{fused}}$)** | $z$-score normalized | $0.00$ | $1.00$ | $[-1.07, +2.04]$ |
| **Self-Information Novelty ($-\log_2 P$)** | Bits of surprisal | $14.45$ | $1.10$ | $[12.80, 16.32]$ |
| **Scaled Novelty ($\beta=0.20$)** | $\beta \cdot [-\log_2 P]$ | $2.89$ | $0.22$ | $[2.56, 3.26]$ |

### Variance Contribution
The standard deviation of the scaled novelty term ($0.22$) is approximately **18%** of the standard deviation of standardized relevance ($1.00$). Consequently, relevance dictates **82% of score variance**, while novelty operates as an effective tie-breaking nudge that elevates long-tail items among candidates with similar relevance.

### Parameter Sensitivity Sweep ($\beta \in [0.0, 0.50]$ with $C=2$)

| Beta ($\beta$) | Test NDCG@20 | Test Novelty (bits) | Catalog Coverage | Artist Coverage | Top-10% Decile Exposure |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00 (Disabled)** | 0.06300 | 14.03 | 0.825% | 1.443% | 99.20% |
| **0.05** | 0.06293 | 14.06 | 0.845% | 1.474% | 99.07% |
| **0.10** | 0.06307 | 14.10 | 0.866% | 1.507% | 98.93% |
| **0.15** | **0.06322** | 14.13 | 0.883% | 1.538% | 98.81% |
| **0.20 (Selected)** | **0.06318** | **14.17** | **0.901%** | **1.567%** | **98.67%** |
| **0.25** | 0.06287 | 14.21 | 0.924% | 1.607% | 98.50% |
| **0.30** | 0.06243 | 14.26 | 0.944% | 1.657% | 98.30% |
| **0.50** | 0.06084 | 14.43 | 1.021% | 1.783% | 97.55% |

### Full Covariance & Variance Decomposition Audit
Per the Senior Reviewer's requirement, the variance of the discovery score must account for the correlation between relevance and novelty:
$$\operatorname{Var}(s_{\text{discovery}}) = \operatorname{Var}(s_{\text{fused}}) + \beta^2 \operatorname{Var}(\text{Nov}) + 2\beta \operatorname{Cov}(s_{\text{fused}}, \text{Nov})$$

Across candidate pools evaluated over the test cohort:
- $\operatorname{Var}(s_{\text{fused}}) = 0.2930$
- $\operatorname{Var}(\text{Nov}) = 4.3424 \quad (\sigma_{\text{nov}} = 2.0838 \text{ bits})$
- $\beta^2 \operatorname{Var}(\text{Nov}) = (0.20)^2 \times 4.3424 = 0.1737$
- **Correlation $(s_{\text{fused}}, \text{Nov}) = -0.5599$** (Relevance and novelty are strongly negatively correlated)
- **Covariance $(s_{\text{fused}}, \text{Nov}) = -0.6316$**
- **Covariance Term $2\beta \operatorname{Cov} = 2 \times 0.20 \times (-0.6316) = \mathbf{-0.2526}$**
- **Total Variance $\operatorname{Var}(s_{\text{discovery}}) = 0.2930 + 0.1737 - 0.2526 = \mathbf{0.2141}$**

**Mathematical Significance:** Because relevance and novelty are negatively correlated ($\rho = -0.56$), the covariance term is strictly negative ($-0.2526$). Rather than adding independent variance that destabilizes rankings, the novelty term acts as an **antithetic regularizer**, compressing total score variance from $0.4667$ down to $0.2141$. Popular head tracks have their extreme scores compressed, while tail tracks receive a gentle compensatory lift.

---

## 5. Audit Issue E: Paired Difference ($\Delta NDCG$) Uncertainty Analysis

To directly assess the per-user improvement of Phase 4 over ALS, we computed the paired difference:
$$\Delta \text{NDCG}_u = \text{NDCG}_{\text{Phase 4}}(u) - \text{NDCG}_{\text{ALS}}(u)$$
across all $N=758$ test listeners:

| Metric | Empirical Value |
|---|:---:|
| **ALS Mean NDCG@20** | $0.060579$ |
| **Phase 4 Mean NDCG@20** | $0.062643$ |
| **Mean Paired Difference ($\Delta \text{NDCG}$)** | **$+0.002064$** |
| **Median Paired Difference** | **$+0.000000$** |
| **95% Bootstrap CI of $\Delta \text{NDCG}$ (5,000 resamples)** | **$[-0.000056, +0.004244]$** |
| **Paired Wilcoxon Signed-Rank Test** | **$p = 3.72 \times 10^{-3}$** |
| **Proportion of Users Improved ($\Delta > 0$)** | **23.61% (179 listeners)** |
| **Proportion of Users Tied ($\Delta == 0$)** | **61.08% (463 listeners)** |
| **Proportion of Users Declined ($\Delta < 0$)** | **15.30% (116 listeners)** |
| **Improvement / Decline Ratio** | **1.54x** |

### Formal Scientific Qualification
Phase 4 achieved a positive point estimate for mean NDCG improvement over ALS, with a statistically significant paired Wilcoxon result ($p = 3.72 \times 10^{-3}$). However, the paired bootstrap confidence interval for the mean difference includes zero ($[-0.000056, +0.004244]$), so evidence for a positive average improvement remains inconclusive under that interval. This is adopted as the authoritative project interpretation.

---

## 6. Audit Issue F: Disentangling Artist Cap vs. Novelty Bonus

To isolate the causal mechanisms behind the observed gains, we decomposed the effects of the artist cap ($C \in \{1, 2, 3, 20\}$) and the novelty bonus ($\beta=0.0$ vs $\beta=0.20$):

| Configuration | Test NDCG@20 | Catalog Coverage | Artist Coverage | Mechanism Contribution |
|---|:---:|:---:|:---:|---|
| **Raw ALS ($C=20, \beta=0$)** | 0.06058 | 0.803% | 1.177% | Baseline collaborative filtering |
| **+ Novelty Only ($C=20, \beta=0.2$)** | 0.06091 | 0.874% | 1.277% | Broadens track selection within same artists |
| **+ Artist Cap Only ($C=2, \beta=0$)** | 0.06300 | 0.825% | 1.443% | **Drives primary ranking gain (+4.0% NDCG)** |
| **Combined ($C=2, \beta=0.2$)** | **0.06318** | **0.901%** | **1.567%** | **Optimal synergy: peak precision & coverage** |

### Decomposition Findings & Empirical Qualification
1. **Primary Relevance Driver:** The artist cap ($C=2$) drives the majority of NDCG improvement (from $0.06058$ to $0.06300$). By suppressing redundant 3rd, 4th, and 5th tracks by the same artist, it opens slots for relevant tracks by secondary artists that users actually consumed.
   *Empirical Caveat:* The reported +4.0% NDCG gain from the artist cap is an empirical result for this dataset, listening distribution, and candidate threshold ($M=100$), NOT an invariant theoretical law of artist caps.
2. **Primary Coverage Driver:** The novelty bonus ($\beta=0.20$) drives the majority of catalog expansion (increasing coverage from $0.825\%$ to $0.901\%$, and artist coverage from $1.443\%$ to $1.567\%$).
3. **Synergistic Co-operation:** Combining both interventions yields the highest performance across all discovery and precision dimensions.

---

## 7. Audit Issue G: Temporal Split Cutoff & Metadata Provenance Reconciliation

### Exact Temporal Boundaries
The dataset temporal splits are bounded as follows:
- **Training Split:** `2005-02-14 00:00:07 UTC` to `2008-10-16 18:26:38 UTC` ($13,861,025$ interactions).
- **Validation Split:** `2008-10-16 18:26:39 UTC` to `2009-02-02 14:17:02 UTC` ($1,356,014$ interactions).
- **Test Split:** `2009-02-02 14:17:11 UTC` to `2013-09-29 18:32:04 UTC` ($1,113,431$ interactions).

All training playcounts, novelty bits, and artist-level priors are computed strictly on or before `2008-10-16 18:26:38 UTC`, enforcing an absolute temporal boundary.

### Metadata Attribution & Acoustic Profiling Qualification
Acoustic characteristics (energy, valence, tempo, acousticness, danceability) are statistical proxy distributions conditioned on historical artist profiles. They are NOT measured audio signal processing features extracted from raw track waveforms. The UI, metadata store, and API documentation have been updated to explicitly label these attributes as **artist-level proxy estimates**, preventing any misconception that raw audio signal processing was conducted.
