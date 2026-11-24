# EUTERPE Phase 7A — Offline Historical Discovery Evaluation (100-Listener Replay Cohort)

**Date:** 2026-11-24 16:34:11  
**Evaluation Scope:** Offline Historical Replay across 100 Stratified Listeners  
**Experimental Design:** Within-Subjects Counterbalanced Triad (Popularity vs. ALS vs. EUTERPE)  
**Ground-Truth Reference:** Test split future interactions (`2009-02-02` to `2013-09-29`)  
**Methodological Note:** This report evaluates historical scrobble replays, NOT prospective active listener responses. Real-world human validation is deferred to the N=36 prospective study (Phase 7B).

---

## 1. Executive Discovery Summary

In accordance with the Senior Scientific Reviewer's Phase 7 review, this offline evaluation measures whether EUTERPE exposes listeners to relevant tracks beyond their historical comfort zone:

| Discovery Metric | Popularity Baseline | Implicit ALS Baseline | EUTERPE Discovery Engine | EUTERPE vs. ALS Delta | Significance / 95% Bootstrap CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **Novel Discovery Rate ($NDR$)** | 90.50% | 74.55% | **77.25%** | **+2.70%** | Paired Wilcoxon $p = 0.125$ |
| **Familiarity Rate ($FR$)** | 9.50% | 25.45% | **22.75%** | **-2.70%** | Baseline familiarity control |
| **Positive Discovery Rate ($PDR$)** | 2.97% | 36.09% | **36.49%** | **+0.41%** | 95% CI: `[-3.20%, +3.91%]`, $p = 0.55$ |
| **Rejection Rate ($RR$)** | 66.55% | 24.55% | **27.15%** | **+2.60%** | Point estimate $< 30\%$ guardrail |
| **Long-Tail Future-Listening Hit Rate ($LTE_{\text{future}}$)** | 0.00% | 4.33% | **26.15%** | **+21.82%** | **6.0x lift ($p = 1.18 \times 10^{-11}$)** |
| **Discovery Satisfaction ($DS, 1-5$)** | 1.19 | 2.90 | **2.90** | **+0.00** | Point estimate equality ($p = 0.61$) |

---

## 2. Formal Non-Inferiority Gate Audit: PDR Non-Inferiority

### Stated Criterion:
$$\text{Lower bound of 95% Bootstrap CI of } \Delta PDR \ge -2.50\%$$

### Empirical Audit:
- **Mean $\Delta PDR$ (EUTERPE - ALS):** `+0.41%`
- **95% Bootstrap Confidence Interval (2,000 resamples):** `[-3.20%, +3.91%]`
- **Lower Confidence Bound:** ` -3.20%`
- **Decision:** Because $-3.20\% < -2.50\%$, the lower confidence bound fails to clear the pre-registered non-inferiority margin.
- **Official Gate Status:** **INCONCLUSIVE / NOT ESTABLISHED**.

The empirical point estimate is positive (+0.41%), and the paired Wilcoxon test ($p = 0.55$) shows no evidence that EUTERPE is inferior to ALS. However, under the pre-registered $-2.5\%$ statistical boundary, non-inferiority cannot be asserted with 95% confidence on this sample size.

---

## 3. Metric Operational Definitions

1. **Long-Tail Future-Listening Hit Rate ($LTE_{\text{future}}$):**  
   $$\operatorname{LTE}_{\text{future}}(u, s) = \frac{\sum_{i \in R_{u,s}} \mathbb{I}[i \in T_u \land \text{Novelty}(i) \ge 16.0\text{ bits}]}{\max\left(1, \sum_{i \in R_{u,s}} \mathbb{I}[i \in T_u]\right)}$$
   Evaluated per listener and averaged across listeners with a paired Wilcoxon signed-rank test. Measures the proportion of relevant recommendation hits that reside in the long-tail (rarity $\ge 16$ bits of self-information, corresponding to the bottom decile of catalogue playcounts).
2. **Positive Discovery Rate ($PDR$):**  
   Proportion of unfamiliar candidate items (tracks outside user's training and top-artist profile) that match future ground-truth test interactions.
3. **Familiarity Rate ($FR$):**  
   Proportion of recommended items that were already in the user's historical training scrobbles or dominant top-artist catalogue.
4. **Rejection Rate ($RR$):**  
   Proportion of recommended tracks that are outside the listener's historical profile and not consumed in future interactions.

---

## 4. Defensible Scientific Conclusion

In offline historical replay, EUTERPE achieves a dramatic, highly statistically significant expansion in **Long-Tail Future-Listening Hit Rate (26.15% vs. 4.33%, $p = 1.18 \times 10^{-11}$)** while keeping taste rejection bounded (27.15% vs. Popularity's 66.55%). 

However, non-inferiority on Positive Discovery Rate remains inconclusive under the 95% bootstrap interval, and historical scrobbles do not substitute for subjective listener ratings. Proof of real-world discovery satisfaction requires the Phase 7B prospective study.
