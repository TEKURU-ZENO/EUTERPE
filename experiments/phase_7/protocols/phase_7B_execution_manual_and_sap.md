# EUTERPE Phase 7B: Prospective Human Listener Study
## Execution Manual, Safeguards Protocol & Statistical Analysis Plan (SAP)
**Document ID:** `EUTERPE-PH7B-SAP-001`  
**Protocol Version:** `v1.1 (Ratified & Operationalized)`  
**Status:** Approved for Execution  
**Lead Investigator:** Dev Mehta  
**Governance & Safety Gate:** Phase 7B Feasibility Pilot Cohort ($N=36$)  

---

### Executive Summary & Methodological Boundary
This manual establishes the rigorous operational execution rules, statistical safeguards, cold-start intake procedures, and pre-specified Statistical Analysis Plan (SAP) for the Phase 7B Prospective Human Listener Study of the **EUTERPE Discovery Engine**.

> [!IMPORTANT]
> **Cohort Classification:** The $N=36$ participant cohort is pre-specified as a **Feasibility Pilot Cohort** designed to establish real-world operational feasibility, audit impression and audio telemetry, validate cross-condition isolation, and estimate preliminary variance for subjective familiarity and discovery preferences. In accordance with the Senior Scientific Reviewer's audit, this study is **not statistically confirmatory**; the non-inferiority margin ($\delta = -2.5\%$) on conditional positive discovery is formally recognized as potentially underpowered.

---

### Section 1: Audio Playback Qualification Rule & Parity Protection
To prevent deceptive engagement signals (e.g., participants rating tracks purely based on metadata, title recognition, or tag aesthetics without listening), EUTERPE operationalizes an automated audio qualification barrier.

```
       [ Recommendation Slate Rendered ]
                      │
                      ▼
         [ 30s Audio Preview Available? ]
                ├── No  ──> [ Metadata-Only Impression ] (Flagged: audio_verified = False)
                └── Yes ──> [ Audio Player Initialized ]
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [ Playback Duration < 5.0s ]   [ Playback Duration >= 5.0s OR Completed ]
                  │                               │
                  ▼                               ▼
      [ Metadata-Only Impression ]    [ Audio-Verified Impression ]
      (Flagged: audio_verified=False) (Flagged: audio_verified = True)
```

#### 1.1 Qualification Rule Specification
1. **Qualification Threshold:** An impression is certified with `audio_verified: true` if and only if:
   $$\text{audio\_play\_duration\_sec} \ge 5.0 \quad \lor \quad \text{audio\_playback\_completed} = \text{True}$$
2. **Disaggregated Outcome Reporting:** All primary and secondary endpoints must be decomposed and reported in parallel across two distinct pools:
   - **Audio-Verified Pool:** Tracks where the listener actively auditioned the acoustic waveform for $\ge 5.0$ seconds.
   - **Metadata-Only Pool:** Tracks where feedback was submitted with $< 5.0$ seconds of playback or where audio previews were unavailable.
3. **Preview Availability Parity:** To eliminate system-level bias caused by differential preview resolution, preview resolution must be logged per condition. If condition preview availability deviates by $> 5.0\%$, cross-condition odds ratios must be adjusted for preview presence via logistic regression.

---

### Section 2: Prospective Long-Tail Superiority ($LTE_{\text{prospective}}$)
Phase 7B establishes a strict, non-ambiguous mathematical definition for prospective long-tail discovery superiority.

#### 2.1 Metric Definition
A recommended item $i$ is classified as belonging to the **Long-Tail Discovery Pool** if its catalog information content (surprisal) satisfies:
$$\mathcal{I}(i) = -\log_2 P(\text{item}_i) \ge 16.0 \text{ bits}$$

The **Prospective Long-Tail Engagement Rate ($LTE_{\text{prospective}}$)** measures the conditional probability that a listener actively embraces an unfamiliar long-tail recommendation:
$$LTE_{\text{prospective}} = \frac{\sum_{i \in \mathcal{S}_{\text{tail}}} \mathbb{I}(\text{fam}_i = \text{UNKNOWN} \;\land\; \text{pref}_i \in \{\text{POSITIVE}, \text{SAVE\_FOR\_LATER}\})}{\sum_{i \in \mathcal{S}_{\text{tail}}} \mathbb{I}(\text{fam}_i = \text{UNKNOWN} \;\land\; \text{pref}_i \neq \text{UNSPECIFIED})} \times 100\%$$

#### 2.2 Gate 2 Passing Criterion
> [!IMPORTANT]
> **Gate 2 Criterion:** EUTERPE must demonstrate an absolute advantage over the baseline of at least **10.0 percentage points**:
> $$\Delta LTE_{\text{prospective}} = LTE_{\text{prospective}}(\text{EUTERPE}) - LTE_{\text{prospective}}(\text{ALS}) \ge +10.0\text{ percentage points}$$
> *Note: This requires an arithmetic difference of $+10.0$ percentage points (e.g., $35.0\%$ vs $25.0\%$), not a $+10\%$ relative lift.*

---

### Section 3: Statistical Analysis Plan (SAP) Protecting Participant Unit of Inference

#### 3.1 Primary Unit of Inference ($N=36$)
- **Unit Protection:** The participant ($N=36$) is the sole independent unit of statistical inference. Impressions within a participant's session are clustered and non-independent.
- **Forbidden Practice:** Pooling 2,160 impressions into an unclustered two-sample proportion test is strictly prohibited.
- **Primary Endpoint Estimator:** For each participant $j \in \{1, \dots, 36\}$, paired difference scores are computed across Latin Square conditions:
  $$D_j = Y_{j, \text{EUTERPE}} - Y_{j, \text{ALS}}$$
  where $Y_j$ is participant $j$'s conditional positive discovery rate ($PDR_{\text{conditional}}$).

#### 3.2 Paired Inference & Cluster-Bootstrap Protocol
1. **Primary Significance Test:** Paired Wilcoxon Signed-Rank Test on $\{D_j\}_{j=1}^{36}$, testing $H_0: \text{Median}(D_j) \le \delta$ vs $H_1: \text{Median}(D_j) > \delta$ with non-inferiority margin $\delta = -0.025$ ($-2.5$ percentage points).
2. **Participant-Cluster Bootstrap:**
   - Resample $N=36$ participants with replacement $B = 10,000$ iterations.
   - For each resample $b$, compute the mean paired difference $\bar{D}^{*(b)}$.
   - Compute the two-sided $95\%$ BCa (bias-corrected and accelerated) confidence interval:
     $$\text{CI}_{95\%} = [\hat{q}_{0.025}^*, \hat{q}_{0.975}^*]$$
   - Gate 1 non-inferiority is supported if the lower bound of $\text{CI}_{95\%}$ exceeds $-2.5\%$.

#### 3.3 Pre-Defined Missing Data Handling
1. **Participant Drop-Out:** If a participant completes $< 50\%$ of assigned slates, their data is excluded from the primary per-protocol cohort and recorded in the study drop-out log.
2. **Unrated Impressions:** Unrated items within a slate are excluded from conditional denominators ($PDR_{\text{conditional}}$) but tracked in the total denominator ($PDR_{\text{yield}}$).
3. **Sensitivity Analysis:** Primary findings must be subjected to worst-case imputation (treating unrated EUTERPE items as negative and unrated ALS items as positive) and inverse-probability-weighting (IPW) sensitivity checks.

---

### Section 4: Cold-Start Onboarding & Fallback Telemetry Logger
Because prospective listeners will submit contemporary artist preferences that may not appear in the static 51,833-artist catalog, EUTERPE implements deterministic seed resolution and fallback tracking.

#### 4.1 Resolution Hierarchy
1. **Exact Canonical Match:** Exact string lookup in `artist_to_idx.json`.
2. **Normalized Match:** Lowercase, punctuation-stripped, whitespace-collapsed lookup.
3. **Substring / Token Invariant Match:** Unique prefix or token overlap (confidence $0.80$).
4. **Resolution Categorization:**
   - `RESOLVED`: Seed matched to a single catalog artist ID.
   - `AMBIGUOUS`: Seed matched multiple catalog candidates; prompt presented for disambiguation.
   - `UNSUPPORTED`: Seed missing from the 51,833-artist catalog.

#### 4.2 Generic Profile Fallback Telemetry
If a participant achieves zero resolved seeds, the system triggers the **Generic Mainstream Centroid Fallback** and logs an intake event. The telemetry logger in `ParticipantOnboardingManager` persists:
- `seeds_submitted`: Raw strings entered by the participant.
- `resolution_rate_pct`: Percentage of submitted seeds successfully resolved.
- `fallback_required`: Boolean flag indicating fallback activation.
- `calibration_displacement`: $\ell_2$ Euclidean displacement $\|\mathbf{u}_{\text{calibrated}} - \mathbf{u}_{\text{initial}}\|_2$ induced by the 3-track acoustic calibration step.

---

### Section 5: Anti-Filter-Bubble Gate & Artist Variety Distribution
To prevent algorithmic rabbit holes and over-concentration on single artists, EUTERPE enforces bounded artist variety ($C=2$).

#### 5.1 Verification Metric
For each 20-track slate $\mathcal{S}$, the unique artist count $A(\mathcal{S}) = |\text{UniqueArtists}(\mathcal{S})|$ must satisfy:
$$A(\mathcal{S}) \ge 10 \quad (\text{since } C \le 2 \text{ across } K=20)$$

#### 5.2 Mandatory Distributional Reporting
In addition to reporting the mean artist count across slates, reports must disclose:
- **Minimum Artist Count:** $\min_{\mathcal{S}} A(\mathcal{S})$
- **25th Percentile ($Q1$):** Lower quartile of unique artist count.
- **Sub-12 Slate Frequency:** Percentage of slates presenting $< 12$ distinct artists:
  $$\%_{\text{slates} < 12} = \frac{1}{|\mathbb{S}|}\sum_{\mathcal{S} \in \mathbb{S}} \mathbb{I}(A(\mathcal{S}) < 12) \times 100\%$$
- **Gate 3 Condition:** Mean $A(\mathcal{S}) \ge 14.0$, Minimum $A(\mathcal{S}) \ge 10$, and $\%_{\text{slates} < 12} \le 5.0\%$.

---

### Section 6: Human Study Governance & Informed Consent Package

#### 6.1 Cohort Recruitment & Stratification
- **Sample Size:** $N=36$ human listeners.
- **Stratification:**
  - **Tier 1 (Sparse $Q1$):** 12 participants listening $< 5$ hours/week or with narrow acoustic breadth.
  - **Tier 2 (Medium $Q2-Q3$):** 12 participants listening 5–15 hours/week.
  - **Tier 3 (Heavy $Q4$):** 12 participants listening $> 15$ hours/week (active music discoverers).
- **Incentive & Ethics:** Voluntary participation, full informed consent, right to withdraw at any time without penalty, anonymized participant IDs.

#### 6.2 Triple-Blinding Protocol
- **Blinding Key:** Maintained by an independent study coordinator not involved in outcome scoring.
- **Labels:** Conditions are triple-labeled in the UI as **Source Blue**, **Source Emerald**, and **Source Amber**.
- **Rotation:** Label mapping rotates per participant to eliminate visual/color biases.

---

### Section 7: First Human Study Report Structure (Interim Template)
When participant data collection commences, results must be presented in the following 10-section research-grade structure:

1. **Executive Sign-Off & Cohort Demographics:** Recruitment yields across Tiers $Q1, Q2-Q3, Q4$, completion rates, and dropout audit.
2. **Audio Verification Telemetry:** Preview availability parity, audition rate ($\ge 5.0$s), replay frequency, and duration distribution.
3. **Primary Gate 1 Evaluation ($PDR_{\text{conditional}}$):** Paired Wilcoxon test, participant-cluster bootstrap $95\%$ CI, and non-inferiority margin check.
4. **Secondary Yield Evaluation ($PDR_{\text{yield}}$):** Positive discovery yield, saved yield, and rejection yield across all 20 slate positions.
5. **Gate 2 Long-Tail Superiority ($LTE_{\text{prospective}}$):** Comparison of $\ge 16.0$ bit engagement across conditions ($\ge +10.0$ pp threshold).
6. **Gate 3 Anti-Filter-Bubble Distribution:** Slate unique artist count mean, min, 25th percentile, and sub-12 slate percentage.
7. **Cold-Start Onboarding Audit:** Seed resolution rates, fallback activations, and calibration displacement magnitudes.
8. **Session 2 Feedback Ablation:** Paired comparison of Session 2 slates (Feedback Active vs Feedback Ablated), order effects, negative suppression vs positive steering.
9. **Qualitative Semantic Synthesis:** Thematic clustering of participant open-ended responses regarding discovery delight, friction, and genre balance.
10. **Research Audit & Phase 8 Roadmap:** Summary of verified claims, protocol deviations (if any), and readiness for Phase 8 scale-up.
