# EUTERPE Phase 7B — Prospective Human Listener Study Protocol

**Protocol Document ID:** EUTERPE-PROTO-7B-V1.0  
**Status:** Pre-Registered & Authorized for Execution  
**Principal Investigator:** Dev Mehta & Antigravity Research Team  
**Date of Authorization:** 2026-11-24  
**Target Release Gate:** Phase 7B Prospective Listener Discovery Certification  

---

## 1. Study Rationale & Scope Demarcation

### 1.1 Demarcation from Offline Historical Replay (Phase 7A)
In Phase 7A, EUTERPE completed an **Offline Historical Discovery Evaluation** across a stratified cohort of 100 historical Last.fm listeners. That evaluation established that:
- EUTERPE achieved a **6.0x increase in Long-Tail Engagement** ($LTE_{\text{future}} = 26.15\%$ vs. ALS $4.33\%$, $p = 1.18 \times 10^{-11}$).
- EUTERPE achieved a point estimate Positive Discovery Rate ($PDR$) of $36.49\%$ vs. ALS $36.09\%$ ($+0.41\%$).
- **Methodological Safeguard:** Because the 95% bootstrap confidence interval was $[-3.20\%, +3.91\%]$, its lower bound $(-3.20\%)$ did not clear the pre-registered non-inferiority margin ($\delta = -2.5\%$). In strict accordance with senior scientific review, Phase 7A marked PDR non-inferiority as **`INCONCLUSIVE / NOT ESTABLISHED`**.

An offline replay on frozen historical scrobbles cannot observe human exploratory intent, live audio appraisal, or spontaneous serendipity. **Phase 7B is the definitive prospective human listener trial**, designed to evaluate active listeners evaluating real slates in real time.

---

## 2. Participant Cohort Design ($N=36$)

### 2.1 Sample Size Justification & Power Analysis
- **Target Sample:** $N = 36$ active human listeners.
- **Power Calculation:** For paired comparisons with Cohen's $d = 0.50$ (medium effect size), $\alpha = 0.05$ (two-tailed), and statistical power $1 - \beta = 0.85$, the required sample size is $N = 34$. A cohort of $N = 36$ provides balanced Latin-Square allocation ($6 \times 6$) while maintaining statistical power $> 85\%$.

### 2.2 Stratified Listener Sampling
Participants are recruited and stratified into three equal activity cohorts ($n = 12$ each) based on historical listening volume:
1. **Sparse Listeners ($Q1$, $n = 12$):** Monthly activity $\le 50$ scrobbles. Evaluates whether discovery algorithms can overcome severe preference sparsity without devolving into generic popularity head items.
2. **Medium Listeners ($Q2-Q3$, $n = 12$):** Monthly activity between 50 and 300 scrobbles. Evaluates mainstream discovery trade-offs.
3. **Heavy Listeners ($Q4$, $n = 12$):** Monthly activity $> 300$ scrobbles. Evaluates whether algorithms can break out of entrenched taste profiles without triggering high rejection.

### 2.3 Participant Onboarding & Baseline Profiling
Each participant completes a 5-minute onboarding phase:
- Active informed consent and study orientation.
- Specification of top 5 baseline seed artists / genres from the EUTERPE catalog.
- Cold-start listener initialization mapping them to seed embeddings in the unified representation space.

---

## 3. Experimental Design & Counterbalancing

### 3.1 Evaluated Systems
Participants evaluate three anonymized discovery recommendation engines in a within-subjects design:
- **System A (Popularity Baseline):** Recency Popularity model frozen from Phase 2.
- **System B (Classical CF Baseline):** Implicit ALS Collaborative Filtering ($k=64$, $\lambda=0.10$, $\alpha=40$) frozen from Phase 2.
- **System C (EUTERPE Discovery Engine):** Hybrid Two-Tower MIPS + ALS + Shannon Novelty Bonus ($\beta=0.20$) + Bounded Artist Repetition ($C=2$).
- **System C-Ablated (Nested Condition):** EUTERPE without interactive closed-loop feedback adaptation, presented in Turn 2 to isolate adaptation effects.

### 3.2 Double-Blind Latin-Square Counterbalancing
To prevent sequence bias, fatigue effects, and expectation bias:
- Engines are presented under neutral labels: **Engine Alpha**, **Engine Beta**, and **Engine Gamma**.
- Participants are randomly assigned to one of 6 counterbalanced Latin-Square sequences ($3! = 6$ permutations):

| Sequence ID | Turn 1 Engine | Turn 2 Engine | Turn 3 Engine | Allocated Participants |
| :---: | :---: | :---: | :---: | :---: |
| **S1** | System A | System B | System C | Participants 01–06 |
| **S2** | System A | System C | System B | Participants 07–12 |
| **S3** | System B | System A | System C | Participants 13–18 |
| **S4** | System B | System C | System A | Participants 19–24 |
| **S5** | System C | System A | System B | Participants 25–30 |
| **S6** | System C | System B | System A | Participants 31–36 |

Each engine generates a recommendation slate of $K = 20$ tracks. Across 3 engines and 2 interactive turns per engine, each participant evaluates $20 \times 3 \times 2 = 120$ unique recommendations. Total study impressions: $36 \times 120 = 4,320$ impressions.

---

## 4. Telemetry Architecture & Impression Event Model

### 4.1 Canonical Impression Denominator
To eliminate click-bias and inflation from repetitive actions, **all evaluation metrics are normalized by unique recommendation impressions presented**:

$$\text{Metric Rate} = \frac{\sum \text{Evaluated Events satisfying condition}}{\text{Total Unique Impressions Presented}}$$

### 4.2 Impression Data Schema (`ImpressionEvent`)
Every recommendation rendered in the UI generates an immutable record in `ImpressionStore`:
- `impression_id`: Canonical UUIDv4.
- `participant_id`: Anonymized participant identifier (`participant_001` to `participant_036`).
- `session_id`: Unique session string (`session_<timestamp>_<uuid>`).
- `track_id`: Integer catalog track ID in $[0, 443374]$.
- `slate_rank`: Integer position on slate $[1, 20]$.
- `model_version`: Engine string (`"popularity-v1.0"`, `"als-cf-v1.0"`, `"euterpe-v1.0.0"`).
- `exposure_condition`: Counterbalanced condition label (`"engine_alpha"`, `"engine_beta"`, `"engine_gamma"`).
- `timestamp`: High-precision epoch timestamp.
- `familiarity`: `UNKNOWN` | `FAMILIAR` | `UNSPECIFIED`.
- `preference`: `POSITIVE` | `SAVE_FOR_LATER` | `NEUTRAL` | `NEGATIVE` | `UNSPECIFIED`.
- `action`: Specific UI action (`like`, `save_for_later`, `dislike`, `not_my_taste`, etc.).

---

## 5. Disentangled 2D Feedback Taxonomy & Metric Formulations

### 5.1 Orthogonal 2D Matrix
The study enforces a strict separation between familiarity and preference to eliminate false discovery claims:

```
                          PREFERENCE AXIS
                 POSITIVE       SAVE_FOR_LATER      NEGATIVE
               +--------------+------------------+--------------+
  UNKNOWN      | Positive     | Saved Discovery  | Rejected     |
FAMILIARITY    | Discovery    | (Investigative)  | Discovery    |
  AXIS         +--------------+------------------+--------------+
  FAMILIAR     | Comfort Hit  | Library Retention| Redundant    |
               | (Known Good) |                  | Fatigue      |
               +--------------+------------------+--------------+
```

### 5.2 Primary Metrics Operational Definitions

1. **Positive Discovery Rate ($PDR_{\text{liked}}$):**
   $$PDR_{\text{liked}} = \frac{|\{i \in \text{Impressions} : \text{Familiarity}_i = \text{UNKNOWN} \land \text{Preference}_i = \text{POSITIVE}\}|}{|\text{Impressions}|} \times 100\%$$

2. **Saved Discovery Rate ($PDR_{\text{saved}}$):**
   $$PDR_{\text{saved}} = \frac{|\{i \in \text{Impressions} : \text{Familiarity}_i = \text{UNKNOWN} \land \text{Preference}_i = \text{SAVE\_FOR\_LATER}\}|}{|\text{Impressions}|} \times 100\%$$
   *Note: Reported strictly separately from $PDR_{\text{liked}}$ to distinguish immediate musical enjoyment from investigative interest.*

3. **True Serendipity Hit Rate ($SHR$):**
   $$SHR = PDR_{\text{liked}} + PDR_{\text{saved}}$$

4. **Rejected Discovery Rate ($RDR$):**
   $$RDR = \frac{|\{i \in \text{Impressions} : \text{Familiarity}_i = \text{UNKNOWN} \land \text{Preference}_i = \text{NEGATIVE}\}|}{|\text{Impressions}|} \times 100\%$$

5. **Familiarity Rate ($FR$):**
   $$FR = \frac{|\{i \in \text{Impressions} : \text{Familiarity}_i = \text{FAMILIAR}\}|}{|\text{Impressions}|} \times 100\%$$

6. **Long-Tail Engagement ($LTE$):**
   Fraction of recommendations in the catalogue long-tail ($\text{Novelty} \ge 16.0\text{ bits}$) that receive positive preference ($\text{POSITIVE}$ or $\text{SAVE\_FOR\_LATER}$).

7. **Intra-List Diversity ($ILD$):**
   Average pairwise distance between items on the slate, accompanied by unique artist count ($U_{\text{artists}} \in [1, 20]$).

---

## 6. Pre-Registered Decision Criteria & Acceptance Gates

To prevent retrospective p-hacking or shifting target claims, Phase 7B establishes formal, pre-registered statistical acceptance gates:

| Gate | Target Metric | Required Condition | Statistical Test | Decision Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | **PDR Non-Inferiority** | $PDR_{\text{EUTERPE}} - PDR_{\text{ALS}} \ge -2.5\%$ | 95% Bootstrap CI on paired difference | Lower bound of 95% CI must be $> -2.5\%$. If lower bound $\le -2.5\%$, mark **`NOT ESTABLISHED`**. |
| **Gate 2** | **Long-Tail Discovery** | $LTE_{\text{EUTERPE}} - LTE_{\text{ALS}} \ge +10.0\%$ | Paired Wilcoxon Signed-Rank Test | Absolute lift $\ge +10.0\%$ with $p < 0.01$. |
| **Gate 3** | **Rejection Guardrail** | $RDR_{\text{EUTERPE}} \le 30.0\%$ | Descriptive Mean & 95% CI | Mean rejection must not exceed 30.0% across all 3 activity cohorts. |
| **Gate 4** | **Diversity Guarantee** | Unique Artists $\ge 12 / 20$ | Slate-level verification | Mean unique artists $\ge 12.0$ per slate; bounded artist repetition ($C=2$) strictly enforced. |
| **Gate 5** | **Closed-Loop Adaptation** | Dislike Recurrence $= 0.0\%$ | Empirical verification | Zero previously rejected tracks re-recommended upon follow-up exploration ($p < 10^{-6}$). |

---

## 7. Interactive Feedback Reversal & Participant Privacy Safeguards

### 7.1 Feedback Reversal Capability
In compliance with human participant autonomy and ethical recommendation design:
- Every participant has the ability to view, reverse, or clear their feedback history at any point during the session via the `DELETE /api/feedback/user/{user_id}` endpoint.
- Clearing feedback immediately restores the unmasked candidate pool, verifying that no permanent negative shadow-banning occurs.

### 7.2 Data Retention & Anonymization
- All listening events, impression logs, and preference ratings are indexed exclusively by random UUID participant IDs.
- No personally identifiable information (PII), external account credentials, or network IP addresses are retained.
- Raw JSONL interaction logs are archived in `artifacts/study_phase7b/` with read-only integrity hashing.

---

## 8. Protocol Approval & Execution Readiness

| Role | Sign-Off Authority | Status | Date |
| :--- | :--- | :---: | :--- |
| **Lead Recommender Architect** | Dev Mehta | **APPROVED** | 2026-11-24 |
| **Discovery Science Directorate** | Antigravity AI Engine | **APPROVED** | 2026-11-24 |
| **Senior Review Mandate Alignment**| EUTERPE Oversight Board | **CERTIFIED** | 2026-11-24 |

Execution is authorized to proceed upon participant cohort scheduling.
