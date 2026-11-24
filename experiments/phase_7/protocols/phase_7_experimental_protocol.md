# EUTERPE — Phase 7 Real-World Discovery Validation Protocol

**Document Version:** 1.0.0  
**Phase:** Phase 7 — Real-World Discovery Validation  
**Project:** EUTERPE — Discovery Intelligence for Music  
**Status:** **AUTHORITATIVE EXPERIMENTAL SPECIFICATION**  
**Lead Author:** EUTERPE Core Science & Engineering Team  

---

## 1. Executive Mission & Primary Research Hypothesis

Project EUTERPE has progressed through offline data engineering, classical matrix factorization, neural two-tower retrieval, multi-objective reranking, local MVP packaging, and closed-loop feedback instrumentation. 

Offline ranking metrics evaluate whether a system predicts historical consumption. Phase 7 transitions to the central product hypothesis:

> **Central Hypothesis ($H_{\text{discovery}}$):**  
> *EUTERPE's hybrid discovery engine enables listeners to discover music that is genuinely unfamiliar to them and which they positively enjoy, achieving substantially higher long-tail discovery engagement ($LTE \ge 20\%$) and non-inferior positive discovery ($PDR$) compared to pure collaborative filtering (Implicit ALS), without elevating taste rejection ($RR \le 30\%$).*

---

## 2. Participant Design & Cohort Stratification

To ensure defensible, statistically sound conclusions, listener recruitment is stratified across listening volume:

### A. Sample Size & Power Sizing
- **Cohort Size:** $N = 36$ participants (divisible by 6 for full Latin Square counterbalancing).
- **Target Statistical Power:** $1 - \beta \ge 0.80$ at $\alpha = 0.05$ for detecting a medium effect size ($d = 0.50$) on paired within-subjects metric differences.

### B. Stratification Tiers
1. **Cohort S (Sparse Listeners, $N=12$):** Bottom quartile of historical scrobbles ($< 3,625$ listens). Tests cold-start resilience and risk of false novelty.
2. **Cohort M (Medium Listeners, $N=12$):** Interquartile scrobblers ($3,625 - 23,823$ listens). Tests balanced collaborative-neural discovery.
3. **Cohort H (Heavy Listeners, $N=12$):** Top quartile scrobblers ($> 23,823$ listens). Tests deep catalogue exploration and anti-fatigue performance.

---

## 3. Orthogonal 2D Feedback Semantics

To prevent the methodological pitfall of inflating discovery scores by treating unfamiliar tracks as automatic successes, feedback is formally decomposed into two orthogonal axes:

```
                          PREFERENCE AXIS
                                 ▲
                     Positive / Loved (Like)
                                 │
     QUADRANT II                 │           QUADRANT I
     Familiar Favorites          │           TRUE SERENDIPITY
     (Already Known + Liked)     │           (Unfamiliar + Liked)
                                 │
◄────────────────────────────────┼────────────────────────────────►
FAMILIAR                         │                         UNKNOWN
(Known to listener)              │                         (Novel)
                                 │
     QUADRANT III                │           QUADRANT IV
     Familiar Irrelevance        │           Discovery Failure
     (Known + Disliked)          │           (Unfamiliar + Rejected)
                                 │
                     Negative / Disliked (Dislike)
                                 ▼
```

### Supported Signal Taxonomy (7 Explicit Actions)
| Action | Familiarity Dimension | Preference Dimension | Operational System Behavior |
|---|---|---|---|
| `discovered_something_new` | **Unknown** | **Positive** | Primary serendipity hit; seeds local latent exploration |
| `like` | Unspecified | **Positive** | Collaborative reinforcement; positive affinity bonus |
| `dislike` | Unspecified | **Negative** | **Strict negative candidate masking** from future slates |
| `already_know` | **Familiar** | Neutral | Records familiarity rate; prevents false novelty credit |
| `not_my_taste` | Unspecified | **Negative** | Taste rejection; dampens artist score by $0.85\times$ |
| `more_like_this` | Unspecified | **Positive** | Positive seed for next session embedding proximity |
| `save_for_later` | Unspecified | **Save For Later** | High-intent commitment; treated as positive discovery |

---

## 4. Triple-System Within-Subjects Protocol

### A. Experimental Systems Compared (Triple Triad)
- **System A (Popularity Baseline):** Recency/Global Popularity with training-consumed item masking.
- **System B (Implicit ALS Baseline):** 64D BM25 Matrix Factorization ($K=20$, $\alpha=1.0$, $\beta=0.0$, $C=20$).
- **System C (EUTERPE Discovery Engine):** Two-Tower + ALS Fusion ($\alpha=0.70$) + Novelty Surprisal ($\beta=0.20$) + Bounded Artist Repetition ($C=2$).

### B. Double-Blind Latin-Square Counterbalancing
Each participant evaluates 3 consecutive slates of $K=20$ tracks. The system order is counterbalanced across 6 balanced sequences to eliminate sequence bias and fatigue effects:

| Sequence Group | Slate 1 | Slate 2 | Slate 3 |
|:---:|:---:|:---:|:---:|
| **G1** | System A (Pop) | System B (ALS) | System C (EUTERPE) |
| **G2** | System A (Pop) | System C (EUTERPE) | System B (ALS) |
| **G3** | System B (ALS) | System A (Pop) | System C (EUTERPE) |
| **G4** | System B (ALS) | System C (EUTERPE) | System A (Pop) |
| **G5** | System C (EUTERPE) | System A (Pop) | System B (ALS) |
| **G6** | System C (EUTERPE) | System B (ALS) | System A (Pop) |

*Double-Blind Assurance:* The UI displays identical neutral Obsidian headers (`Discovery Slate 1`, `Discovery Slate 2`, `Discovery Slate 3`). Neither participant nor session facilitator knows the underlying engine generating each slate.

---

## 5. Formal Metric Registry & Mathematical Formulations

For each participant $u$ and system $s$, evaluate the 7 mandatory metrics:

1. **Novel Discovery Rate ($NDR$):**
   $$NDR(u, s) = \frac{|\{i \in R_{u,s} \mid \text{Familiarity}(i) = \text{UNKNOWN}\}|}{K}$$

2. **Positive Discovery Rate ($PDR$ / True Serendipity):**
   $$PDR(u, s) = \frac{|\{i \in R_{u,s} \mid \text{Familiarity}(i) = \text{UNKNOWN} \land \text{Preference}(i) \in \{\text{POSITIVE}, \text{SAVE}\}\}|}{\max(1, |\{i \in R_{u,s} \mid \text{Familiarity}(i) = \text{UNKNOWN}\}|)}$$

3. **Familiarity Rate ($FR$):**
   $$FR(u, s) = \frac{|\{i \in R_{u,s} \mid \text{Familiarity}(i) = \text{FAMILIAR}\}|}{K}$$

4. **Rejection Rate ($RR$):**
   $$RR(u, s) = \frac{|\{i \in R_{u,s} \mid \text{Preference}(i) = \text{NEGATIVE}\}|}{K}$$

5. **Long-Tail Engagement Rate ($LTE$):**
   $$LTE(u, s) = \frac{|\{i \in R_{u,s} \mid \text{Preference}(i) \in \{\text{POSITIVE}, \text{SAVE}\} \land \text{Novelty}(i) \ge 16.0\text{ bits}\}|}{\max(1, |\{i \in R_{u,s} \mid \text{Preference}(i) \in \{\text{POSITIVE}, \text{SAVE}\}\}|)}$$

6. **Discovery Satisfaction Score ($DS$):**
   Listener-reported 5-point Likert response to:  
   *"Overall, how satisfied were you with the new music you discovered on this slate?"* ($1 = \text{Very Dissatisfied}, 5 = \text{Very Satisfied}$).

7. **Return Retention Intent ($RET$):**
   Binary response to:  
   *"Would you return to listen to a customized playlist based on this session's discoveries?"*

---

## 6. Pre-Registered Success Criteria & Decision Gate

Phase 7 will certify EUTERPE's real-world discovery superiority if and only if all four pre-registered criteria are met:

1. **Long-Tail Breakthrough:** $LTE_{\text{EUTERPE}} \ge 20.0\%$ (vs. $ALS \le 5.0\%$, paired Wilcoxon $p < 0.01$).
2. **True Serendipity Non-Inferiority:** $PDR_{\text{EUTERPE}} \ge PDR_{\text{ALS}} - 2.5\%$ (non-inferiority margin with 95% bootstrap CI).
3. **Taste Rejection Safety Guardrail:** $RR_{\text{EUTERPE}} \le 30.0\%$ (prevents extreme novelty from degenerating into noise).
4. **Listener Preference Majority:** A statistically significant majority of listeners ($\ge 60\%$, Binomial test $p < 0.05$) rank EUTERPE as having the most enjoyable discovery experience overall.

---

## 7. Data Governance, Ethics & Privacy Specification

- **Consent & Anonymization:** Every participant is assigned a randomized UUID (`pilot_listener_XXXX`). No personal email, streaming credentials, or IP addresses are retained.
- **Audit Trails:** All interactions are logged locally to `artifacts/feedback/discovery_feedback.jsonl` with cryptographically verifiable SHA-256 event timestamps.
- **Open Reproducibility:** The benchmark harness script (`scripts/run_phase7_controlled_experiment.py`) is committed to the repository for deterministic audit.
