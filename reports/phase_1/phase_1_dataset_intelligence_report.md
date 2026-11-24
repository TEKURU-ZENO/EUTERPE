# EUTERPE — Phase 1 Dataset Intelligence & Listening Analysis Report
**Version:** 1.0.0  
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Corpus:** Last.fm Dataset - 1K Users (`lastfm-dataset-1K`)  
**Verified Source Events:** 19,150,868 rows  

---

## 1. Executive Reconciliation & Event Accounting

The raw 19,150,868 scrobble events from Last.fm have been fully reconciled across every ingestion, filtering, and partitioning stage with zero unexplained record loss:

```
Total Raw Harvested Scrobbles:                19,150,868 (100.000%)
  ├── Missing Required Fields:                        0 (  0.000%)
  ├── Unparseable Timestamps:                         0 (  0.000%)
  └── Retained Ingested Events:               19,150,868 (100.000%)
        ├── Filtered Rapid Duplicates (Δt < 30s): 13,577 (  0.071%)
        └── Clean Interaction Stream:         19,137,291 ( 99.929%)
              ├── Training Partition (80% t): 15,309,832
              │     ├── Warm Eligible Events: 13,861,025 (884 users, 443,375 tracks)
              │     └── Tail (<5 plays / <50u):1,448,807
              ├── Validation Partition (10%):  1,913,699
              │     ├── Warm Eligible Events:  1,356,014
              │     └── Cold-Start Events:       557,685
              └── Test Partition (10%):        1,913,760
                    ├── Warm Eligible Events:  1,113,431
                    └── Cold-Start Events:       800,329
```

---

## 2. Listener Activity Distribution

Analysis of 992 unique listeners reveals substantial engagement heterogeneity, spanning from moderate listeners to heavy power scrobblers:

| Metric | Scrobbles / Listener | Interpretation |
|---|:---:|---|
| **Mean Activity** | **19,291.6** | Substantial longitudinal depth per user |
| **Standard Deviation** | **23,177.6** | High variance across user listening volumes |
| **Minimum Activity** | **2** | Lowest scrobble count in unfiltered stream |
| **25th Percentile ($P_{25}$)** | **3,774.2** | Active listener lower quartile |
| **Median Activity ($P_{50}$)** | **11,550.5** | Typical listener scrobble volume |
| **75th Percentile ($P_{75}$)** | **25,707.0** | Upper quartile listener volume |
| **90th Percentile ($P_{90}$)** | **47,159.1** | Highly engaged cohort |
| **99th Percentile ($P_{99}$)** | **113,181.8** | Top 1% super-power users |
| **Maximum Activity** | **183,102** | Peak scrobbler record |
| **Mean Unique Tracks / User** | **4,648.1** | Repertoire breadth per listener |
| **Mean Unique Artists / User** | **905.3** | Distinct artists explored per listener |

---

## 3. Popularity Skew & The Matthew Effect

Quantitative confirmation of **Hypothesis $H_1$** (severe popularity concentration in raw music listening):

| Dimension | Measure | Value | Scientific Significance |
|---|---|:---:|---|
| **Track Level** | Total Catalog Tracks | **1,489,456** | Raw conservative `recording_key` items |
| **Track Concentration** | **Top 1% Tracks** | **31.93% of plays** | Extreme concentration in top percentile |
| **Track Concentration** | **Top 5% Tracks** | **58.75% of plays** | $>50\%$ of all scrobbles driven by top 5% items |
| **Track Concentration** | **Top 10% Tracks** | **71.24% of plays** | Confirms $H_1$ decile concentration target |
| **Track Exposure** | **Track Gini Coefficient** | **0.7848** | **$G > 0.75$**, extreme listening inequality |
| **Artist Level** | Total Catalog Artists | **174,091** | Unique artists identified |
| **Artist Concentration**| **Top 1% Artists** | **62.64% of plays** | 1% of artists claim nearly half of all listening |
| **Artist Concentration**| **Top 10% Artists** | **91.93% of plays** | Broad mainstream dominance |
| **Artist Exposure** | **Artist Gini Coefficient** | **0.9358** | **$G > 0.75$**, acute artist distribution inequality |

---

## 4. Long-Tail Partitioning & Pareto Dynamics

Empirical analysis establishes that listening histories exhibit a classic heavy-tailed power-law distribution:

| Partition | Catalog Share (%) | Catalog Track Count | Scrobbles Claimed (%) | Recommender Implication |
|---|:---:|:---:|:---:|---|
| **Head (Top 1%)** | **1.0%** | **14,894** | **31.93%** | High familiarity, low discovery value |
| **Mid-Tier (Next 19%)** | **19.0%** | **282,996** | **51.03%** | Established favorites and genre pillars |
| **Long Tail (Remaining 80%)**| **80.0%** | **1,191,564** | **17.03%** | **Primary target for EUTERPE discovery engine** |

* **The Pareto 80/20 Threshold:** Exactly **16.63%** of catalog tracks (247,681 tracks) account for **80.0%** of all historical listening events.

---

## 5. Longitudinal & Temporal Stability

* **Earliest Scrobble:** `2005-02-01`
* **Latest Scrobble:** `2013-09-01`
* **Total Longitudinal Span:** **55 months** (~4.5 years of continuous scrobbling)
* **Mean Monthly Volume:** **347,951 scrobbles/month**
* **Mean Active Listeners:** **416 listeners/month**
* **Peak Monthly Activity:** `2009-03-01` with **622,528 scrobbles**

---

## 6. Sparse Matrix Topology (Training Benchmark)

* **Eligible User Space ($|\mathcal{U}|$)**: **884 listeners**
* **Eligible Item Space ($|\mathcal{I}|$)**: **443,375 track recordings**
* **Total Potential Matrix Coordinates**: **391,943,500 cells**
* **Training Non-Zero Pairs (`nnz`)**: **2,776,274 interactions**
* **Matrix Density**: **0.7083%** (Sparsity: **99.2917%**)
* **In-Memory Footprint**: **21.18 MB** (Compressed Sparse Row `csr_matrix`)
* **Implicit Confidence Scaling**: $c_{ui} = 1 + 40 \cdot \ln(1 + r_{ui})$

---

## 7. Compliance with Senior Scientific Review Mandates

1. **Conventional Recall@20 ($RQ_1$):** Mathematically aligned with literature ($|R_u \cap T_u| / |T_u|$) with capped variant retained for saturation diagnosis.
2. **Temporal Eligibility Invariant ($RQ_2$):** 50-scrobble user and 5-scrobble item filters were calculated **exclusively on interactions prior to $T_{\text{train}}$** (`2008-10-16 18:26:39`). Zero future information influenced training vocabulary.
3. **Dual Identity Resolution ($RQ_3$):** Conservative `recording_key` preserves 1,489,456 distinct acoustic recordings; broad `normalized_track_key` provides 1,452,747 collapsed grouping keys.
4. **Scrobble Spam Filtering ($RQ_6$):** 30s threshold removed 13,577 duplicate client glitches (0.071%), while preserving 100% of raw events in an immutable audit trail.
5. **Licensing Verification:** Verified official Last.fm non-commercial academic license terms and archived in `docs/DATASET_LICENSE.md`.

---

## 8. Phase 1 Sign-Off Readiness

All seven workstreams of Phase 1 are fully executed, verified with 26 passing automated unit tests, and backed by persisted Parquet datasets, NPZ matrices, and JSON mappings.
EUTERPE is primed for **Phase 2: Classical Recommendation Intelligence**.
