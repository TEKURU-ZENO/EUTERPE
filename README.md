# EUTERPE — The Intelligent Music Discovery Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial Research](https://img.shields.io/badge/License-Non--Commercial-lightgrey.svg)](docs/01_dataset_decision_record.md)
[![Status: Phase 0 Foundation Complete](https://img.shields.io/badge/Status-Phase%200%20Approved-gold.svg)](#)

> **Full Name:** EUTERPE — Discovery Intelligence for Music  
> **Tagline:** *Beyond what you listen to. Into what you'll love.*  

---

## 1. System Overview

**EUTERPE** is an end-to-end machine learning system and discovery engine designed to bridge the tension between predictive recommendation accuracy and long-tail musical exploration.

The project is structured around three interconnected systems:
1. **Recommendation Intelligence:** Scalable candidate generation capturing historical listening preferences through classical collaborative filtering (Popularity, Implicit ALS, BPR-MF) and neural dual-encoder retrieval (PyTorch Two-Tower).
2. **Discovery Intelligence:** Diversity-aware and constraint-guided reranking (Maximal Marginal Relevance and Discovery-Constrained Optimization) expanding catalog coverage and surfacing high-relevance hidden gems.
3. **Music Discovery Experience:** High-performance serving architecture (FastAPI) and an interactive web dashboard (React + TypeScript) offering 5 core listening experiences and evidence-based recommendation explanations.

---

## 2. Master Development Roadmap

| Phase | Focus | Status | Key Deliverables |
|:---:|:---|:---:|:---|
| **0** | **Scientific Specification & Foundation** | **COMPLETE** | Research questions, formal hypotheses, dataset audit, metric implementations, architecture |
| **1** | **Data Engineering & Listening Intelligence** | *Up Next* | Ingestion, canonical track resolution, temporal splitting, interaction matrices |
| **2** | **Classical Recommendation Baselines** | *Planned* | Global Popularity, Recency-Decayed Popularity, Implicit ALS, BPR-MF |
| **3** | **Neural Recommendation Intelligence** | *Planned* | Two-Tower dual-encoder (PyTorch), in-batch sampled softmax, FAISS ANN indexing |
| **4** | **Discovery Science & Reranking** | *Planned* | Novelty@20, ILD, Gini exposure, Catalog coverage, MMR, Pareto frontier |
| **5** | **Product Engineering (API & Frontend)** | *Planned* | FastAPI service, React dashboard, 5 discovery experiences, explanation engine |
| **6** | **Validation, Benchmarking & MVP Release** | *Planned* | Bootstrap uncertainty testing, unit testing, leakage audit, artifact packaging |
| **7–9**| **Advanced Sequential & Production Intelligence** | *Future* | Sequential models (SASRec/GRU4Rec), live experimentation, production serving |

---

## 3. Project Structure

```
EUTERPE/
├── docs/                                # Scientific specifications & design records
│   ├── 00_research_specification.md     # Research questions, theoretical framing, hypotheses
│   ├── 01_dataset_decision_record.md    # Last.fm 1K audit, licensing, schema & entity resolution
│   ├── 02_model_comparison_plan.md      # Baselines, Two-Tower neural model & experimental matrix
│   ├── 03_evaluation_methodology.md     # Chronological splitting & exact discovery metric formulas
│   └── 04_architecture_specification.md # System components, API routes & UI palette
├── src/
│   └── euterpe/
│       ├── __init__.py
│       ├── config.py                    # Central typed configuration (paths, hyperparameters)
│       ├── data/                        # Ingestion, canonical resolution & splitting
│       │   ├── ingestion.py
│       │   ├── normalization.py         # Deterministic title/artist cleaning & composite keys
│       │   └── splitting.py
│       ├── models/                      # Recommendation engines
│       │   ├── classical/               # ALS & BPR implementations
│       │   └── neural/                  # Two-Tower PyTorch model & FAISS retrieval
│       ├── discovery/                   # Discovery science & reranking
│       │   ├── metrics.py               # Recall@K, NDCG@K, Novelty@K, ILD, Gini, Serendipity
│       │   ├── reranking.py             # MMR and discovery-constrained optimizers
│       │   └── segmentation.py          # User cohort analysis (mainstream, niche, exploratory)
│       └── api/                         # FastAPI REST application
├── tests/                               # Rigorous unit & integration tests
│   ├── test_normalization.py
│   └── test_metrics.py
├── scripts/                             # CLI execution utilities
├── data/                                # Data storage (raw, processed, splits)
├── requirements.txt                     # Pinned project dependencies
└── pyproject.toml                       # Python package configuration
```

---

## 4. Scientific Rigor & Principles

EUTERPE strictly enforces the following engineering guidelines:
- **No Data Leakage:** All evaluation is conducted strictly via global chronological cutoffs ($t < T_{\text{train}} \le t_{\text{val}} < t_{\text{test}}$). Future interactions never influence training features, popularity rankings, or candidate pools.
- **Frozen Baselines:** Baseline performance thresholds (ALS, BPR) are fixed prior to neural model evaluation; goalposts are never moved retroactively.
- **Evidence-Based Explanations:** Explanations cite actual user-item interaction affinities and embedding distances—never fabricated text.
- **Statistical Significance:** All metrics report 95% confidence intervals derived from $B = 1,000$ paired user bootstrap iterations with paired Wilcoxon signed-rank tests ($p < 0.01$).

---

## 5. Quickstart & Verification

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run test suite to verify metrics and normalization
pytest tests/ -v
```

---

## 6. Citation

If utilizing this dataset or research pipeline, cite:
```bibtex
@book{celma2010music,
  title={Music Recommendation and Discovery in the Long Tail},
  author={Celma, {\`O}scar},
  year={2010},
  publisher={Springer Berlin Heidelberg}
}
```
