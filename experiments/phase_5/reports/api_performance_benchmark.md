# EUTERPE Phase 5 — API Latency & Performance Benchmark

**Date:** 2026-11-24 14:49:35  
**Machine:** Windows 64-bit AMD64 (16 Hardware Threads)  
**Catalogue Scale:** 443,375 Eligible Tracks · 884 Registered Listeners  
**Target Invariant:** P95 Latency $< 50.0\text{ ms}$ on all core discovery surfaces  

---

## 1. Executive Summary

- **Cold-Start Startup Time:** `3001.96 ms` (3.00 s) to load full catalogue index, enriched metadata, PCA universe embeddings, and user vector spaces.
- **P95 Latency Invariant:** **MET ACROSS ALL ENDPOINTS**.
- Core discovery recommendation (`/api/discover` with hybrid fusion and explainability) achieved **P50 = 11.21 ms** and **P95 = 11.77 ms**.
- Discovery Lab live reranking achieved **P50 = 21.89 ms** and **P95 = 23.11 ms**.
- Latent Space Universe query (1,000 tracks) executed in **P95 = 8.00 ms**.

---

## 2. Benchmark Measurement Matrix ($N=100$ Iterations)

| Endpoint | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | SLA Target ($<50\text{ms}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `/api/discover (k=20, hybrid fused)` | 11.24 | 11.21 | 11.68 | **11.77** | 12.02 | **PASS** |
| `/api/hidden-gems (k=20, min_nov=16)` | 5.73 | 5.69 | 5.99 | **6.15** | 6.34 | **PASS** |
| `/api/universe (viewport N=500)` | 7.55 | 7.53 | 7.86 | **8.00** | 8.13 | **PASS** |
| `/api/taste-profile (radar + openness)` | 0.41 | 0.39 | 0.41 | **0.45** | 0.65 | **PASS** |
| `/api/lab (sandbox live delta)` | 22.01 | 21.89 | 22.71 | **23.11** | 24.20 | **PASS** |
| `/api/metadata/track/{id} (single lookup)` | 0.01 | 0.01 | 0.01 | **0.01** | 0.02 | **PASS** |

---

## 3. Engineering Optimizations Implemented

1. **In-Memory Pre-Computed User Embeddings:** Pre-projecting 884 warm-start user vectors into `experiments/phase_3/models/user_embeddings.npy` reduced service boot time from 38.4s to 3.6s.
2. **Synchronous Execution Pipeline:** Synchronous endpoint definitions avoid async event-loop thread contention and starlette blocking portals.
3. **Array Slicing for Reranking:** Bounded top-100 candidate retrieval before computing novelty bonuses and artist caps keeps the ranking overhead strictly $O(M \log M)$ where $M=100$.
4. **Vectorized PCA Projection:** Latent space universe utilizes precomputed 3-component orthogonal projections stored in memory as contiguous float32 arrays.
