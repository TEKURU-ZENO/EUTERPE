# EUTERPE Phase 6 — Concurrent Load & Throughput Benchmark

**Date:** 2026-11-24 15:27:17  
**Machine:** Windows AMD64 Host (16 Hardware Threads)  
**Catalogue Scale:** 443,375 Eligible Tracks · 884 Registered Listeners  
**Workload:** Mixed multi-endpoint requests (/api/discover, /api/hidden-gems, /api/taste-profile, /api/lab) across 50 distinct listener accounts.  

---

## 1. Concurrency Benchmark Results

| Concurrency Tier | Total Requests | Duration (s) | Throughput (QPS) | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Errors |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5 Workers** | 200 | 1.93s | **103.39 req/s** | 47.51 | 36.55 | 106.12 | **110.62** | **119.17** | 0 (0.0%) |
| **20 Workers** | 200 | 1.97s | **101.47 req/s** | 174.74 | 144.89 | 391.66 | **437.41** | **506.35** | 0 (0.0%) |
| **50 Workers** | 200 | 1.95s | **102.55 req/s** | 220.98 | 177.11 | 505.9 | **583.38** | **663.91** | 0 (0.0%) |

---

## 2. Operational Qualifications & Scope

1. **Burst Test Nature:** This benchmark evaluates 200 requests per tier over ~2 seconds. It represents a concurrent burst stress test, not sustained multi-hour throughput under production conditions.
2. **Latency Inflation Under Load:** While multi-threaded throughput remains steady at ~102 req/s with a 0% error rate, queuing delays elevate P95 latency to **583.38 ms** at 50 concurrent workers. The sub-50 ms P95 SLA achieved in Phase 5 is valid strictly for sequential, warm in-memory queries.
3. **Local MVP Boundary:** The service is verified for local, multi-user development and pilot evaluations. Public production deployment requires asynchronous queues, caching, rate limiting, authentication, and sustained load testing.
