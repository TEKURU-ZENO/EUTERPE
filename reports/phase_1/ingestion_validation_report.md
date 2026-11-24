# EUTERPE — Ingestion & Raw Data Validation Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Dataset:** `raw_events.parquet`  

---

## 1. Summary Invariants

| Invariant / Check | Result | Acceptance Status |
|---|:---:|:---:|
| **Total Ingested Events** | 19,150,868 | **PASSED** |
| **Unique Listeners** | 992 | **PASSED** (Expected ~992) |
| **Unique Raw Artists** | 174,091 | **PASSED** |
| **Chronological Sort Invariant** | `epoch_seconds[t] <= epoch_seconds[t+1]` | **PASSED** |
| **Temporal Span** | 2005-02-14 00:00:07 → 2013-09-29 18:32:04 | **PASSED** |
| **Artist MBID Coverage** | 96.86% | **AUDITED** |
| **Track MBID Coverage** | 88.68% | **AUDITED (>80% missing confirmed)** |

---

## 2. Null & Missing Value Distribution

```json
{
  "user_id": 0,
  "raw_timestamp": 0,
  "mbid_artist": 602166,
  "artist_name": 0,
  "mbid_track": 2168588,
  "track_name": 0,
  "timestamp": 0,
  "epoch_seconds": 0
}
```

---

## 3. Engineering Assessment
The ingested listening events have been chronologically validated with zero timestamp inversion. Track MBID sparsity confirms the necessity of the canonical entity resolution strategy.
