# EUTERPE — Scrobble Filtering & Spam Audit Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Threshold:** $\Delta t < 30$ seconds (consecutive identical `recording_key` per user)  

---

## 1. Executive Summary & Impact Analysis

In accordance with **Senior Scientific Review 06**, raw listening events are permanently preserved, and filtering decisions are quantitatively audited to verify that legitimate musical habits are not discarded.

| Event Category | Count | Proportion | Scientific Meaning |
|---|:---:|:---:|---|
| **Total Ingested Events** | 19,150,868 | 100.00% | Full verified scrobble history |
| **Filtered Duplicate Scrobbles** | 13,577 | **0.071%** | Rapid multi-scrobbles ($\Delta t < 30$s) |
| **Retained Clean Scrobbles** | 19,137,291 | **99.929%** | Valid interaction base for collaborative filtering |

---

## 2. Duplicate Interval Distribution

```json
{
  "0_to_5_seconds (immediate loop/glitch)": 10836,
  "6_to_15_seconds (rapid skip)": 805,
  "16_to_29_seconds (preview skip)": 1936
}
```

---

## 3. Findings
Filtering removes a modest 0.071% of noise caused by scrobbler client loops and media player preview skips while preserving 99%+ of user interactions.
