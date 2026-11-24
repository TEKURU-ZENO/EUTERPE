# EUTERPE — Temporal Splitting & Leakage Audit Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Protocol:** Global Chronological Partitioning (80% / 10% / 10%)  

---

## 1. Boundary Timestamps & Chronological Cutoffs

* **$T_{\text{train}}$ Cutoff:** `2008-10-16 18:26:39` (Epoch `1224181599`)
* **$T_{\text{val}}$ Cutoff:** `2009-02-02 14:17:11` (Epoch `1233584231`)
* **Temporal Leakage Test:** `max(Train) < min(Val) < min(Test)`
  * $\max(t_{\text{train}})$: `2008-10-16 18:26:38`
  * $\min(t_{\text{val}})$: `2008-10-16 18:26:39`
  * $\max(t_{\text{val}})$: `2009-02-02 14:17:02`
  * $\min(t_{\text{test}})$: `2009-02-02 14:17:11`
  * **Status:** **PASSED (Strict Chronological Ordering Verified)**

---

## 2. Training-Exclusion Eligibility Enforcement (Senior Scientific Review 02)

To prevent future target leakage, all filtering thresholds were calculated exclusively on interactions prior to $T_{\text{train}}$:
* **Eligible Listeners ($\ge 50$ training scrobbles):** 884
* **Eligible Tracks ($\ge 5$ training scrobbles):** 443,375

---

## 3. Partition Event Breakdown

| Partition | Total Events | Warm-Start Events | Cold-Start Events | Unique Users | Unique Tracks |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training ($t < T_{\text{train}})** | 15,309,832 | 13,861,025 | — | 884 | 443,375 |
| **Validation ($T_{\text{train}} \le t < T_{\text{val}})** | 1,913,699 | 1,356,014 | 557,685 | 689 | 215,399 |
| **Test ($t \ge T_{\text{val}})** | 1,913,760 | 1,113,431 | 800,329 | 758 | 193,506 |

---

## 4. Engineering Sign-off
Temporal splits saved to `C:\Users\Dev Mehta\Desktop\EUTERPE\data\splits` in 9.30 seconds.
