"""Temporal splitting module for EUTERPE.

Workstream 1.5 & Senior Scientific Review 02:
Executes strict global chronological partitioning (80% Train / 10% Val / 10% Test).
Calculates user and item eligibility EXCLUSIVELY on training-period events (t < T_train),
permanently preventing future data leakage into collaborative filtering baselines.
"""

import json
import time
from pathlib import Path
import polars as pl
from euterpe.config import default_config

BASE_DIR = Path(__file__).resolve().parents[2]
FILTERED_PARQUET = BASE_DIR / "data" / "interim" / "filtered_events.parquet"
SPLITS_DIR = BASE_DIR / "data" / "splits"
SPLIT_REPORT = BASE_DIR / "reports" / "phase_1" / "temporal_split_audit.md"


def execute_chronological_split(
    input_parquet: Path,
    splits_dir: Path,
    train_pct: float = 0.80,
    val_pct: float = 0.10,
    min_user_train_scrobbles: int = 50,
    min_item_train_scrobbles: int = 5
) -> dict:
    """Partition scrobbles chronologically and enforce training-only eligibility."""
    print("Beginning global chronological partitioning and eligibility enforcement...")
    t0 = time.time()
    splits_dir.mkdir(parents=True, exist_ok=True)
    SPLIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading filtered events from {input_parquet}...")
    df = pl.read_parquet(input_parquet)
    total_events = len(df)
    
    # Verify events are sorted chronologically
    df = df.sort(["epoch_seconds", "user_id"])
    
    # Determine exact quantile cutoff timestamps
    train_idx = int(total_events * train_pct)
    val_idx = int(total_events * (train_pct + val_pct))
    
    # Boundary timestamps
    t_train_cutoff = df[train_idx]["epoch_seconds"][0]
    t_val_cutoff = df[val_idx]["epoch_seconds"][0]
    
    ts_train_str = str(df[train_idx]["timestamp"][0])
    ts_val_str = str(df[val_idx]["timestamp"][0])
    
    print(f"Cutoff 1 (T_train, {train_pct*100:.0f}%): {ts_train_str} (epoch {t_train_cutoff})")
    print(f"Cutoff 2 (T_val, {(train_pct+val_pct)*100:.0f}%): {ts_val_str} (epoch {t_val_cutoff})")
    
    # Partition raw temporal windows
    train_window = df.filter(pl.col("epoch_seconds") < t_train_cutoff)
    val_window = df.filter((pl.col("epoch_seconds") >= t_train_cutoff) & (pl.col("epoch_seconds") < t_val_cutoff))
    test_window = df.filter(pl.col("epoch_seconds") >= t_val_cutoff)
    
    print(f"Raw window counts: Train={len(train_window):,}, Val={len(val_window):,}, Test={len(test_window):,}")
    
    # SENIOR SCIENTIFIC REVIEW 02:
    # Eligibility thresholds MUST be computed strictly using training-period information!
    print("Computing user and item eligibility exclusively from training window...")
    user_train_counts = train_window.group_by("user_id").len()
    eligible_users = set(
        user_train_counts.filter(pl.col("len") >= min_user_train_scrobbles)["user_id"].to_list()
    )
    
    track_train_counts = train_window.group_by("recording_key").len()
    eligible_items = set(
        track_train_counts.filter(pl.col("len") >= min_item_train_scrobbles)["recording_key"].to_list()
    )
    
    print(f"Eligible training users (>= {min_user_train_scrobbles} train scrobbles): {len(eligible_users):,}")
    print(f"Eligible training tracks (>= {min_item_train_scrobbles} train scrobbles): {len(eligible_items):,}")
    
    # Filter train split to eligible users and items
    train_clean = train_window.filter(
        pl.col("user_id").is_in(eligible_users) & pl.col("recording_key").is_in(eligible_items)
    )
    
    # Partition validation into warm (both user & track eligible in train) and cold
    val_warm = val_window.filter(
        pl.col("user_id").is_in(eligible_users) & pl.col("recording_key").is_in(eligible_items)
    )
    val_cold = val_window.filter(
        (~pl.col("user_id").is_in(eligible_users)) | (~pl.col("recording_key").is_in(eligible_items))
    )
    
    # Partition test into warm (both user & track eligible in train) and cold
    test_warm = test_window.filter(
        pl.col("user_id").is_in(eligible_users) & pl.col("recording_key").is_in(eligible_items)
    )
    test_cold = test_window.filter(
        (~pl.col("user_id").is_in(eligible_users)) | (~pl.col("recording_key").is_in(eligible_items))
    )
    
    print(f"Clean splits: Train={len(train_clean):,}, Val(warm)={len(val_warm):,}, Test(warm)={len(test_warm):,}")
    print(f"Cold-start held-out: Val(cold)={len(val_cold):,}, Test(cold)={len(test_cold):,}")
    
    # Verify Temporal Invariants
    max_train_ts = train_clean["epoch_seconds"].max()
    min_val_ts = val_warm["epoch_seconds"].min()
    max_val_ts = val_warm["epoch_seconds"].max()
    min_test_ts = test_warm["epoch_seconds"].min()
    
    leakage_passed = (max_train_ts < min_val_ts) and (max_val_ts < min_test_ts)
    print(f"Temporal Leakage Invariant: {'PASSED (Zero Leakage)' if leakage_passed else 'FAILED'}")
    
    # Save parquet splits
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "validation.parquet"
    test_path = splits_dir / "test.parquet"
    val_cold_path = splits_dir / "validation_cold.parquet"
    test_cold_path = splits_dir / "test_cold.parquet"
    
    train_clean.write_parquet(train_path, compression="zstd")
    val_warm.write_parquet(val_path, compression="zstd")
    test_warm.write_parquet(test_path, compression="zstd")
    val_cold.write_parquet(val_cold_path, compression="zstd")
    test_cold.write_parquet(test_cold_path, compression="zstd")
    
    elapsed = time.time() - t0
    
    audit_report = f"""# EUTERPE — Temporal Splitting & Leakage Audit Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Protocol:** Global Chronological Partitioning ({train_pct*100:.0f}% / {val_pct*100:.0f}% / {(1.0-train_pct-val_pct)*100:.0f}%)  

---

## 1. Boundary Timestamps & Chronological Cutoffs

* **$T_{{\\text{{train}}}}$ Cutoff:** `{ts_train_str}` (Epoch `{t_train_cutoff}`)
* **$T_{{\\text{{val}}}}$ Cutoff:** `{ts_val_str}` (Epoch `{t_val_cutoff}`)
* **Temporal Leakage Test:** `max(Train) < min(Val) < min(Test)`
  * $\\max(t_{{\\text{{train}}}})$: `{train_clean['timestamp'].max()}`
  * $\\min(t_{{\\text{{val}}}})$: `{val_warm['timestamp'].min()}`
  * $\\max(t_{{\\text{{val}}}})$: `{val_warm['timestamp'].max()}`
  * $\\min(t_{{\\text{{test}}}})$: `{test_warm['timestamp'].min()}`
  * **Status:** **{'PASSED (Strict Chronological Ordering Verified)' if leakage_passed else 'FAILED'}**

---

## 2. Training-Exclusion Eligibility Enforcement (Senior Scientific Review 02)

To prevent future target leakage, all filtering thresholds were calculated exclusively on interactions prior to $T_{{\\text{{train}}}}$:
* **Eligible Listeners ($\ge {min_user_train_scrobbles}$ training scrobbles):** {len(eligible_users):,}
* **Eligible Tracks ($\ge {min_item_train_scrobbles}$ training scrobbles):** {len(eligible_items):,}

---

## 3. Partition Event Breakdown

| Partition | Total Events | Warm-Start Events | Cold-Start Events | Unique Users | Unique Tracks |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training ($t < T_{{\\text{{train}}}})** | {len(train_window):,} | {len(train_clean):,} | — | {train_clean['user_id'].n_unique():,} | {train_clean['recording_key'].n_unique():,} |
| **Validation ($T_{{\\text{{train}}}} \le t < T_{{\\text{{val}}}})** | {len(val_window):,} | {len(val_warm):,} | {len(val_cold):,} | {val_warm['user_id'].n_unique():,} | {val_warm['recording_key'].n_unique():,} |
| **Test ($t \ge T_{{\\text{{val}}}})** | {len(test_window):,} | {len(test_warm):,} | {len(test_cold):,} | {test_warm['user_id'].n_unique():,} | {test_warm['recording_key'].n_unique():,} |

---

## 4. Engineering Sign-off
Temporal splits saved to `{splits_dir}` in {elapsed:.2f} seconds.
"""
    with open(SPLIT_REPORT, "w", encoding="utf-8") as f:
        f.write(audit_report)
        
    print(f"Temporal split audit report written: {SPLIT_REPORT}")
    return {
        "t_train_cutoff": t_train_cutoff,
        "t_val_cutoff": t_val_cutoff,
        "ts_train": ts_train_str,
        "ts_val": ts_val_str,
        "train_events": len(train_clean),
        "val_warm_events": len(val_warm),
        "test_warm_events": len(test_warm),
        "val_cold_events": len(val_cold),
        "test_cold_events": len(test_cold),
        "eligible_users": len(eligible_users),
        "eligible_items": len(eligible_items),
        "leakage_passed": leakage_passed,
        "elapsed_seconds": round(elapsed, 2)
    }


if __name__ == "__main__":
    if FILTERED_PARQUET.exists():
        execute_chronological_split(FILTERED_PARQUET, SPLITS_DIR)
    else:
        print(f"Required file {FILTERED_PARQUET} not found.")
