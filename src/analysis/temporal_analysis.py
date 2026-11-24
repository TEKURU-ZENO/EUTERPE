"""Temporal listening dynamics and longitudinal volume analysis for EUTERPE.

Workstream 1.7:
Analyzes listening trends across years (2005-2009), active users per month,
and verifies longitudinal stability across the temporal splits.
"""

from pathlib import Path
import polars as pl


def analyze_temporal_dynamics(df_events: pl.DataFrame) -> dict:
    """Analyze monthly activity volume and user retention across time."""
    print("Computing longitudinal monthly activity...")
    
    # Extract year-month
    monthly_df = df_events.with_columns([
        pl.col("timestamp").dt.truncate("1mo").alias("month")
    ]).group_by("month").agg([
        pl.len().alias("scrobbles"),
        pl.col("user_id").n_unique().alias("active_users"),
        pl.col("recording_key").n_unique().alias("unique_tracks")
    ]).sort("month")
    
    months = [str(m)[:10] for m in monthly_df["month"]]
    scrobbles = monthly_df["scrobbles"].to_list()
    active_users = monthly_df["active_users"].to_list()
    
    return {
        "start_month": months[0] if months else "",
        "end_month": months[-1] if months else "",
        "total_months": len(months),
        "peak_month": months[scrobbles.index(max(scrobbles))] if scrobbles else "",
        "peak_monthly_scrobbles": max(scrobbles) if scrobbles else 0,
        "mean_monthly_scrobbles": float(sum(scrobbles) / len(scrobbles)) if scrobbles else 0,
        "mean_monthly_active_users": float(sum(active_users) / len(active_users)) if active_users else 0,
    }
