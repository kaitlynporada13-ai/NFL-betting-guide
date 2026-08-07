"""
Pull ALL remaining data we have access to but haven't grabbed yet.
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

print("=" * 70)
print("PULLING ALL REMAINING DATA")
print("=" * 70)

# 1. SNAP COUNTS
print("\n[1/8] Pulling snap count data...")
try:
    snaps = nfl.import_snap_counts([2023, 2024])
    snaps.to_parquet(RAW_DIR / "snap_counts.parquet", index=False)
    print(f"  Snap counts: {len(snaps)} records")
except Exception as e:
    print(f"  Snap counts error: {e}")

# 2-8: Build from PBP
print("\n[2/8] Loading PBP for feature extraction...")
pbp = nfl.import_pbp_data([2024])
plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
print(f"  Plays loaded: {len(plays)}")

# 3. RED ZONE TARGET SHARE
print("\n[3/8] Red zone target share by player...")
rz = plays[(plays["yardline_100"] <= 20) & (plays["pass_attempt"] == 1) &
           (plays["receiver_player_name"].notna())]
rz_player = rz.groupby(["receiver_player_name", "posteam"]).agg(
    rz_targets=("pass_attempt", "sum"),
    rz_tds=("touchdown", "sum"),
).reset_index()
team_rz = rz.groupby("posteam")["pass_attempt"].sum().reset_index(name="team_rz")
rz_player = rz_player.merge(team_rz, on="posteam")
rz_player["rz_target_share"] = rz_player["rz_targets"] / rz_player["team_rz"]
rz_player.to_parquet(PROC_DIR / "player_redzone_targets.parquet", index=False)
print(f"  {len(rz_player)} players with RZ targets")

# 4. GOAL-LINE CARRIES
print("\n[4/8] Goal-line carry share...")
gl = plays[(plays["yardline_100"] <= 5) & (plays["rush_attempt"] == 1) &
           (plays["rusher_player_name"].notna())]
gl_player = gl.groupby(["rusher_player_name", "posteam"]).agg(
    gl_carries=("rush_attempt", "sum"),
    gl_tds=("touchdown", "sum"),
).reset_index()
gl_player.to_parquet(PROC_DIR / "player_goalline_carries.parquet", index=False)
print(f"  {len(gl_player)} players with goal-line carries")

# 5. GARBAGE TIME
print("\n[5/8] Garbage time detection...")
plays["is_garbage"] = (plays["qtr"] == 4) & (plays["score_differential"].abs() >= 21)
pass_plays = plays[plays["pass_attempt"] == 1]
gt_splits = pass_plays[pass_plays["receiver_player_name"].notna()].groupby(
    ["receiver_player_name", "is_garbage"]
).agg(targets=("pass_attempt", "sum"), yards=("receiving_yards", "sum")).reset_index()
gt_splits.to_parquet(PROC_DIR / "player_garbage_time_splits.parquet", index=False)
print(f"  Garbage time plays: {plays['is_garbage'].sum()}")

# 6. AIR YARDS
print("\n[6/8] Air yards / target depth...")
targets = pass_plays[pass_plays["receiver_player_name"].notna()]
air = targets.groupby("receiver_player_name").agg(
    total_targets=("pass_attempt", "sum"),
    avg_air_yards=("air_yards", "mean"),
    avg_yac=("yards_after_catch", "mean"),
    deep_targets=("air_yards", lambda x: (x >= 15).sum()),
).reset_index()
air["deep_rate"] = air["deep_targets"] / air["total_targets"]
air = air[air["total_targets"] >= 30].sort_values("avg_air_yards", ascending=False)
air.to_parquet(PROC_DIR / "player_air_yards_profile.parquet", index=False)
print(f"  {len(air)} players with air yards profiles")

# 7. QB PRESSURE
print("\n[7/8] QB pressure profiles...")
qb = plays[plays["passer_player_name"].notna()].groupby("passer_player_name").agg(
    dropbacks=("pass_attempt", "sum"),
    sacks=("sack", "sum"),
    qb_hits=("qb_hit", "sum"),
).reset_index()
qb["sack_rate"] = qb["sacks"] / qb["dropbacks"]
qb["hit_rate"] = qb["qb_hits"] / qb["dropbacks"]
qb = qb[qb["dropbacks"] >= 100]
qb.to_parquet(PROC_DIR / "qb_pressure_profile.parquet", index=False)
print(f"  {len(qb)} QBs with pressure profiles")

# 8. PLAY-ACTION
print("\n[8/8] Play-action rate...")
if "play_action" in plays.columns:
    pa = plays[plays["pass_attempt"] == 1].groupby("posteam").agg(
        passes=("pass_attempt", "sum"),
        pa_passes=("play_action", "sum"),
    ).reset_index()
    pa["pa_rate"] = pa["pa_passes"] / pa["passes"]
    pa.to_parquet(PROC_DIR / "team_play_action_rate.parquet", index=False)
    print(f"  Play-action: {len(pa)} teams")
else:
    print("  Play-action column not in 2024 PBP")

# 9. INJURY RETURN PERFORMANCE
print("\n[BONUS] Injury return performance cross-reference...")
injuries = pd.read_parquet(RAW_DIR / "injuries_historical.parquet")
stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"

out_records = injuries[injuries["report_status"] == "Out"][["season","week","full_name","report_primary_injury"]].copy()
out_records["return_week"] = out_records["week"] + 1
out_records["player_lower"] = out_records["full_name"].str.lower()
stats["player_lower"] = stats[name_col].str.lower()

# For each "Out" record, find their stats the following week
returns = out_records.merge(
    stats[["player_lower","season","week","passing_yards","rushing_yards","receiving_yards","receptions"]],
    left_on=["player_lower","season","return_week"],
    right_on=["player_lower","season","week"],
    how="inner", suffixes=("_inj","_stat"),
)
if not returns.empty:
    returns.to_parquet(PROC_DIR / "injury_return_performance.parquet", index=False)
    print(f"  Injury returns matched: {len(returns)} player-games")
    by_injury = returns.groupby("report_primary_injury").agg(
        count=("full_name","count"),
        avg_rec_yds=("receiving_yards","mean"),
        avg_rush_yds=("rushing_yards","mean"),
    ).reset_index().sort_values("count", ascending=False)
    print(f"  Top injury types on return:")
    print(f"  {by_injury.head(8).to_string()}")

print("\n" + "=" * 70)
print("DONE. All remaining data pulled.")
print("=" * 70)
