"""
Build scheme profiles + test coordinator hypothesis + save.
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

print("=" * 70)
print("BUILDING SCHEME PROFILES + COORDINATOR HYPOTHESIS TEST")
print("=" * 70)

# ========================================
# PART 1: Build team defensive scheme profiles
# ========================================
print("\n[1/3] Building team defensive scheme profiles (2024-2025)...")

pbp_24 = nfl.import_pbp_data([2024])
pbp_25 = nfl.import_pbp_data([2025])
pbp = pd.concat([pbp_24, pbp_25], ignore_index=True)
plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

# Defensive scheme per team per season
def_scheme = plays[plays["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"])].groupby(
    ["season", "defteam"]
).agg(
    total_classified=("defense_man_zone_type", "count"),
    man_plays=("defense_man_zone_type", lambda x: (x == "MAN_COVERAGE").sum()),
    zone_plays=("defense_man_zone_type", lambda x: (x == "ZONE_COVERAGE").sum()),
).reset_index()

def_scheme["man_rate"] = def_scheme["man_plays"] / def_scheme["total_classified"]
def_scheme["zone_rate"] = def_scheme["zone_plays"] / def_scheme["total_classified"]
def_scheme["scheme_lean"] = np.where(def_scheme["man_rate"] > 0.55, "MAN_HEAVY",
                            np.where(def_scheme["zone_rate"] > 0.55, "ZONE_HEAVY", "BALANCED"))

def_scheme.to_parquet(PROC_DIR / "team_defensive_schemes.parquet", index=False)
print(f"  Saved defensive scheme profiles: {len(def_scheme)} team-seasons")
print(f"\n  2024 Man-heavy teams: {def_scheme[(def_scheme['season']==2024) & (def_scheme['scheme_lean']=='MAN_HEAVY')]['defteam'].tolist()}")
print(f"  2024 Zone-heavy teams: {def_scheme[(def_scheme['season']==2024) & (def_scheme['scheme_lean']=='ZONE_HEAVY')]['defteam'].tolist()}")
print(f"  2025 Man-heavy teams: {def_scheme[(def_scheme['season']==2025) & (def_scheme['scheme_lean']=='MAN_HEAVY')]['defteam'].tolist()}")
print(f"  2025 Zone-heavy teams: {def_scheme[(def_scheme['season']==2025) & (def_scheme['scheme_lean']=='ZONE_HEAVY')]['defteam'].tolist()}")

# Offensive scheme per team
off_scheme = plays.groupby(["season", "posteam"]).agg(
    total_plays=("play_type", "count"),
    pass_plays=("pass_attempt", "sum"),
    rush_plays=("rush_attempt", "sum"),
    shotgun_plays=("shotgun", "sum"),
    no_huddle_plays=("no_huddle", "sum"),
).reset_index()
off_scheme["pass_rate"] = off_scheme["pass_plays"] / off_scheme["total_plays"]
off_scheme["shotgun_rate"] = off_scheme["shotgun_plays"] / off_scheme["total_plays"]
off_scheme["no_huddle_rate"] = off_scheme["no_huddle_plays"] / off_scheme["total_plays"]

off_scheme.to_parquet(PROC_DIR / "team_offensive_schemes.parquet", index=False)
print(f"\n  Saved offensive scheme profiles: {len(off_scheme)} team-seasons")

# ========================================
# PART 2: Player performance vs Man/Zone
# ========================================
print("\n[2/3] Building player performance vs man/zone...")

pass_plays = plays[(plays["pass_attempt"] == 1) & 
                   (plays["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))].copy()

# Receiver performance by coverage type
rec_by_coverage = pass_plays[pass_plays["receiver_player_name"].notna()].groupby(
    ["receiver_player_name", "defense_man_zone_type"]
).agg(
    targets=("pass_attempt", "sum"),
    completions=("complete_pass", "sum"),
    yards=("receiving_yards", "sum"),
    tds=("touchdown", "sum"),
    epa=("epa", "sum"),
).reset_index()

rec_by_coverage["yards_per_target"] = rec_by_coverage["yards"] / rec_by_coverage["targets"]
rec_by_coverage["catch_rate"] = rec_by_coverage["completions"] / rec_by_coverage["targets"]

# Pivot to get man vs zone side by side
rec_pivot = rec_by_coverage.pivot_table(
    index="receiver_player_name",
    columns="defense_man_zone_type",
    values=["yards_per_target", "catch_rate", "targets"],
    aggfunc="first",
).reset_index()
rec_pivot.columns = ["_".join(c).strip("_") for c in rec_pivot.columns]

# Find players who dominate man OR zone
if "yards_per_target_MAN_COVERAGE" in rec_pivot.columns and "yards_per_target_ZONE_COVERAGE" in rec_pivot.columns:
    rec_pivot["man_zone_diff"] = rec_pivot["yards_per_target_MAN_COVERAGE"] - rec_pivot["yards_per_target_ZONE_COVERAGE"]
    rec_pivot["man_specialist"] = rec_pivot["man_zone_diff"] > 3  # 3+ more yards vs man
    rec_pivot["zone_specialist"] = rec_pivot["man_zone_diff"] < -3  # 3+ more yards vs zone
    
    # Filter to players with decent sample
    min_targets = rec_pivot.filter(like="targets").min(axis=1)
    reliable = rec_pivot[min_targets >= 20]
    
    reliable.to_parquet(PROC_DIR / "player_man_zone_splits.parquet", index=False)
    print(f"  Players with man/zone splits (20+ targets each): {len(reliable)}")
    
    man_killers = reliable[reliable["man_specialist"]].sort_values("man_zone_diff", ascending=False)
    zone_killers = reliable[reliable["zone_specialist"]].sort_values("man_zone_diff")
    
    print(f"\n  TOP MAN COVERAGE KILLERS (much better vs man):")
    for _, row in man_killers.head(10).iterrows():
        print(f"    {row['receiver_player_name']}: +{row['man_zone_diff']:.1f} yds/target vs man")
    
    print(f"\n  TOP ZONE COVERAGE KILLERS (much better vs zone):")
    for _, row in zone_killers.head(10).iterrows():
        print(f"    {row['receiver_player_name']}: +{abs(row['man_zone_diff']):.1f} yds/target vs zone")

# ========================================
# PART 3: Test coordinator change hypothesis
# ========================================
print(f"\n\n[3/3] Testing coordinator change hypothesis against props...")

# Load props + game context
props = pd.read_parquet(PROC_DIR / "bets_fully_enriched.parquet")

# 2024 teams with new OCs (from our research)
new_oc_2024 = ["ATL", "CAR", "CHI", "CIN", "CLE", "NE", "NO", "PHI", "PIT", "SEA", "TEN", "LV", "WAS"]

# Mark props for teams with new OCs
FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

props["home_abbr"] = props["home_team"].map(FULL_TO_ABBR)
props["away_abbr"] = props["away_team"].map(FULL_TO_ABBR)

# For 2024 season props, flag teams with new OCs
props_2024 = props[props["season"] == 2024].copy()
props_2024["team_has_new_oc"] = (
    props_2024["home_abbr"].isin(new_oc_2024) | props_2024["away_abbr"].isin(new_oc_2024)
)

# Compare: new OC teams early season vs rest
props_2024["bet_won"] = (
    ((props_2024["signal"] == "bet_over") & (props_2024["result"] == "won")) |
    ((props_2024["signal"] == "bet_under") & (props_2024["result"] == "lost"))
)

# Early season (weeks 1-4) for new OC teams
new_oc_early = props_2024[(props_2024["team_has_new_oc"]) & (props_2024["week"] <= 4)]
new_oc_all = props_2024[props_2024["team_has_new_oc"]]
stable_early = props_2024[(~props_2024["team_has_new_oc"]) & (props_2024["week"] <= 4)]

if len(new_oc_early) > 30:
    noe_under = new_oc_early[new_oc_early["signal"] == "bet_under"]
    noe_over = new_oc_early[new_oc_early["signal"] == "bet_over"]
    stable_under = stable_early[stable_early["signal"] == "bet_under"]
    
    if len(noe_under) > 20:
        hit = noe_under["bet_won"].mean()
        roi = ((noe_under["bet_won"].sum() * 100) - ((len(noe_under) - noe_under["bet_won"].sum()) * 110)) / (len(noe_under) * 110) * 100
        print(f"\n  New OC teams + UNDER + weeks 1-4 (2024):")
        print(f"    Hit rate: {hit:.1%} | ROI: {roi:+.1f}% | N={len(noe_under)}")
    
    if len(stable_under) > 20:
        hit_s = stable_under["bet_won"].mean()
        roi_s = ((stable_under["bet_won"].sum() * 100) - ((len(stable_under) - stable_under["bet_won"].sum()) * 110)) / (len(stable_under) * 110) * 100
        print(f"  Stable OC teams + UNDER + weeks 1-4 (2024):")
        print(f"    Hit rate: {hit_s:.1%} | ROI: {roi_s:+.1f}% | N={len(stable_under)}")

    # Overall new OC performance all season
    all_under = new_oc_all[new_oc_all["signal"] == "bet_under"]
    if len(all_under) > 50:
        hit_all = all_under["bet_won"].mean()
        roi_all = ((all_under["bet_won"].sum() * 100) - ((len(all_under) - all_under["bet_won"].sum()) * 110)) / (len(all_under) * 110) * 100
        print(f"  New OC teams + UNDER + ALL WEEKS (2024):")
        print(f"    Hit rate: {hit_all:.1%} | ROI: {roi_all:+.1f}% | N={len(all_under)}")

print("\n" + "=" * 70)
print("DONE. Scheme data + coordinator test complete.")
print("=" * 70)
