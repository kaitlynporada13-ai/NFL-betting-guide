"""
Play-by-Play Ingestion & Quarter/Half Aggregation.
Pulls full PBP data from nflverse, then builds quarter-level and half-level
stats for teams and players.

This powers:
- 1st half / 2nd half over/under models
- Quarter-level scoring trends
- Player production by half
- Fast starter / slow starter profiles
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl

from pipeline.config_loader import load_settings, get_data_dir


def pull_pbp_data(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull play-by-play data from nflverse.
    WARNING: This is large (~500MB per season). Only pulls what's needed.
    """
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[pbp] Pulling play-by-play for seasons: {seasons}")
    
    # Only pull columns we need to keep memory manageable
    cols_needed = [
        "game_id", "season", "week", "game_half", "qtr", "posteam", "defteam",
        "play_type", "yards_gained", "touchdown", "interception", "fumble_lost",
        "pass_attempt", "rush_attempt", "complete_pass", "incomplete_pass",
        "passing_yards", "rushing_yards", "receiving_yards",
        "passer_player_id", "passer_player_name",
        "rusher_player_id", "rusher_player_name",
        "receiver_player_id", "receiver_player_name",
        "td_team", "posteam_score", "defteam_score",
        "posteam_score_post", "defteam_score_post",
        "score_differential", "half_seconds_remaining",
        "ep", "epa", "wp", "wpa",
    ]

    all_pbp = []
    for season in seasons:
        try:
            print(f"  Pulling {season}...")
            pbp = nfl.import_pbp_data([season], columns=cols_needed)
            all_pbp.append(pbp)
            print(f"    {len(pbp)} plays")
        except Exception as e:
            print(f"    Could not pull {season}: {e}")

    if all_pbp:
        combined = pd.concat(all_pbp, ignore_index=True)
        print(f"  Total: {len(combined)} plays across {len(all_pbp)} seasons")
        return combined
    return pd.DataFrame()


def build_team_quarter_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PBP into team stats by game AND quarter.
    
    Produces per-team, per-game, per-quarter:
    - Points scored
    - Pass yards / rush yards
    - Pass attempts / rush attempts
    - Turnovers
    - EPA (expected points added)
    """
    # Filter to actual plays (not penalties, timeouts, etc.)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # Points scored per quarter: derive from score changes
    # Use posteam_score_post - posteam_score for points on each play
    plays["points_scored"] = (plays["posteam_score_post"] - plays["posteam_score"]).clip(lower=0)

    team_qtr = plays.groupby(["game_id", "season", "week", "posteam", "qtr"]).agg(
        points=("points_scored", "sum"),
        total_yards=("yards_gained", "sum"),
        pass_yards=("passing_yards", "sum"),
        rush_yards=("rushing_yards", "sum"),
        pass_attempts=("pass_attempt", "sum"),
        rush_attempts=("rush_attempt", "sum"),
        completions=("complete_pass", "sum"),
        touchdowns=("touchdown", "sum"),
        interceptions=("interception", "sum"),
        fumbles_lost=("fumble_lost", "sum"),
        total_plays=("play_type", "count"),
        epa_total=("epa", "sum"),
        epa_per_play=("epa", "mean"),
    ).reset_index()

    team_qtr.rename(columns={"posteam": "team"}, inplace=True)
    return team_qtr


def build_team_half_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PBP into team stats by game AND half.
    
    Halves:
    - "Half1" = quarters 1-2
    - "Half2" = quarters 3-4 (includes OT)
    """
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    plays["points_scored"] = (plays["posteam_score_post"] - plays["posteam_score"]).clip(lower=0)

    # Map quarters to halves
    plays["half"] = plays["qtr"].map({1: "H1", 2: "H1", 3: "H2", 4: "H2", 5: "H2"})

    team_half = plays.groupby(["game_id", "season", "week", "posteam", "half"]).agg(
        points=("points_scored", "sum"),
        total_yards=("yards_gained", "sum"),
        pass_yards=("passing_yards", "sum"),
        rush_yards=("rushing_yards", "sum"),
        pass_attempts=("pass_attempt", "sum"),
        rush_attempts=("rush_attempt", "sum"),
        completions=("complete_pass", "sum"),
        touchdowns=("touchdown", "sum"),
        interceptions=("interception", "sum"),
        fumbles_lost=("fumble_lost", "sum"),
        total_plays=("play_type", "count"),
        epa_total=("epa", "sum"),
        epa_per_play=("epa", "mean"),
    ).reset_index()

    team_half.rename(columns={"posteam": "team"}, inplace=True)
    return team_half


def build_player_half_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PBP into player stats by game AND half.
    Covers passing, rushing, receiving broken down by 1st half / 2nd half.
    """
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    plays["half"] = plays["qtr"].map({1: "H1", 2: "H1", 3: "H2", 4: "H2", 5: "H2"})

    # --- Passing stats by half ---
    pass_plays = plays[plays["pass_attempt"] == 1].copy()
    passer_half = pass_plays.groupby(
        ["game_id", "season", "week", "passer_player_id", "passer_player_name", "posteam", "half"]
    ).agg(
        pass_yards=("passing_yards", "sum"),
        pass_attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        pass_tds=("touchdown", "sum"),
        interceptions=("interception", "sum"),
        pass_epa=("epa", "sum"),
    ).reset_index()
    passer_half.rename(columns={
        "passer_player_id": "player_id",
        "passer_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)
    passer_half["position"] = "QB"

    # --- Rushing stats by half ---
    rush_plays = plays[plays["rush_attempt"] == 1].copy()
    rusher_half = rush_plays.groupby(
        ["game_id", "season", "week", "rusher_player_id", "rusher_player_name", "posteam", "half"]
    ).agg(
        rush_yards=("rushing_yards", "sum"),
        rush_attempts=("rush_attempt", "sum"),
        rush_tds=("touchdown", "sum"),
        rush_epa=("epa", "sum"),
    ).reset_index()
    rusher_half.rename(columns={
        "rusher_player_id": "player_id",
        "rusher_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)

    # --- Receiving stats by half ---
    rec_plays = pass_plays[pass_plays["complete_pass"] == 1].copy()
    receiver_half = rec_plays.groupby(
        ["game_id", "season", "week", "receiver_player_id", "receiver_player_name", "posteam", "half"]
    ).agg(
        rec_yards=("receiving_yards", "sum"),
        receptions=("complete_pass", "sum"),
        rec_tds=("touchdown", "sum"),
        rec_epa=("epa", "sum"),
    ).reset_index()
    receiver_half.rename(columns={
        "receiver_player_id": "player_id",
        "receiver_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)

    # Also count targets (complete + incomplete to that receiver)
    all_targets = pass_plays[pass_plays["receiver_player_id"].notna()].copy()
    target_half = all_targets.groupby(
        ["game_id", "season", "week", "receiver_player_id", "receiver_player_name", "posteam", "half"]
    ).agg(
        targets=("pass_attempt", "sum"),
    ).reset_index()
    target_half.rename(columns={
        "receiver_player_id": "player_id",
        "receiver_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)

    # Merge receptions with targets
    receiver_half = receiver_half.merge(
        target_half, on=["game_id", "season", "week", "player_id", "player_name", "team", "half"],
        how="outer",
    )

    return {
        "passing": passer_half,
        "rushing": rusher_half,
        "receiving": receiver_half,
    }


def build_player_quarter_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PBP into player stats by game AND quarter.
    More granular than half - useful for finding Q1/Q3 scorers.
    """
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # --- Receiving by quarter (most useful for props) ---
    pass_plays = plays[plays["pass_attempt"] == 1].copy()

    # Targets per quarter
    targets_qtr = pass_plays[pass_plays["receiver_player_id"].notna()].groupby(
        ["game_id", "season", "week", "receiver_player_id", "receiver_player_name", "posteam", "qtr"]
    ).agg(
        targets=("pass_attempt", "sum"),
        receptions=("complete_pass", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("touchdown", "sum"),
    ).reset_index()
    targets_qtr.rename(columns={
        "receiver_player_id": "player_id",
        "receiver_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)

    # Rushing by quarter
    rush_plays = plays[plays["rush_attempt"] == 1].copy()
    rushing_qtr = rush_plays.groupby(
        ["game_id", "season", "week", "rusher_player_id", "rusher_player_name", "posteam", "qtr"]
    ).agg(
        rush_attempts=("rush_attempt", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("touchdown", "sum"),
    ).reset_index()
    rushing_qtr.rename(columns={
        "rusher_player_id": "player_id",
        "rusher_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)

    # Passing by quarter
    passing_qtr = pass_plays.groupby(
        ["game_id", "season", "week", "passer_player_id", "passer_player_name", "posteam", "qtr"]
    ).agg(
        pass_attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_tds=("touchdown", "sum"),
        interceptions=("interception", "sum"),
    ).reset_index()
    passing_qtr.rename(columns={
        "passer_player_id": "player_id",
        "passer_player_name": "player_name",
        "posteam": "team",
    }, inplace=True)

    return {
        "receiving_qtr": targets_qtr,
        "rushing_qtr": rushing_qtr,
        "passing_qtr": passing_qtr,
    }


def build_rolling_quarter_features(team_qtr: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling averages for team quarter stats.
    E.g., "avg points scored in Q1 over last 5 games"
    """
    df = team_qtr.sort_values(["team", "season", "week"]).copy()

    metrics = ["points", "total_yards", "pass_yards", "rush_yards",
               "touchdowns", "epa_per_play", "total_plays"]

    for metric in metrics:
        if metric in df.columns:
            df[f"{metric}_roll5"] = (
                df.groupby(["team", "qtr"])[metric]
                .transform(lambda x: x.rolling(5, min_periods=2).mean())
            )
            df[f"{metric}_roll3"] = (
                df.groupby(["team", "qtr"])[metric]
                .transform(lambda x: x.rolling(3, min_periods=2).mean())
            )

    return df


def build_rolling_half_features(team_half: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling averages for team half stats.
    E.g., "avg 1st half points over last 5 games"
    """
    df = team_half.sort_values(["team", "season", "week"]).copy()

    metrics = ["points", "total_yards", "pass_yards", "rush_yards",
               "touchdowns", "epa_per_play", "total_plays",
               "interceptions", "fumbles_lost"]

    for metric in metrics:
        if metric in df.columns:
            df[f"{metric}_roll5"] = (
                df.groupby(["team", "half"])[metric]
                .transform(lambda x: x.rolling(5, min_periods=2).mean())
            )
            df[f"{metric}_roll3"] = (
                df.groupby(["team", "half"])[metric]
                .transform(lambda x: x.rolling(3, min_periods=2).mean())
            )

    # Derived: 1st half scoring rate (% of total game points scored in 1st half)
    # This requires pivoting
    return df


def build_half_scoring_splits(team_half: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-team-per-game view showing 1st half vs 2nd half splits.
    Useful for identifying fast starters vs. second half teams.
    """
    pivot = team_half.pivot_table(
        index=["game_id", "season", "week", "team"],
        columns="half",
        values=["points", "total_yards", "pass_yards", "rush_yards",
                "touchdowns", "epa_per_play"],
        aggfunc="sum",
    ).reset_index()

    # Flatten column names
    pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot.columns]

    # Calculate ratios
    if "points_H1" in pivot.columns and "points_H2" in pivot.columns:
        total_pts = pivot["points_H1"] + pivot["points_H2"]
        pivot["h1_points_pct"] = pivot["points_H1"] / total_pts.replace(0, np.nan)
        pivot["h2_points_pct"] = pivot["points_H2"] / total_pts.replace(0, np.nan)

    if "total_yards_H1" in pivot.columns and "total_yards_H2" in pivot.columns:
        total_yds = pivot["total_yards_H1"] + pivot["total_yards_H2"]
        pivot["h1_yards_pct"] = pivot["total_yards_H1"] / total_yds.replace(0, np.nan)

    return pivot


def save_quarter_half_data():
    """
    Pull PBP, build quarter/half aggregations, and save.
    This is the main entry point for quarter/half data.
    """
    settings = load_settings()
    seasons = settings["data"]["historical_seasons"]
    raw_dir = get_data_dir("raw")
    processed_dir = get_data_dir("processed")

    print("=" * 60)
    print("BUILDING QUARTER/HALF STATS FROM PLAY-BY-PLAY")
    print("=" * 60)

    # Pull PBP
    pbp = pull_pbp_data(seasons)
    if pbp.empty:
        print("ERROR: No PBP data pulled")
        return

    # --- Team quarter stats ---
    print("\n[1/5] Building team quarter stats...")
    team_qtr = build_team_quarter_stats(pbp)
    team_qtr.to_parquet(raw_dir / "team_quarter_stats.parquet", index=False)
    print(f"  Saved: {len(team_qtr)} team-quarter records")

    # --- Team half stats ---
    print("[2/5] Building team half stats...")
    team_half = build_team_half_stats(pbp)
    team_half.to_parquet(raw_dir / "team_half_stats.parquet", index=False)
    print(f"  Saved: {len(team_half)} team-half records")

    # --- Half scoring splits ---
    print("[3/5] Building half scoring splits...")
    splits = build_half_scoring_splits(team_half)
    splits.to_parquet(processed_dir / "team_half_splits.parquet", index=False)
    print(f"  Saved: {len(splits)} game-team half splits")

    # --- Player half stats ---
    print("[4/5] Building player half stats...")
    player_half = build_player_half_stats(pbp)
    for key, df in player_half.items():
        df.to_parquet(raw_dir / f"player_{key}_half.parquet", index=False)
        print(f"  Saved player {key} half stats: {len(df)} records")

    # --- Player quarter stats ---
    print("[5/5] Building player quarter stats...")
    player_qtr = build_player_quarter_stats(pbp)
    for key, df in player_qtr.items():
        df.to_parquet(raw_dir / f"player_{key}.parquet", index=False)
        print(f"  Saved player {key}: {len(df)} records")

    # --- Rolling features ---
    print("\nBuilding rolling features...")
    team_qtr_rolling = build_rolling_quarter_features(team_qtr)
    team_qtr_rolling.to_parquet(processed_dir / "team_quarter_features.parquet", index=False)

    team_half_rolling = build_rolling_half_features(team_half)
    team_half_rolling.to_parquet(processed_dir / "team_half_features.parquet", index=False)

    print("\n" + "=" * 60)
    print("DONE - Quarter/half data saved")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  data/raw/team_quarter_stats.parquet")
    print(f"  data/raw/team_half_stats.parquet")
    print(f"  data/raw/player_passing_half.parquet")
    print(f"  data/raw/player_rushing_half.parquet")
    print(f"  data/raw/player_receiving_half.parquet")
    print(f"  data/raw/player_receiving_qtr.parquet")
    print(f"  data/raw/player_rushing_qtr.parquet")
    print(f"  data/raw/player_passing_qtr.parquet")
    print(f"  data/processed/team_half_splits.parquet")
    print(f"  data/processed/team_quarter_features.parquet")
    print(f"  data/processed/team_half_features.parquet")


if __name__ == "__main__":
    save_quarter_half_data()
