"""
Red Zone & Target Share Features.
Builds from PBP data:
1. Red zone trip rate + TD conversion % per team
2. Target share changes (delta) for redistribution detection
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl

from pipeline.config_loader import load_settings, get_data_dir


def build_redzone_features(pbp: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build red zone features from PBP data.
    
    Features per team per game:
    - Red zone trips (drives reaching inside the 20)
    - Red zone TDs (TDs scored from inside the 20)
    - Red zone TD % (conversion rate)
    - Red zone pass vs rush ratio
    - Defensive red zone stops
    """
    if pbp is None:
        raw_dir = get_data_dir("raw")
        # Load from saved PBP quarter data (we already have the plays)
        # Re-pull PBP if needed
        settings = load_settings()
        seasons = settings["data"]["historical_seasons"]
        print("[redzone] Pulling PBP data for red zone analysis...")
        
        cols_needed = [
            "game_id", "season", "week", "posteam", "defteam",
            "play_type", "yardline_100", "touchdown", "field_goal_attempt",
            "field_goal_result", "pass_attempt", "rush_attempt",
            "yards_gained", "drive", "epa",
        ]
        pbp = nfl.import_pbp_data(seasons, columns=cols_needed)

    print("[redzone] Building red zone features...")
    
    # Filter to real plays in the red zone (within 20 yards of end zone)
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    rz_plays = plays[plays["yardline_100"] <= 20].copy()

    # --- Red zone trips per team per game ---
    # A "trip" is a unique drive that reaches the red zone
    rz_drives = rz_plays.drop_duplicates(subset=["game_id", "posteam", "drive"])
    rz_trips = rz_drives.groupby(["game_id", "season", "week", "posteam"]).agg(
        rz_trips=("drive", "nunique"),
    ).reset_index()

    # --- Red zone TDs ---
    rz_tds = rz_plays[rz_plays["touchdown"] == 1].groupby(
        ["game_id", "season", "week", "posteam"]
    ).agg(
        rz_tds=("touchdown", "sum"),
    ).reset_index()

    # --- Red zone play breakdown ---
    rz_breakdown = rz_plays.groupby(["game_id", "season", "week", "posteam"]).agg(
        rz_plays=("play_type", "count"),
        rz_pass_attempts=("pass_attempt", "sum"),
        rz_rush_attempts=("rush_attempt", "sum"),
        rz_yards=("yards_gained", "sum"),
        rz_epa=("epa", "sum"),
    ).reset_index()

    # Merge all red zone stats
    rz_stats = rz_trips.merge(rz_tds, on=["game_id", "season", "week", "posteam"], how="left")
    rz_stats = rz_stats.merge(rz_breakdown, on=["game_id", "season", "week", "posteam"], how="left")
    rz_stats["rz_tds"] = rz_stats["rz_tds"].fillna(0)
    
    # Conversion rate
    rz_stats["rz_td_pct"] = rz_stats["rz_tds"] / rz_stats["rz_trips"].replace(0, np.nan)
    rz_stats["rz_pass_rate"] = rz_stats["rz_pass_attempts"] / rz_stats["rz_plays"].replace(0, np.nan)

    rz_stats.rename(columns={"posteam": "team"}, inplace=True)

    # --- Defensive red zone stats (what teams ALLOW) ---
    def_rz_trips = rz_drives.groupby(["game_id", "season", "week", "defteam"]).agg(
        def_rz_trips_allowed=("drive", "nunique"),
    ).reset_index()
    def_rz_tds = rz_plays[rz_plays["touchdown"] == 1].groupby(
        ["game_id", "season", "week", "defteam"]
    ).agg(
        def_rz_tds_allowed=("touchdown", "sum"),
    ).reset_index()

    def_rz = def_rz_trips.merge(def_rz_tds, on=["game_id", "season", "week", "defteam"], how="left")
    def_rz["def_rz_tds_allowed"] = def_rz["def_rz_tds_allowed"].fillna(0)
    def_rz["def_rz_td_pct_allowed"] = def_rz["def_rz_tds_allowed"] / def_rz["def_rz_trips_allowed"].replace(0, np.nan)
    def_rz.rename(columns={"defteam": "team"}, inplace=True)

    # Merge offensive + defensive
    rz_combined = rz_stats.merge(
        def_rz[["game_id", "season", "week", "team", "def_rz_trips_allowed", "def_rz_tds_allowed", "def_rz_td_pct_allowed"]],
        on=["game_id", "season", "week", "team"],
        how="outer",
    )

    # Rolling averages
    rz_combined = rz_combined.sort_values(["team", "season", "week"])
    for metric in ["rz_trips", "rz_tds", "rz_td_pct", "rz_pass_rate",
                   "def_rz_trips_allowed", "def_rz_tds_allowed", "def_rz_td_pct_allowed"]:
        if metric in rz_combined.columns:
            rz_combined[f"{metric}_roll5"] = (
                rz_combined.groupby("team")[metric]
                .transform(lambda x: x.rolling(5, min_periods=2).mean())
            )

    print(f"  Red zone features: {len(rz_combined)} team-game records")
    return rz_combined


def build_target_share_deltas(player_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build target share change features.
    Detects when a player's target share is rising or falling,
    which indicates role changes, injuries to teammates, or new additions.
    
    Features:
    - target_share_delta_3: change in target share last 3 vs previous 3
    - target_share_trend: slope of target share over recent games
    - is_rising: boolean - target share increasing significantly
    - is_falling: boolean - target share decreasing significantly
    """
    if player_stats is None:
        raw_dir = get_data_dir("raw")
        player_stats = pd.read_parquet(raw_dir / "player_stats_historical.parquet")

    print("[targets] Building target share delta features...")

    # Filter to pass catchers
    receivers = player_stats[player_stats["position"].isin(["WR", "TE", "RB"])].copy()
    
    if "target_share" not in receivers.columns:
        # Calculate target share from targets if not available
        if "targets" in receivers.columns:
            team_targets = receivers.groupby(["season", "week", "recent_team"])["targets"].transform("sum")
            receivers["target_share"] = receivers["targets"] / team_targets.replace(0, np.nan)
        else:
            print("  WARNING: No target_share or targets column available")
            return pd.DataFrame()

    receivers = receivers.sort_values(["player_id", "season", "week"])

    # Rolling target share — MUST shift(1) so the window uses only PRIOR games,
    # otherwise the current week's target share leaks into the feature (target leakage).
    receivers["ts_roll3"] = (
        receivers.groupby("player_id")["target_share"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    )
    receivers["ts_roll5"] = (
        receivers.groupby("player_id")["target_share"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    )
    receivers["ts_roll_prev5"] = (
        receivers.groupby("player_id")["target_share"]
        .transform(lambda x: x.shift(4).rolling(5, min_periods=3).mean())
    )

    # Delta: recent vs previous window
    receivers["target_share_delta"] = receivers["ts_roll3"] - receivers["ts_roll5"]
    receivers["target_share_delta_wide"] = receivers["ts_roll3"] - receivers["ts_roll_prev5"]

    # Trend (simplified slope over last 5 games)
    def rolling_slope(series, window=5):
        """Calculate approximate slope over a rolling window."""
        return series.rolling(window, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0,
            raw=True,
        )

    receivers["target_share_trend"] = (
        receivers.groupby("player_id")["target_share"]
        .transform(lambda x: rolling_slope(x.shift(1)))  # prior games only, no leakage
    )

    # Flags
    receivers["ts_rising"] = receivers["target_share_delta"] > 0.03  # 3%+ increase
    receivers["ts_falling"] = receivers["target_share_delta"] < -0.03  # 3%+ decrease

    # Also build team-level target concentration (Herfindahl index)
    # High concentration = one player dominates targets
    if "targets" in receivers.columns:
        team_totals = receivers.groupby(["season", "week", "recent_team"])["targets"].transform("sum")
        receivers["player_target_pct"] = receivers["targets"] / team_totals.replace(0, np.nan)
        
        team_hhi = receivers.groupby(["season", "week", "recent_team"]).apply(
            lambda x: (x["player_target_pct"] ** 2).sum() if len(x) > 0 else 0,
            include_groups=False,
        ).reset_index(name="team_target_hhi")
        
        receivers = receivers.merge(team_hhi, on=["season", "week", "recent_team"], how="left")

    print(f"  Target share features: {len(receivers)} player-game records")
    print(f"  Players with rising share: {receivers['ts_rising'].sum()}")
    print(f"  Players with falling share: {receivers['ts_falling'].sum()}")

    return receivers


def save_redzone_target_data():
    """Build and save red zone + target share features."""
    raw_dir = get_data_dir("raw")
    processed_dir = get_data_dir("processed")

    print("=" * 60)
    print("BUILDING RED ZONE & TARGET SHARE FEATURES")
    print("=" * 60)

    # Red zone features
    rz = build_redzone_features()
    if not rz.empty:
        rz.to_parquet(processed_dir / "redzone_features.parquet", index=False)
        print(f"\n  Saved red zone features: {len(rz)} records")

    # Target share deltas
    ts = build_target_share_deltas()
    if not ts.empty:
        ts.to_parquet(processed_dir / "target_share_features.parquet", index=False)
        print(f"  Saved target share features: {len(ts)} records")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    save_redzone_target_data()
