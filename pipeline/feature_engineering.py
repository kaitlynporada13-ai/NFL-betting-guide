"""
Feature Engineering Pipeline.
Transforms raw data into model-ready features incorporating:
- Rolling team performance metrics
- Situational factors (rest, travel, surface, weather)
- Scheme/pace indicators
- Player-level prop features
"""

import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config_loader import load_settings, load_stadiums, get_data_dir


# ============================================================
# TEAM-LEVEL FEATURES (for spreads, totals, moneyline)
# ============================================================

def build_rolling_team_stats(team_stats: pd.DataFrame, windows: list[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Build rolling averages for team performance metrics.
    Creates features like "avg passing yards last 3 games" etc.
    
    Args:
        team_stats: Weekly team stats from ingest_stats
        windows: Rolling window sizes (games)
    """
    df = team_stats.sort_values(["team", "season", "week"]).copy()

    # Metrics to roll
    metrics = [
        "total_pass_yards", "total_rush_yards", "total_pass_tds",
        "total_rush_tds", "total_interceptions", "total_fumbles_lost",
        "total_carries", "total_targets", "total_receptions",
        "total_attempts", "total_completions",
    ]

    for window in windows:
        for metric in metrics:
            if metric in df.columns:
                col_name = f"{metric}_roll{window}"
                df[col_name] = (
                    df.groupby("team")[metric]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

    # Derived rolling metrics
    for window in windows:
        prefix = f"roll{window}"
        # Completion percentage
        if f"total_completions_roll{window}" in df.columns:
            df[f"completion_pct_{prefix}"] = (
                df[f"total_completions_roll{window}"]
                / df[f"total_attempts_roll{window}"].replace(0, np.nan)
            )

        # Pass/rush ratio
        total_yards_col = f"total_pass_yards_roll{window}"
        rush_yards_col = f"total_rush_yards_roll{window}"
        if total_yards_col in df.columns and rush_yards_col in df.columns:
            df[f"pass_rush_ratio_{prefix}"] = (
                df[total_yards_col]
                / (df[total_yards_col] + df[rush_yards_col]).replace(0, np.nan)
            )

        # Turnover rate
        if f"total_interceptions_roll{window}" in df.columns:
            df[f"turnovers_{prefix}"] = (
                df[f"total_interceptions_roll{window}"]
                + df[f"total_fumbles_lost_roll{window}"]
            )

    return df


def build_situational_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Build situational features for each game:
    - Rest days differential
    - Time zone travel penalty
    - Division rivalry flag
    - Surface change flag
    - Altitude change
    """
    stadiums_data = load_stadiums()
    stadiums = stadiums_data["stadiums"]
    team_map = stadiums_data["team_stadium_map"]

    df = games.copy()

    # Rest differential (home - away)
    if "home_rest" in df.columns and "away_rest" in df.columns:
        df["rest_differential"] = df["home_rest"] - df["away_rest"]
        df["short_rest_home"] = df["home_rest"] <= 6
        df["short_rest_away"] = df["away_rest"] <= 6

    # Time zone travel
    timezone_offsets = {
        "America/New_York": -5,
        "America/Indiana/Indianapolis": -5,
        "America/Detroit": -5,
        "America/Chicago": -6,
        "America/Denver": -7,
        "America/Phoenix": -7,
        "America/Los_Angeles": -8,
    }

    def get_tz_offset(team):
        stadium_key = team_map.get(team)
        if not stadium_key:
            return -6  # default Central
        stadium_info = stadiums.get(stadium_key, {})
        tz = stadium_info.get("timezone", "America/Chicago")
        return timezone_offsets.get(tz, -6)

    df["home_tz"] = df["home_team"].map(get_tz_offset)
    df["away_tz"] = df["away_team"].map(get_tz_offset)
    df["tz_travel"] = abs(df["away_tz"] - df["home_tz"])
    df["east_to_west_travel"] = (df["away_tz"] - df["home_tz"]) > 0  # traveling west
    df["west_to_east_early"] = (df["away_tz"] - df["home_tz"]) < 0  # traveling east (early body clock)

    # Surface features
    def get_surface(team):
        stadium_key = team_map.get(team)
        if not stadium_key:
            return "turf"
        return stadiums.get(stadium_key, {}).get("surface", "turf")

    def get_roof(team):
        stadium_key = team_map.get(team)
        if not stadium_key:
            return "dome"
        return stadiums.get(stadium_key, {}).get("roof", "open")

    df["home_surface"] = df["home_team"].map(get_surface)
    df["away_home_surface"] = df["away_team"].map(get_surface)
    df["surface_change"] = df["home_surface"] != df["away_home_surface"]
    df["game_on_grass"] = df["home_surface"] == "grass"
    df["game_in_dome"] = df["home_team"].map(get_roof).isin(["dome", "retractable"])

    # Altitude (Denver factor)
    def get_altitude(team):
        stadium_key = team_map.get(team)
        if not stadium_key:
            return 500
        return stadiums.get(stadium_key, {}).get("altitude_ft", 500)

    df["game_altitude"] = df["home_team"].map(get_altitude)
    df["high_altitude"] = df["game_altitude"] > 4000  # basically Denver
    df["away_altitude_diff"] = df["home_team"].map(get_altitude) - df["away_team"].map(get_altitude)

    # Division game (more competitive, lower margins)
    if "div_game" in df.columns:
        df["is_division_game"] = df["div_game"].astype(bool)
    else:
        df["is_division_game"] = False

    return df


def build_pace_features(team_stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Build pace-of-play features that affect volume (important for props).
    - Plays per game
    - Average seconds per play
    - Neutral script pass rate
    """
    df = team_stats.copy()

    # Plays per game proxy: attempts + carries
    if "total_attempts" in df.columns and "total_carries" in df.columns:
        df["total_plays"] = df["total_attempts"] + df["total_carries"]

        # Rolling pace
        df["pace_roll5"] = (
            df.groupby("team")["total_plays"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )

    return df


# ============================================================
# PLAYER-LEVEL FEATURES (for props)
# ============================================================

def build_player_rolling_stats(player_stats: pd.DataFrame, windows: list[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Build rolling averages for player prop features.
    Tracks per-game production, target share, and consistency.
    """
    df = player_stats.sort_values(["player_id", "season", "week"]).copy()

    # Metrics to roll by position
    qb_metrics = ["passing_yards", "passing_tds", "interceptions", "completions", "attempts", "rushing_yards"]
    rb_metrics = ["rushing_yards", "carries", "rushing_tds", "receptions", "receiving_yards", "targets"]
    wr_te_metrics = ["receiving_yards", "receptions", "targets", "receiving_tds", "target_share"]

    all_metrics = list(set(qb_metrics + rb_metrics + wr_te_metrics))

    # CRITICAL: rolling windows MUST exclude the current week, otherwise the feature
    # leaks the target (a roll that includes this week's stat is partly the answer).
    # .shift(1) inside each player group makes every rolling value use ONLY prior games.
    def prior_roll(series, window, how):
        s = series.shift(1)  # drop current week before aggregating
        r = s.rolling(window, min_periods=1)
        return getattr(r, how)()

    for window in windows:
        for metric in all_metrics:
            if metric in df.columns:
                df[f"{metric}_roll{window}"] = (
                    df.groupby("player_id")[metric]
                    .transform(lambda x: prior_roll(x, window, "mean"))
                )

    # Consistency (std dev) — important for props (prior games only)
    for window in [5, 10]:
        for metric in ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]:
            if metric in df.columns:
                df[f"{metric}_std{window}"] = (
                    df.groupby("player_id")[metric]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
                )

    # Ceiling/floor (max/min in window) — prior games only
    for metric in ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]:
        if metric in df.columns:
            df[f"{metric}_max5"] = (
                df.groupby("player_id")[metric]
                .transform(lambda x: prior_roll(x, 5, "max"))
            )
            df[f"{metric}_min5"] = (
                df.groupby("player_id")[metric]
                .transform(lambda x: prior_roll(x, 5, "min"))
            )

    return df


def build_matchup_features(
    player_stats: pd.DataFrame,
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build opponent-based matchup features.
    - How does this defense rank against this position?
    - Historical performance vs. this specific team?
    """
    # Calculate defensive rankings (yards allowed by position)
    # This is a simplified version — could be much more granular

    # Merge player stats with game info to know opponent
    player_games = player_stats.merge(
        games[["season", "week", "home_team", "away_team"]].drop_duplicates(),
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "home_team"],
        how="left",
    )
    # For away players
    player_games_away = player_stats.merge(
        games[["season", "week", "home_team", "away_team"]].drop_duplicates(),
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "away_team"],
        how="left",
    )

    # Determine opponent
    player_games["opponent"] = np.where(
        player_games["recent_team"] == player_games["home_team"],
        player_games["away_team"],
        player_games["home_team"],
    )

    # Calculate how much each defense allows per game by position
    def_allowed = player_games.groupby(
        ["season", "week", "opponent", "position"]
    ).agg(
        pass_yards_allowed=("passing_yards", "sum"),
        rush_yards_allowed=("rushing_yards", "sum"),
        rec_yards_allowed=("receiving_yards", "sum"),
        receptions_allowed=("receptions", "sum"),
    ).reset_index()

    # Rolling defensive performance
    def_allowed = def_allowed.sort_values(["opponent", "position", "season", "week"])
    for metric in ["pass_yards_allowed", "rush_yards_allowed", "rec_yards_allowed", "receptions_allowed"]:
        def_allowed[f"{metric}_roll5"] = (
            def_allowed.groupby(["opponent", "position"])[metric]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )

    return def_allowed


# ============================================================
# WEATHER IMPACT FEATURES
# ============================================================

def build_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw weather into model-ready impact features.
    """
    df = weather_df.copy()

    # Passing suppression factor (0 to 1, where 1 = no impact)
    df["pass_suppression"] = 1.0
    if "wind_speed" in df.columns:
        # 10% reduction per 5 mph over 10
        wind_penalty = ((df["wind_speed"].fillna(0) - 10).clip(lower=0) / 5) * 0.10
        df["pass_suppression"] -= wind_penalty

    if "precip_prob" in df.columns:
        # 5% reduction for rain likely games
        rain_penalty = (df["precip_prob"].fillna(0) / 100) * 0.05
        df["pass_suppression"] -= rain_penalty

    df["pass_suppression"] = df["pass_suppression"].clip(lower=0.7)

    # Rushing boost (inverse of pass suppression — bad weather helps run game)
    df["rush_boost"] = 1.0 + (1.0 - df["pass_suppression"]) * 0.5

    # Total suppression (bad weather = lower scoring)
    df["total_suppression"] = 1.0
    if "wind_speed" in df.columns:
        total_wind = ((df["wind_speed"].fillna(0) - 15).clip(lower=0) / 5) * 0.05
        df["total_suppression"] -= total_wind
    if "temp" in df.columns:
        cold_penalty = ((32 - df["temp"].fillna(60)).clip(lower=0) / 20) * 0.05
        df["total_suppression"] -= cold_penalty
    df["total_suppression"] = df["total_suppression"].clip(lower=0.80)

    return df


# ============================================================
# REFEREE FEATURES
# ============================================================

def build_ref_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Build referee tendency features.
    Requires ref assignment data (can be added from nflverse or scraped).
    
    Key metrics:
    - Average flags per game
    - Pass interference rate
    - Holding call rate
    - Impact on total points
    """
    # Placeholder — ref data needs to be sourced separately
    # nflverse has penalty data that can be grouped by referee
    # For now, return empty features to be filled later
    print("[features] Referee features: placeholder (requires penalty data)")
    return games


# ============================================================
# MASTER FEATURE BUILDER
# ============================================================

def build_game_features(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Master function to build all game-level features.
    Combines rolling stats, situational, pace, and weather.
    """
    print("[features] Building game-level features...")

    # 0. Ensure derived target columns exist
    if "home_margin" not in games.columns:
        if "home_score" in games.columns and "away_score" in games.columns:
            games = games.copy()
            games["home_margin"] = games["home_score"] - games["away_score"]
            games["total_points"] = games["home_score"] + games["away_score"]
            if "spread_line" in games.columns:
                games["home_cover"] = games["home_margin"] + games["spread_line"] > 0
            if "total_line" in games.columns:
                games["over_hit"] = games["total_points"] > games["total_line"]

    # 1. Situational features
    game_features = build_situational_features(games)
    print("  ✓ Situational features (rest, travel, surface, altitude)")

    # 2. Rolling team stats
    rolling_team = build_rolling_team_stats(team_stats)
    print("  ✓ Rolling team performance metrics")

    # 3. Pace features
    pace = build_pace_features(team_stats, games)
    print("  ✓ Pace-of-play features")

    # 4. Merge home team stats
    home_cols = {c: f"home_{c}" for c in rolling_team.columns if c not in ["team", "season", "week"]}
    home_stats = rolling_team.rename(columns=home_cols)
    game_features = game_features.merge(
        home_stats,
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left",
    )

    # 5. Merge away team stats
    away_cols = {c: f"away_{c}" for c in rolling_team.columns if c not in ["team", "season", "week"]}
    away_stats = rolling_team.rename(columns=away_cols)
    game_features = game_features.merge(
        away_stats,
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left",
    )

    # 6. Weather features (if available)
    if weather is not None and not weather.empty:
        weather_features = build_weather_features(weather)
        game_features = game_features.merge(
            weather_features,
            on="game_id",
            how="left",
        )
        print("  ✓ Weather impact features")

    # 7. Derived differential features
    # Offensive power differential (home - away)
    for window in [3, 5]:
        pass_h = f"home_total_pass_yards_roll{window}"
        pass_a = f"away_total_pass_yards_roll{window}"
        rush_h = f"home_total_rush_yards_roll{window}"
        rush_a = f"away_total_rush_yards_roll{window}"

        if pass_h in game_features.columns and pass_a in game_features.columns:
            game_features[f"pass_yards_diff_roll{window}"] = (
                game_features[pass_h] - game_features[pass_a]
            )
        if rush_h in game_features.columns and rush_a in game_features.columns:
            game_features[f"rush_yards_diff_roll{window}"] = (
                game_features[rush_h] - game_features[rush_a]
            )

    print(f"  Final feature set: {game_features.shape[1]} columns, {len(game_features)} rows")
    return game_features


def build_player_prop_features(
    player_stats: pd.DataFrame,
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Master function to build all player-prop features.
    Incorporates: rolling stats, matchups, situational context,
    red zone features, air yards, snap counts, QB pressure.
    """
    print("[features] Building player-prop features...")

    # 1. Rolling player stats
    player_features = build_player_rolling_stats(player_stats)
    print("  ✓ Rolling player metrics + consistency")

    # 2. Matchup features (defensive rankings)
    matchup_features = build_matchup_features(player_stats, games, team_stats)
    print("  ✓ Defensive matchup features")

    # 3. Merge game situational features
    game_sit = build_situational_features(games)
    player_features = player_features.merge(
        game_sit[["season", "week", "home_team", "away_team",
                  "game_in_dome", "game_on_grass", "high_altitude",
                  "rest_differential", "tz_travel", "is_division_game"]].drop_duplicates(),
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "home_team"],
        how="left",
    )

    # 4. Merge red zone features (team-level RZ opportunity)
    proc_dir = get_data_dir("processed")
    rz_path = proc_dir / "redzone_features.parquet"
    if rz_path.exists():
        rz = pd.read_parquet(rz_path)
        rz_cols = ["season", "week", "team", "rz_td_pct_roll5", "rz_pass_rate_roll5",
                   "rz_trips_roll5", "rz_tds_roll5"]
        rz_avail = [c for c in rz_cols if c in rz.columns]
        player_features = player_features.merge(
            rz[rz_avail].drop_duplicates(),
            left_on=["season", "week", "recent_team"],
            right_on=["season", "week", "team"],
            how="left",
        )
        if "team" in player_features.columns:
            player_features.drop(columns=["team"], inplace=True, errors="ignore")
        print("  ✓ Red zone features (team-level)")

    # 5. Merge player red zone targets
    player_rz_path = proc_dir / "player_redzone_targets.parquet"
    if player_rz_path.exists():
        player_rz = pd.read_parquet(player_rz_path)
        if "receiver_player_name" in player_rz.columns:
            player_rz = player_rz.rename(columns={"receiver_player_name": "player_name"})
        rz_player_cols = ["player_name", "rz_targets", "rz_tds", "rz_target_share"]
        rz_player_avail = [c for c in rz_player_cols if c in player_rz.columns]
        if rz_player_avail:
            # This is a season-level profile, merge on player name
            name_col = "player_display_name" if "player_display_name" in player_features.columns else "player_name"
            player_features = player_features.merge(
                player_rz[rz_player_avail].drop_duplicates(subset=["player_name"]),
                left_on=name_col,
                right_on="player_name",
                how="left",
                suffixes=("", "_rz"),
            )
            print("  ✓ Player red zone targets")

    # 6. Merge air yards profile
    air_path = proc_dir / "player_air_yards_profile.parquet"
    if air_path.exists():
        air = pd.read_parquet(air_path)
        if "receiver_player_name" in air.columns:
            air = air.rename(columns={"receiver_player_name": "player_name"})
        air_cols = ["player_name", "avg_air_yards", "avg_yac", "deep_rate"]
        air_avail = [c for c in air_cols if c in air.columns]
        if air_avail:
            name_col = "player_display_name" if "player_display_name" in player_features.columns else "player_name"
            player_features = player_features.merge(
                air[air_avail].drop_duplicates(subset=["player_name"]),
                left_on=name_col,
                right_on="player_name",
                how="left",
                suffixes=("", "_air"),
            )
            print("  ✓ Air yards profile (depth, YAC)")

    # 7. Merge QB pressure profile
    qbp_path = proc_dir / "qb_pressure_profile.parquet"
    if qbp_path.exists():
        qbp = pd.read_parquet(qbp_path)
        if "passer_player_name" in qbp.columns:
            qbp = qbp.rename(columns={"passer_player_name": "player_name"})
        qbp_cols = ["player_name", "sack_rate", "hit_rate"]
        qbp_avail = [c for c in qbp_cols if c in qbp.columns]
        if qbp_avail:
            name_col = "player_display_name" if "player_display_name" in player_features.columns else "player_name"
            player_features = player_features.merge(
                qbp[qbp_avail].drop_duplicates(subset=["player_name"]),
                left_on=name_col,
                right_on="player_name",
                how="left",
                suffixes=("", "_pressure"),
            )
            print("  ✓ QB pressure profile")

    # 8. Merge snap count data (workload stability)
    raw_dir = get_data_dir("raw")
    snap_path = raw_dir / "snap_counts.parquet"
    if snap_path.exists():
        snaps = pd.read_parquet(snap_path)
        # Build rolling snap% for each player
        snaps = snaps.sort_values(["player", "season", "week"])
        snaps["snap_pct_roll3"] = (
            snaps.groupby("player")["offense_pct"]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        snaps["snap_pct_roll5"] = (
            snaps.groupby("player")["offense_pct"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        snap_features = snaps[["player", "season", "week", "offense_pct",
                               "snap_pct_roll3", "snap_pct_roll5"]].drop_duplicates()
        # Merge on player name + season + week
        name_col = "player_display_name" if "player_display_name" in player_features.columns else "player_name"
        player_features = player_features.merge(
            snap_features,
            left_on=[name_col, "season", "week"],
            right_on=["player", "season", "week"],
            how="left",
        )
        if "player" in player_features.columns and "player" != name_col:
            player_features.drop(columns=["player"], inplace=True, errors="ignore")
        print("  ✓ Snap count features (workload stability)")

    # 9. Merge goal-line carries (for RBs/TD model)
    gl_path = proc_dir / "player_goalline_carries.parquet"
    if gl_path.exists():
        gl = pd.read_parquet(gl_path)
        if "rusher_player_name" in gl.columns:
            gl = gl.rename(columns={"rusher_player_name": "player_name"})
        gl_cols = ["player_name", "gl_carries", "gl_tds"]
        gl_avail = [c for c in gl_cols if c in gl.columns]
        if gl_avail:
            name_col = "player_display_name" if "player_display_name" in player_features.columns else "player_name"
            player_features = player_features.merge(
                gl[gl_avail].drop_duplicates(subset=["player_name"]),
                left_on=name_col,
                right_on="player_name",
                how="left",
                suffixes=("", "_gl"),
            )
            print("  ✓ Goal-line carries")

    print(f"  Final feature set: {player_features.shape[1]} columns, {len(player_features)} rows")
    return player_features


# ============================================================
# SAVE PROCESSED FEATURES
# ============================================================

def save_features():
    """Load raw data, build features, and save to processed/."""
    raw_dir = get_data_dir("raw")
    processed_dir = get_data_dir("processed")

    # Load raw data
    games = pd.read_parquet(raw_dir / "games_historical.parquet")
    team_stats = pd.read_parquet(raw_dir / "team_stats_historical.parquet")
    player_stats = pd.read_parquet(raw_dir / "player_stats_historical.parquet")

    # Build features
    game_features = build_game_features(games, team_stats)
    game_features.to_parquet(processed_dir / "game_features.parquet", index=False)
    print(f"Saved game features: {game_features.shape}")

    player_features = build_player_prop_features(player_stats, games, team_stats)
    player_features.to_parquet(processed_dir / "player_features.parquet", index=False)
    print(f"Saved player features: {player_features.shape}")


if __name__ == "__main__":
    save_features()
