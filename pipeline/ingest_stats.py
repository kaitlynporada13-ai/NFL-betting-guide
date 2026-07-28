"""
Historical NFL Stats Ingestion via nfl_data_py (nflverse).
Pulls game-level and player-level data for model training and live predictions.
"""

import pandas as pd
import nfl_data_py as nfl

from pipeline.config_loader import load_settings, get_data_dir


# New nflverse data URL format (2025+)
NFLVERSE_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"


def _import_weekly_data_with_fallback(seasons: list[int]) -> pd.DataFrame:
    """
    Pull weekly player stats, using nfl_data_py for older seasons
    and direct GitHub URLs for newer seasons (2025+).
    """
    old_seasons = [s for s in seasons if s <= 2024]
    new_seasons = [s for s in seasons if s >= 2025]

    dfs = []

    # Pull older seasons via nfl_data_py (legacy format)
    if old_seasons:
        try:
            legacy = nfl.import_weekly_data(old_seasons)
            dfs.append(legacy)
        except Exception as e:
            print(f"  Warning: could not pull legacy seasons {old_seasons}: {e}")

    # Pull newer seasons directly from nflverse-data releases
    for year in new_seasons:
        url = NFLVERSE_STATS_URL.format(year=year)
        try:
            df = pd.read_parquet(url)
            # Normalize column names to match legacy format
            col_renames = {
                "player_display_name": "player_display_name",
                "player_name": "player_name",
                "recent_team": "recent_team",
            }
            # The new format has same column names mostly
            dfs.append(df)
            print(f"  Pulled {year} from nflverse-data: {len(df)} rows")
        except Exception as e:
            print(f"  Warning: could not pull {year} data: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    return pd.DataFrame()


def pull_game_data(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull game-level schedule and results for specified seasons.
    Includes scores, spreads, totals, and game metadata.
    """
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[ingest_stats] Pulling game schedules for seasons: {seasons}")
    games = nfl.import_schedules(seasons)

    # Keep relevant columns
    cols = [
        "game_id", "season", "game_type", "week", "gameday", "weekday",
        "gametime", "away_team", "home_team", "away_score", "home_score",
        "home_rest", "away_rest", "spread_line", "total_line",
        "away_spread_odds", "home_spread_odds", "away_moneyline", "home_moneyline",
        "div_game", "roof", "surface", "temp", "wind", "stadium_id", "stadium"
    ]
    available_cols = [c for c in cols if c in games.columns]
    games = games[available_cols].copy()

    # Derived fields
    if "home_score" in games.columns and "away_score" in games.columns:
        games["total_points"] = games["home_score"] + games["away_score"]
        games["home_margin"] = games["home_score"] - games["away_score"]
        games["home_cover"] = games["home_margin"] + games["spread_line"] > 0
        games["over_hit"] = games["total_points"] > games["total_line"]

    return games


def pull_team_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull weekly team-level stats (offensive and defensive).
    Covers yards, turnovers, efficiency, etc.
    """
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[ingest_stats] Pulling weekly team stats for seasons: {seasons}")
    weekly = _import_weekly_data_with_fallback(seasons)

    # Aggregate to team-game level
    team_stats = weekly.groupby(
        ["season", "week", "recent_team"]
    ).agg(
        total_pass_yards=("passing_yards", "sum"),
        total_rush_yards=("rushing_yards", "sum"),
        total_receiving_yards=("receiving_yards", "sum"),
        total_pass_tds=("passing_tds", "sum"),
        total_rush_tds=("rushing_tds", "sum"),
        total_receiving_tds=("receiving_tds", "sum"),
        total_interceptions=("interceptions", "sum"),
        total_fumbles_lost=("sack_fumbles_lost", "sum"),
        total_carries=("carries", "sum"),
        total_targets=("targets", "sum"),
        total_receptions=("receptions", "sum"),
        total_attempts=("attempts", "sum"),
        total_completions=("completions", "sum"),
    ).reset_index()

    team_stats.rename(columns={"recent_team": "team"}, inplace=True)
    return team_stats


def pull_player_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull weekly player-level stats for prop modeling.
    Covers passing, rushing, receiving for all skill positions.
    """
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[ingest_stats] Pulling weekly player stats for seasons: {seasons}")
    weekly = _import_weekly_data_with_fallback(seasons)

    # Keep skill position players
    skill_positions = ["QB", "RB", "WR", "TE"]
    players = weekly[weekly["position"].isin(skill_positions)].copy()

    # Key columns for prop modeling
    cols = [
        "player_id", "player_name", "player_display_name", "position",
        "recent_team", "season", "week",
        # Passing
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "sacks", "sack_yards",
        # Rushing
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
        # Receiving
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_fumbles_lost",
        # General
        "fantasy_points_ppr", "target_share", "air_yards_share",
    ]
    available_cols = [c for c in cols if c in players.columns]
    players = players[available_cols].copy()

    return players


def pull_rosters(seasons: list[int] | None = None) -> pd.DataFrame:
    """Pull roster data for player metadata (height, weight, experience)."""
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[ingest_stats] Pulling roster data for seasons: {seasons}")
    rosters = nfl.import_players()
    return rosters


def save_all_historical_data():
    """Pull and save all historical data to parquet files."""
    settings = load_settings()
    seasons = settings["data"]["historical_seasons"]
    raw_dir = get_data_dir("raw")

    print("=" * 60)
    print("PULLING ALL HISTORICAL NFL DATA")
    print("=" * 60)

    # Game data
    games = pull_game_data(seasons)
    games.to_parquet(raw_dir / "games_historical.parquet", index=False)
    print(f"  Saved {len(games)} games to games_historical.parquet")

    # Team stats
    team_stats = pull_team_stats(seasons)
    team_stats.to_parquet(raw_dir / "team_stats_historical.parquet", index=False)
    print(f"  Saved {len(team_stats)} team-week records to team_stats_historical.parquet")

    # Player stats
    player_stats = pull_player_stats(seasons)
    player_stats.to_parquet(raw_dir / "player_stats_historical.parquet", index=False)
    print(f"  Saved {len(player_stats)} player-week records to player_stats_historical.parquet")

    # Rosters
    rosters = pull_rosters(seasons)
    rosters.to_parquet(raw_dir / "rosters_historical.parquet", index=False)
    print(f"  Saved {len(rosters)} roster entries to rosters_historical.parquet")

    print("=" * 60)
    print("DONE - All historical data saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    save_all_historical_data()
