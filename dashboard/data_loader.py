"""
Dashboard Data Loader.
Handles loading prediction outputs and formatting for display.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_best_bets() -> pd.DataFrame | None:
    """Load the latest best bets output."""
    filepath = DATA_DIR / "best_bets_latest.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


def load_spread_predictions() -> pd.DataFrame | None:
    """Load spread predictions with edges."""
    filepath = DATA_DIR / "spread_edges_latest.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    # Fall back to raw spread predictions
    filepath = DATA_DIR / "spreads_latest.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


def load_totals_predictions() -> pd.DataFrame | None:
    """Load totals predictions with edges."""
    filepath = DATA_DIR / "totals_edges_latest.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    filepath = DATA_DIR / "totals_latest.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


def load_player_prop_predictions(prop_type: str) -> pd.DataFrame | None:
    """Load predictions for a specific prop type."""
    filepath = DATA_DIR / f"{prop_type}_latest.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


def load_all_prop_predictions() -> dict[str, pd.DataFrame]:
    """Load all available player prop predictions."""
    props = {}
    prop_types = [
        "passing_yards", "rushing_yards", "receiving_yards",
        "receptions", "touchdowns_anytime",
    ]
    for prop_type in prop_types:
        df = load_player_prop_predictions(prop_type)
        if df is not None:
            props[prop_type] = df
    return props


def load_game_predictions() -> pd.DataFrame | None:
    """Load combined game-level predictions (spread + total + ML)."""
    spreads = load_spread_predictions()
    totals = load_totals_predictions()

    if spreads is None and totals is None:
        return None

    if spreads is not None and totals is not None:
        # Merge on game_id
        combined = spreads.merge(
            totals,
            on="game_id",
            how="outer",
            suffixes=("_spread", "_total"),
        )
        return combined
    return spreads if spreads is not None else totals


def get_data_freshness() -> str | None:
    """Check when prediction data was last updated."""
    filepath = DATA_DIR / "best_bets_latest.parquet"
    if filepath.exists():
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        return mtime.strftime("%B %d, %Y at %I:%M %p")
    return None


def get_player_list(prop_type: str | None = None) -> list[str]:
    """Get list of players with predictions available."""
    if prop_type:
        df = load_player_prop_predictions(prop_type)
        if df is not None and "player_display_name" in df.columns:
            return sorted(df["player_display_name"].dropna().unique().tolist())
        if df is not None and "player_name" in df.columns:
            return sorted(df["player_name"].dropna().unique().tolist())

    # Search across all prop types
    all_players = set()
    for prop, df in load_all_prop_predictions().items():
        name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
        if name_col in df.columns:
            all_players.update(df[name_col].dropna().unique())
    return sorted(all_players)


def get_player_data(player_name: str) -> dict[str, pd.DataFrame]:
    """Get all prediction data for a specific player across all prop types."""
    player_data = {}
    all_props = load_all_prop_predictions()

    for prop_type, df in all_props.items():
        name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
        if name_col in df.columns:
            player_rows = df[df[name_col].str.contains(player_name, case=False, na=False)]
            if not player_rows.empty:
                player_data[prop_type] = player_rows

    return player_data
