"""
Line Movement Capture System.
Records prop lines at multiple timestamps to calculate Closing Line Value (CLV).

CLV = the difference between the line when you bet vs the line at kickoff.
Consistently beating the closing line is the #1 predictor of long-term profitability.

Schedule:
  - Tuesday 6pm ET: First capture (lines just posted for most games)
  - Wednesday 8am ET: Second capture (overnight movement)
  - Thursday 8am ET: Third capture (main betting day)
  - Sunday 9am ET: Final capture before kickoff (closing lines)

Each capture is saved with a timestamp. After games, we compare:
  - Opening line vs closing line (market movement)
  - Our bet line vs closing line (our CLV)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json

from pipeline.config_loader import get_data_dir, load_settings
from pipeline.ingest_odds import pull_game_odds, pull_player_props, pull_all_props_for_week


LINES_DIR = get_data_dir("lines")


def capture_game_lines() -> pd.DataFrame:
    """Capture current game-level lines (spreads, totals, moneylines)."""
    print("[capture] Pulling game-level lines...")
    try:
        odds = pull_game_odds(markets="h2h,spreads,totals")
        if odds.empty:
            print("  No game lines available")
            return pd.DataFrame()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        odds["capture_timestamp"] = timestamp
        odds["capture_utc"] = datetime.now(timezone.utc).isoformat()

        # Save timestamped snapshot
        filename = f"game_lines_{timestamp}.parquet"
        odds.to_parquet(LINES_DIR / filename, index=False)
        print(f"  Saved {len(odds)} lines to {filename}")

        # Also save as 'latest'
        odds.to_parquet(LINES_DIR / "game_lines_latest.parquet", index=False)

        return odds
    except Exception as e:
        print(f"  ERROR capturing game lines: {e}")
        return pd.DataFrame()


def capture_prop_lines() -> pd.DataFrame:
    """Capture current player prop lines for all available games."""
    print("[capture] Pulling player prop lines...")
    try:
        props = pull_all_props_for_week()
        if props.empty:
            print("  No prop lines available (may be too early in the week)")
            return pd.DataFrame()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        props["capture_timestamp"] = timestamp
        props["capture_utc"] = datetime.now(timezone.utc).isoformat()

        # Save timestamped snapshot
        filename = f"prop_lines_{timestamp}.parquet"
        props.to_parquet(LINES_DIR / filename, index=False)
        print(f"  Saved {len(props)} prop lines to {filename}")

        # Also save as 'latest'
        props.to_parquet(LINES_DIR / "prop_lines_latest.parquet", index=False)

        return props
    except Exception as e:
        print(f"  ERROR capturing prop lines: {e}")
        return pd.DataFrame()


def capture_all():
    """Run full line capture (game + props)."""
    print("=" * 60)
    print(f"LINE CAPTURE — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    game_lines = capture_game_lines()
    prop_lines = capture_prop_lines()

    # Summary
    print(f"\n  Games captured: {game_lines['game_id'].nunique() if not game_lines.empty else 0}")
    print(f"  Props captured: {len(prop_lines) if not prop_lines.empty else 0}")
    print("=" * 60)

    return game_lines, prop_lines


def build_line_history(market: str = "props") -> pd.DataFrame:
    """
    Combine all captured snapshots into a line history DataFrame.
    Shows how each line moved over time.
    """
    pattern = "prop_lines_*.parquet" if market == "props" else "game_lines_*.parquet"
    files = sorted(LINES_DIR.glob(pattern))

    if not files:
        print(f"  No {market} line captures found")
        return pd.DataFrame()

    # Skip 'latest' files
    files = [f for f in files if "latest" not in f.name]

    all_captures = []
    for f in files:
        df = pd.read_parquet(f)
        all_captures.append(df)

    if not all_captures:
        return pd.DataFrame()

    history = pd.concat(all_captures, ignore_index=True)
    print(f"  Line history: {len(history)} records across {len(files)} captures")
    return history


def calculate_clv(bet_line: float, closing_line: float, direction: str) -> float:
    """
    Calculate Closing Line Value.
    Positive CLV = you got a better line than the market closed at.

    Args:
        bet_line: The line when you placed your bet
        closing_line: The line at kickoff
        direction: 'over' or 'under'

    Returns:
        CLV in points/yards (positive = good)
    """
    if direction == "over":
        # For overs, you want the closing line to be HIGHER than your bet line
        # (market moved toward over after you bet)
        return closing_line - bet_line
    else:
        # For unders, you want the closing line to be LOWER than your bet line
        # (market moved toward under after you bet)
        return bet_line - closing_line


def analyze_line_movement(player_name: str = None) -> pd.DataFrame:
    """
    Analyze how lines moved for a specific player or all players.
    Returns opening, closing, and movement direction.
    """
    history = build_line_history("props")
    if history.empty:
        return pd.DataFrame()

    if player_name:
        history = history[history["player_name"].str.lower() == player_name.lower()]

    # Group by player + market + game, get first and last capture
    if "outcome_point" not in history.columns:
        print("  No point data in captures")
        return pd.DataFrame()

    # Only 'Over' lines (Over/Under are mirrors)
    overs = history[history["outcome_name"] == "Over"].copy()

    grouped = overs.groupby(["player_name", "market", "event_id"]).agg(
        opening_line=("outcome_point", "first"),
        closing_line=("outcome_point", "last"),
        opening_price=("outcome_price", "first"),
        closing_price=("outcome_price", "last"),
        first_capture=("capture_timestamp", "first"),
        last_capture=("capture_timestamp", "last"),
        num_captures=("capture_timestamp", "count"),
    ).reset_index()

    grouped["line_movement"] = grouped["closing_line"] - grouped["opening_line"]
    grouped["moved_up"] = grouped["line_movement"] > 0
    grouped["moved_down"] = grouped["line_movement"] < 0

    return grouped.sort_values("line_movement", key=abs, ascending=False)


if __name__ == "__main__":
    capture_all()
