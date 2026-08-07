"""
Automated Bet Card Generator.
Pulls current week's props, runs each through the strategy engine,
and outputs a ranked bet card saved to the dashboard.

Run schedule: Tuesday evening (earliest props available) + Thursday (final card)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, date
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config_loader import load_settings, load_stadiums, get_data_dir
from pipeline.ingest_odds import pull_game_odds, pull_all_props_for_week
from pipeline.strategy_engine import evaluate_prop
from pipeline.ingest_contracts import is_contract_year_player


PROC_DIR = get_data_dir("processed")
NOTES_DIR = Path(__file__).parent.parent / "data" / "human_notes"


def get_current_week() -> int:
    """Estimate current NFL week based on date."""
    season_start = date(2026, 9, 5)  # Week 1 Thursday
    today = date.today()
    if today < season_start:
        return 0  # preseason
    days = (today - season_start).days
    return min(max(1, days // 7 + 1), 22)


def load_game_context() -> dict:
    """Load game-level context (dome, division, totals) for strategy engine."""
    stadiums_data = load_stadiums()
    stadiums = stadiums_data.get("stadiums", {})
    team_map = stadiums_data.get("team_stadium_map", {})

    # Dome teams
    dome_teams = set()
    for team, stadium_key in team_map.items():
        info = stadiums.get(stadium_key, {})
        if info.get("roof") in ("dome", "retractable"):
            dome_teams.add(team)

    # Division mapping
    divisions = {
        "AFC_East": ["BUF", "MIA", "NE", "NYJ"],
        "AFC_North": ["BAL", "CIN", "CLE", "PIT"],
        "AFC_South": ["HOU", "IND", "JAX", "TEN"],
        "AFC_West": ["DEN", "KC", "LAC", "LV"],
        "NFC_East": ["DAL", "NYG", "PHI", "WAS"],
        "NFC_North": ["CHI", "DET", "GB", "MIN"],
        "NFC_South": ["ATL", "CAR", "NO", "TB"],
        "NFC_West": ["ARI", "LAR", "SEA", "SF"],
    }
    team_to_div = {}
    for div, teams in divisions.items():
        for t in teams:
            team_to_div[t] = div

    return {
        "dome_teams": dome_teams,
        "team_to_div": team_to_div,
    }


def load_player_history() -> pd.DataFrame:
    """Load recent player stats for rolling averages."""
    path = get_data_dir("raw") / "player_stats_historical.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def get_player_rolling_avg(stats: pd.DataFrame, player_name: str, stat_col: str, window: int = 5) -> float:
    """Get a player's rolling average for a stat."""
    if stats.empty:
        return 0.0
    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    player = stats[stats[name_col].str.lower() == player_name.lower()]
    if player.empty:
        return 0.0
    player = player.sort_values(["season", "week"])
    recent = player[stat_col].dropna().tail(window)
    return recent.mean() if len(recent) > 0 else 0.0


def generate_bet_card():
    """
    Main function: pull props, evaluate each, output bet card.
    """
    settings = load_settings()
    week = get_current_week()
    context = load_game_context()
    stats = load_player_history()

    print("=" * 70)
    print(f"BET CARD GENERATOR — Week {week}")
    print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    if week == 0:
        print("\n  Season hasn't started yet. No per-game props available.")
        print("  Check the Season Props page for preseason futures.")
        return

    # Pull current odds
    print("\n[1/4] Pulling game odds...")
    game_odds = pull_game_odds(markets="h2h,spreads,totals")
    if game_odds.empty:
        print("  No games found. May be bye week or offseason.")
        return

    # Extract game totals for strategy engine
    totals = game_odds[game_odds["market"] == "totals"]
    game_totals = {}
    if not totals.empty:
        for _, row in totals[totals["outcome_name"] == "Over"].iterrows():
            game_totals[row["game_id"]] = row.get("outcome_point", 45.0)

    # Extract spreads for big dog/fav identification
    spreads = game_odds[game_odds["market"] == "spreads"]
    game_spreads = {}
    if not spreads.empty:
        for _, row in spreads.iterrows():
            game_spreads.setdefault(row["game_id"], {})[row["outcome_name"]] = row.get("outcome_point", 0)

    print(f"  {game_odds['game_id'].nunique()} games this week")

    # Pull props
    print("\n[2/4] Pulling player props...")
    props = pull_all_props_for_week()
    if props.empty:
        print("  No player props available yet. Try again closer to gameday.")
        return

    print(f"  {len(props)} prop lines pulled")

    # Market mapping for strategy engine
    stat_col_map = {
        "player_pass_yds": "passing_yards",
        "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards",
        "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards",
    }

    # Evaluate each prop
    print("\n[3/4] Running strategy engine on each prop...")
    bet_card = []

    # Only evaluate 'Over' lines (mirrors of Under)
    over_props = props[props["outcome_name"] == "Over"].copy()

    for _, prop in over_props.iterrows():
        player_name = prop.get("player_name", "")
        market = prop.get("market", "")
        line = prop.get("outcome_point", 0)
        price = prop.get("outcome_price", -110)
        event_id = prop.get("event_id", "")
        home_team = prop.get("home_team", "")
        away_team = prop.get("away_team", "")

        if not player_name or not market or not line:
            continue

        # Get player's rolling average
        stat_col = stat_col_map.get(market, "")
        rolling_avg = get_player_rolling_avg(stats, player_name, stat_col) if stat_col else 0

        # Determine game context
        game_total = game_totals.get(event_id, 45.0)
        home_spread = game_spreads.get(event_id, {}).get(home_team, 0)

        # Determine if home or away team
        is_home = True  # simplified; would need roster lookup for precision

        # Game context flags
        is_dome = home_team in context["dome_teams"] if home_team else False
        home_div = context["team_to_div"].get(home_team, "")
        away_div = context["team_to_div"].get(away_team, "")
        is_division = home_div == away_div and home_div != ""

        # Spread-based flags
        is_big_underdog = abs(home_spread) >= 7 if home_spread else False
        is_big_favorite = abs(home_spread) >= 7 if home_spread else False

        # Run strategy engine
        result = evaluate_prop(
            player_name=player_name,
            market=market,
            fanduel_line=line,
            fanduel_price=price,
            player_rolling_avg=rolling_avg,
            week=week,
            home_team=home_team,
            away_team=away_team,
            is_dome=is_dome,
            is_division=is_division,
            game_total=game_total,
            is_big_underdog=is_big_underdog,
            is_big_favorite=is_big_favorite,
            player_position="",  # would need roster data
            season=2026,
        )

        if result["action"] != "no_bet":
            bet_card.append({
                "player": player_name,
                "market": market,
                "line": line,
                "price": price,
                "direction": result["direction"],
                "confidence": result["confidence"],
                "confidence_tier": result["confidence_tier"],
                "units": result["units"],
                "strategy": result["strategy"],
                "reasoning": result["reasoning"],
                "strategies_triggered": result["strategies_triggered"],
                "home_team": home_team,
                "away_team": away_team,
                "rolling_avg": rolling_avg,
            })

    # Sort by confidence
    bet_card_df = pd.DataFrame(bet_card)
    if bet_card_df.empty:
        print("  No actionable bets found this week.")
        return

    bet_card_df = bet_card_df.sort_values("confidence", ascending=False)

    # Save bet card
    print(f"\n[4/4] Saving bet card ({len(bet_card_df)} plays)...")

    # Save as parquet for dashboard
    bet_card_df.to_parquet(PROC_DIR / "bet_card_latest.parquet", index=False)

    # Save timestamped copy
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    bet_card_df.to_parquet(PROC_DIR / f"bet_card_week{week}_{ts}.parquet", index=False)

    # Also save as readable YAML for quick review
    top_plays = bet_card_df.head(15)
    card_yaml = {
        "week": week,
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_plays": len(bet_card_df),
        "top_plays": [],
    }
    for _, row in top_plays.iterrows():
        card_yaml["top_plays"].append({
            "player": row["player"],
            "market": row["market"],
            "line": float(row["line"]),
            "direction": row["direction"],
            "units": float(row["units"]),
            "confidence_tier": row["confidence_tier"],
            "strategy": row["strategy"],
            "reasoning": row["reasoning"],
        })

    yaml_path = PROC_DIR / "bet_card_latest.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(card_yaml, f, default_flow_style=False, allow_unicode=True)

    # Print summary
    print(f"\n{'='*70}")
    print(f"WEEK {week} BET CARD — {len(bet_card_df)} total plays")
    print(f"{'='*70}")
    print(f"\n{'Tier':<6} {'Dir':<6} {'Player':<22} {'Market':<20} {'Line':>6} {'Units':>5} {'Strategy'}")
    print("-" * 90)
    for _, row in top_plays.iterrows():
        print(f"{row['confidence_tier']:<6} {row['direction']:<6} {row['player']:<22} "
              f"{row['market']:<20} {row['line']:>6.1f} {row['units']:>5.1f} {row['strategy']}")

    print(f"\nSaved to: data/processed/bet_card_latest.parquet")
    print(f"Also: data/processed/bet_card_latest.yaml")

    return bet_card_df


if __name__ == "__main__":
    generate_bet_card()
