"""
Historical Prop Line Ingestion via PropLine API.
Pulls graded (resolved) player prop lines from FanDuel for backtesting.

API: https://prop-line.com
- Free tier: 1,000 requests/day (enough to backfill over ~1 week)
- Hobby $9/mo: 5,000 req/day + prop resolution + historical movement
- Pro $19/mo: 25,000 req/day + 90-day archive + bulk CSV export

Prop resolution = each prop graded as won/lost/push against actual stats.
This is gold for backtesting CLV (closing line value).

Setup:
1. Sign up at https://prop-line.com (free, no credit card)
2. Add PROPLINE_API_KEY to config/.env
3. Run: python -m pipeline.ingest_prop_history
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

from pipeline.config_loader import load_settings, get_data_dir


BASE_URL = "https://api.prop-line.com/v1"

# NFL prop market keys (PropLine uses the-odds-api compatible format)
NFL_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
    "player_pass_interceptions",
]


def _get_api_key() -> str:
    """Get PropLine API key from environment."""
    import os
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(env_path)

    key = os.environ.get("PROPLINE_API_KEY", "")
    if not key:
        raise ValueError(
            "PROPLINE_API_KEY not set. Sign up free at https://prop-line.com "
            "and add the key to config/.env"
        )
    return key


def get_nfl_events(date: str = None) -> list[dict]:
    """
    Get NFL events (games) for a given date or upcoming.
    
    Args:
        date: Optional date string (YYYY-MM-DD) for historical lookup
    """
    api_key = _get_api_key()
    url = f"{BASE_URL}/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key}
    if date:
        params["date"] = date

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def get_event_props(event_id: str, markets: list[str] = None, bookmakers: str = "fanduel") -> dict:
    """
    Get player prop odds for a specific game.
    
    Args:
        event_id: Game identifier from get_nfl_events()
        markets: List of prop markets to pull
        bookmakers: Which sportsbook (default: fanduel)
    """
    api_key = _get_api_key()
    if markets is None:
        markets = NFL_PROP_MARKETS

    url = f"{BASE_URL}/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "bookmakers": bookmakers,
    }

    response = requests.get(url, params=params, timeout=15)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    return response.json()


def get_resolved_props(event_id: str) -> list[dict]:
    """
    Get resolved (graded) props for a completed game.
    Each prop shows: player, market, line, outcome (won/lost/push), actual stat.
    
    This is the key backtesting data — tells you what the line was AND if it hit.
    Requires Hobby ($9/mo) or higher plan.
    """
    api_key = _get_api_key()
    url = f"{BASE_URL}/sports/americanfootball_nfl/events/{event_id}/resolution"
    params = {"apiKey": api_key}

    response = requests.get(url, params=params, timeout=15)
    if response.status_code == 404:
        return []
    if response.status_code == 403:
        print("  [propline] Resolution endpoint requires Hobby plan ($9/mo)")
        return []
    response.raise_for_status()
    return response.json()


def pull_historical_props_for_week(season: int, week: int) -> pd.DataFrame:
    """
    Pull all FanDuel prop lines and resolutions for a specific NFL week.
    
    Strategy:
    1. Get all games for that week
    2. For each game, pull prop odds (closing lines) 
    3. If available, pull resolution (graded results)
    """
    # NFL weeks typically fall on specific date ranges
    # This is an approximation — actual dates vary by season
    # Better to pull events by date range or use schedule data

    print(f"  [propline] Pulling props for {season} Week {week}...")
    
    # We'll use the schedule to find game dates
    raw_dir = get_data_dir("raw")
    games_path = raw_dir / "games_historical.parquet"

    if not games_path.exists():
        print("  ERROR: games_historical.parquet not found. Run ingest_stats first.")
        return pd.DataFrame()

    games = pd.read_parquet(games_path)
    week_games = games[(games["season"] == season) & (games["week"] == week)]

    if week_games.empty:
        print(f"  No games found for {season} Week {week}")
        return pd.DataFrame()

    all_props = []

    for _, game in week_games.iterrows():
        game_date = str(game.get("gameday", ""))
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # Get events for that date
        try:
            events = get_nfl_events(date=game_date)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"    Error getting events for {game_date}: {e}")
            continue

        # Find matching event
        matched_event = None
        for event in events:
            if (event.get("home_team", "").endswith(home) or
                home in event.get("home_team", "")):
                matched_event = event
                break

        if not matched_event:
            continue

        event_id = matched_event["id"]

        # Pull props
        try:
            props_data = get_event_props(event_id)
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error getting props for {away}@{home}: {e}")
            continue

        # Parse props into rows
        for bookmaker in props_data.get("bookmakers", []):
            if bookmaker.get("key") != "fanduel":
                continue
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    row = {
                        "season": season,
                        "week": week,
                        "game_date": game_date,
                        "home_team": home,
                        "away_team": away,
                        "event_id": event_id,
                        "market": market["key"],
                        "player_name": outcome.get("description", outcome.get("name", "")),
                        "outcome": outcome.get("name"),  # Over/Under
                        "line": outcome.get("point"),
                        "price": outcome.get("price"),
                        "last_update": bookmaker.get("last_update"),
                    }
                    all_props.append(row)

        # Try to get resolution (graded results)
        try:
            resolutions = get_resolved_props(event_id)
            time.sleep(0.5)
            for res in resolutions:
                # Merge resolution with prop data
                for prop in all_props:
                    if (prop["event_id"] == event_id and
                        prop["player_name"] == res.get("player_name") and
                        prop["market"] == res.get("market")):
                        prop["actual_stat"] = res.get("actual")
                        prop["result"] = res.get("result")  # won/lost/push
        except Exception as e:
            pass  # Resolution may not be available on free tier

    df = pd.DataFrame(all_props)
    if not df.empty:
        print(f"    Got {len(df)} prop lines for Week {week}")
    return df


def pull_season_props(season: int, start_week: int = 1, end_week: int = 18) -> pd.DataFrame:
    """
    Pull all prop lines for an entire NFL season.
    
    On free tier (1,000 req/day), this will take multiple days.
    On Pro ($19/mo, 25,000 req/day), entire season in one session.
    
    Args:
        season: NFL season year
        start_week: First week to pull (default: 1)
        end_week: Last week to pull (default: 18)
    """
    print(f"\n{'='*60}")
    print(f"PULLING HISTORICAL PROPS: {season} Season (Weeks {start_week}-{end_week})")
    print(f"{'='*60}")

    raw_dir = get_data_dir("raw")
    all_weeks = []

    for week in range(start_week, end_week + 1):
        # Check if we already have this week
        week_file = raw_dir / f"props_{season}_week{week:02d}.parquet"
        if week_file.exists():
            print(f"  Week {week}: already cached, skipping")
            existing = pd.read_parquet(week_file)
            all_weeks.append(existing)
            continue

        week_data = pull_historical_props_for_week(season, week)
        if not week_data.empty:
            week_data.to_parquet(week_file, index=False)
            all_weeks.append(week_data)

        # Small delay between weeks
        time.sleep(1)

    if all_weeks:
        combined = pd.concat(all_weeks, ignore_index=True)
        combined.to_parquet(raw_dir / f"props_historical_{season}.parquet", index=False)
        print(f"\nSaved: {len(combined)} total prop lines for {season}")
        return combined

    return pd.DataFrame()


def pull_all_historical_props():
    """
    Pull props for all available seasons (2023-2025).
    PropLine has data from ~April 2023 onward.
    
    Free tier strategy: ~1,000 requests/day
    - Each week uses ~20-40 requests (1 events call + 1 odds per game + 1 resolution per game)
    - NFL has 16 games/week = ~48 requests per week
    - 18 weeks per season = ~864 requests per season
    - So 1 full season per day on free tier!
    """
    raw_dir = get_data_dir("raw")

    print("\n" + "=" * 60)
    print("HISTORICAL PROP LINE BACKFILL")
    print("Free tier: ~1 season/day | Pro: all 3 seasons in one run")
    print("=" * 60)

    seasons_to_pull = [2023, 2024, 2025]
    
    for season in seasons_to_pull:
        # Check if already done
        season_file = raw_dir / f"props_historical_{season}.parquet"
        if season_file.exists():
            existing = pd.read_parquet(season_file)
            print(f"\n{season}: Already have {len(existing)} prop lines cached")
            continue

        pull_season_props(season)

    # Combine all seasons into one master file
    all_seasons = []
    for season in seasons_to_pull:
        season_file = raw_dir / f"props_historical_{season}.parquet"
        if season_file.exists():
            all_seasons.append(pd.read_parquet(season_file))

    if all_seasons:
        master = pd.concat(all_seasons, ignore_index=True)
        master.to_parquet(raw_dir / "props_historical_all.parquet", index=False)
        print(f"\n{'='*60}")
        print(f"MASTER FILE: {len(master)} total prop lines across {len(all_seasons)} seasons")
        print(f"{'='*60}")


def build_prop_backtest_features(props_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build backtesting features from historical prop data.
    Self-grades props by matching lines against actual nflverse player stats.
    No paid tier needed — we have the actual stats already.
    
    Key metrics:
    - Hit rate by market type
    - Hit rate by line size (easier vs harder props)
    - CLV potential (where our model would have had edge)
    """
    if props_df is None:
        raw_dir = get_data_dir("raw")
        props_path = raw_dir / "props_historical_all.parquet"
        if not props_path.exists():
            print("No historical props available. Run pull_all_historical_props() first.")
            return pd.DataFrame()
        props_df = pd.read_parquet(props_path)

    print("[backtest] Self-grading props against actual stats...")

    # Load actual player stats
    raw_dir = get_data_dir("raw")
    stats_path = raw_dir / "player_stats_historical.parquet"
    if not stats_path.exists():
        print("  ERROR: player_stats_historical.parquet not found")
        return pd.DataFrame()

    stats = pd.read_parquet(stats_path)

    # Map PropLine market keys to nflverse stat columns
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards",
        "player_rush_attempts": "carries",
        "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards",
        "player_pass_interceptions": "interceptions",
    }

    df = props_df.copy()

    # Only grade "Over" lines (Over/Under are mirror — grading one is enough)
    overs = df[df["outcome"] == "Over"].copy()

    if overs.empty:
        print("  No 'Over' props found to grade")
        return pd.DataFrame()

    # Match props to actual stats by player name + season + week
    # Normalize player names for matching
    overs["player_name_clean"] = overs["player_name"].str.strip().str.lower()

    if "player_display_name" in stats.columns:
        stats["player_name_clean"] = stats["player_display_name"].str.strip().str.lower()
    elif "player_name" in stats.columns:
        stats["player_name_clean"] = stats["player_name"].str.strip().str.lower()
    else:
        print("  ERROR: No player name column in stats")
        return pd.DataFrame()

    # Merge
    graded = overs.merge(
        stats[["player_name_clean", "season", "week"] + 
              [c for c in market_to_stat.values() if c in stats.columns]],
        on=["player_name_clean", "season", "week"],
        how="left",
    )

    # Grade each prop
    results = []
    for _, row in graded.iterrows():
        market = row.get("market")
        line = row.get("line")
        stat_col = market_to_stat.get(market)

        if stat_col is None or pd.isna(line):
            continue

        actual = row.get(stat_col)
        if pd.isna(actual):
            continue

        # Grade: Over wins if actual > line, loses if actual < line, push if equal
        if actual > line:
            result = "won"
        elif actual < line:
            result = "lost"
        else:
            result = "push"

        results.append({
            "season": row["season"],
            "week": row["week"],
            "player_name": row["player_name"],
            "market": market,
            "line": line,
            "price": row.get("price"),
            "actual_stat": actual,
            "result": result,
            "edge": actual - line,  # positive = over hit
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
        })

    graded_df = pd.DataFrame(results)

    if graded_df.empty:
        print("  No props could be matched to actual stats")
        return pd.DataFrame()

    # --- Summary stats ---
    print(f"\n  SELF-GRADED RESULTS:")
    print(f"  Total props graded: {len(graded_df)}")
    print(f"  Won: {(graded_df['result'] == 'won').sum()}")
    print(f"  Lost: {(graded_df['result'] == 'lost').sum()}")
    print(f"  Push: {(graded_df['result'] == 'push').sum()}")
    print(f"  Overall Over hit rate: {(graded_df['result'] == 'won').mean():.1%}")

    # By market
    print(f"\n  Hit rate by market:")
    market_stats = graded_df.groupby("market").agg(
        total=("result", "count"),
        wins=("result", lambda x: (x == "won").sum()),
        avg_edge=("edge", "mean"),
    ).reset_index()
    market_stats["hit_rate"] = market_stats["wins"] / market_stats["total"]

    for _, row in market_stats.iterrows():
        print(f"    {row['market']}: {row['hit_rate']:.1%} over hit rate "
              f"({row['wins']}/{row['total']}) | avg edge: {row['avg_edge']:+.1f}")

    # Save graded data
    processed_dir = get_data_dir("processed")
    graded_df.to_parquet(processed_dir / "props_graded.parquet", index=False)
    print(f"\n  Saved graded props to data/processed/props_graded.parquet")

    return graded_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        build_prop_backtest_features()
    else:
        pull_all_historical_props()
