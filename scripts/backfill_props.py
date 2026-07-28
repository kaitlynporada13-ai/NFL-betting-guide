"""
Historical Player Prop Backfill from The Odds API.
Pulls closing FanDuel prop lines for all NFL games 2023-2025.
Self-grades against nflverse actual stats.

Credit cost: 10 per region per market per event
Strategy: Pull 1 snapshot per game (close to kickoff) for each prop market.
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

API_KEY = "1c2e7d0377ac3dd72171dc52a8382260"
BASE_URL = "https://api.the-odds-api.com/v4"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
]

# Map nflverse abbreviations to full team names used by The Odds API
TEAM_ABBR_TO_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def get_historical_events(date: str) -> list[dict]:
    """Get NFL events available at a historical timestamp."""
    url = f"{BASE_URL}/historical/sports/americanfootball_nfl/events"
    params = {"apiKey": API_KEY, "date": date}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  [events] Credits remaining: {remaining}")
    return data.get("data", [])


def get_historical_event_props(event_id: str, date: str, markets: list[str]) -> dict:
    """
    Get historical prop odds for a specific event.
    Cost: 10 credits per region per market.
    """
    url = f"{BASE_URL}/historical/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "bookmakers": "fanduel",
        "date": date,
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 404:
        return {}
    if r.status_code == 422:
        # Event may not have prop data available
        return {}
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    return r.json()


def parse_prop_response(response: dict, season: int, week: int) -> list[dict]:
    """Parse the historical odds response into flat rows."""
    rows = []
    event_data = response.get("data", {})
    if not event_data:
        return rows

    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")
    commence_time = event_data.get("commence_time", "")
    event_id = event_data.get("id", "")
    timestamp = response.get("timestamp", "")

    for bookmaker in event_data.get("bookmakers", []):
        if bookmaker.get("key") != "fanduel":
            continue
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                row = {
                    "season": season,
                    "week": week,
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence_time,
                    "snapshot_time": timestamp,
                    "market": market_key,
                    "player_name": outcome.get("description", ""),
                    "outcome": outcome.get("name", ""),  # Over/Under
                    "line": outcome.get("point"),
                    "price": outcome.get("price"),
                }
                rows.append(row)

    return rows


def backfill_season(season: int, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull historical props for all games in a season.
    Uses game schedule to find event dates, then queries ~2 hours before kickoff
    to get near-closing lines.
    """
    season_games = games_df[games_df["season"] == season].copy()
    season_games = season_games[season_games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
    
    print(f"\n{'='*60}")
    print(f"BACKFILLING: {season} NFL Season ({len(season_games)} games)")
    print(f"{'='*60}")

    # Check for cached data
    cache_file = DATA_DIR / f"historical_props_{season}.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        print(f"  Found cached data: {len(cached)} rows, {cached['event_id'].nunique()} games")
        return cached

    all_rows = []
    games_processed = 0
    games_skipped = 0

    # Group games by gameday to reduce API calls
    season_games["gameday_str"] = season_games["gameday"].astype(str)
    grouped = season_games.groupby("gameday_str")

    events_cache = {}  # Cache events by date to avoid duplicate calls

    for gameday, day_games in grouped:
        if not gameday or gameday == "None" or gameday == "NaT":
            continue

        # Get events for this day (query at noon)
        query_dt = f"{gameday}T18:00:00Z"

        if query_dt not in events_cache:
            try:
                events = get_historical_events(query_dt)
                events_cache[query_dt] = events
                time.sleep(0.3)
            except Exception as e:
                print(f"  ERROR getting events for {gameday}: {e}")
                continue
        else:
            events = events_cache[query_dt]

        # Process each game on this day
        for _, game in day_games.iterrows():
            week = game.get("week", 0)
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            gametime = game.get("gametime", "13:00")

            # Match to our game using full team names
            home_full = TEAM_ABBR_TO_FULL.get(home, home)
            away_full = TEAM_ABBR_TO_FULL.get(away, away)

            matched_event = None
            for event in events:
                e_home = event.get("home_team", "")
                e_away = event.get("away_team", "")
                if e_home == home_full and e_away == away_full:
                    matched_event = event
                    break

            if not matched_event:
                games_skipped += 1
                continue

            event_id = matched_event["id"]

            # Query ~2 hours before kickoff for near-closing lines
            try:
                hour = int(str(gametime).split(":")[0]) if pd.notna(gametime) else 13
                prop_dt = f"{gameday}T{max(hour-2, 10):02d}:00:00Z"
            except:
                prop_dt = f"{gameday}T15:00:00Z"

            # Pull props for this event
            try:
                response = get_historical_event_props(event_id, prop_dt, PROP_MARKETS)
                time.sleep(0.3)
            except Exception as e:
                print(f"  ERROR getting props for {away}@{home} Wk{week}: {e}")
                continue

            if response:
                rows = parse_prop_response(response, season, week)
                all_rows.extend(rows)
                games_processed += 1

                if games_processed % 10 == 0:
                    print(f"  Processed {games_processed} games... ({len(all_rows)} props)")
            else:
                games_skipped += 1

    # Save
    if all_rows:
        combined = pd.DataFrame(all_rows)
        combined.to_parquet(cache_file, index=False)
        print(f"\n  Season {season}: {len(combined)} prop lines, {combined['event_id'].nunique()} games")
        print(f"  Processed: {games_processed}, Skipped: {games_skipped}")
        return combined

    print(f"\n  Season {season}: No data retrieved (processed={games_processed}, skipped={games_skipped})")
    return pd.DataFrame()


def run_full_backfill():
    """Run the complete historical prop backfill for 2023-2025."""
    print("\n" + "=" * 70)
    print("NFL HISTORICAL PLAYER PROP BACKFILL")
    print("Source: The Odds API | Book: FanDuel")
    print("=" * 70)

    # Load game schedule
    games_path = DATA_DIR / "games_all.parquet"
    if not games_path.exists():
        games_path = DATA_DIR / "games_historical.parquet"
    games = pd.read_parquet(games_path)

    # Only pull from May 2023 onward (when prop data became available)
    seasons = [2023, 2024, 2025]
    all_seasons = []

    for season in seasons:
        season_df = backfill_season(season, games)
        if not season_df.empty:
            all_seasons.append(season_df)

    # Combine all
    if all_seasons:
        master = pd.concat(all_seasons, ignore_index=True)
        master.to_parquet(DATA_DIR / "historical_props_all.parquet", index=False)
        print(f"\n{'='*70}")
        print(f"COMPLETE: {len(master)} total prop lines across {master['season'].nunique()} seasons")
        print(f"Games covered: {master['event_id'].nunique()}")
        print(f"Markets: {master['market'].nunique()}")
        print(f"Players: {master['player_name'].nunique()}")
        print(f"Credits remaining: check API response headers")
        print(f"{'='*70}")
    else:
        print("\nNo data retrieved. Check API key and credits.")


if __name__ == "__main__":
    run_full_backfill()
