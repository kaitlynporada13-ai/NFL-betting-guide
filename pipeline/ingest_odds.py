"""
Live Odds Ingestion via The Odds API.
Pulls current FanDuel lines for spreads, totals, moneylines, and player props.

API docs: https://the-odds-api.com/liveapi/guides/v4/
Free tier: 500 requests/month
"""

import requests
import pandas as pd
from datetime import datetime

from pipeline.config_loader import load_settings, get_data_dir


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"


def _get_api_key() -> str:
    """Get the Odds API key from settings."""
    settings = load_settings()
    key = settings["api_keys"]["odds_api"]
    if not key or key == "your_odds_api_key_here":
        raise ValueError(
            "ODDS_API_KEY not set. Add it to config/.env file. "
            "Get a free key at https://the-odds-api.com/"
        )
    return key


def pull_game_odds(markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """
    Pull current game-level odds from FanDuel.
    
    Markets:
        - h2h: moneyline
        - spreads: point spread
        - totals: over/under
    
    Returns DataFrame with one row per game per market per outcome.
    """
    api_key = _get_api_key()
    settings = load_settings()
    bookmaker = settings["sportsbook"]

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "bookmakers": bookmaker,
        "oddsFormat": settings["data"]["odds_format"],
    }

    print(f"[ingest_odds] Pulling game odds from The Odds API (markets: {markets})")
    response = requests.get(f"{BASE_URL}/sports/{SPORT}/odds", params=params)
    response.raise_for_status()

    data = response.json()
    print(f"  API requests remaining: {response.headers.get('x-requests-remaining', 'unknown')}")

    rows = []
    for game in data:
        game_info = {
            "game_id": game["id"],
            "sport": game["sport_key"],
            "commence_time": game["commence_time"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
        }

        for bookmaker_data in game.get("bookmakers", []):
            if bookmaker_data["key"] != bookmaker:
                continue

            for market in bookmaker_data.get("markets", []):
                market_key = market["key"]
                for outcome in market["outcomes"]:
                    row = {
                        **game_info,
                        "bookmaker": bookmaker_data["key"],
                        "market": market_key,
                        "outcome_name": outcome["name"],
                        "outcome_price": outcome["price"],
                        "outcome_point": outcome.get("point"),
                        "last_update": bookmaker_data["last_update"],
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
        df["last_update"] = pd.to_datetime(df["last_update"])
        df["pulled_at"] = datetime.utcnow()

    print(f"  Retrieved odds for {df['game_id'].nunique()} games")
    return df


def pull_player_props(event_id: str, prop_markets: list[str] | None = None) -> pd.DataFrame:
    """
    Pull player prop odds for a specific game.
    
    Prop markets available (varies by game/timing):
        - player_pass_yds
        - player_pass_tds
        - player_rush_yds
        - player_rush_attempts
        - player_receptions
        - player_reception_yds
        - player_anytime_td
        - player_pass_interceptions
    
    Note: Player props endpoint requires a paid plan on The Odds API.
    Free tier only covers game-level markets.
    """
    api_key = _get_api_key()
    settings = load_settings()
    bookmaker = settings["sportsbook"]

    if prop_markets is None:
        prop_markets = [
            "player_pass_yds",
            "player_pass_tds",
            "player_rush_yds",
            "player_rush_attempts",
            "player_receptions",
            "player_reception_yds",
            "player_anytime_td",
        ]

    all_rows = []
    for market in prop_markets:
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": market,
            "bookmakers": bookmaker,
            "oddsFormat": settings["data"]["odds_format"],
        }

        print(f"[ingest_odds] Pulling player props: {market} for game {event_id}")
        response = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
            params=params,
        )

        if response.status_code == 404:
            print(f"  No props available for {market}")
            continue
        response.raise_for_status()

        data = response.json()

        for bookmaker_data in data.get("bookmakers", []):
            if bookmaker_data["key"] != bookmaker:
                continue

            for mkt in bookmaker_data.get("markets", []):
                # last_update can live at market or bookmaker level (varies by endpoint)
                last_update = mkt.get("last_update") or bookmaker_data.get("last_update")
                for outcome in mkt.get("outcomes", []):
                    row = {
                        "event_id": event_id,
                        "home_team": data.get("home_team"),
                        "away_team": data.get("away_team"),
                        "commence_time": data.get("commence_time"),
                        "market": mkt["key"],
                        "player_name": outcome.get("description", outcome.get("name")),
                        "outcome_name": outcome.get("name"),  # Over/Under
                        "outcome_price": outcome.get("price"),
                        "outcome_point": outcome.get("point"),
                        "last_update": last_update,
                    }
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
        df["pulled_at"] = datetime.utcnow()

    print(f"  Retrieved {len(df)} player prop lines")
    return df


def pull_all_props_for_week() -> pd.DataFrame:
    """
    Pull player props for all upcoming games this week.
    First gets the game list, then pulls props for each.
    """
    # Get upcoming games
    game_odds = pull_game_odds(markets="h2h")  # minimal call just to get game IDs
    if game_odds.empty:
        print("[ingest_odds] No upcoming games found")
        return pd.DataFrame()

    game_ids = game_odds["game_id"].unique()
    print(f"[ingest_odds] Pulling player props for {len(game_ids)} games")

    all_props = []
    for game_id in game_ids:
        props = pull_player_props(game_id)
        if not props.empty:
            all_props.append(props)

    if all_props:
        return pd.concat(all_props, ignore_index=True)
    return pd.DataFrame()


def save_current_odds():
    """Pull and save current odds snapshot to data/raw/."""
    raw_dir = get_data_dir("raw")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("PULLING CURRENT FANDUEL ODDS")
    print("=" * 60)

    # Game-level odds
    try:
        game_odds = pull_game_odds()
        if not game_odds.empty:
            filename = f"odds_games_{timestamp}.parquet"
            game_odds.to_parquet(raw_dir / filename, index=False)
            print(f"  Saved game odds to {filename}")

            # Also save a 'latest' copy for easy access
            game_odds.to_parquet(raw_dir / "odds_games_latest.parquet", index=False)
    except ValueError as e:
        print(f"  Skipping game odds: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"  API error pulling game odds: {e}")

    # Player props (will only work with paid API tier)
    try:
        props = pull_all_props_for_week()
        if not props.empty:
            filename = f"odds_props_{timestamp}.parquet"
            props.to_parquet(raw_dir / filename, index=False)
            print(f"  Saved player props to {filename}")
            props.to_parquet(raw_dir / "odds_props_latest.parquet", index=False)
    except ValueError as e:
        print(f"  Skipping player props: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"  API error pulling props: {e}")

    print("=" * 60)
    print("DONE - Odds saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    save_current_odds()
