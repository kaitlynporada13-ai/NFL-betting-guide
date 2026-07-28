"""
Weather Ingestion via Visual Crossing API.
Pulls game-day weather forecasts for outdoor stadiums.

API docs: https://www.visualcrossing.com/resources/documentation/weather-api/
Free tier: 1000 requests/day
"""

import requests
import pandas as pd
from datetime import datetime

from pipeline.config_loader import load_settings, load_stadiums, get_data_dir


BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


def _get_api_key() -> str:
    """Get the Visual Crossing API key from settings."""
    settings = load_settings()
    key = settings["api_keys"]["weather_api"]
    if not key or key == "your_weather_api_key_here":
        raise ValueError(
            "WEATHER_API_KEY not set. Add it to config/.env file. "
            "Get a free key at https://www.visualcrossing.com/"
        )
    return key


def _needs_weather(stadium_info: dict) -> bool:
    """Determine if a stadium needs weather data (open-air only)."""
    roof = stadium_info.get("roof", "open")
    return roof == "open"


def get_game_weather(
    latitude: float,
    longitude: float,
    game_date: str,
    game_hour: int = 13,
) -> dict:
    """
    Get weather forecast for a specific location and date.
    
    Args:
        latitude: Stadium latitude
        longitude: Stadium longitude
        game_date: Date string in YYYY-MM-DD format
        game_hour: Hour of kickoff (24h format) to get closest forecast
    
    Returns:
        Dict with temp, wind, precip probability, conditions
    """
    api_key = _get_api_key()

    location = f"{latitude},{longitude}"
    url = f"{BASE_URL}/{location}/{game_date}/{game_date}"

    params = {
        "key": api_key,
        "unitGroup": "us",  # Fahrenheit, mph
        "include": "hours",
        "elements": "datetime,temp,feelslike,humidity,precip,precipprob,snow,"
                    "windspeed,windgust,winddir,cloudcover,conditions,description",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Find the hour closest to game time
    hours = data.get("days", [{}])[0].get("hours", [])
    game_hour_data = None
    for hour in hours:
        hour_num = int(hour["datetime"].split(":")[0])
        if hour_num == game_hour:
            game_hour_data = hour
            break

    if game_hour_data is None and hours:
        # Fall back to the day summary
        day_data = data["days"][0]
        return {
            "temp": day_data.get("temp"),
            "feels_like": day_data.get("feelslike"),
            "wind_speed": day_data.get("windspeed"),
            "wind_gust": day_data.get("windgust"),
            "wind_dir": day_data.get("winddir"),
            "precip_prob": day_data.get("precipprob"),
            "precip_inches": day_data.get("precip"),
            "snow_inches": day_data.get("snow"),
            "humidity": day_data.get("humidity"),
            "cloud_cover": day_data.get("cloudcover"),
            "conditions": day_data.get("conditions"),
        }

    return {
        "temp": game_hour_data.get("temp"),
        "feels_like": game_hour_data.get("feelslike"),
        "wind_speed": game_hour_data.get("windspeed"),
        "wind_gust": game_hour_data.get("windgust"),
        "wind_dir": game_hour_data.get("winddir"),
        "precip_prob": game_hour_data.get("precipprob"),
        "precip_inches": game_hour_data.get("precip"),
        "snow_inches": game_hour_data.get("snow"),
        "humidity": game_hour_data.get("humidity"),
        "cloud_cover": game_hour_data.get("cloudcover"),
        "conditions": game_hour_data.get("conditions"),
    }


def get_weather_for_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get weather forecasts for a set of games.
    Only fetches for open-air stadiums.
    
    Args:
        games_df: DataFrame with columns: game_id, home_team, gameday, gametime
        
    Returns:
        DataFrame with weather data merged in
    """
    stadiums_data = load_stadiums()
    stadiums = stadiums_data["stadiums"]
    team_map = stadiums_data["team_stadium_map"]

    weather_rows = []

    for _, game in games_df.iterrows():
        home_team = game.get("home_team")
        game_date = game.get("gameday")
        game_time = game.get("gametime", "13:00")

        # Look up stadium
        stadium_key = team_map.get(home_team)
        if not stadium_key:
            print(f"  [weather] No stadium found for team: {home_team}")
            weather_rows.append({"game_id": game.get("game_id")})
            continue

        stadium_info = stadiums.get(stadium_key, {})

        # Skip domed stadiums
        if not _needs_weather(stadium_info):
            weather_rows.append({
                "game_id": game.get("game_id"),
                "weather_relevant": False,
                "dome": True,
                "temp": 72,  # assume dome temp
                "wind_speed": 0,
                "precip_prob": 0,
            })
            continue

        # Parse game hour
        try:
            game_hour = int(str(game_time).split(":")[0])
        except (ValueError, AttributeError):
            game_hour = 13

        # Get weather
        try:
            coords = stadium_info.get("coordinates", [0, 0])
            weather = get_game_weather(coords[0], coords[1], str(game_date), game_hour)
            weather["game_id"] = game.get("game_id")
            weather["weather_relevant"] = True
            weather["dome"] = False
            weather_rows.append(weather)
        except Exception as e:
            print(f"  [weather] Error getting weather for {home_team}: {e}")
            weather_rows.append({"game_id": game.get("game_id")})

    return pd.DataFrame(weather_rows)


def categorize_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical weather impact flags for model features.
    
    Categories:
        - wind_impact: wind >= 15 mph (suppresses passing)
        - heavy_wind: wind >= 20 mph (significant suppression)
        - rain_likely: precip prob >= 50%
        - cold_game: temp <= 32F
        - extreme_cold: temp <= 20F
        - snow_game: snow > 0
        - perfect_conditions: 50-80F, wind < 10, no precip
    """
    df = weather_df.copy()

    df["wind_impact"] = df.get("wind_speed", 0).fillna(0) >= 15
    df["heavy_wind"] = df.get("wind_speed", 0).fillna(0) >= 20
    df["rain_likely"] = df.get("precip_prob", 0).fillna(0) >= 50
    df["cold_game"] = df.get("temp", 72).fillna(72) <= 32
    df["extreme_cold"] = df.get("temp", 72).fillna(72) <= 20
    df["snow_game"] = df.get("snow_inches", 0).fillna(0) > 0

    # Perfect conditions: mild temp, low wind, dry
    df["perfect_conditions"] = (
        (df.get("temp", 72).fillna(72).between(50, 80))
        & (df.get("wind_speed", 0).fillna(0) < 10)
        & (df.get("precip_prob", 0).fillna(0) < 20)
    )

    return df


if __name__ == "__main__":
    # Example usage - would normally be called by the pipeline
    print("Weather ingestion module loaded.")
    print("Use get_weather_for_games(games_df) with a games DataFrame.")
    print("Requires WEATHER_API_KEY in config/.env")
