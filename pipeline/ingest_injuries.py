"""
Injury Report Ingestion via ESPN API (unofficial, no key required).
Pulls current injury reports, practice participation, and game status.

Also pulls from nflverse injury data for historical injury context.
"""

import requests
import pandas as pd
from datetime import datetime

from pipeline.config_loader import get_data_dir


ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

# NFL team ID mapping for ESPN API
ESPN_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3,
    "CIN": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GB": 9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC": 12, "LAC": 24, "LAR": 14,
    "LV": 13, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26, "SF": 25, "TB": 27,
    "TEN": 10, "WAS": 28,
}


def pull_team_injuries(team_abbr: str) -> pd.DataFrame:
    """
    Pull current injury report for a specific team from ESPN.
    
    Returns DataFrame with player name, position, injury type, status, practice status.
    """
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        print(f"  [injuries] Unknown team: {team_abbr}")
        return pd.DataFrame()

    url = f"{ESPN_BASE_URL}/teams/{team_id}/injuries"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [injuries] Error fetching {team_abbr}: {e}")
        return pd.DataFrame()

    rows = []
    for category in data.get("injuries", []):
        for item in category.get("injuries", []):
            athlete = item.get("athlete", {})
            row = {
                "team": team_abbr,
                "player_name": athlete.get("displayName", ""),
                "player_id_espn": athlete.get("id"),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "injury_type": item.get("type", {}).get("description", ""),
                "injury_detail": item.get("details", {}).get("detail", ""),
                "status": item.get("status", ""),
                "game_status": item.get("details", {}).get("fantasyStatus", {}).get("description", ""),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def pull_all_injuries() -> pd.DataFrame:
    """Pull injury reports for all 32 teams."""
    print("[injuries] Pulling injury reports for all teams...")
    all_injuries = []

    for team_abbr in ESPN_TEAM_IDS:
        injuries = pull_team_injuries(team_abbr)
        if not injuries.empty:
            all_injuries.append(injuries)

    if all_injuries:
        combined = pd.concat(all_injuries, ignore_index=True)
        combined["pulled_at"] = datetime.utcnow()
        print(f"  Total injuries found: {len(combined)} across {combined['team'].nunique()} teams")
        return combined

    print("  No injuries found (possibly offseason)")
    return pd.DataFrame()


def pull_nflverse_injuries(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull historical injury data from nflverse.
    Useful for understanding how injuries historically impacted performance.
    """
    import nfl_data_py as nfl

    if seasons is None:
        from pipeline.config_loader import load_settings
        settings = load_settings()
        seasons = settings["data"]["historical_seasons"]

    print(f"[injuries] Pulling nflverse injury data for {seasons}...")
    try:
        injuries = nfl.import_injuries(seasons)
        print(f"  Historical injuries: {len(injuries)} records")
        return injuries
    except Exception as e:
        print(f"  Could not pull nflverse injuries: {e}")
        return pd.DataFrame()


def get_injury_impact_features(injuries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert injury report into model-ready features.
    
    Key signals:
    - Starting QB questionable/out → huge spread/total impact
    - Multiple O-line injuries → rushing suppression
    - Key WR out → target redistribution
    - Key pass rusher out → opposing QB boost
    """
    if injuries_df.empty:
        return pd.DataFrame()

    df = injuries_df.copy()

    # Categorize by impact level
    out_statuses = ["Out", "Injured Reserve", "Physically Unable to Perform", "Doubtful"]
    questionable_statuses = ["Questionable", "Day-To-Day"]

    df["is_out"] = df["status"].isin(out_statuses) | df["game_status"].str.contains("Out", case=False, na=False)
    df["is_questionable"] = df["status"].isin(questionable_statuses)

    # Key position flags
    df["is_qb"] = df["position"] == "QB"
    df["is_oline"] = df["position"].isin(["OT", "OG", "C", "OL", "T", "G"])
    df["is_skill"] = df["position"].isin(["WR", "RB", "TE"])
    df["is_pass_rusher"] = df["position"].isin(["DE", "OLB", "EDGE", "DT"])
    df["is_db"] = df["position"].isin(["CB", "S", "FS", "SS"])

    # Team-level aggregations
    team_impact = df.groupby("team").agg(
        total_injuries=("player_name", "count"),
        players_out=("is_out", "sum"),
        players_questionable=("is_questionable", "sum"),
        qb_injured=("is_qb", lambda x: (x & df.loc[x.index, "is_out"]).any()),
        oline_injuries=("is_oline", lambda x: (x & df.loc[x.index, "is_out"]).sum()),
        skill_players_out=("is_skill", lambda x: (x & df.loc[x.index, "is_out"]).sum()),
        pass_rushers_out=("is_pass_rusher", lambda x: (x & df.loc[x.index, "is_out"]).sum()),
        dbs_out=("is_db", lambda x: (x & df.loc[x.index, "is_out"]).sum()),
    ).reset_index()

    # Derived impact scores (simple heuristics)
    team_impact["offensive_impact"] = (
        team_impact["qb_injured"].astype(int) * 10
        + team_impact["oline_injuries"] * 3
        + team_impact["skill_players_out"] * 2
    )
    team_impact["defensive_impact"] = (
        team_impact["pass_rushers_out"] * 3
        + team_impact["dbs_out"] * 2
    )

    return team_impact


def pull_espn_news(team_abbr: str = None, limit: int = 20) -> pd.DataFrame:
    """
    Pull latest NFL news from ESPN.
    Can filter by team or get league-wide news.
    """
    if team_abbr:
        team_id = ESPN_TEAM_IDS.get(team_abbr)
        url = f"{ESPN_BASE_URL}/teams/{team_id}/news"
    else:
        url = f"{ESPN_BASE_URL}/news"

    params = {"limit": limit}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [news] Error: {e}")
        return pd.DataFrame()

    rows = []
    for article in data.get("articles", []):
        row = {
            "headline": article.get("headline", ""),
            "description": article.get("description", ""),
            "published": article.get("published", ""),
            "type": article.get("type", ""),
            "team": team_abbr or "NFL",
            "url": article.get("links", {}).get("web", {}).get("href", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    return df


def pull_all_news() -> pd.DataFrame:
    """Pull league-wide NFL news."""
    print("[news] Pulling latest NFL news from ESPN...")
    news = pull_espn_news(limit=50)
    print(f"  Retrieved {len(news)} articles")
    return news


def save_injury_data():
    """Pull and save current injury reports."""
    raw_dir = get_data_dir("raw")

    print("=" * 60)
    print("PULLING INJURY REPORTS")
    print("=" * 60)

    # Current injuries from ESPN
    injuries = pull_all_injuries()
    if not injuries.empty:
        injuries.to_parquet(raw_dir / "injuries_current.parquet", index=False)
        print(f"  Saved current injuries: {len(injuries)} entries")

        # Build impact features
        impact = get_injury_impact_features(injuries)
        if not impact.empty:
            impact.to_parquet(raw_dir / "injury_impact_features.parquet", index=False)
            print(f"  Saved injury impact features for {len(impact)} teams")

    # News
    news = pull_all_news()
    if not news.empty:
        news.to_parquet(raw_dir / "news_latest.parquet", index=False)
        print(f"  Saved {len(news)} news articles")

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    save_injury_data()
