"""
Team Hierarchy & Coaching Data Ingestion.
Pulls coaching staff, scheme identifiers, and organizational data.
Used for tracking scheme changes, coordinator impacts, and "like player" analysis.

Sources:
- nflverse (coaching data, draft picks, combine)
- ESPN API (depth charts, roster transactions)
"""

import requests
import pandas as pd
import nfl_data_py as nfl

from pipeline.config_loader import load_settings, get_data_dir


ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

ESPN_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3,
    "CIN": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GB": 9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC": 12, "LAC": 24, "LAR": 14,
    "LV": 13, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26, "SF": 25, "TB": 27,
    "TEN": 10, "WAS": 28,
}


def pull_depth_charts(season: int | None = None) -> pd.DataFrame:
    """
    Pull depth charts from nflverse.
    Shows starter/backup designations for every position.
    """
    settings = load_settings()
    if season is None:
        season = settings["data"]["current_season"]

    print(f"[hierarchy] Pulling depth charts for {season}...")
    try:
        depth = nfl.import_depth_charts([season])
        print(f"  Depth chart entries: {len(depth)}")
        return depth
    except Exception as e:
        print(f"  Could not pull depth charts: {e}")
        return pd.DataFrame()


def pull_draft_data(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull draft history — useful for "pedigree" features.
    High draft picks tend to get more opportunities.
    """
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[hierarchy] Pulling draft data for {seasons}...")
    try:
        draft = nfl.import_draft_picks(seasons)
        print(f"  Draft picks: {len(draft)}")
        return draft
    except Exception as e:
        print(f"  Could not pull draft data: {e}")
        return pd.DataFrame()


def pull_combine_data(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Pull NFL combine data — player athleticism profiles.
    Used for "like player" comparisons (similar body type + athletic profile).
    """
    settings = load_settings()
    if seasons is None:
        seasons = settings["data"]["historical_seasons"]

    print(f"[hierarchy] Pulling combine data for {seasons}...")
    try:
        combine = nfl.import_combine_data(seasons)
        print(f"  Combine entries: {len(combine)}")
        return combine
    except Exception as e:
        print(f"  Could not pull combine data: {e}")
        return pd.DataFrame()


def pull_espn_depth_chart(team_abbr: str) -> pd.DataFrame:
    """
    Pull current depth chart from ESPN for a specific team.
    More up-to-date than nflverse during the season.
    """
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return pd.DataFrame()

    url = f"{ESPN_BASE_URL}/teams/{team_id}/depthcharts"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [depth] Error for {team_abbr}: {e}")
        return pd.DataFrame()

    rows = []
    for group in data.get("items", []):
        group_name = group.get("name", "")  # e.g. "offense", "defense", "specialTeams"
        for position in group.get("positions", {}).values():
            pos_name = position.get("position", {}).get("abbreviation", "")
            for rank, athlete_entry in enumerate(position.get("athletes", []), 1):
                athlete = athlete_entry.get("athlete", {})
                row = {
                    "team": team_abbr,
                    "group": group_name,
                    "position": pos_name,
                    "depth_rank": rank,
                    "player_name": athlete.get("displayName", ""),
                    "player_id_espn": athlete.get("id"),
                    "jersey": athlete.get("jersey", ""),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def pull_all_depth_charts_espn() -> pd.DataFrame:
    """Pull current ESPN depth charts for all 32 teams."""
    print("[hierarchy] Pulling ESPN depth charts for all teams...")
    all_charts = []

    for team_abbr in ESPN_TEAM_IDS:
        chart = pull_espn_depth_chart(team_abbr)
        if not chart.empty:
            all_charts.append(chart)

    if all_charts:
        combined = pd.concat(all_charts, ignore_index=True)
        print(f"  Total depth chart entries: {len(combined)}")
        return combined
    return pd.DataFrame()


def pull_espn_roster(team_abbr: str) -> pd.DataFrame:
    """Pull full roster with contract/experience info from ESPN."""
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return pd.DataFrame()

    url = f"{ESPN_BASE_URL}/teams/{team_id}/roster"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [roster] Error for {team_abbr}: {e}")
        return pd.DataFrame()

    rows = []
    for group in data.get("athletes", []):
        for athlete in group.get("items", []):
            row = {
                "team": team_abbr,
                "player_name": athlete.get("displayName", ""),
                "player_id_espn": athlete.get("id"),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "jersey": athlete.get("jersey", ""),
                "age": athlete.get("age"),
                "height": athlete.get("displayHeight", ""),
                "weight": athlete.get("displayWeight", ""),
                "experience": athlete.get("experience", {}).get("years", 0),
                "college": athlete.get("college", {}).get("name", ""),
                "status": athlete.get("status", {}).get("type", ""),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def build_coaching_features() -> pd.DataFrame:
    """
    Build coaching/scheme features from available data.
    
    Key features:
    - Is this a new coordinator? (first year in role = adjustment period)
    - Offensive scheme tendency (run-heavy vs. pass-heavy from play data)
    - Defensive scheme (4-3 vs 3-4, zone vs man tendency)
    
    Note: Detailed coaching data requires manual maintenance in a config file.
    nflverse provides some through play-by-play tendencies.
    """
    # For now, build from play-type frequencies
    settings = load_settings()
    current_season = settings["data"]["current_season"]

    try:
        # Pull play-by-play to derive scheme tendencies
        pbp = nfl.import_pbp_data([current_season - 1])  # last full season

        # Team-level play type tendencies
        team_plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
        scheme = team_plays.groupby("posteam").agg(
            total_plays=("play_type", "count"),
            pass_plays=("play_type", lambda x: (x == "pass").sum()),
            run_plays=("play_type", lambda x: (x == "run").sum()),
        ).reset_index()

        scheme["pass_rate"] = scheme["pass_plays"] / scheme["total_plays"]
        scheme["run_rate"] = scheme["run_plays"] / scheme["total_plays"]

        # Categorize scheme tendency
        scheme["scheme_tendency"] = pd.cut(
            scheme["pass_rate"],
            bins=[0, 0.48, 0.52, 0.56, 1.0],
            labels=["run_heavy", "balanced", "pass_lean", "pass_heavy"],
        )

        scheme.rename(columns={"posteam": "team"}, inplace=True)
        print(f"[hierarchy] Built scheme features for {len(scheme)} teams")
        return scheme

    except Exception as e:
        print(f"[hierarchy] Could not build coaching features: {e}")
        return pd.DataFrame()


def build_like_player_profiles(combine: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """
    Build "like player" comparison profiles.
    Groups players by physical archetype + position for matchup analysis.
    
    Example: "How do 6'1, 200lb slot WRs with 4.4 speed perform against Cover 3?"
    """
    if combine.empty:
        return pd.DataFrame()

    df = combine.copy()

    # Create archetype bins
    if "ht" in df.columns:
        df["height_bin"] = pd.cut(df["ht"], bins=[60, 70, 73, 76, 84], labels=["short", "average", "tall", "very_tall"])
    if "wt" in df.columns:
        df["weight_bin"] = pd.cut(df["wt"], bins=[150, 200, 220, 250, 350], labels=["light", "medium", "heavy", "very_heavy"])
    if "forty" in df.columns:
        df["speed_tier"] = pd.cut(df["forty"], bins=[4.2, 4.4, 4.5, 4.6, 5.5], labels=["elite", "fast", "average", "slow"])

    # Create archetype string
    archetype_cols = ["pos", "height_bin", "weight_bin", "speed_tier"]
    available = [c for c in archetype_cols if c in df.columns]
    if available:
        df["archetype"] = df[available].astype(str).agg("_".join, axis=1)

    return df


def save_hierarchy_data():
    """Pull and save all team hierarchy data."""
    raw_dir = get_data_dir("raw")
    settings = load_settings()

    print("=" * 60)
    print("PULLING TEAM HIERARCHY DATA")
    print("=" * 60)

    # Depth charts (nflverse)
    depth = pull_depth_charts()
    if not depth.empty:
        depth.to_parquet(raw_dir / "depth_charts.parquet", index=False)
        print(f"  Saved depth charts: {len(depth)} entries")

    # ESPN depth charts (more current during season)
    espn_depth = pull_all_depth_charts_espn()
    if not espn_depth.empty:
        espn_depth.to_parquet(raw_dir / "depth_charts_espn.parquet", index=False)
        print(f"  Saved ESPN depth charts: {len(espn_depth)} entries")

    # Draft data
    draft = pull_draft_data()
    if not draft.empty:
        draft.to_parquet(raw_dir / "draft_history.parquet", index=False)
        print(f"  Saved draft history: {len(draft)} entries")

    # Combine data
    combine = pull_combine_data()
    if not combine.empty:
        combine.to_parquet(raw_dir / "combine_data.parquet", index=False)
        print(f"  Saved combine data: {len(combine)} entries")

        # Build like-player profiles
        profiles = build_like_player_profiles(combine, pd.DataFrame())
        if not profiles.empty:
            profiles.to_parquet(raw_dir / "player_archetypes.parquet", index=False)
            print(f"  Saved player archetypes: {len(profiles)} entries")

    # Coaching/scheme features
    scheme = build_coaching_features()
    if not scheme.empty:
        scheme.to_parquet(raw_dir / "team_scheme_features.parquet", index=False)
        print(f"  Saved scheme features: {len(scheme)} teams")

    print("=" * 60)
    print("DONE - Hierarchy data saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    save_hierarchy_data()
