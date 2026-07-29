"""
Query Engine: Natural language lookups against our data.
Parses questions and returns formatted answers from nflverse stats,
scheme data, injury reports, and player intel.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# Team name mappings
TEAM_ALIASES = {
    "cardinals": "ARI", "arizona": "ARI", "ari": "ARI",
    "falcons": "ATL", "atlanta": "ATL", "atl": "ATL",
    "ravens": "BAL", "baltimore": "BAL", "bal": "BAL",
    "bills": "BUF", "buffalo": "BUF", "buf": "BUF",
    "panthers": "CAR", "carolina": "CAR", "car": "CAR",
    "bears": "CHI", "chicago": "CHI", "chi": "CHI",
    "bengals": "CIN", "cincinnati": "CIN", "cin": "CIN",
    "browns": "CLE", "cleveland": "CLE", "cle": "CLE",
    "cowboys": "DAL", "dallas": "DAL", "dal": "DAL",
    "broncos": "DEN", "denver": "DEN", "den": "DEN",
    "lions": "DET", "detroit": "DET", "det": "DET",
    "packers": "GB", "green bay": "GB", "gb": "GB",
    "texans": "HOU", "houston": "HOU", "hou": "HOU",
    "colts": "IND", "indianapolis": "IND", "ind": "IND",
    "jaguars": "JAX", "jacksonville": "JAX", "jax": "JAX",
    "chiefs": "KC", "kansas city": "KC", "kc": "KC",
    "chargers": "LAC", "lac": "LAC",
    "rams": "LAR", "lar": "LAR", "la rams": "LAR",
    "raiders": "LV", "las vegas": "LV", "lv": "LV",
    "dolphins": "MIA", "miami": "MIA", "mia": "MIA",
    "vikings": "MIN", "minnesota": "MIN", "min": "MIN",
    "patriots": "NE", "new england": "NE", "ne": "NE",
    "saints": "NO", "new orleans": "NO", "no": "NO",
    "giants": "NYG", "nyg": "NYG",
    "jets": "NYJ", "nyj": "NYJ",
    "eagles": "PHI", "philadelphia": "PHI", "phi": "PHI",
    "steelers": "PIT", "pittsburgh": "PIT", "pit": "PIT",
    "seahawks": "SEA", "seattle": "SEA", "sea": "SEA",
    "49ers": "SF", "niners": "SF", "san francisco": "SF", "sf": "SF",
    "buccaneers": "TB", "bucs": "TB", "tampa": "TB", "tb": "TB",
    "titans": "TEN", "tennessee": "TEN", "ten": "TEN",
    "commanders": "WAS", "washington": "WAS", "was": "WAS",
}


# Cache loaded data
_cache = {}

def _load(name):
    """Lazy-load and cache dataframes."""
    if name in _cache:
        return _cache[name]
    
    paths = {
        "stats": RAW_DIR / "player_stats_historical.parquet",
        "games": RAW_DIR / "games_historical.parquet",
        "injuries": RAW_DIR / "injuries_historical.parquet",
        "bankable": PROC_DIR / "bankable_players.parquet",
        "avoid": PROC_DIR / "avoid_players.parquet",
        "man_zone": PROC_DIR / "player_man_zone_splits.parquet",
        "def_scheme": PROC_DIR / "team_defensive_schemes.parquet",
        "off_scheme": PROC_DIR / "team_offensive_schemes.parquet",
        "half_stats": RAW_DIR / "player_receiving_half.parquet",
        "qtr_stats": RAW_DIR / "player_receiving_qtr.parquet",
    }
    
    path = paths.get(name)
    if path and path.exists():
        df = pd.read_parquet(path)
        _cache[name] = df
        return df
    return None


def _find_player(query_lower, stats_df):
    """Find player name from partial query."""
    if stats_df is None:
        return None, None
    
    name_col = "player_display_name" if "player_display_name" in stats_df.columns else "player_name"
    names = stats_df[name_col].dropna().unique()
    
    # Try exact match first
    for name in names:
        if query_lower == name.lower():
            return name, name_col
    
    # Then partial match
    matches = [n for n in names if query_lower in n.lower()]
    if len(matches) == 1:
        return matches[0], name_col
    elif len(matches) > 1:
        # Pick the one with most games (most likely the one they mean)
        counts = stats_df[stats_df[name_col].isin(matches)].groupby(name_col).size()
        return counts.idxmax(), name_col
    
    # Try last name only
    matches = [n for n in names if query_lower in n.lower().split()[-1]]
    if matches:
        counts = stats_df[stats_df[name_col].isin(matches)].groupby(name_col).size()
        return counts.idxmax(), name_col
    
    return None, name_col


def _find_team(query_lower):
    """Extract team abbreviation from query."""
    for alias, abbr in TEAM_ALIASES.items():
        if alias in query_lower:
            return abbr
    return None


def _get_player_stats(player_name, name_col, stats_df):
    """Get all stats for a player."""
    return stats_df[stats_df[name_col] == player_name].sort_values(["season", "week"])


def _merge_game_context(player_games, games_df):
    """Merge game context (opponent, venue, weather) onto player stats."""
    if games_df is None:
        return player_games
    
    # Merge on season + week + team
    merged = player_games.merge(
        games_df[["season", "week", "home_team", "away_team", "roof", 
                  "surface", "temp", "wind", "div_game", "weekday"]].drop_duplicates(),
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "home_team"],
        how="left",
        suffixes=("", "_h"),
    )
    # Fill in away games
    away = player_games.merge(
        games_df[["season", "week", "home_team", "away_team", "roof",
                  "surface", "temp", "wind", "div_game", "weekday"]].drop_duplicates(),
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "away_team"],
        how="left",
        suffixes=("", "_a"),
    )
    
    # Combine - use home merge where available, fill with away
    if "roof" not in merged.columns and "roof" in away.columns:
        merged["roof"] = away["roof"]
    
    # Determine opponent
    merged["opponent"] = np.where(
        merged["recent_team"] == merged.get("home_team", ""),
        merged.get("away_team", ""),
        merged.get("home_team", ""),
    )
    merged["is_home"] = merged["recent_team"] == merged.get("home_team", "")
    
    return merged


def _format_stats_table(df, stat_cols=None):
    """Format a stats dataframe into readable output."""
    if df.empty:
        return "No data found."
    
    if stat_cols is None:
        stat_cols = []
        for c in ["passing_yards", "passing_tds", "rushing_yards", "carries",
                  "receiving_yards", "receptions", "targets", "interceptions"]:
            if c in df.columns and df[c].notna().any():
                stat_cols.append(c)
    
    if not stat_cols:
        return "No relevant stats found."
    
    lines = []
    lines.append(f"**{len(df)} games found:**\n")
    
    # Summary stats
    for col in stat_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            lines.append(f"- **{col.replace('_', ' ').title()}:** "
                        f"Avg {vals.mean():.1f} | Med {vals.median():.1f} | "
                        f"High {vals.max():.0f} | Low {vals.min():.0f}")
    
    # Last few games detail
    if len(df) <= 8:
        lines.append(f"\n**Game log:**")
        for _, row in df.iterrows():
            parts = [f"Wk{int(row.get('week', 0))}"]
            if "opponent" in row and pd.notna(row.get("opponent")):
                parts.append(f"vs {row['opponent']}")
            for col in stat_cols[:4]:
                if pd.notna(row.get(col)):
                    parts.append(f"{col.replace('_',' ').split()[-1]}:{int(row[col])}")
            lines.append(f"  {'  |  '.join(parts)}")
    
    return "\n".join(lines)


def _detect_stat_focus(query_lower):
    """Detect which stat the user is asking about."""
    if any(x in query_lower for x in ["reception", "catch", "rec "]):
        return ["receptions", "targets"]
    elif any(x in query_lower for x in ["receiving yard", "rec yard", "rec yd"]):
        return ["receiving_yards", "receptions", "targets"]
    elif any(x in query_lower for x in ["rush", "carry", "carries", "running"]):
        return ["rushing_yards", "carries"]
    elif any(x in query_lower for x in ["pass yard", "passing yard", "throw"]):
        return ["passing_yards", "completions", "attempts"]
    elif any(x in query_lower for x in ["pass td", "passing td", "touchdown"]):
        return ["passing_tds", "interceptions"]
    elif "target" in query_lower:
        return ["targets", "receptions", "target_share"]
    return None  # Return all relevant stats


def _detect_num_games(query_lower):
    """Detect 'last N games' pattern."""
    match = re.search(r"last (\d+)", query_lower)
    if match:
        return int(match.group(1))
    return None


def query_player_vs_team(player_name, name_col, opponent_abbr, stats_df, games_df, stat_focus=None):
    """Player stats against a specific opponent."""
    player_games = _get_player_stats(player_name, name_col, stats_df)
    merged = _merge_game_context(player_games, games_df)
    
    vs_team = merged[merged["opponent"] == opponent_abbr]
    if vs_team.empty:
        return f"No games found for **{player_name}** vs {opponent_abbr}."
    
    return f"**{player_name} vs {opponent_abbr}:**\n\n" + _format_stats_table(vs_team, stat_focus)


def query_player_venue(player_name, name_col, venue_type, stats_df, games_df, stat_focus=None):
    """Player stats in dome vs outdoor."""
    player_games = _get_player_stats(player_name, name_col, stats_df)
    merged = _merge_game_context(player_games, games_df)
    
    if venue_type == "dome":
        filtered = merged[merged["roof"].isin(["dome", "closed"])]
        label = "dome/indoor"
    elif venue_type == "outdoor":
        filtered = merged[merged["roof"] == "outdoors"]
        label = "outdoor"
    elif venue_type == "grass":
        filtered = merged[merged["surface"].str.contains("grass", case=False, na=False)]
        label = "grass"
    elif venue_type == "turf":
        filtered = merged[~merged["surface"].str.contains("grass", case=False, na=False)]
        label = "turf"
    else:
        return "Unknown venue type."
    
    if filtered.empty:
        return f"No {label} games found for **{player_name}**."
    
    return f"**{player_name} in {label} games:**\n\n" + _format_stats_table(filtered, stat_focus)


def query_player_weather(player_name, name_col, weather_type, stats_df, games_df, stat_focus=None):
    """Player stats in specific weather conditions."""
    player_games = _get_player_stats(player_name, name_col, stats_df)
    merged = _merge_game_context(player_games, games_df)
    
    if weather_type == "cold":
        filtered = merged[merged["temp"].fillna(65) <= 35]
        label = "cold (<=35°F)"
    elif weather_type == "wind":
        filtered = merged[merged["wind"].fillna(0) >= 15]
        label = "windy (15+ mph)"
    elif weather_type == "hot":
        filtered = merged[merged["temp"].fillna(65) >= 85]
        label = "hot (85°F+)"
    else:
        return "Unknown weather type."
    
    if filtered.empty:
        return f"No {label} games found for **{player_name}**."
    
    return f"**{player_name} in {label} games:**\n\n" + _format_stats_table(filtered, stat_focus)


def query_player_last_n(player_name, name_col, n, stats_df, stat_focus=None):
    """Player's last N games."""
    player_games = _get_player_stats(player_name, name_col, stats_df)
    last_n = player_games.tail(n)
    
    if last_n.empty:
        return f"No recent games found for **{player_name}**."
    
    return f"**{player_name} last {n} games:**\n\n" + _format_stats_table(last_n, stat_focus)


def query_player_game_context(player_name, name_col, context, stats_df, games_df, stat_focus=None):
    """Player stats in division games, primetime, home/away, as dog/favorite."""
    player_games = _get_player_stats(player_name, name_col, stats_df)
    merged = _merge_game_context(player_games, games_df)
    
    if context == "division":
        filtered = merged[merged["div_game"] == 1]
        label = "division games"
    elif context == "primetime":
        filtered = merged[merged["weekday"].str.contains("Thursday|Monday", na=False)]
        label = "primetime (Thu/Mon)"
    elif context == "home":
        filtered = merged[merged["is_home"] == True]
        label = "home games"
    elif context == "away":
        filtered = merged[merged["is_home"] == False]
        label = "away games"
    elif context == "favorite":
        # Team was favored (negative spread for home, or positive if away)
        # Simplified: just check if team won
        label = "as favorite"
        filtered = merged  # Would need spread data merged properly
    elif context == "underdog":
        label = "as underdog"
        filtered = merged
    else:
        return "Unknown context."
    
    if filtered.empty:
        return f"No {label} found for **{player_name}**."
    
    return f"**{player_name} in {label}:**\n\n" + _format_stats_table(filtered, stat_focus)


def query_player_coverage(player_name):
    """Player performance vs man vs zone coverage."""
    splits = _load("man_zone")
    if splits is None:
        return "Coverage split data not available."
    
    # Find player in splits
    name_col = "receiver_player_name" if "receiver_player_name" in splits.columns else splits.columns[0]
    match = splits[splits[name_col].str.contains(player_name, case=False, na=False)]
    
    if match.empty:
        return f"No man/zone split data found for **{player_name}**. (Need 20+ targets vs each coverage type)"
    
    row = match.iloc[0]
    lines = [f"**{player_name} — Man vs Zone Splits:**\n"]
    
    man_yds = row.get("yards_per_target_MAN_COVERAGE", np.nan)
    zone_yds = row.get("yards_per_target_ZONE_COVERAGE", np.nan)
    diff = row.get("man_zone_diff", np.nan)
    
    if pd.notna(man_yds):
        lines.append(f"- **vs Man Coverage:** {man_yds:.1f} yards/target")
    if pd.notna(zone_yds):
        lines.append(f"- **vs Zone Coverage:** {zone_yds:.1f} yards/target")
    if pd.notna(diff):
        if diff > 2:
            lines.append(f"- **Verdict:** MAN KILLER (+{diff:.1f} yds/target vs man)")
        elif diff < -2:
            lines.append(f"- **Verdict:** ZONE KILLER (+{abs(diff):.1f} yds/target vs zone)")
        else:
            lines.append(f"- **Verdict:** Neutral (diff: {diff:+.1f})")
    
    return "\n".join(lines)


def query_player_without_teammate(player_name, name_col, teammate_query, stats_df, games_df, injuries_df):
    """Player stats when a specific teammate was OUT (injured)."""
    if injuries_df is None:
        return "Injury data not available."
    
    # Find teammate in injury reports
    teammate_matches = injuries_df[
        injuries_df["full_name"].str.contains(teammate_query, case=False, na=False) &
        (injuries_df["report_status"] == "Out")
    ]
    
    if teammate_matches.empty:
        return f"No injury records found for '{teammate_query}'."
    
    teammate_name = teammate_matches["full_name"].mode().iloc[0]
    teammate_team = teammate_matches["team"].mode().iloc[0]
    
    # Get weeks when teammate was out
    out_weeks = teammate_matches[["season", "week"]].drop_duplicates()
    
    # Get player's stats during those weeks
    player_games = _get_player_stats(player_name, name_col, stats_df)
    
    # Merge to find overlapping weeks (same team)
    player_same_team = player_games[player_games["recent_team"] == teammate_team]
    
    without = player_same_team.merge(out_weeks, on=["season", "week"], how="inner")
    with_tm = player_same_team.merge(out_weeks, on=["season", "week"], how="left", indicator=True)
    with_tm = with_tm[with_tm["_merge"] == "left_only"].drop("_merge", axis=1)
    
    if without.empty:
        return f"No games found where **{player_name}** played while **{teammate_name}** was out."
    
    lines = [f"**{player_name} WITHOUT {teammate_name}** ({len(without)} games):"]
    lines.append(_format_stats_table(without))
    lines.append(f"\n**{player_name} WITH {teammate_name}** ({len(with_tm)} games):")
    lines.append(_format_stats_table(with_tm))
    
    return "\n".join(lines)


def query_team_scheme(team_abbr):
    """Get team's defensive scheme profile."""
    def_scheme = _load("def_scheme")
    off_scheme = _load("off_scheme")
    
    lines = [f"**{team_abbr} Scheme Profile:**\n"]
    
    if def_scheme is not None:
        team_def = def_scheme[def_scheme["defteam"] == team_abbr]
        if not team_def.empty:
            latest = team_def.sort_values("season").iloc[-1]
            lines.append(f"**Defense (latest season):**")
            lines.append(f"- Man rate: {latest.get('man_rate', 0):.0%}")
            lines.append(f"- Zone rate: {latest.get('zone_rate', 0):.0%}")
            lines.append(f"- Lean: {latest.get('scheme_lean', 'Unknown')}")
    
    if off_scheme is not None:
        team_off = off_scheme[off_scheme["posteam"] == team_abbr]
        if not team_off.empty:
            latest = team_off.sort_values("season").iloc[-1]
            lines.append(f"\n**Offense (latest season):**")
            lines.append(f"- Pass rate: {latest.get('pass_rate', 0):.0%}")
            lines.append(f"- Shotgun rate: {latest.get('shotgun_rate', 0):.0%}")
            lines.append(f"- No-huddle rate: {latest.get('no_huddle_rate', 0):.0%}")
    
    return "\n".join(lines) if len(lines) > 1 else f"No scheme data found for {team_abbr}."


def query_player_half(player_name, half):
    """Player stats by half (1st half vs 2nd half)."""
    half_data = _load("half_stats")
    if half_data is None:
        return "Half split data not available."
    
    match = half_data[half_data["player_name"].str.contains(player_name, case=False, na=False)]
    if match.empty:
        return f"No half split data for **{player_name}**."
    
    h1 = match[match["half"] == "H1"]
    h2 = match[match["half"] == "H2"]
    
    lines = [f"**{player_name} — Half Splits:**\n"]
    
    if not h1.empty:
        h1_yds = h1["rec_yards"].mean() if "rec_yards" in h1.columns else 0
        h1_rec = h1["receptions"].mean() if "receptions" in h1.columns else 0
        lines.append(f"- **1st Half:** {h1_yds:.1f} rec yds | {h1_rec:.1f} receptions (per game)")
    if not h2.empty:
        h2_yds = h2["rec_yards"].mean() if "rec_yards" in h2.columns else 0
        h2_rec = h2["receptions"].mean() if "receptions" in h2.columns else 0
        lines.append(f"- **2nd Half:** {h2_yds:.1f} rec yds | {h2_rec:.1f} receptions (per game)")
    
    return "\n".join(lines)


def query_player_season(player_name, name_col, season, stats_df, stat_focus=None):
    """Player stats for a specific season."""
    player_games = _get_player_stats(player_name, name_col, stats_df)
    season_games = player_games[player_games["season"] == season]
    
    if season_games.empty:
        return f"No data for **{player_name}** in {season}."
    
    return f"**{player_name} — {season} Season:**\n\n" + _format_stats_table(season_games, stat_focus)


# ================================================================
# MAIN QUERY ROUTER
# ================================================================

def process_query(query: str) -> str:
    """
    Parse a natural language query and route to the appropriate lookup.
    Returns formatted markdown response.
    """
    q = query.lower().strip()
    
    if not q:
        return ""
    
    # Load core data
    stats_df = _load("stats")
    games_df = _load("games")
    injuries_df = _load("injuries")
    
    if stats_df is None:
        return "Player stats data not loaded. Run the data pipeline first."
    
    # Detect stat focus
    stat_focus = _detect_stat_focus(q)
    
    # Detect number of games
    num_games = _detect_num_games(q)
    
    # Detect team
    team_mentioned = _find_team(q)
    
    # Detect venue/weather keywords
    is_dome = any(x in q for x in ["dome", "indoor", "indoors"])
    is_outdoor = any(x in q for x in ["outdoor", "outdoors", "outside"])
    is_cold = "cold" in q
    is_wind = any(x in q for x in ["wind", "windy"])
    is_hot = "hot" in q
    is_grass = "grass" in q
    is_turf = "turf" in q
    
    # Detect game context
    is_division = any(x in q for x in ["division", "divisional", "div "])
    is_primetime = any(x in q for x in ["primetime", "prime time", "monday night", "thursday"])
    is_home = " home" in q and "home" not in q[:4]
    is_away = " away" in q or " road" in q
    
    # Detect coverage
    is_man = any(x in q for x in ["man coverage", "man defense", " man", "vs man"])
    is_zone = any(x in q for x in ["zone coverage", "zone defense", " zone", "vs zone"])
    is_coverage = is_man or is_zone or "coverage" in q or "scheme" in q
    
    # Detect "without" / "out" pattern
    without_match = re.search(r"without (.+?)(?:\s|$)|with (.+?) out|(.+?) out", q)
    
    # Detect half/quarter
    is_first_half = any(x in q for x in ["first half", "1st half", "h1"])
    is_second_half = any(x in q for x in ["second half", "2nd half", "h2"])
    
    # Detect season
    season_match = re.search(r"20(2[0-9])", q)
    season_year = int("20" + season_match.group(1)) if season_match else None
    
    # ---- TEAM-ONLY QUERIES ----
    if team_mentioned and not any(c.isalpha() and c not in "".join(TEAM_ALIASES.keys()) for _ in []):
        # Check if query is ONLY about a team (no player)
        words_without_team = q
        for alias in TEAM_ALIASES:
            words_without_team = words_without_team.replace(alias, "")
        words_without_team = words_without_team.strip()
        
        team_only_keywords = ["defense", "scheme", "offensive", "coverage", "pass rate"]
        if any(k in q for k in team_only_keywords) and len(words_without_team.split()) <= 3:
            return query_team_scheme(team_mentioned)
    
    # ---- FIND THE PLAYER ----
    # Remove known keywords to isolate player name
    clean_q = q
    remove_words = ["vs", "against", "in", "last", "games", "game", "stats",
                    "dome", "outdoor", "cold", "wind", "hot", "grass", "turf",
                    "division", "primetime", "home", "away", "man", "zone",
                    "coverage", "scheme", "first half", "second half", "without",
                    "receptions", "receiving", "rushing", "passing", "yards", "targets",
                    "catches", "carries", "touchdowns", "tds"]
    
    name_search = clean_q
    for word in remove_words:
        name_search = name_search.replace(word, " ")
    # Remove team names
    for alias in TEAM_ALIASES:
        name_search = name_search.replace(alias, " ")
    # Remove numbers
    name_search = re.sub(r"\d+", "", name_search).strip()
    name_search = " ".join(name_search.split())  # Clean whitespace
    
    if not name_search or len(name_search) < 3:
        # Maybe the whole query is a player name
        name_search = q.split(" vs ")[0].split(" against ")[0].strip()
    
    player_name, name_col = _find_player(name_search, stats_df)
    
    if player_name is None:
        # Try the original query as player name
        player_name, name_col = _find_player(q.split()[0] + " " + q.split()[1] if len(q.split()) > 1 else q, stats_df)
    
    if player_name is None:
        # Check bankable/avoid as fallback
        bankable = _load("bankable")
        avoid = _load("avoid")
        if bankable is not None:
            match = bankable[bankable["player_clean"].str.contains(q.split()[0], na=False)]
            if not match.empty:
                row = match.iloc[0]
                return f"✅ **{row['player_clean'].title()}** — BANKABLE\nHit rate: {row['hit_rate']:.1%} | ROI: {row['roi']:+.1f}%"
        if avoid is not None:
            match = avoid[avoid["player_clean"].str.contains(q.split()[0], na=False)]
            if not match.empty:
                row = match.iloc[0]
                return f"🚫 **{row['player_clean'].title()}** — AVOID\nHit rate: {row['hit_rate']:.1%} | ROI: {row['roi']:+.1f}%"
        
        # Strategy keyword fallback
        strategies = {
            "week 1": "🔴 **Week 1:** SLAM UNDER everything. Pass TDs 75.8%, Pass yds 70.6%. Biggest edge of the year.",
            "dome": "🏟️ **Dome:** Pass TDs OVER +EV at plus money. 45.9% hit vs 36.9% implied.",
            "cold": "🥶 **Cold:** Rush OVER (54.5%). Rec UNDER + outdoor + cold (56.1%).",
            "wind": "💨 **Wind:** Pass UNDER (55.7% hit at 15+ mph).",
            "monday": "🌙 **Monday:** UNDER lean (53.5%, 3/3 seasons).",
            "division": "🏈 **Division:** UNDER lean. Rec UNDER + div + outdoor = 57.4%.",
            "injury": "🏥 **High injury (8+ out):** UNDER hits 63.2%.",
            "new team": "🔄 **New team:** UNDER weeks 1-4 (55.2%).",
            "boom": "📈 **After boom:** Rush UNDER 54.8%, Rec UNDER 54.3%.",
        }
        for kw, resp in strategies.items():
            if kw in q:
                return resp
        
        return f"Couldn't find a player matching '{name_search}'. Try a full name like 'Patrick Mahomes' or a keyword like 'dome', 'week 1'."
    
    # ---- ROUTE TO SPECIFIC QUERY TYPE ----
    
    # Without teammate
    if without_match or "without" in q or "out" in q:
        tm_query = without_match.group(1) or without_match.group(2) or without_match.group(3) if without_match else ""
        if tm_query:
            return query_player_without_teammate(player_name, name_col, tm_query.strip(), stats_df, games_df, injuries_df)
    
    # Coverage/scheme
    if is_coverage:
        return query_player_coverage(player_name)
    
    # Half splits
    if is_first_half or is_second_half:
        return query_player_half(player_name, "H1" if is_first_half else "H2")
    
    # Vs specific team
    if team_mentioned and any(x in q for x in ["vs", "against", "versus"]):
        return query_player_vs_team(player_name, name_col, team_mentioned, stats_df, games_df, stat_focus)
    elif team_mentioned:
        return query_player_vs_team(player_name, name_col, team_mentioned, stats_df, games_df, stat_focus)
    
    # Venue
    if is_dome:
        return query_player_venue(player_name, name_col, "dome", stats_df, games_df, stat_focus)
    if is_outdoor:
        return query_player_venue(player_name, name_col, "outdoor", stats_df, games_df, stat_focus)
    if is_grass:
        return query_player_venue(player_name, name_col, "grass", stats_df, games_df, stat_focus)
    if is_turf:
        return query_player_venue(player_name, name_col, "turf", stats_df, games_df, stat_focus)
    
    # Weather
    if is_cold:
        return query_player_weather(player_name, name_col, "cold", stats_df, games_df, stat_focus)
    if is_wind:
        return query_player_weather(player_name, name_col, "wind", stats_df, games_df, stat_focus)
    if is_hot:
        return query_player_weather(player_name, name_col, "hot", stats_df, games_df, stat_focus)
    
    # Game context
    if is_division:
        return query_player_game_context(player_name, name_col, "division", stats_df, games_df, stat_focus)
    if is_primetime:
        return query_player_game_context(player_name, name_col, "primetime", stats_df, games_df, stat_focus)
    if is_home:
        return query_player_game_context(player_name, name_col, "home", stats_df, games_df, stat_focus)
    if is_away:
        return query_player_game_context(player_name, name_col, "away", stats_df, games_df, stat_focus)
    
    # Specific season
    if season_year:
        return query_player_season(player_name, name_col, season_year, stats_df, stat_focus)
    
    # Last N games
    if num_games:
        return query_player_last_n(player_name, name_col, num_games, stats_df, stat_focus)
    
    # Default: last 5 games
    return query_player_last_n(player_name, name_col, 5, stats_df, stat_focus)
