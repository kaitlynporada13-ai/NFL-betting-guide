"""
Season-Long Props Analysis: FanDuel 2026-27 Regular Season Totals
Compares FanDuel season lines to historical production + situational adjustments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
NOTES_DIR = DATA_DIR / "human_notes"

# ============================================================
# SEASON LINES FROM FANDUEL (captured 2026-08-07)
# ============================================================

pass_yards = {
    "Aaron Rodgers": 3050.5, "Baker Mayfield": 3500.5, "Bo Nix": 3400.5,
    "Brock Purdy": 3775.5, "Bryce Young": 3000.5, "Caleb Williams": 3575.5,
    "Cam Ward": 3250.5, "C.J. Stroud": 3550.5, "Dak Prescott": 4000.5,
    "Daniel Jones": 3300.5, "Drake Maye": 3750.5, "Fernando Mendoza": 2300.5,
    "Jalen Hurts": 3150.5, "Jared Goff": 4050.5, "Jaxson Dart": 3100.5,
    "Jayden Daniels": 3200.5, "Joe Burrow": 3900.5, "Josh Allen": 3600.5,
    "Jordan Love": 3500.5, "Justin Herbert": 3500.5, "Lamar Jackson": 3200.5,
    "Malik Willis": 2925.5, "Matthew Stafford": 3825.5, "Patrick Mahomes": 3600.5,
    "Sam Darnold": 3600.5, "Trevor Lawrence": 3750.5, "Tyler Shough": 3450.5,
}

pass_tds = {
    "Aaron Rodgers": (20.5, -114, -114), "Baker Mayfield": (25.5, -102, -130),
    "Bo Nix": (23.5, -122, -108), "Brock Purdy": (27.5, -114, -114),
    "Bryce Young": (20.5, -114, -114), "Caleb Williams": (24.5, 100, -132),
    "Cam Ward": (19.5, 102, -136), "C.J. Stroud": (22.5, -122, -108),
    "Dak Prescott": (27.5, -108, -122), "Daniel Jones": (18.5, 102, -136),
    "Drake Maye": (25.5, -144, 108), "Fernando Mendoza": (12.5, -114, -114),
    "Jalen Hurts": (21.5, -122, -108), "Jared Goff": (29.5, -108, -122),
    "Jaxson Dart": (19.5, -108, -122), "Jayden Daniels": (21.5, -102, -130),
    "Joe Burrow": (32.5, -114, -114), "Jordan Love": (24.5, -102, -130),
    "Josh Allen": (24.5, -120, -110), "Justin Herbert": (25.5, -108, -122),
    "Lamar Jackson": (24.5, 104, -138), "Malik Willis": (13.5, -148, 112),
    "Matthew Stafford": (29.5, -132, 100), "Patrick Mahomes": (24.5, -102, -130),
    "Sam Darnold": (23.5, -102, -130), "Trevor Lawrence": (25.5, -102, -130),
    "Tyler Shough": (19.5, -148, 112),
}

rush_yards = {
    "Ashton Jeanty": 975.5, "Bhayshul Tuten": 725.5, "Bijan Robinson": 1150.5,
    "Breece Hall": 900.5, "Bucky Irving": 825.5, "Chase Brown": 825.5,
    "Christian McCaffrey": 900.5, "Chuba Hubbard": 700.5, "D'Andre Swift": 800.5,
    "David Montgomery": 825.5, "De'Von Achane": 975.5, "Derrick Henry": 1225.5,
    "Drake Maye": 400.5, "J.K. Dobbins": 700.5, "Jahmyr Gibbs": 1200.5,
    "Jalen Hurts": 400.5, "James Cook": 1175.5, "Jadarian Price": 700.5,
    "Jacory Croskey-Merritt": 625.5, "Javonte Williams": 925.5,
    "Jaxson Dart": 475.5, "Jayden Daniels": 550.5, "Jaylen Warren": 625.5,
    "Jeremiyah Love": 825.5, "Josh Allen": 500.5, "Jonathan Taylor": 1225.5,
    "Kyren Williams": 925.5, "Kenneth Walker": 925.5, "Lamar Jackson": 525.5,
    "Malik Willis": 550.5, "Omarion Hampton": 925.5, "Quinshon Judkins": 875.5,
    "Rhamondre Stevenson": 600.5, "Saquon Barkley": 1050.5, "Tony Pollard": 775.5,
    "Travis Etienne": 850.5, "TreVeyon Henderson": 700.5,
}

rec_yards = {
    "Amon-Ra St. Brown": 1225.5, "A.J. Brown": 1100.5, "Bijan Robinson": 550.5,
    "Brian Thomas Jr.": 725.5, "Brock Bowers": 900.5, "Carnell Tate": 775.5,
    "CeeDee Lamb": 1200.5, "Chris Godwin": 625.5, "Chris Olave": 1050.5,
    "Christian McCaffrey": 550.5, "Christian Watson": 750.5,
    "Colston Loveland": 775.5, "Courtland Sutton": 775.5,
    "Dallas Goedert": 575.5, "Davante Adams": 775.5, "DeVonta Smith": 1025.5,
    "D.J. Moore": 800.5, "DK Metcalf": 800.5, "Drake London": 1100.5,
    "Emeka Egbuka": 900.5, "Garrett Wilson": 975.5, "George Pickens": 1050.5,
    "George Kittle": 700.5, "Harold Fannin Jr.": 700.5, "Jahmyr Gibbs": 475.5,
    "Jameson Williams": 900.5, "Ja'Marr Chase": 1325.5,
    "Jaxon Smith-Njigba": 1350.5, "Jaylen Waddle": 925.5,
    "Jayden Reed": 600.5, "Jakobi Meyers": 700.5, "Jordan Addison": 725.5,
    "Jordyn Tyson": 750.5, "Justin Jefferson": 1150.5,
    "KC Concepcion": 600.5, "Kyle Pitts": 725.5, "Ladd McConkey": 875.5,
    "Luther Burden III": 900.5, "Makai Lemon": 625.5, "Mark Andrews": 500.5,
    "Marvin Harrison Jr.": 825.5, "Matthew Golden": 600.5,
    "Michael Pittman Jr.": 725.5, "Michael Wilson": 725.5,
    "Mike Evans": 850.5, "Nico Collins": 1050.5, "Omar Cooper Jr.": 500.5,
    "Parker Washington": 800.5, "Puka Nacua": 1375.5,
    "Quentin Johnston": 675.5, "Rashee Rice": 975.5, "Rashid Shaheed": 475.5,
    "Rome Odunze": 800.5, "Romeo Doubs": 650.5, "Sam LaPorta": 650.5,
    "Tee Higgins": 875.5, "Terry McLaurin": 900.5,
    "Tetairoa McMillan": 950.5, "Travis Kelce": 675.5, "Trey McBride": 950.5,
    "Tucker Kraft": 775.5, "Tyler Warren": 750.5, "Wan'Dale Robinson": 650.5,
    "Zay Flowers": 1000.5,
}

# ============================================================
# CONTEXT DATA
# ============================================================

# New team QBs (confirmed from preseason intel)
NEW_TEAM_QBS = {
    "Kyler Murray": "MIN", "Malik Willis": "MIA", "Geno Smith": "NYJ",
    "Kirk Cousins": "LV", "Sam Darnold": "MIN",  # Darnold was MIN last year, now unknown
}

# New team WRs/TEs
NEW_TEAM_RECEIVERS = {
    "A.J. Brown": "NE", "D.J. Moore": "BUF", "Jaylen Waddle": "DEN",
    "Mike Evans": "SF", "Justin Jefferson": "MIN",  # new QB (Murray)
}

# First-time OC play-callers (high disruption)
FIRST_TIME_PLAYCALLERS = ["BAL", "WAS", "PHI", "LV"]

# Team disruption levels (from coaching_changes)
HIGH_DISRUPTION = {"MIA", "WAS", "NYG", "BAL", "NYJ", "LV", "TEN"}

# SOS rankings (1=easiest, 32=hardest)
SOS = {
    "DET": 1, "CIN": 2, "NO": 3, "NYJ": 4, "HOU": 5, "BAL": 6,
    "CLE": 7, "SF": 8, "MIN": 9, "IND": 10, "PHI": 11, "BUF": 12,
    "LV": 13, "KC": 14, "PIT": 15, "TEN": 16, "JAX": 17, "ATL": 18,
    "TB": 19, "NYG": 20, "MIA": 21, "CHI": 22, "SEA": 23, "LAC": 24,
    "GB": 25, "LAR": 26, "DAL": 27, "CAR": 28, "WAS": 29, "ARI": 30,
}

# Player -> team mapping (2026)
PLAYER_TEAMS = {
    "Aaron Rodgers": "NYJ", "Baker Mayfield": "TB", "Bo Nix": "DEN",
    "Brock Purdy": "SF", "Bryce Young": "CAR", "Caleb Williams": "CHI",
    "Cam Ward": "TEN", "C.J. Stroud": "HOU", "Dak Prescott": "DAL",
    "Daniel Jones": "NYG", "Drake Maye": "NE", "Fernando Mendoza": "LV",
    "Jalen Hurts": "PHI", "Jared Goff": "DET", "Jaxson Dart": "NYG",
    "Jayden Daniels": "WAS", "Joe Burrow": "CIN", "Josh Allen": "BUF",
    "Jordan Love": "GB", "Justin Herbert": "LAC", "Lamar Jackson": "BAL",
    "Malik Willis": "MIA", "Matthew Stafford": "LAR", "Patrick Mahomes": "KC",
    "Sam Darnold": "MIN", "Trevor Lawrence": "JAX", "Tyler Shough": "PIT",
    # RBs
    "Ashton Jeanty": "DEN", "Bijan Robinson": "ATL", "Breece Hall": "NYJ",
    "Bucky Irving": "TB", "Chase Brown": "CIN", "Christian McCaffrey": "SF",
    "Chuba Hubbard": "CAR", "D'Andre Swift": "CHI", "David Montgomery": "DET",
    "De'Von Achane": "MIA", "Derrick Henry": "BAL", "Jahmyr Gibbs": "DET",
    "James Cook": "BUF", "Jonathan Taylor": "IND", "Kyren Williams": "LAR",
    "Kenneth Walker": "KC", "Javonte Williams": "DEN", "Saquon Barkley": "PHI",
    "Tony Pollard": "TEN", "Travis Etienne": "JAX", "Quinshon Judkins": "GB",
    "Omarion Hampton": "CAR", "Jeremiyah Love": "ARI", "Jaylen Warren": "PIT",
    "Rhamondre Stevenson": "NE", "TreVeyon Henderson": "CIN",
    "Bhayshul Tuten": "JAX", "Jadarian Price": "MIN",
    "Jacory Croskey-Merritt": "ARI", "J.K. Dobbins": "LAC",
    # WRs/TEs
    "Amon-Ra St. Brown": "DET", "A.J. Brown": "NE", "CeeDee Lamb": "DAL",
    "Ja'Marr Chase": "CIN", "Puka Nacua": "LAR", "Jaxon Smith-Njigba": "SEA",
    "Justin Jefferson": "MIN", "Drake London": "ATL", "Nico Collins": "HOU",
    "Chris Olave": "NO", "George Pickens": "PIT", "DeVonta Smith": "PHI",
    "Garrett Wilson": "NYJ", "Tetairoa McMillan": "ARI", "Rashee Rice": "KC",
    "Jaylen Waddle": "DEN", "Tee Higgins": "CIN", "Terry McLaurin": "WAS",
    "Zay Flowers": "BAL", "Jameson Williams": "DET", "Trey McBride": "ARI",
    "Emeka Egbuka": "SEA", "Luther Burden III": "CLE",
    "Ladd McConkey": "LAC", "D.J. Moore": "BUF", "DK Metcalf": "PIT",
    "Mike Evans": "SF", "Brock Bowers": "LV", "Davante Adams": "NYJ",
    "George Kittle": "SF", "Kyle Pitts": "ATL", "Travis Kelce": "KC",
    "Sam LaPorta": "DET", "Mark Andrews": "BAL", "Tucker Kraft": "GB",
    "Tyler Warren": "PIT", "Courtland Sutton": "DEN",
    "Michael Pittman Jr.": "IND", "Marvin Harrison Jr.": "ARI",
    "Chris Godwin": "TB", "Christian Watson": "GB", "Rome Odunze": "CHI",
    "Romeo Doubs": "GB", "Jordan Addison": "MIN", "Makai Lemon": "PHI",
    "Wan'Dale Robinson": "NYG", "Dallas Goedert": "PHI",
    "Brian Thomas Jr.": "JAX", "Carnell Tate": "CIN",
    "Colston Loveland": "DET", "Harold Fannin Jr.": "CLE",
    "Jakobi Meyers": "LV", "Jordyn Tyson": "ARI",
    "KC Concepcion": "KC", "Matthew Golden": "NYJ",
    "Michael Wilson": "ARI", "Omar Cooper Jr.": "BUF",
    "Parker Washington": "PIT", "Quentin Johnston": "LAC",
    "Rashid Shaheed": "NO", "Jayden Reed": "GB",
}

# Contract year players (from our data)
CONTRACT_YEAR = {
    "Ja'Marr Chase", "CeeDee Lamb", "Amon-Ra St. Brown", "Nico Collins",
    "Tee Higgins", "George Kittle", "Dalton Kincaid", "Sam LaPorta",
    "Courtland Sutton", "Terry McLaurin", "Michael Pittman Jr.",
    "Jaylen Warren", "Chuba Hubbard", "Alvin Kamara",
}


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def load_historical_stats():
    """Load player career stats for comparison."""
    path = RAW_DIR / "player_stats_historical.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


def get_player_season_totals(stats: pd.DataFrame, player_name: str, stat_col: str):
    """Get a player's season totals across available years."""
    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    player = stats[stats[name_col].str.lower() == player_name.lower()]
    if player.empty:
        return {}
    season_totals = player.groupby("season")[stat_col].sum()
    games_played = player.groupby("season")["week"].count()
    return {
        "seasons": season_totals.to_dict(),
        "games": games_played.to_dict(),
        "avg_season": season_totals.mean(),
        "last_season": season_totals.iloc[-1] if len(season_totals) > 0 else 0,
        "max_season": season_totals.max(),
        "avg_per_game": (season_totals / games_played).mean(),
    }


def analyze_all():
    """Run the full analysis."""
    stats = load_historical_stats()
    results = []

    print("=" * 80)
    print("SEASON PROPS ANALYSIS — FanDuel 2026-27")
    print("=" * 80)

    # --- PASSING YARDS ---
    print("\n" + "=" * 80)
    print("PASSING YARDS")
    print("=" * 80)
    for player, line in sorted(pass_yards.items(), key=lambda x: x[1], reverse=True):
        team = PLAYER_TEAMS.get(player, "?")
        sos = SOS.get(team, 16)
        hist = get_player_season_totals(stats, player, "passing_yards")
        avg = hist.get("avg_season", 0)
        last = hist.get("last_season", 0)
        per_game = hist.get("avg_per_game", 0)

        # Adjustment factors
        adjustments = []
        projected_games = 17  # assume full season

        # New team / disruption
        if team in HIGH_DISRUPTION:
            adjustments.append(("HIGH DISRUPTION team", -0.05))
        if player in NEW_TEAM_QBS:
            adjustments.append(("New team QB", -0.04))

        # SOS impact (easy schedule = more yards opportunity)
        if sos <= 8:
            adjustments.append((f"Easy SOS (#{sos})", +0.03))
        elif sos >= 25:
            adjustments.append((f"Hard SOS (#{sos})", -0.03))

        # Health/age concerns
        if player == "Dak Prescott":
            adjustments.append(("Injury concern (hamstring)", -0.08))
        if player == "Aaron Rodgers":
            adjustments.append(("Age 43", -0.05))

        # Calculate projection
        if per_game > 0:
            base_projection = per_game * projected_games
        elif avg > 0:
            base_projection = avg
        else:
            base_projection = line  # no data, assume line is fair

        total_adj = sum(a[1] for a in adjustments)
        adjusted_projection = base_projection * (1 + total_adj)
        edge = adjusted_projection - line
        edge_pct = edge / line * 100

        direction = "OVER" if edge > 0 else "UNDER"
        confidence = abs(edge_pct)

        if confidence > 5:
            tier = "STRONG"
        elif confidence > 2.5:
            tier = "LEAN"
        else:
            tier = "skip"

        results.append({
            "market": "pass_yards", "player": player, "team": team,
            "line": line, "projection": adjusted_projection,
            "edge": edge, "edge_pct": edge_pct,
            "direction": direction, "tier": tier,
            "adjustments": adjustments,
        })

        if tier != "skip":
            adj_str = ", ".join(f"{a[0]}" for a in adjustments) if adjustments else "none"
            print(f"  {tier:6s} {direction:5s} | {player:20s} ({team}) | "
                  f"Line: {line:.0f} | Proj: {adjusted_projection:.0f} | "
                  f"Edge: {edge:+.0f} ({edge_pct:+.1f}%) | Adj: {adj_str}")

    # --- RUSHING YARDS ---
    print("\n" + "=" * 80)
    print("RUSHING YARDS")
    print("=" * 80)
    for player, line in sorted(rush_yards.items(), key=lambda x: x[1], reverse=True):
        team = PLAYER_TEAMS.get(player, "?")
        sos = SOS.get(team, 16)
        hist = get_player_season_totals(stats, player, "rushing_yards")
        avg = hist.get("avg_season", 0)
        last = hist.get("last_season", 0)
        per_game = hist.get("avg_per_game", 0)

        adjustments = []
        projected_games = 17

        if team in HIGH_DISRUPTION:
            adjustments.append(("HIGH DISRUPTION", -0.04))
        if sos <= 8:
            adjustments.append((f"Easy SOS (#{sos})", +0.02))
        elif sos >= 25:
            adjustments.append((f"Hard SOS (#{sos})", -0.03))
        if player in CONTRACT_YEAR:
            adjustments.append(("Contract year", +0.03))
        if player == "Jeremiyah Love":
            adjustments.append(("Rookie + hardest schedule", -0.06))
        if player == "Kenneth Walker":
            adjustments.append(("New team (KC)", -0.03))
        if player == "Christian McCaffrey":
            adjustments.append(("Age 30 + injury history", -0.05))

        if per_game > 0:
            base_projection = per_game * projected_games
        elif avg > 0:
            base_projection = avg
        else:
            base_projection = line

        total_adj = sum(a[1] for a in adjustments)
        adjusted_projection = base_projection * (1 + total_adj)
        edge = adjusted_projection - line
        edge_pct = edge / line * 100

        direction = "OVER" if edge > 0 else "UNDER"
        confidence = abs(edge_pct)
        tier = "STRONG" if confidence > 5 else "LEAN" if confidence > 2.5 else "skip"

        results.append({
            "market": "rush_yards", "player": player, "team": team,
            "line": line, "projection": adjusted_projection,
            "edge": edge, "edge_pct": edge_pct,
            "direction": direction, "tier": tier,
            "adjustments": adjustments,
        })

        if tier != "skip":
            adj_str = ", ".join(f"{a[0]}" for a in adjustments) if adjustments else "none"
            print(f"  {tier:6s} {direction:5s} | {player:20s} ({team}) | "
                  f"Line: {line:.0f} | Proj: {adjusted_projection:.0f} | "
                  f"Edge: {edge:+.0f} ({edge_pct:+.1f}%) | Adj: {adj_str}")

    # --- RECEIVING YARDS ---
    print("\n" + "=" * 80)
    print("RECEIVING YARDS")
    print("=" * 80)
    for player, line in sorted(rec_yards.items(), key=lambda x: x[1], reverse=True):
        team = PLAYER_TEAMS.get(player, "?")
        sos = SOS.get(team, 16)
        hist = get_player_season_totals(stats, player, "receiving_yards")
        avg = hist.get("avg_season", 0)
        last = hist.get("last_season", 0)
        per_game = hist.get("avg_per_game", 0)

        adjustments = []
        projected_games = 17

        if player in NEW_TEAM_RECEIVERS:
            adjustments.append(("New team", -0.06))
        if team in HIGH_DISRUPTION:
            adjustments.append(("HIGH DISRUPTION", -0.04))
        if sos <= 8:
            adjustments.append((f"Easy SOS (#{sos})", +0.03))
        elif sos >= 25:
            adjustments.append((f"Hard SOS (#{sos})", -0.03))
        if player in CONTRACT_YEAR:
            adjustments.append(("Contract year", +0.03))

        if per_game > 0:
            base_projection = per_game * projected_games
        elif avg > 0:
            base_projection = avg
        else:
            base_projection = line

        total_adj = sum(a[1] for a in adjustments)
        adjusted_projection = base_projection * (1 + total_adj)
        edge = adjusted_projection - line
        edge_pct = edge / line * 100

        direction = "OVER" if edge > 0 else "UNDER"
        confidence = abs(edge_pct)
        tier = "STRONG" if confidence > 5 else "LEAN" if confidence > 2.5 else "skip"

        results.append({
            "market": "rec_yards", "player": player, "team": team,
            "line": line, "projection": adjusted_projection,
            "edge": edge, "edge_pct": edge_pct,
            "direction": direction, "tier": tier,
            "adjustments": adjustments,
        })

        if tier != "skip":
            adj_str = ", ".join(f"{a[0]}" for a in adjustments) if adjustments else "none"
            print(f"  {tier:6s} {direction:5s} | {player:20s} ({team}) | "
                  f"Line: {line:.0f} | Proj: {adjusted_projection:.0f} | "
                  f"Edge: {edge:+.0f} ({edge_pct:+.1f}%) | Adj: {adj_str}")

    # --- SUMMARY ---
    print("\n" + "=" * 80)
    print("TOP PLAYS (Strongest Edges)")
    print("=" * 80)
    df = pd.DataFrame(results)
    strong = df[df["tier"].isin(["STRONG", "LEAN"])].sort_values("edge_pct", key=abs, ascending=False)
    print(f"\n{'Direction':<6} {'Market':<12} {'Player':<22} {'Team':<5} {'Line':>7} {'Proj':>7} {'Edge%':>7}")
    print("-" * 75)
    for _, r in strong.head(25).iterrows():
        print(f"{r['direction']:<6} {r['market']:<12} {r['player']:<22} {r['team']:<5} "
              f"{r['line']:>7.0f} {r['projection']:>7.0f} {r['edge_pct']:>+6.1f}%")


if __name__ == "__main__":
    analyze_all()
