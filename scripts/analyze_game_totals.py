"""
Game Totals Analyzer — Over/Under lean for each game on a target slate.
Built for Week 1 where the ML totals model can't run (no current-season data),
so this uses situational signals: coaching disruption, dome, key injuries.

HONEST CAVEAT: Game totals are a sharp market. These leans are lower-conviction
than our player-prop edges. Use for entertainment parlays, not as locks.
"""
import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.ingest_odds import pull_game_odds
from pipeline.config_loader import load_stadiums

NOTES_DIR = Path(__file__).parent.parent / "data" / "human_notes"

# Teams with heaviest coaching/scheme disruption (new HC + new OC / first-time playcaller)
# From coaching_changes_2026.yaml — these lean UNDER early season
HIGH_DISRUPTION = {"MIA", "WAS", "NYG", "BAL", "NYJ", "LV", "TEN"}

# Teams whose offense should be smooth (continuity or elite scheme) — slight OVER tolerance
CONTINUITY_OFFENSE = {"DET", "CIN", "BUF", "PHI", "LAR", "SF", "GB", "KC"}

# Full team-name -> abbreviation
NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


# Teams missing a KEY offensive player for Week 1 (from injury intel) -> UNDER lean
# Kamara (NO) out, Tyson (NO) out, key WRs/RBs hurt
INJURY_UNDER_TEAMS = {
    "NO": "Kamara (MCL) + Tyson (hamstring) both out — offense depleted",
    "WAS": "LT Laremy Tunsil (triceps) out — pass protection compromised",
    "LAC": "C Tyler Biadasz (ACL) out — interior line concern",
}


def get_dome_teams():
    data = load_stadiums()
    stadiums = data.get("stadiums", {})
    team_map = data.get("team_stadium_map", {})
    domes = set()
    for team, key in team_map.items():
        if stadiums.get(key, {}).get("roof") in ("dome", "retractable"):
            domes.add(team)
    return domes


# 2026 new head coaches (source: coaching_changes_2026.yaml) — 69% under historically
NEW_HC_2026 = {"BAL", "BUF", "LV", "MIA", "PIT", "TEN", "NYJ", "NYG"}


def analyze_game(home, away, total_line, dome_teams, is_division=False,
                 abs_spread=0, home_favored=False, kickoff_hour_et=None, temp=None):
    """
    CONSENSUS model: every layer with a historical Week 1 signal votes
    UNDER (-1), OVER (+1), or neutral (0). Conviction = how many layers agree.
    Each vote is tagged with its historical hit rate.
    """
    home_ab = NAME_TO_ABBR.get(home, home)
    away_ab = NAME_TO_ABBR.get(away, away)

    votes = []  # (layer_name, direction, note)

    # 1. Total line bucket
    if total_line <= 42:
        votes.append(("total_line", 0, f"low total {total_line} — no edge"))
    elif total_line <= 47:
        votes.append(("total_line", -1, f"mid total {total_line} (67% U)"))
    elif total_line <= 49.5:
        votes.append(("total_line", -1, f"high total {total_line} (73% U)"))
    else:
        votes.append(("total_line", +1, f"very high total {total_line} (55% O)"))

    # 2. Roof (outdoors 67% U; dome/closed ~56% no edge)
    if home_ab in dome_teams:
        votes.append(("roof", 0, "indoor (no edge, 56%)"))
    else:
        votes.append(("roof", -1, "outdoors (67% U)"))

    # 3. Division (70% U)
    if is_division:
        votes.append(("division", -1, "division game (70% U)"))
    else:
        votes.append(("division", 0, "non-division"))

    # 4. Spread size (moderate 3.5-6.5 = 67%, big 7+ = 70%; close <=3 no edge)
    if abs_spread >= 7:
        votes.append(("spread", -1, f"big fav {abs_spread} (70% U)"))
    elif abs_spread > 3:
        votes.append(("spread", -1, f"moderate spread {abs_spread} (67% U)"))
    else:
        votes.append(("spread", 0, f"close spread {abs_spread} (no edge)"))

    # 5. New head coach (69% U)
    new_hcs = [t for t in (home_ab, away_ab) if t in NEW_HC_2026]
    if new_hcs:
        votes.append(("new_coach", -1, f"new HC: {','.join(new_hcs)} (69% U)"))
    else:
        votes.append(("new_coach", 0, "no coaching change"))

    # 6. Kickoff slot (early 1pm = 71% U; primetime = 47% = OVER lean)
    if kickoff_hour_et is not None:
        if kickoff_hour_et <= 13:
            votes.append(("slot", -1, "1pm ET kickoff (71% U)"))
        elif kickoff_hour_et >= 18:
            votes.append(("slot", +1, "primetime (47% U = over lean)"))
        else:
            votes.append(("slot", 0, "late afternoon (no edge)"))

    # 7. Home/away favorite (home fav 69% U; away fav 56% no edge)
    if home_favored:
        votes.append(("favorite", -1, "home favorite (69% U)"))
    else:
        votes.append(("favorite", 0, "away favorite (56%, weak)"))

    # 8. Hot weather (80F+, 69% U) — only if we have a forecast
    if temp is not None and temp >= 80 and home_ab not in dome_teams:
        votes.append(("weather", -1, f"hot {int(temp)}F (69% U)"))

    # 9. Key injuries (not a historical layer but material)
    for t in (home_ab, away_ab):
        if t in INJURY_UNDER_TEAMS:
            votes.append(("injury", -1, INJURY_UNDER_TEAMS[t]))

    # Tally
    under_votes = sum(1 for _, d, _ in votes if d == -1)
    over_votes = sum(1 for _, d, _ in votes if d == +1)
    net = over_votes - under_votes  # negative = under

    if under_votes > 0 and over_votes == 0:
        pick = "UNDER"
    elif over_votes > 0 and under_votes == 0:
        pick = "OVER"
    elif under_votes > over_votes:
        pick = "UNDER (mixed)"
    elif over_votes > under_votes:
        pick = "OVER (mixed)"
    else:
        pick = "PASS"

    # Conviction = agreement strength
    agree = max(under_votes, over_votes)
    conflict = min(under_votes, over_votes)
    if conflict == 0 and agree >= 5:
        conf = "STRONG"
    elif conflict == 0 and agree >= 3:
        conf = "LEAN"
    elif agree - conflict >= 3:
        conf = "LEAN"
    elif agree - conflict >= 2:
        conf = "SLIGHT LEAN"
    else:
        conf = "PASS / COIN FLIP"

    summary = f"{under_votes}U/{over_votes}O layers | " + "; ".join(n for _, _, n in votes)
    return pick, conf, net, summary


def main(target_date="2026-09-13"):
    print("=" * 78)
    print(f"GAME TOTALS ANALYSIS — target slate: {target_date}")
    print("=" * 78)

    dome_teams = get_dome_teams()

    # Pull all game odds with totals
    odds = pull_game_odds(markets="h2h,spreads,totals")
    if odds.empty:
        print("No games/odds available.")
        return

    # Filter to target date's games
    odds["date"] = pd.to_datetime(odds["commence_time"]).dt.date.astype(str)
    slate = odds[odds["date"] == target_date]
    if slate.empty:
        print(f"No games found for {target_date}. Available dates:")
        print("  " + ", ".join(sorted(odds["date"].unique())))
        print("\nRunning on ALL available games instead:")
        slate = odds

    # Get totals (Over rows carry the line) and spreads for context
    totals = slate[(slate["market"] == "totals") & (slate["outcome_name"] == "Over")]
    spreads = slate[slate["market"] == "spreads"]
    # Map game_id -> abs spread
    spread_map = {}
    for _, s in spreads.iterrows():
        pt = s.get("outcome_point")
        if pt is not None:
            gid = s["game_id"]
            spread_map[gid] = max(spread_map.get(gid, 0), abs(pt))

    # Division lookup
    divisions = {
        "AFCE": {"BUF", "MIA", "NE", "NYJ"}, "AFCN": {"BAL", "CIN", "CLE", "PIT"},
        "AFCS": {"HOU", "IND", "JAX", "TEN"}, "AFCW": {"DEN", "KC", "LAC", "LV"},
        "NFCE": {"DAL", "NYG", "PHI", "WAS"}, "NFCN": {"CHI", "DET", "GB", "MIN"},
        "NFCS": {"ATL", "CAR", "NO", "TB"}, "NFCW": {"ARI", "LAR", "SEA", "SF"},
    }
    def same_div(a, b):
        for teams in divisions.values():
            if a in teams and b in teams:
                return True
        return False

    # Home-favorite lookup: spread point on the home team's row (negative = favored)
    home_fav_map = {}
    kickoff_map = {}
    for _, s in spreads.iterrows():
        if s.get("outcome_name") == s.get("home_team"):
            pt = s.get("outcome_point")
            if pt is not None:
                home_fav_map[s["game_id"]] = pt < 0  # negative spread = favored
    for _, row in totals.iterrows():
        ct = row.get("commence_time")
        if ct is not None:
            # commence_time is UTC; ET = UTC-4 (Sept, EDT)
            hour_et = (pd.to_datetime(ct).hour - 4) % 24
            kickoff_map[row["game_id"]] = hour_et

    results = []
    for _, row in totals.iterrows():
        home, away = row["home_team"], row["away_team"]
        line = row.get("outcome_point")
        if line is None:
            continue
        home_ab = NAME_TO_ABBR.get(home, home)
        away_ab = NAME_TO_ABBR.get(away, away)
        is_div = same_div(home_ab, away_ab)
        abs_spread = spread_map.get(row["game_id"], 0)
        home_favored = home_fav_map.get(row["game_id"], False)
        kickoff_hour = kickoff_map.get(row["game_id"])
        pick, conf, score, reason = analyze_game(
            home, away, line, dome_teams,
            is_division=is_div, abs_spread=abs_spread,
            home_favored=home_favored, kickoff_hour_et=kickoff_hour, temp=None,
        )
        results.append({
            "matchup": f"{away_ab} @ {home_ab}",
            "total": line, "pick": pick, "confidence": conf,
            "score": score, "reason": reason,
        })

    if not results:
        print("No totals lines posted yet for this slate.")
        return

    df = pd.DataFrame(results).sort_values("score", key=abs, ascending=False)

    # Output
    print(f"\n{len(df)} games analyzed. Ranked by layer consensus:\n")
    print(f"{'Matchup':<14} {'Total':>6} {'Pick':<14} {'Conf':<10} Layer tally")
    print("-" * 90)
    for _, r in df.iterrows():
        tally = r["reason"].split(" | ")[0]  # the "XU/YO layers" part
        print(f"{r['matchup']:<14} {r['total']:>6.1f} {r['pick']:<14} {r['confidence']:<10} {tally}")

    # Detail of layers for the strongest plays
    print("\n" + "-" * 90)
    print("LAYER DETAIL (top plays):")
    for _, r in df.head(6).iterrows():
        print(f"\n  {r['matchup']} — {r['pick']} {r['total']} [{r['confidence']}]")
        layers = r["reason"].split(" | ", 1)[1].split("; ")
        for lyr in layers:
            print(f"     - {lyr}")

    # Parlay math reality check
    n = len(df)
    print("\n" + "=" * 78)
    print("PARLAY REALITY CHECK")
    print("=" * 78)
    print(f"  Legs: {n}")
    for hit in (0.50, 0.53, 0.55):
        prob = hit ** n
        print(f"  If each leg hits {hit:.0%}: full parlay hits {prob:.4%}  (~1 in {int(1/prob) if prob>0 else 0:,})")
    print("\n  NOTE: A full-slate totals parlay is a lottery ticket, not a +EV bet.")
    print("  The math compounds against you fast. Consider a smaller 3-4 leg parlay")
    print("  of only the strongest leans, or bet them straight.")

    # Save
    out = Path(__file__).parent.parent / "data" / "processed" / "game_totals_latest.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved to {out}")
    return df


if __name__ == "__main__":
    tgt = sys.argv[1] if len(sys.argv) > 1 else "2026-09-13"
    main(tgt)
