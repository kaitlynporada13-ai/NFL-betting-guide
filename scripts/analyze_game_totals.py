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


def analyze_game(home, away, total_line, dome_teams, is_division=False, abs_spread=0, temp=None):
    """
    Return (pick, confidence, score, reason) for a single game total.
    Scoring is grounded in historical Week 1 totals data (2021-2025, 80 games).
    Positive score = OVER lean, negative = UNDER lean.
    """
    home_ab = NAME_TO_ABBR.get(home, home)
    away_ab = NAME_TO_ABBR.get(away, away)

    score = 0
    reasons = []

    # --- PRIMARY: total-line bucket (the strongest historical signal) ---
    if total_line <= 42:
        # Low totals: NO under edge (historically +2.2 margin, ~57% but goes over)
        reasons.append(f"low total ({total_line}) — no historical Week 1 edge, PASS")
        # score stays neutral
    elif total_line <= 47:
        score -= 2
        reasons.append(f"mid total ({total_line}) — 67% under historically")
    elif total_line <= 49.5:
        score -= 3
        reasons.append(f"high total ({total_line}) — 73% under (sweet spot)")
    else:  # 50+
        score += 2
        reasons.append(f"very high total ({total_line}) — 55% OVER historically (+3.9 margin)")

    # --- Division games: 70% under ---
    if is_division:
        score -= 1
        reasons.append("division game (70% under)")

    # --- Big favorite (7+): 70% under (blowout, starters pulled) ---
    if abs_spread >= 7:
        score -= 1
        reasons.append(f"big spread ({abs_spread}) — blowout risk, 70% under")

    # --- Hot weather (80F+): 69% under ---
    if temp is not None and temp >= 80:
        score -= 1
        reasons.append(f"hot ({int(temp)}F) — 69% under")

    # --- Coaching disruption (secondary) ---
    disrupted = [t for t in (home_ab, away_ab) if t in HIGH_DISRUPTION]
    if len(disrupted) == 2:
        score -= 1
        reasons.append("both offenses disrupted (new scheme)")

    # --- Key injuries ---
    for t in (home_ab, away_ab):
        if t in INJURY_UNDER_TEAMS:
            score -= 1
            reasons.append(INJURY_UNDER_TEAMS[t])

    # Decide pick
    if score <= -1:
        pick = "UNDER"
    elif score >= 1:
        pick = "OVER"
    else:
        pick = "PASS"

    strength = abs(score)
    if strength >= 4:
        conf = "STRONG LEAN"
    elif strength >= 3:
        conf = "LEAN"
    elif strength >= 2:
        conf = "SLIGHT LEAN"
    else:
        conf = "COIN FLIP / PASS"

    return pick, conf, score, "; ".join(reasons)


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
        pick, conf, score, reason = analyze_game(
            home, away, line, dome_teams,
            is_division=is_div, abs_spread=abs_spread, temp=None,
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
    print(f"\n{len(df)} games analyzed. Picks ranked by conviction:\n")
    print(f"{'Matchup':<14} {'Total':>6} {'Pick':<11} {'Conf':<12} Reason")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{r['matchup']:<14} {r['total']:>6.1f} {r['pick']:<11} {r['confidence']:<12} {r['reason'][:60]}")

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
