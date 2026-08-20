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


# 2026 new head coaches (source: coaching_changes_2026.yaml)
NEW_HC_2026 = {"BAL", "BUF", "LV", "MIA", "PIT", "TEN", "NYJ", "NYG"}

# Load data-driven layer weights (computed by analyze_week1_layers.py)
def load_weights():
    import yaml
    path = Path(__file__).parent.parent / "config" / "week1_totals_weights.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f).get("layer_weights", {})


def analyze_game(home, away, total_line, dome_teams, weights, is_division=False,
                 abs_spread=0, home_favored=False, kickoff_hour_et=None, temp=None):
    """
    WEIGHTED model: each layer contributes its historical lift-over-baseline
    (computed from 2021-2025 data). Positive weight = UNDER push, negative = OVER.
    Net score = sum of applicable weights. This lets below-baseline layers
    (primetime, indoor, very-high total, away fav) push a game toward OVER.
    """
    home_ab = NAME_TO_ABBR.get(home, home)
    away_ab = NAME_TO_ABBR.get(away, away)

    def w(layer, value):
        return weights.get(layer, {}).get(value, 0.0)

    contribs = []  # (layer_value, weight, note)

    # 1. Total line bucket
    if total_line <= 42:
        val = "low"
    elif total_line <= 47:
        val = "mid"
    elif total_line <= 49.5:
        val = "high"
    else:
        val = "very_high"
    contribs.append((f"total={total_line}", w("total_line", val), f"total {total_line} ({val})"))

    # 2. Roof
    roof_val = "indoor" if home_ab in dome_teams else "outdoors"
    contribs.append(("roof", w("roof", roof_val), roof_val))

    # 3. Division
    div_val = "yes" if is_division else "no"
    contribs.append(("division", w("division", div_val), f"division={div_val}"))

    # 4. Spread size
    if abs_spread >= 7:
        sp_val = "big"
    elif abs_spread > 3:
        sp_val = "moderate"
    else:
        sp_val = "close"
    contribs.append(("spread", w("spread", sp_val), f"spread {abs_spread} ({sp_val})"))

    # 5. New head coach
    new_hcs = [t for t in (home_ab, away_ab) if t in NEW_HC_2026]
    nc_val = "yes" if new_hcs else "no"
    note = f"new HC: {','.join(new_hcs)}" if new_hcs else "no coaching change"
    contribs.append(("new_coach", w("new_coach", nc_val), note))

    # 6. Kickoff slot
    if kickoff_hour_et is not None:
        if kickoff_hour_et <= 13:
            slot_val = "early"
        elif kickoff_hour_et >= 18:
            slot_val = "primetime"
        else:
            slot_val = "afternoon"
        contribs.append(("slot", w("slot", slot_val), f"{slot_val} kickoff"))

    # 7. Favorite location
    fav_val = "home" if home_favored else "away"
    contribs.append(("favorite", w("favorite", fav_val), f"{fav_val} favorite"))

    # 8. Weather (outdoor only)
    if temp is not None and home_ab not in dome_teams:
        wx = "hot" if temp >= 80 else "mild" if temp < 70 else None
        if wx:
            contribs.append(("weather", w("weather", wx), f"{wx} ({int(temp)}F)"))

    # Sum weights
    net = sum(c[1] for c in contribs)

    # Injuries: material adjustment on top (not in historical weights)
    inj_notes = []
    for t in (home_ab, away_ab):
        if t in INJURY_UNDER_TEAMS:
            net += 4.0  # push under
            inj_notes.append(INJURY_UNDER_TEAMS[t])

    # Decide pick + conviction from net score
    if net >= 15:
        pick, conf = "UNDER", "STRONG"
    elif net >= 8:
        pick, conf = "UNDER", "LEAN"
    elif net >= 4:
        pick, conf = "UNDER", "SLIGHT LEAN"
    elif net <= -15:
        pick, conf = "OVER", "STRONG"
    elif net <= -8:
        pick, conf = "OVER", "LEAN"
    elif net <= -4:
        pick, conf = "OVER", "SLIGHT LEAN"
    else:
        pick, conf = "PASS", "COIN FLIP"

    # Build detail sorted by absolute contribution
    parts = sorted(contribs, key=lambda c: abs(c[1]), reverse=True)
    detail = "; ".join(f"{note} {wt:+.0f}" for _, wt, note in parts)
    if inj_notes:
        detail += " | INJ: " + "; ".join(inj_notes) + " +4"
    summary = f"net {net:+.0f} | " + detail
    return pick, conf, net, summary


def main(target_date="2026-09-13"):
    print("=" * 78)
    print(f"GAME TOTALS ANALYSIS — target slate: {target_date}")
    print("=" * 78)

    dome_teams = get_dome_teams()
    weights = load_weights()
    if not weights:
        print("WARNING: no weights file found. Run analyze_week1_layers.py first.")
        return

    # Pull all game odds with totals
    odds = pull_game_odds(markets="h2h,spreads,totals")
    if odds.empty:
        print("No games/odds available.")
        return

    # Filter to target date's games — convert UTC to ET first so late Sunday
    # night games (8:20pm ET = 00:20 UTC next day) stay on the correct date.
    ct = pd.to_datetime(odds["commence_time"], utc=True, errors="coerce")
    odds["date"] = (ct - pd.Timedelta(hours=4)).dt.date.astype(str)  # EDT in September
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
            home, away, line, dome_teams, weights,
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

    df = pd.DataFrame(results).sort_values("score", ascending=False)  # + = under, - = over

    # Output — unders at top, overs at bottom, passes in middle
    print(f"\n{len(df)} games analyzed. Ranked: strongest UNDER (top) -> strongest OVER (bottom)\n")
    print(f"{'Matchup':<14} {'Total':>6} {'Pick':<8} {'Conf':<12} {'Net':>5}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['matchup']:<14} {r['total']:>6.1f} {r['pick']:<8} {r['confidence']:<12} {r['score']:>+5.0f}")

    # Detail: strongest unders and strongest overs
    print("\n" + "-" * 90)
    print("LAYER DETAIL (strongest leans, both directions):")
    strong = pd.concat([df.head(4), df.tail(3)]).drop_duplicates(subset=["matchup"])
    for _, r in strong.iterrows():
        print(f"\n  {r['matchup']} — {r['pick']} {r['total']} [{r['confidence']}]")
        detail = r["reason"].split(" | ", 1)[1]
        for part in detail.split("; "):
            print(f"     - {part}")

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
