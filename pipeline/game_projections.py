"""
GAME TOTALS + SPREADS projection engine.
Produces weekly pick tables in the same shape as the prop engine:
  matchup, line, call, confidence, why  -> saved for the dashboard pages.

Grounded ONLY in out-of-sample validated edges:
  - TOTALS: Week 1 games lean UNDER (flat, ~59% OOS). No in-season or matchup edge
    survived testing, so weeks 2+ are PASS.
  - SPREADS: home underdog vs a road favorite of 7+ points (~60-67% ATS OOS).
    Nothing else survived, so all other games are PASS.
"""
import pandas as pd
from pathlib import Path
from datetime import date

from pipeline.config_loader import get_data_dir
from pipeline.ingest_odds import pull_game_odds

PROC = get_data_dir("processed")

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


def current_week():
    ss = date(2026, 9, 10)
    t = date.today()
    if t < ss:
        return 1 if (ss - t).days <= 28 else 0
    return min(max(1, (t - ss).days // 7 + 1), 22)


def build_totals(week):
    odds = pull_game_odds(markets="totals")
    if odds.empty:
        return pd.DataFrame()
    tot = odds[(odds["market"] == "totals") & (odds["outcome_name"] == "Over")]
    rows = []
    for _, r in tot.iterrows():
        away = NAME_TO_ABBR.get(r["away_team"], r["away_team"])
        home = NAME_TO_ABBR.get(r["home_team"], r["home_team"])
        line = r.get("outcome_point")
        if line is None:
            continue
        if week == 1:
            call, conf = "UNDER", "MEDIUM"
            why = ("Week 1 games lean under (~59% out-of-sample) — offenses underperform in "
                   "openers. Flat bet; total level/matchup did NOT add validated edge.")
        else:
            call, conf = "PASS", "PASS"
            why = ("No validated totals edge in Weeks 2+ — the closing market is efficient and "
                   "the under/over bias is unstable year to year. Pass unless late injury/weather news.")
        rows.append({"matchup": f"{away} @ {home}", "line": line,
                     "call": call, "confidence": conf, "why": why})
    df = pd.DataFrame(rows)
    df.to_parquet(PROC / "totals_picks_latest.parquet", index=False)
    return df


def build_spreads(week):
    odds = pull_game_odds(markets="spreads")
    if odds.empty:
        return pd.DataFrame()
    spreads = odds[odds["market"] == "spreads"]
    rows = []
    for gid, grp in spreads.groupby("game_id"):
        home = grp["home_team"].iloc[0]
        away = grp["away_team"].iloc[0]
        hrow = grp[grp["outcome_name"] == home]
        if hrow.empty:
            continue
        home_pt = hrow["outcome_point"].iloc[0]
        h, a = NAME_TO_ABBR.get(home, home), NAME_TO_ABBR.get(away, away)
        # Validated edge: road favorite 7+ -> bet home underdog
        if home_pt >= 7:
            call = f"{h} +{home_pt:.1f}"
            conf = "MEDIUM-HIGH"
            why = (f"Home underdog vs a road favorite of {home_pt:.0f}+ — validated ATS edge "
                   f"(~60-67% out-of-sample). Public overvalues strong road teams; home dogs cover.")
        else:
            call = "PASS"
            conf = "PASS"
            why = ("No validated ATS edge — only home dogs vs 7+ road favorites survived testing. "
                   "This game doesn't qualify; the spread market is otherwise efficient.")
        rows.append({"matchup": f"{a} @ {h}", "line": f"home {home_pt:+.1f}",
                     "call": call, "confidence": conf, "why": why})
    df = pd.DataFrame(rows)
    df.to_parquet(PROC / "spreads_picks_latest.parquet", index=False)
    return df


def main():
    week = current_week()
    print(f"Building game totals + spreads for Week {week}...")
    t = build_totals(week)
    s = build_spreads(week)
    print(f"  Totals: {len(t)} games | Spreads: {len(s)} games")
    if not s.empty:
        quals = s[s["confidence"] != "PASS"]
        print(f"  Qualifying spread plays: {len(quals)}")
        for _, r in quals.iterrows():
            print(f"    {r['call']} ({r['matchup']})")


if __name__ == "__main__":
    main()
