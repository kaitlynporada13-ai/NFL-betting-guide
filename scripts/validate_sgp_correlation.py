"""
VALIDATE correlated same-game parlays (SGP) out-of-sample.

Thesis: in a low-scoring game, EVERYONE underperforms together, so stacking
UNDER legs from the same game should hit as a parlay MORE often than if the
legs were independent. If true -> real SGP edge. If false -> stacking just
compounds the vig.

Method (train 2023-24, test 2025):
  1. Grade every individual UNDER leg in our validated markets.
  2. Group legs by game (event_id).
  3. For 2-leg and 3-leg same-game UNDER combos, compare:
       actual joint hit rate  vs  product of individual leg rates (independence)
     A positive gap = correlation edge. Also test low-total games specifically.
  4. Compare a same-game stack vs a cross-game stack (random legs from
     different games) to isolate the same-game correlation effect.
"""
import sys
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
rng = np.random.default_rng(42)

MARKET_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}

# Full team name -> abbreviation, to join props (full names) to games (abbr).
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def grade_legs():
    """Every individual UNDER leg with its actual result + game total context."""
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    props = props[props["market"].isin(MARKET_STAT) & (props["outcome"] == "Over")].copy()
    props["pname"] = props["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()

    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[nc].str.lower().str.replace(".", "", regex=False).str.strip()

    frames = []
    for market, stat in MARKET_STAT.items():
        pm = props[props["market"] == market]
        s = stats[["pname", "season", "week", stat]].dropna(subset=[stat]).copy()
        m = pm.merge(s, on=["pname", "season", "week"], how="inner")
        m["actual"] = m[stat]
        m = m[m["actual"] != m["line"]]
        m["under_win"] = (m["actual"] < m["line"]).astype(int)
        frames.append(m[["season", "week", "event_id", "home_team", "away_team",
                          "market", "player_name", "line", "under_win"]])
    legs = pd.concat(frames, ignore_index=True)

    # attach game total line
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[["season", "week", "home_team", "away_team", "total_line"]].copy()
    legs["home_abbr"] = legs["home_team"].map(TEAM_ABBR)
    legs["away_abbr"] = legs["away_team"].map(TEAM_ABBR)
    legs = legs.merge(g, left_on=["season", "week", "home_abbr", "away_abbr"],
                      right_on=["season", "week", "home_team", "away_team"],
                      how="left", suffixes=("", "_g"))
    return legs


def joint_vs_independent(pool, n_legs, n_samples=20000, low_total=None):
    """Sample n_leg combos; return actual joint-hit rate vs independence product."""
    # group legs by game
    by_game = {eid: grp for eid, grp in pool.groupby("event_id") if len(grp) >= n_legs}
    if not by_game:
        return None
    overall_leg_rate = pool["under_win"].mean()

    same_hits, cross_hits = [], []
    games = list(by_game.keys())
    for _ in range(n_samples):
        # SAME-GAME combo
        eid = games[rng.integers(len(games))]
        grp = by_game[eid]
        idx = rng.choice(len(grp), size=n_legs, replace=False)
        legs = grp.iloc[idx]
        same_hits.append(int(legs["under_win"].sum() == n_legs))
        # CROSS-GAME combo (legs from n_legs DIFFERENT games) - independence benchmark
        if len(games) >= n_legs:
            eids = rng.choice(len(games), size=n_legs, replace=False)
            picks = []
            for e in eids:
                gg = by_game[games[e]]
                picks.append(gg.iloc[rng.integers(len(gg))]["under_win"])
            cross_hits.append(int(sum(picks) == n_legs))

    same = np.mean(same_hits)
    cross = np.mean(cross_hits) if cross_hits else np.nan
    independent = overall_leg_rate ** n_legs
    return {"leg_rate": overall_leg_rate, "n_games": len(by_game),
            "same_game_joint": same, "cross_game_joint": cross,
            "independent_expect": independent}


def report(pool, label):
    print(f"\n{'='*78}\n{label}  (single-leg under rate = {pool['under_win'].mean():.1%}, "
          f"{pool['event_id'].nunique()} games)\n{'='*78}")
    print(f"  {'Legs':<5}{'SameGame Joint':>16}{'CrossGame Joint':>17}{'Independent':>14}  Correlation?")
    print("  " + "-" * 70)
    for n in (2, 3):
        r = joint_vs_independent(pool, n)
        if r is None:
            print(f"  {n:<5}  (not enough games with {n}+ legs)")
            continue
        # positive corr if same-game beats cross-game (independence)
        gap = r["same_game_joint"] - (r["cross_game_joint"] if not np.isnan(r["cross_game_joint"]) else r["independent_expect"])
        verdict = "POSITIVE (edge)" if gap > 0.02 else "NEGATIVE (trap)" if gap < -0.02 else "~neutral"
        print(f"  {n:<5}{r['same_game_joint']:>15.1%}{r['cross_game_joint']:>17.1%}"
              f"{r['independent_expect']:>14.1%}  {verdict}  (gap {gap:+.1%})")


def main():
    legs = grade_legs()
    legs.to_parquet(PROC / "sgp_graded_legs.parquet", index=False)
    print(f"Graded {len(legs)} individual under legs across "
          f"{legs['event_id'].nunique()} games (2023-25).")

    # Week 1 only (our current edge), all games
    w1 = legs[legs["week"] == 1]
    for split_lbl, sub in [("TRAIN 2023-24 Week 1", w1[w1["season"] <= 2024]),
                           ("TEST 2025 Week 1", w1[w1["season"] == 2025])]:
        if len(sub) >= 30:
            report(sub, split_lbl)

    # Low-total Week 1 games (thesis is strongest here) — combine years for sample
    lt = w1[w1["total_line"] <= 43]
    if len(lt) >= 30:
        report(lt, "WEEK 1 LOW-TOTAL games (total <= 43), all years")

    print("\nInterpretation:")
    print("  SameGame Joint > CrossGame Joint  => same-game unders are POSITIVELY")
    print("  correlated (a real SGP edge: bad games sink everyone together).")
    print("  SameGame < CrossGame => stacking hurts (legs offset; pay vig for nothing).")


if __name__ == "__main__":
    main()
