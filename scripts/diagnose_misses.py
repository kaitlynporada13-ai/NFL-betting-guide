"""
Diagnose WHY the bottoms-up projection missed on 2024 & 2025 Week 1 games.
Decomposes each game to team level: projected pts vs actual pts per team,
flags the bigger deviation, and tags coaching changes (stale-profile risk).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from bottoms_up_totals import team_week_offense, season_profiles, fit_points_model, project_points
RAW = Path(__file__).parent.parent / "data" / "raw"


def coach_changes():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    rows = []
    for _, r in g.iterrows():
        rows.append({"team": r["home_team"], "season": r["season"], "coach": r.get("home_coach")})
        rows.append({"team": r["away_team"], "season": r["season"], "coach": r.get("away_coach")})
    tc = pd.DataFrame(rows).dropna(subset=["coach"])
    tsc = tc.groupby(["team", "season"])["coach"].agg(lambda x: x.mode().iloc[0])
    changed = {}
    for (team, season), coach in tsc.items():
        prev = tsc.get((team, season - 1))
        changed[(team, season)] = (prev is not None and prev != coach)
    return changed


def main():
    tw = team_week_offense()
    coef = fit_points_model(tw)
    cc = coach_changes()
    g = pd.read_parquet(RAW / "games_historical.parquet")

    for season in [2024, 2025]:
        prior = season - 1
        off, deff, league = season_profiles(tw, prior)
        wk1 = g[(g["week"] == 1) & (g["season"] == season) & g["home_score"].notna()]

        print("=" * 96)
        print(f"{season} WEEK 1 — team-level projection vs actual (profiles from {prior})")
        print("=" * 96)
        print(f"{'Matchup':<12}{'Line':>5}{'Proj':>6}{'Act':>5}{'Miss?':>6}  "
              f"{'AwayProj>Act':>16}{'HomeProj>Act':>16}  Culprit")
        print("-" * 96)

        rows = []
        for _, r in wk1.iterrows():
            h, a = r["home_team"], r["away_team"]
            ph = project_points(h, a, off, deff, league, coef)
            pa = project_points(a, h, off, deff, league, coef)
            if ph is None or pa is None:
                continue
            proj = ph + pa
            act_h, act_a = r["home_score"], r["away_score"]
            actual = act_h + act_a
            line = r["total_line"]
            proj_says = "U" if proj < line else "O"
            result = "U" if actual < line else "O" if actual > line else "P"
            miss = "MISS" if proj_says != result and result != "P" else "ok"

            a_err = pa - act_a   # + = over-projected away
            h_err = ph - act_h
            # culprit = team with bigger absolute error, tag coach change
            if abs(a_err) >= abs(h_err):
                culp_team, culp_err = a, a_err
            else:
                culp_team, culp_err = h, h_err
            nc = " (NEW HC)" if cc.get((culp_team, season)) else ""
            culprit = f"{culp_team} proj{'+' if culp_err>0 else ''}{culp_err:.0f}{nc}"

            print(f"{a+'@'+h:<12}{line:>5.0f}{proj:>6.1f}{actual:>5.0f}{miss:>6}  "
                  f"{pa:>6.1f}/{act_a:<3.0f}{'':>4}{ph:>6.1f}/{act_h:<3.0f}{'':>4}  {culprit}")
            rows.append({"miss": miss == "MISS", "err": actual - proj,
                         "culp_team": culp_team, "culp_new_hc": bool(cc.get((culp_team, season)))})

        d = pd.DataFrame(rows)
        misses = d[d["miss"]]
        print(f"\n  {season} misses: {len(misses)}/{len(d)}")
        print(f"  Avg projection error (actual - proj): {d['err'].mean():+.1f} "
              f"(+ = model under-projected scoring)")
        print(f"  Misses where the culprit team had a NEW HEAD COACH: "
              f"{misses['culp_new_hc'].sum()}/{len(misses)}")
        print()


if __name__ == "__main__":
    main()
