"""
BOTTOMS-UP TOTALS PROJECTION.
Build a game total from the ground up:
  1. Each team's offense by type (pass yds, rush yds, pass TDs, rush TDs) - full prior season.
  2. Each opponent's defense allowed by type - full prior season.
  3. Project each team's output = blend(team offense, opponent defense) vs league average.
  4. Convert projected yards + TDs -> points via a regression calibrated on real team-games.
  5. Sum -> projected game total. Compare to the line.

Then VALIDATE on historical Week 1 (2022-2025): does the projection beat the closing line?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"


def team_week_offense():
    """Per team-week offensive output from player game logs, with opponent + points scored."""
    ps = pd.read_parquet(RAW / "player_stats_historical.parquet")
    grp = ps.groupby(["recent_team", "season", "week"]).agg(
        pass_yds=("passing_yards", "sum"),
        rush_yds=("rushing_yards", "sum"),
        pass_td=("passing_tds", "sum"),
        rush_td=("rushing_tds", "sum"),
    ).reset_index().rename(columns={"recent_team": "team"})

    # Attach opponent + points scored from schedule
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[g["home_score"].notna()]
    home = g[["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["season", "week", "team", "opp", "pts", "opp_pts"]
    away = g[["season", "week", "away_team", "home_team", "away_score", "home_score"]].copy()
    away.columns = ["season", "week", "team", "opp", "pts", "opp_pts"]
    sched = pd.concat([home, away], ignore_index=True)

    tw = grp.merge(sched, on=["season", "week", "team"], how="inner")
    return tw


def season_profiles(tw, season):
    """Team offense (per game) and defense (allowed per game) for a season, + league avgs."""
    s = tw[tw["season"] == season]
    off = s.groupby("team").agg(
        off_pass_yds=("pass_yds", "mean"), off_rush_yds=("rush_yds", "mean"),
        off_pass_td=("pass_td", "mean"), off_rush_td=("rush_td", "mean"),
    )
    # Defense allowed = what opponents did vs this team (group by opp)
    deff = s.groupby("opp").agg(
        def_pass_yds=("pass_yds", "mean"), def_rush_yds=("rush_yds", "mean"),
        def_pass_td=("pass_td", "mean"), def_rush_td=("rush_td", "mean"),
    )
    deff.index.name = "team"
    league = {
        "pass_yds": s["pass_yds"].mean(), "rush_yds": s["rush_yds"].mean(),
        "pass_td": s["pass_td"].mean(), "rush_td": s["rush_td"].mean(),
    }
    return off, deff, league


def fit_points_model(tw):
    """Regress actual points on offensive inputs. Returns coefficients."""
    d = tw.dropna(subset=["pts"])
    X = np.column_stack([
        np.ones(len(d)), d["pass_td"], d["rush_td"], d["pass_yds"], d["rush_yds"],
    ])
    y = d["pts"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef  # [intercept, pass_td, rush_td, pass_yds, rush_yds]


def project_points(team, opp, off, deff, league, coef):
    """Projected points for `team` facing `opp`, blending offense vs opponent defense."""
    if team not in off.index or opp not in deff.index:
        return None
    o = off.loc[team]
    d = deff.loc[opp]
    # Blend: team offense adjusted by how opponent's D compares to league avg
    def blend(off_val, def_val, lg):
        # projected = offense * (opponent defense / league avg)  [matchup adjustment]
        factor = def_val / lg if lg > 0 else 1.0
        return off_val * factor
    p_pass_yds = blend(o["off_pass_yds"], d["def_pass_yds"], league["pass_yds"])
    p_rush_yds = blend(o["off_rush_yds"], d["def_rush_yds"], league["rush_yds"])
    p_pass_td = blend(o["off_pass_td"], d["def_pass_td"], league["pass_td"])
    p_rush_td = blend(o["off_rush_td"], d["def_rush_td"], league["rush_td"])
    pts = (coef[0] + coef[1]*p_pass_td + coef[2]*p_rush_td
           + coef[3]*p_pass_yds + coef[4]*p_rush_yds)
    return pts


SLATE = [
    ("NYJ", "TEN", 39.5), ("MIA", "LV", 40.5), ("CLE", "JAX", 40.5),
    ("ATL", "PIT", 41.5), ("GB", "MIN", 44.5), ("BUF", "HOU", 44.5),
    ("WAS", "PHI", 46.5), ("ARI", "LAC", 46.5), ("CHI", "CAR", 47.5),
    ("DAL", "NYG", 48.5), ("BAL", "IND", 48.5), ("NO", "DET", 49.5),
    ("TB", "CIN", 51.5),
]


def validate_week1(tw):
    """For each Week 1 game 2022-2025, project using PRIOR season profiles; grade vs line."""
    coef = fit_points_model(tw)
    g = pd.read_parquet(RAW / "games_historical.parquet")
    wk1 = g[(g["week"] == 1) & g["total_line"].notna() & g["home_score"].notna()]

    rows = []
    for _, r in wk1.iterrows():
        prior = r["season"] - 1
        off, deff, league = season_profiles(tw, prior)
        ph = project_points(r["home_team"], r["away_team"], off, deff, league, coef)
        pa = project_points(r["away_team"], r["home_team"], off, deff, league, coef)
        if ph is None or pa is None:
            continue
        proj_total = ph + pa
        actual = r["home_score"] + r["away_score"]
        if actual == r["total_line"]:
            continue
        rows.append({
            "season": r["season"], "matchup": f"{r['away_team']}@{r['home_team']}",
            "line": r["total_line"], "proj": round(proj_total, 1), "actual": actual,
            "proj_says": "UNDER" if proj_total < r["total_line"] else "OVER",
            "result": "UNDER" if actual < r["total_line"] else "OVER",
        })
    v = pd.DataFrame(rows)
    v["correct"] = v["proj_says"] == v["result"]
    v["proj_vs_line"] = (v["proj"] - v["line"]).round(1)

    print("=" * 92)
    print(f"VALIDATION DETAIL — every Week 1 game 2022-25: projection vs line vs actual")
    print("=" * 92)
    print(f"{'Season':<7}{'Matchup':<12}{'Line':>6}{'Proj':>7}{'Actual':>8}{'ProjSays':>10}{'Result':>8}{'Hit':>5}")
    print("-" * 92)
    for _, r in v.sort_values(["season", "matchup"]).iterrows():
        hit = "OK" if r["correct"] else "X"
        print(f"{int(r['season']):<7}{r['matchup']:<12}{r['line']:>6.1f}{r['proj']:>7.1f}"
              f"{int(r['actual']):>8}{r['proj_says']:>10}{r['result']:>8}{hit:>5}")

    v.to_csv(Path(__file__).parent.parent / "data" / "processed" / "bottoms_up_validation.csv", index=False)

    print("\n" + "=" * 92)
    print("SCORECARD")
    print("=" * 92)
    print(f"  Projection's over/under call was right: {v['correct'].mean()*100:.1f}%  "
          f"({v['correct'].sum()}/{len(v)} games)")
    print(f"  If you'd just bet UNDER every game:      {(v['result']=='UNDER').mean()*100:.1f}%")
    corr = np.corrcoef(v["proj"], v["actual"])[0, 1]
    print(f"  Correlation(projected total, actual):    {corr:.3f}  (1.0 = perfect, 0 = random)")
    v["edge"] = (v["proj"] - v["line"]).abs()
    strong = v[v["edge"] >= 4]
    if len(strong):
        print(f"  When projection differs from line 4+ pts ({len(strong)} games): "
              f"{strong['correct'].mean()*100:.1f}% right")
    print("  Saved full table: data/processed/bottoms_up_validation.csv")
    return v, coef, tw


def project_2026(tw, coef):
    off, deff, league = season_profiles(tw, 2025)  # prior season profiles
    print("\n" + "=" * 78)
    print("2026 WEEK 1 BOTTOMS-UP PROJECTION (using 2025 profiles) — CURIOSITY")
    print("  Ignores 2026 roster/coach changes. Validation result above tells you if it's trustworthy.")
    print("=" * 78)
    print(f"\n{'Matchup':<14} {'Line':>6} {'Proj':>6} {'Gap':>6} {'Lean':<8}")
    print("-" * 50)
    for away, home, line in SLATE:
        ph = project_points(home, away, off, deff, league, coef)
        pa = project_points(away, home, off, deff, league, coef)
        if ph is None or pa is None:
            print(f"{away+' @ '+home:<14} {line:>6.1f}  (missing team data)")
            continue
        proj = ph + pa
        gap = line - proj
        lean = "UNDER" if gap > 2 else "OVER" if gap < -2 else "~line"
        print(f"{away+' @ '+home:<14} {line:>6.1f} {proj:>6.1f} {gap:>+6.1f} {lean:<8}")


def main():
    tw = team_week_offense()
    v, coef, tw = validate_week1(tw)
    print(f"\nPoints model: pts = {coef[0]:.1f} + {coef[1]:.1f}*passTD + {coef[2]:.1f}*rushTD "
          f"+ {coef[3]:.3f}*passYds + {coef[4]:.3f}*rushYds")
    project_2026(tw, coef)


if __name__ == "__main__":
    main()
