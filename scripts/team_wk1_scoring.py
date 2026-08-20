"""
Part 1: Each team's Week 1 scoring vs the line, last 2 years (2024-2025).
Part 2: Project each 2026 Sept 13 game's total from those team averages (CURIOSITY ONLY —
        team scoring history was shown NOT to predict Week 1 totals).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"

# 2026 Week 1 (Sept 13) slate: away, home, current total line
SLATE = [
    ("NYJ", "TEN", 39.5), ("MIA", "LV", 40.5), ("CLE", "JAX", 40.5),
    ("ATL", "PIT", 41.5), ("GB", "MIN", 44.5), ("BUF", "HOU", 44.5),
    ("WAS", "PHI", 46.5), ("ARI", "LAC", 46.5), ("CHI", "CAR", 47.5),
    ("DAL", "NYG", 48.5), ("BAL", "IND", 48.5), ("NO", "DET", 49.5),
    ("TB", "CIN", 51.5),
]


def team_wk1_rows():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[(g["week"] == 1) & (g["season"] >= 2024) & g["home_score"].notna()].copy()
    rows = []
    for _, r in g.iterrows():
        tot = r["home_score"] + r["away_score"]
        line = r["total_line"]
        ou = "U" if tot < line else "O" if tot > line else "P"
        rows.append({"team": r["home_team"], "season": r["season"], "opp": r["away_team"],
                     "pf": r["home_score"], "pa": r["away_score"], "total": tot,
                     "line": line, "ou": ou, "loc": "vs"})
        rows.append({"team": r["away_team"], "season": r["season"], "opp": r["home_team"],
                     "pf": r["away_score"], "pa": r["home_score"], "total": tot,
                     "line": line, "ou": ou, "loc": "@"})
    return pd.DataFrame(rows)


def main():
    df = team_wk1_rows()

    # PART 1: team Week 1 scoring table
    print("=" * 80)
    print("PART 1 — EACH TEAM'S WEEK 1 SCORING vs LINE (2024 & 2025)")
    print("=" * 80)
    teams_in_slate = sorted({t for g in SLATE for t in g[:2]})
    print(f"\n{'Team':<5} {'Yr':<6} {'Loc':<3} {'Opp':<4} {'PF':>3} {'PA':>3} {'Tot':>4} {'Line':>5} {'O/U':>3}")
    print("-" * 80)
    avg_pf = {}
    for t in teams_in_slate:
        sub = df[df["team"] == t].sort_values("season")
        for _, r in sub.iterrows():
            print(f"{t:<5} {int(r['season']):<6} {r['loc']:<3} {r['opp']:<4} "
                  f"{int(r['pf']):>3} {int(r['pa']):>3} {int(r['total']):>4} "
                  f"{r['line']:>5.1f} {r['ou']:>3}")
        avg_pf[t] = sub["pf"].mean() if len(sub) else np.nan

    # PART 2: project 2026 slate from team Week 1 scoring averages
    print("\n" + "=" * 80)
    print("PART 2 — 2026 PROJECTION (CURIOSITY): sum of each team's avg Wk1 points")
    print("  NOTE: team scoring history does NOT predict Wk1 totals (validated). For fun only.")
    print("=" * 80)
    print(f"\n{'Matchup':<14} {'Line':>6} {'Proj':>6} {'Gap':>6} {'Lean':<8} {'(away avg + home avg)'}")
    print("-" * 80)
    rows = []
    for away, home, line in SLATE:
        a = avg_pf.get(away, np.nan)
        h = avg_pf.get(home, np.nan)
        proj = (a + h) if not (np.isnan(a) or np.isnan(h)) else np.nan
        gap = (line - proj) if not np.isnan(proj) else np.nan
        lean = ("UNDER" if gap > 2 else "OVER" if gap < -2 else "~line") if not np.isnan(gap) else "n/a"
        av = f"({a:.0f} + {h:.0f})" if not np.isnan(proj) else "(missing)"
        print(f"{away+' @ '+home:<14} {line:>6.1f} {proj:>6.1f} {gap:>+6.1f} {lean:<8} {av}")
        rows.append({"matchup": f"{away} @ {home}", "line": line,
                     "projected": round(proj, 1) if not np.isnan(proj) else None,
                     "gap": round(gap, 1) if not np.isnan(gap) else None, "lean": lean})

    out = Path(__file__).parent.parent / "data" / "processed" / "wk1_team_projection.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nSaved projection to {out}")
    print("\nReminder: projection uses only 2 games/team and ignores 2026 roster/coach changes.")


if __name__ == "__main__":
    main()
