"""
VALIDATION: Does prior-season scoring profile predict Week 1 over/under?

For each Week 1 game (2022-2025), use the PRIOR season's scoring for both teams
to compute an expected total, compare to the actual Week 1 line, and check whether
the resulting gap actually predicted the over/under result.

If bigger gaps -> higher under hit rate (monotonic), the tiebreaker is trustworthy.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"


def team_season_scoring(games):
    """Per (team, season) offense/defense PPG from all regular-season games."""
    g = games[games["week"] <= 18].copy()
    rows = []
    for _, r in g.iterrows():
        rows.append({"team": r["home_team"], "season": r["season"],
                     "pf": r["home_score"], "pa": r["away_score"]})
        rows.append({"team": r["away_team"], "season": r["season"],
                     "pf": r["away_score"], "pa": r["home_score"]})
    df = pd.DataFrame(rows)
    agg = df.groupby(["team", "season"]).agg(off=("pf", "mean"), deff=("pa", "mean")).reset_index()
    return {(r["team"], r["season"]): (r["off"], r["deff"]) for _, r in agg.iterrows()}


def main():
    games = pd.read_parquet(RAW / "games_historical.parquet")
    scoring = team_season_scoring(games)

    wk1 = games[games["week"] == 1].copy()
    wk1["actual_total"] = wk1["home_score"] + wk1["away_score"]
    wk1 = wk1[wk1["total_line"].notna()]

    rows = []
    for _, r in wk1.iterrows():
        prior = r["season"] - 1
        h, a = r["home_team"], r["away_team"]
        if (h, prior) not in scoring or (a, prior) not in scoring:
            continue
        h_off, h_def = scoring[(h, prior)]
        a_off, a_def = scoring[(a, prior)]
        expected = (h_off + a_def) / 2 + (a_off + h_def) / 2
        gap = r["total_line"] - expected  # + = line above expected -> predict UNDER
        actual_result = "UNDER" if r["actual_total"] < r["total_line"] else \
                        "OVER" if r["actual_total"] > r["total_line"] else "PUSH"
        rows.append({
            "season": r["season"], "matchup": f"{a}@{h}",
            "line": r["total_line"], "expected": round(expected, 1),
            "gap": round(gap, 1), "actual": r["actual_total"], "result": actual_result,
        })

    df = pd.DataFrame(rows)
    df = df[df["result"] != "PUSH"]
    print("=" * 84)
    print(f"VALIDATION: prior-season scoring gap vs Week 1 result ({len(df)} games, 2022-2025)")
    print("=" * 84)

    # Predicted under when gap > 0
    df["predicted"] = np.where(df["gap"] > 0, "UNDER", "OVER")
    df["correct"] = df["predicted"] == df["result"]
    acc = df["correct"].mean() * 100
    print(f"\nOverall directional accuracy (gap sign vs result): {acc:.1f}%  ({df['correct'].sum()}/{len(df)})")
    print("(50% = no predictive value)")

    # Bucket by gap magnitude
    print("\nUnder hit rate by gap bucket (does a bigger gap = more unders?):")
    print(f"  {'Gap bucket':<16} {'n':>3} {'Under%':>7} {'AvgActualMargin':>16}")
    print("  " + "-" * 46)
    bins = [(-99, -5, "line <<exp (<-5)"), (-5, -2, "line < exp (-5..-2)"),
            (-2, 2, "~matches (-2..2)"), (2, 5, "line > exp (2..5)"),
            (5, 99, "line >>exp (5+)")]
    for lo, hi, label in bins:
        sub = df[(df["gap"] > lo) & (df["gap"] <= hi)]
        if len(sub) == 0:
            continue
        u = (sub["result"] == "UNDER").sum()
        under_pct = u / len(sub) * 100
        margin = (sub["actual"] - sub["line"]).mean()
        print(f"  {label:<16} {len(sub):>3} {under_pct:>6.1f}% {margin:>+15.1f}")

    print("\nInterpretation:")
    print("  If 'line > exp' buckets show HIGH under% and 'line < exp' show LOW under%,")
    print("  the scoring-environment gap is predictive and safe to use as a tiebreaker.")

    out = Path(__file__).parent.parent / "data" / "processed" / "scoring_env_validation.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} graded games to {out}")


if __name__ == "__main__":
    main()
