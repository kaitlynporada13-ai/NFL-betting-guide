"""
Team Scoring Environment tiebreaker for game totals.
Uses each team's PRIOR-SEASON (2025) actual scoring to compute an expected total,
then compares to the current line. Independent of the situational layers.

Caveat (per user): only fully reliable when roster/coach/scheme is stable.
We discount teams with a new HC or new starting QB in 2026.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"

# 2026 teams with major change (new HC and/or new starting QB) -> discount reliability
MAJOR_CHANGE_2026 = {
    "MIA": "new HC + QB (Willis)", "LV": "new HC + rookie QB (Mendoza)",
    "NYJ": "new HC + QB (Geno)", "TEN": "new HC + yr2 QB (Ward)",
    "NYG": "new HC + QB battle (Dart)", "MIN": "new QB (Murray)",
    "PIT": "new HC (McCarthy) + QB", "BAL": "new HC (Minter)",
    "CHI": "new HC (Ben Johnson)", "ARI": "new QB (post-Kyler)",
    "NO": "Kamara out + WR injuries",
}


def team_2025_scoring():
    """Per-team 2025 offense/defense PPG from actual results."""
    games = pd.read_parquet(RAW / "games_historical.parquet")
    g = games[(games["season"] == 2025) & (games["week"] <= 18)].copy()
    rows = []
    for _, r in g.iterrows():
        rows.append({"team": r["home_team"], "pf": r["home_score"], "pa": r["away_score"],
                     "gt": r["home_score"] + r["away_score"]})
        rows.append({"team": r["away_team"], "pf": r["away_score"], "pa": r["home_score"],
                     "gt": r["home_score"] + r["away_score"]})
    df = pd.DataFrame(rows)
    agg = df.groupby("team").agg(
        games=("pf", "count"), off_ppg=("pf", "mean"),
        def_ppg=("pa", "mean"), avg_game_total=("gt", "mean"),
    ).reset_index()
    return agg.set_index("team")


# League avg total ~2025 baseline for context
def expected_total(home_ab, away_ab, scoring):
    """Expected total = (A_off + B_def)/2 + (B_off + A_def)/2, using 2025 PPG."""
    if home_ab not in scoring.index or away_ab not in scoring.index:
        return None, None
    h, a = scoring.loc[home_ab], scoring.loc[away_ab]
    exp_home = (h["off_ppg"] + a["def_ppg"]) / 2
    exp_away = (a["off_ppg"] + h["def_ppg"]) / 2
    exp = exp_home + exp_away
    # Reliability: discount if either team has major 2026 change
    changed = [t for t in (home_ab, away_ab) if t in MAJOR_CHANGE_2026]
    reliability = 1.0 - 0.35 * len(changed)  # -35% per changed team
    return exp, reliability


# The 5 games we passed on + others — target slate lines (Sept 13, 2026)
SLATE = {
    "TB @ CIN": ("TB", "CIN", 51.5),
    "ARI @ LAC": ("ARI", "LAC", 46.5),
    "CHI @ CAR": ("CHI", "CAR", 47.5),
    "GB @ MIN": ("GB", "MIN", 44.5),
    "MIA @ LV": ("MIA", "LV", 40.5),
    # include the leans for context
    "WAS @ PHI": ("WAS", "PHI", 46.5),
    "NO @ DET": ("NO", "DET", 49.5),
    "DAL @ NYG": ("DAL", "NYG", 48.5),
}


def main():
    scoring = team_2025_scoring()
    print("=" * 88)
    print("SCORING ENVIRONMENT TIEBREAKER — expected total (2025 PPG) vs current line")
    print("Positive gap = line is HIGH vs their scoring -> UNDER lean. Negative = OVER lean.")
    print("Reliability discounts teams with new HC/QB in 2026.")
    print("=" * 88)
    print(f"\n{'Matchup':<12} {'Line':>5} {'Exp':>5} {'Gap':>6} {'Rel':>5} {'Lean':<14} Notes")
    print("-" * 88)

    results = []
    for label, (away, home, line) in SLATE.items():
        exp, rel = expected_total(home, away, scoring)
        if exp is None:
            print(f"{label:<12} {line:>5} missing team data")
            continue
        gap = line - exp  # + = line above expected -> under
        eff_gap = gap * rel  # reliability-adjusted
        if eff_gap >= 3:
            lean = "UNDER"
        elif eff_gap <= -3:
            lean = "OVER"
        else:
            lean = "no edge"
        changed = [t for t in (home, away) if t in MAJOR_CHANGE_2026]
        note = ("discount: " + ", ".join(f"{t}" for t in changed)) if changed else "stable rosters"
        print(f"{label:<12} {line:>5.1f} {exp:>5.1f} {gap:>+6.1f} {rel:>5.2f} {lean:<14} {note}")
        results.append({"matchup": label, "line": line, "expected": round(exp, 1),
                        "gap": round(gap, 1), "reliability": rel, "lean": lean})

    print("\n" + "-" * 88)
    print("Team 2025 scoring profiles (for the slate):")
    teams = sorted({t for _, (a, h, _) in SLATE.items() for t in (a, h)})
    print(f"  {'Team':<5} {'Off':>5} {'Def':>5} {'AvgTot':>7}")
    for t in teams:
        if t in scoring.index:
            s = scoring.loc[t]
            flag = " *changed" if t in MAJOR_CHANGE_2026 else ""
            print(f"  {t:<5} {s['off_ppg']:>5.1f} {s['def_ppg']:>5.1f} {s['avg_game_total']:>7.1f}{flag}")

    out = Path(__file__).parent.parent / "data" / "processed" / "scoring_env_latest.parquet"
    pd.DataFrame(results).to_parquet(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
