"""
Deep Week 1 Totals Layer Analysis (2021-2025).
Tests EVERY situational layer independently to find which have real over/under signal.
Output feeds the consensus model in analyze_game_totals.py.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"


def build_coach_change_flags(games):
    """Flag teams whose head coach changed vs the prior season."""
    # Build team -> {season: coach} from all games (home perspective + away)
    rows = []
    for _, g in games.iterrows():
        rows.append({"team": g["home_team"], "season": g["season"], "coach": g.get("home_coach")})
        rows.append({"team": g["away_team"], "season": g["season"], "coach": g.get("away_coach")})
    tc = pd.DataFrame(rows).dropna(subset=["coach"])
    # First coach seen per team-season
    team_season_coach = tc.groupby(["team", "season"])["coach"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
    changed = {}  # (team, season) -> bool new coach
    for (team, season), coach in team_season_coach.items():
        prev = team_season_coach.get((team, season - 1))
        changed[(team, season)] = (prev is not None and prev != coach)
    return changed


def load_week1():
    games = pd.read_parquet(RAW / "games_historical.parquet")
    coach_changed = build_coach_change_flags(games)
    wk1 = games[games["week"] == 1].copy()
    wk1["actual_total"] = wk1["home_score"] + wk1["away_score"]
    wk1 = wk1[wk1["total_line"].notna() & wk1["actual_total"].notna()]
    wk1["margin"] = wk1["actual_total"] - wk1["total_line"]
    wk1["result"] = np.where(wk1["margin"] > 0, "OVER", np.where(wk1["margin"] < 0, "UNDER", "PUSH"))
    wk1["home_new_coach"] = wk1.apply(lambda r: coach_changed.get((r["home_team"], r["season"]), False), axis=1)
    wk1["away_new_coach"] = wk1.apply(lambda r: coach_changed.get((r["away_team"], r["season"]), False), axis=1)
    wk1["any_new_coach"] = wk1["home_new_coach"] | wk1["away_new_coach"]
    wk1["both_new_coach"] = wk1["home_new_coach"] & wk1["away_new_coach"]
    return wk1


def rpt(label, df):
    o = (df["result"] == "OVER").sum()
    u = (df["result"] == "UNDER").sum()
    dec = o + u
    if dec == 0:
        return
    up = u / dec * 100
    tag = ""
    if dec >= 8:
        if up >= 62:
            tag = " <-- UNDER signal"
        elif up <= 38:
            tag = " <-- OVER signal"
    print(f"  {label:<32} n={len(df):<3} U{u:>2}-O{o:<2} ({up:>5.1f}% U) marg {df['margin'].mean():+5.1f}{tag}")


def main():
    wk1 = load_week1()
    print("=" * 82)
    print(f"WEEK 1 TOTALS — LAYER-BY-LAYER SIGNAL TEST ({len(wk1)} games, 2021-2025)")
    print("Baseline: 63.7% under overall. A layer matters if it moves meaningfully off that.")
    print("=" * 82)

    print("\n[TOTAL LINE LEVEL]")
    rpt("Low (<=42)", wk1[wk1["total_line"] <= 42])
    rpt("Mid (42.5-47)", wk1[(wk1["total_line"] > 42) & (wk1["total_line"] <= 47)])
    rpt("High (47.5-49.5)", wk1[(wk1["total_line"] > 47) & (wk1["total_line"] <= 49.5)])
    rpt("Very high (50+)", wk1[wk1["total_line"] >= 50])

    print("\n[ROOF]")
    rpt("Outdoors", wk1[wk1["roof"] == "outdoors"])
    rpt("Dome", wk1[wk1["roof"] == "dome"])
    rpt("Closed (retractable shut)", wk1[wk1["roof"] == "closed"])
    rpt("Any indoor (dome+closed)", wk1[wk1["roof"].isin(["dome", "closed"])])

    print("\n[DIVISION]")
    rpt("Division game", wk1[wk1["div_game"] == True])
    rpt("Non-division", wk1[wk1["div_game"] == False])

    print("\n[SPREAD SIZE]")
    wk1["abs_spread"] = wk1["spread_line"].abs()
    rpt("Pick'em / close (<=3)", wk1[wk1["abs_spread"] <= 3])
    rpt("Moderate (3.5-6.5)", wk1[(wk1["abs_spread"] > 3) & (wk1["abs_spread"] <= 6.5)])
    rpt("Big fav (7+)", wk1[wk1["abs_spread"] >= 7])

    print("\n[NEW HEAD COACH]")
    rpt("Any team new coach", wk1[wk1["any_new_coach"]])
    rpt("Both teams new coach", wk1[wk1["both_new_coach"]])
    rpt("No coaching change", wk1[~wk1["any_new_coach"]])

    print("\n[WEEKDAY / SLOT]")
    if "weekday" in wk1.columns:
        for d in wk1["weekday"].dropna().unique():
            rpt(f"{d}", wk1[wk1["weekday"] == d])

    print("\n[PRIMETIME (kickoff hour)]")
    if "gametime" in wk1.columns:
        wk1["hour"] = pd.to_datetime(wk1["gametime"], format="%H:%M", errors="coerce").dt.hour
        rpt("Early (<=13:00 ET)", wk1[wk1["hour"] <= 13])
        rpt("Late afternoon (14-17)", wk1[(wk1["hour"] >= 14) & (wk1["hour"] <= 17)])
        rpt("Primetime (18:00+)", wk1[wk1["hour"] >= 18])

    print("\n[WEATHER — outdoor only]")
    out = wk1[wk1["roof"] == "outdoors"]
    rpt("Hot (>=80F)", out[out["temp"] >= 80])
    rpt("Warm (70-79F)", out[(out["temp"] >= 70) & (out["temp"] < 80)])
    rpt("Mild (<70F)", out[out["temp"] < 70])
    rpt("Windy (>=12mph)", out[out["wind"] >= 12])

    print("\n[HOME FAVORITE vs AWAY FAVORITE]")
    rpt("Home favored", wk1[wk1["spread_line"] > 0])
    rpt("Away favored", wk1[wk1["spread_line"] < 0])

    print("\n" + "=" * 82)
    print("Use layers tagged with a signal (n>=8) in the consensus model.")


if __name__ == "__main__":
    main()
