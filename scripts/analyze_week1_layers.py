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


BASELINE_UNDER = 63.7  # overall Week 1 under rate 2021-2025

# Collect computed lifts here for saving to the weights file
LAYER_WEIGHTS = {}


def rpt(label, df, layer=None, value=None):
    o = (df["result"] == "OVER").sum()
    u = (df["result"] == "UNDER").sum()
    dec = o + u
    if dec == 0:
        return
    up = u / dec * 100
    lift = up - BASELINE_UNDER  # + = more under than baseline, - = relative OVER
    # Reliability: shrink lift toward 0 for small samples (regularization)
    reliability = min(dec / 20.0, 1.0)  # full weight at n>=20
    weighted_lift = lift * reliability
    direction = "UNDER" if weighted_lift > 3 else "OVER" if weighted_lift < -3 else "neutral"
    tag = ""
    if dec >= 8:
        if weighted_lift > 3:
            tag = f" <-- UNDER (+{weighted_lift:.0f} lift)"
        elif weighted_lift < -3:
            tag = f" <-- OVER ({weighted_lift:.0f} lift)"
    print(f"  {label:<30} n={len(df):<3} {up:>5.1f}%U  lift {lift:+5.1f}  wt {weighted_lift:+5.1f}{tag}")
    if layer and value:
        LAYER_WEIGHTS.setdefault(layer, {})[value] = float(round(weighted_lift, 1))


def main():
    import yaml
    wk1 = load_week1()
    wk1["abs_spread"] = wk1["spread_line"].abs()
    wk1["hour"] = pd.to_datetime(wk1["gametime"], format="%H:%M", errors="coerce").dt.hour

    print("=" * 88)
    print(f"WEEK 1 TOTALS — WEIGHTED LAYER ANALYSIS ({len(wk1)} games, 2021-2025)")
    print(f"Baseline: {BASELINE_UNDER}% under. Weight = (layer under% - baseline) x reliability.")
    print("  Positive weight = UNDER push. Negative weight = OVER push (vs a typical Wk1 game).")
    print("=" * 88)

    print("\n[TOTAL LINE LEVEL]")
    rpt("Low (<=42)", wk1[wk1["total_line"] <= 42], "total_line", "low")
    rpt("Mid (42.5-47)", wk1[(wk1["total_line"] > 42) & (wk1["total_line"] <= 47)], "total_line", "mid")
    rpt("High (47.5-49.5)", wk1[(wk1["total_line"] > 47) & (wk1["total_line"] <= 49.5)], "total_line", "high")
    rpt("Very high (50+)", wk1[wk1["total_line"] >= 50], "total_line", "very_high")

    print("\n[ROOF]")
    rpt("Outdoors", wk1[wk1["roof"] == "outdoors"], "roof", "outdoors")
    rpt("Indoor (dome+closed)", wk1[wk1["roof"].isin(["dome", "closed"])], "roof", "indoor")

    print("\n[DIVISION]")
    rpt("Division game", wk1[wk1["div_game"] == True], "division", "yes")
    rpt("Non-division", wk1[wk1["div_game"] == False], "division", "no")

    print("\n[SPREAD SIZE]")
    rpt("Close (<=3)", wk1[wk1["abs_spread"] <= 3], "spread", "close")
    rpt("Moderate (3.5-6.5)", wk1[(wk1["abs_spread"] > 3) & (wk1["abs_spread"] <= 6.5)], "spread", "moderate")
    rpt("Big fav (7+)", wk1[wk1["abs_spread"] >= 7], "spread", "big")

    print("\n[NEW HEAD COACH]")
    rpt("Any team new coach", wk1[wk1["any_new_coach"]], "new_coach", "yes")
    rpt("No coaching change", wk1[~wk1["any_new_coach"]], "new_coach", "no")

    print("\n[KICKOFF SLOT]")
    rpt("Early (<=13 ET)", wk1[wk1["hour"] <= 13], "slot", "early")
    rpt("Afternoon (14-17)", wk1[(wk1["hour"] >= 14) & (wk1["hour"] <= 17)], "slot", "afternoon")
    rpt("Primetime (18+)", wk1[wk1["hour"] >= 18], "slot", "primetime")

    print("\n[FAVORITE LOCATION]")
    rpt("Home favored", wk1[wk1["spread_line"] > 0], "favorite", "home")
    rpt("Away favored", wk1[wk1["spread_line"] < 0], "favorite", "away")

    print("\n[WEATHER — outdoor only]")
    out = wk1[wk1["roof"] == "outdoors"]
    rpt("Hot (>=80F)", out[out["temp"] >= 80], "weather", "hot")
    rpt("Mild (<70F)", out[out["temp"] < 70], "weather", "mild")

    # Save computed weights for the game model to consume
    weights_path = Path(__file__).parent.parent / "config" / "week1_totals_weights.yaml"
    with open(weights_path, "w") as f:
        yaml.dump({"baseline_under_pct": BASELINE_UNDER, "layer_weights": LAYER_WEIGHTS},
                  f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 88)
    print("WEIGHTS (sorted strongest to weakest):")
    flat = [(f"{lyr}.{val}", w) for lyr, vals in LAYER_WEIGHTS.items() for val, w in vals.items()]
    for name, w in sorted(flat, key=lambda x: abs(x[1]), reverse=True):
        direction = "UNDER" if w > 0 else "OVER" if w < 0 else "neutral"
        print(f"  {name:<24} {w:+6.1f}  {direction}")
    print(f"\nSaved weights to {weights_path}")


if __name__ == "__main__":
    main()
