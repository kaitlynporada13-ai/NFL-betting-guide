"""
Historical Week 1 Game Totals Analysis (2021-2025).
Goal: find real, applicable patterns in Week 1 over/under results
beyond a blanket "Week 1 rust" lean.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW = Path(__file__).parent.parent / "data" / "raw"


def load_week1():
    games = pd.read_parquet(RAW / "games_historical.parquet")
    wk1 = games[games["week"] == 1].copy()
    # Actual total points and grade vs line
    wk1["actual_total"] = wk1["home_score"] + wk1["away_score"]
    wk1 = wk1[wk1["total_line"].notna() & wk1["actual_total"].notna()]
    wk1["margin"] = wk1["actual_total"] - wk1["total_line"]
    wk1["result"] = np.where(wk1["margin"] > 0, "OVER",
                     np.where(wk1["margin"] < 0, "UNDER", "PUSH"))
    return wk1


def rate(df):
    """Over/under counts and under hit rate excluding pushes."""
    o = (df["result"] == "OVER").sum()
    u = (df["result"] == "UNDER").sum()
    p = (df["result"] == "PUSH").sum()
    decided = o + u
    under_pct = u / decided * 100 if decided else 0
    return o, u, p, under_pct


def line(label, df):
    o, u, p, under_pct = rate(df)
    n = len(df)
    if n == 0:
        return
    print(f"  {label:<34} n={n:<3} UNDER {u:>2}-{o:<2}  ({under_pct:>5.1f}% under)  avg margin {df['margin'].mean():+.1f}")


def main():
    wk1 = load_week1()
    print("=" * 80)
    print(f"WEEK 1 GAME TOTALS ANALYSIS — {len(wk1)} games (2021-2025)")
    print("=" * 80)

    # 1. Overall
    print("\n[1] OVERALL")
    line("All Week 1 games", wk1)
    print(f"      Avg line: {wk1['total_line'].mean():.1f} | Avg actual: {wk1['actual_total'].mean():.1f}")

    # 2. By season (is it stable?)
    print("\n[2] BY SEASON (stability check)")
    for s in sorted(wk1["season"].unique()):
        line(f"{s}", wk1[wk1["season"] == s])

    # 3. By total line bucket
    print("\n[3] BY TOTAL LINE LEVEL")
    line("Low total (<=42)", wk1[wk1["total_line"] <= 42])
    line("Mid total (42.5-47)", wk1[(wk1["total_line"] > 42) & (wk1["total_line"] <= 47)])
    line("High total (47.5-49.5)", wk1[(wk1["total_line"] > 47) & (wk1["total_line"] <= 49.5)])
    line("Very high total (50+)", wk1[wk1["total_line"] >= 50])

    # 4. Roof
    print("\n[4] BY ROOF / ENVIRONMENT")
    if "roof" in wk1.columns:
        for r in wk1["roof"].dropna().unique():
            line(f"roof = {r}", wk1[wk1["roof"] == r])
        line("Dome/closed (dome+closed)", wk1[wk1["roof"].isin(["dome", "closed"])])
        line("Outdoors", wk1[wk1["roof"] == "outdoors"])

    # 5. Division games
    print("\n[5] DIVISION vs NON-DIVISION")
    if "div_game" in wk1.columns:
        line("Division games", wk1[wk1["div_game"] == True])
        line("Non-division games", wk1[wk1["div_game"] == False])

    # 6. Spread / favorite size
    print("\n[6] BY SPREAD SIZE (competitiveness)")
    wk1["abs_spread"] = wk1["spread_line"].abs()
    line("Close (spread <=3)", wk1[wk1["abs_spread"] <= 3])
    line("Moderate (3.5-6.5)", wk1[(wk1["abs_spread"] > 3) & (wk1["abs_spread"] <= 6.5)])
    line("Big fav (7+)", wk1[wk1["abs_spread"] >= 7])

    # 7. Weather (Sept heat / wind)
    print("\n[7] WEATHER (outdoor games only)")
    out = wk1[wk1["roof"] == "outdoors"] if "roof" in wk1.columns else wk1
    if "temp" in out.columns and out["temp"].notna().any():
        line("Hot (temp >= 80F)", out[out["temp"] >= 80])
        line("Mild (60-79F)", out[(out["temp"] >= 60) & (out["temp"] < 80)])
    if "wind" in out.columns and out["wind"].notna().any():
        line("Windy (>= 12 mph)", out[out["wind"] >= 12])
        line("Calm (< 12 mph)", out[out["wind"] < 12])

    # 8. Rest (Thu/Mon openers have extra rest)
    print("\n[8] BY REST")
    if "home_rest" in wk1.columns:
        line("Standard rest (both ~7 days)", wk1[(wk1["home_rest"].between(6, 8)) & (wk1["away_rest"].between(6, 8))])

    # Save
    out_path = Path(__file__).parent.parent / "data" / "processed" / "week1_totals_history.parquet"
    wk1.to_parquet(out_path, index=False)
    print(f"\nSaved detail to {out_path}")


if __name__ == "__main__":
    main()
