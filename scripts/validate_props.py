"""
VALIDATE the player-prop UNDER edge out-of-sample.
Grade every historical Over/Under prop (2023-25) against actual player stats,
then test: does betting UNDER clear break-even (52.4%) — by market, Week 1 vs rest,
and train (2023-24) vs holdout (2025)?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"
BREAKEVEN = 52.4

MARKET_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}


def grade_props():
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    props = props[props["market"].isin(MARKET_STAT) & (props["outcome"] == "Over")].copy()
    props["pname"] = props["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()

    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    name_col = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[name_col].str.lower().str.replace(".", "", regex=False).str.strip()

    graded = []
    for market, stat in MARKET_STAT.items():
        pm = props[props["market"] == market]
        s = stats[["pname", "season", "week", stat]].dropna(subset=[stat])
        m = pm.merge(s, on=["pname", "season", "week"], how="inner")
        m["actual"] = m[stat]
        m = m[m["actual"] != m["line"]]  # drop pushes
        m["under_win"] = m["actual"] < m["line"]
        m["stat_market"] = market
        graded.append(m[["season", "week", "market", "player_name", "line", "actual", "under_win"]])
    g = pd.concat(graded, ignore_index=True)
    return g


def rate(df):
    n = len(df)
    return (df["under_win"].mean() * 100 if n else 0), n


def main():
    g = grade_props()
    print("=" * 88)
    print(f"PLAYER PROP UNDER VALIDATION — {len(g)} graded Over/Under props (2023-25)")
    print(f"Break-even at -110 = {BREAKEVEN}%. Train = 2023-24, Test = 2025.")
    print("=" * 88)

    g["is_wk1"] = g["week"] == 1
    g["split"] = np.where(g["season"] <= 2024, "train", "test")

    print(f"\n{'Market':<22}{'Scope':<10}{'Train U%':>9}{'n':>5}{'Test U%':>9}{'n':>5}  Verdict")
    print("-" * 88)
    for market in MARKET_STAT:
        for scope, mask in [("Week 1", g["is_wk1"]), ("All weeks", g["is_wk1"] | ~g["is_wk1"])]:
            sub = g[(g["market"] == market) & mask]
            tr, trn = rate(sub[sub["split"] == "train"])
            te, ten = rate(sub[sub["split"] == "test"])
            if trn < 15 or ten < 8:
                verdict = "(small sample)"
            elif tr >= 54 and te >= 52.4:
                verdict = "*** UNDER holds"
            elif tr <= 46 and te <= 47.6:
                verdict = "OVER edge"
            else:
                verdict = "no edge"
            mk = market.replace("player_", "")
            print(f"{mk:<22}{scope:<10}{tr:>8.1f}%{trn:>5}{te:>8.1f}%{ten:>5}  {verdict}")

    # Overall Week 1 under across all markets
    w1 = g[g["is_wk1"]]
    tr, trn = rate(w1[w1["split"] == "train"])
    te, ten = rate(w1[w1["split"] == "test"])
    print("\n" + "-" * 88)
    print(f"ALL MARKETS, Week 1 UNDER: train {tr:.1f}% ({trn}) | test {te:.1f}% ({ten})")
    allw = g
    tr2, _ = rate(allw[allw["split"] == "train"])
    te2, _ = rate(allw[allw["split"] == "test"])
    print(f"ALL MARKETS, All weeks UNDER: train {tr2:.1f}% | test {te2:.1f}%  "
          f"(is the edge Week-1-specific or season-long?)")

    g.to_parquet(Path(__file__).parent.parent / "data" / "processed" / "props_graded_full.parquet", index=False)
    print("\nSaved graded props to data/processed/props_graded_full.parquet")


if __name__ == "__main__":
    main()
