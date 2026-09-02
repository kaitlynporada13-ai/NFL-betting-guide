"""
VALIDATE whether ANY prop edge exists past Week 1 (out-of-sample).

Week 1 = rust edge (unders win). Question: does that edge persist, decay, or
reverse in Weeks 2-18? We test each week-bucket x market, train 2023-24 / test 2025,
and also test the "line inflation" edge (line above baseline) beyond Week 1 since
that mechanism (mispriced line) isn't rust-specific and might travel.

Break-even = 52.4%. An edge must clear it in BOTH train and test to be bettable.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
BREAKEVEN = 52.4

MARKET_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}


def build():
    """Graded under legs w/ prior-season baseline (for inflation) + week."""
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    props = props[props["market"].isin(MARKET_STAT) & (props["outcome"] == "Over")].copy()
    props["pname"] = props["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()

    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[nc].str.lower().str.replace(".", "", regex=False).str.strip()

    # prior-season per-game baseline
    base_rows = []
    for market, stat in MARKET_STAT.items():
        s = stats[["pname", "season", stat]].dropna(subset=[stat])
        b = s.groupby(["pname", "season"])[stat].mean().reset_index()
        b = b.rename(columns={stat: "baseline", "season": "base_season"})
        b["market"] = market
        base_rows.append(b)
    baselines = pd.concat(base_rows, ignore_index=True)

    frames = []
    for market, stat in MARKET_STAT.items():
        pm = props[props["market"] == market]
        s = stats[["pname", "season", "week", stat]].dropna(subset=[stat]).copy()
        m = pm.merge(s, on=["pname", "season", "week"], how="inner")
        m["actual"] = m[stat]
        m = m[m["actual"] != m["line"]]
        m["under_win"] = (m["actual"] < m["line"]).astype(int)
        b = baselines[baselines["market"] == market][["pname", "base_season", "baseline"]]
        m = m.merge(b, on="pname", how="left")
        m = m[m["base_season"] == m["season"] - 1]
        frames.append(m[["season", "week", "market", "under_win", "line", "baseline"]])
    g = pd.concat(frames, ignore_index=True)
    g["inflation"] = g["line"] - g["baseline"]
    g["split"] = np.where(g["season"] <= 2024, "train", "test")
    return g


def rate(df):
    return (df["under_win"].mean() * 100 if len(df) else 0.0), len(df)


def show(g, mask, label):
    sub = g[mask]
    tr, trn = rate(sub[sub["split"] == "train"])
    te, ten = rate(sub[sub["split"] == "test"])
    verdict = ""
    if trn >= 40 and ten >= 20:
        if tr >= 54 and te >= BREAKEVEN:
            verdict = "*** UNDER edge holds"
        elif tr <= 46 and te <= (100 - BREAKEVEN):
            verdict = "*** OVER edge holds"
        else:
            verdict = "no edge"
    else:
        verdict = "(small sample)"
    print(f"  {label:<34}{tr:>7.1f}%{trn:>6}{te:>7.1f}%{ten:>6}  {verdict}")


def main():
    g = build()
    print("=" * 86)
    print(f"WEEKS 2+ EDGE VALIDATION — {len(g)} graded props w/ prior baseline")
    print(f"Break-even {BREAKEVEN}%. Train 2023-24 / Test 2025. UNDER win rates shown.")
    print("=" * 86)

    print("\n[A] UNDER rate by WEEK BUCKET (all markets pooled)")
    print(f"  {'Week bucket':<34}{'Train U%':>7}{'n':>6}{'Test U%':>7}{'n':>6}  Verdict")
    print("  " + "-" * 74)
    buckets = [("Week 1", g["week"] == 1), ("Weeks 2-4", g["week"].between(2, 4)),
               ("Weeks 5-9", g["week"].between(5, 9)), ("Weeks 10-14", g["week"].between(10, 14)),
               ("Weeks 15-18", g["week"].between(15, 18)), ("Weeks 2-18 (all non-opener)", g["week"] >= 2)]
    for lbl, m in buckets:
        show(g, m, lbl)

    print("\n[B] Weeks 2-18: UNDER rate by MARKET")
    print(f"  {'Market (weeks 2+)':<34}{'Train U%':>7}{'n':>6}{'Test U%':>7}{'n':>6}  Verdict")
    print("  " + "-" * 74)
    w2 = g["week"] >= 2
    for mk in MARKET_STAT:
        show(g, w2 & (g["market"] == mk), mk.replace("player_", ""))

    print("\n[C] Weeks 2-18: LINE-INFLATION edge (does mispriced-line under travel past Wk1?)")
    print(f"  {'Condition (weeks 2+)':<34}{'Train U%':>7}{'n':>6}{'Test U%':>7}{'n':>6}  Verdict")
    print("  " + "-" * 74)
    show(g, w2 & (g["inflation"] > 3), "line ABOVE norm (+3)")
    show(g, w2 & (g["inflation"] > 8), "line WELL above norm (+8)")
    show(g, w2 & (g["inflation"] < -3), "line BELOW norm (-3)  [over?]")

    print("\n[D] Weeks 2+ line-inflation edge BY MARKET (line > +3 above baseline)")
    print(f"  {'Market + line>+3 (wk2+)':<34}{'Train U%':>7}{'n':>6}{'Test U%':>7}{'n':>6}  Verdict")
    print("  " + "-" * 74)
    for mk in MARKET_STAT:
        show(g, w2 & (g["market"] == mk) & (g["inflation"] > 3), mk.replace("player_", "") + " line>+3")

    g.to_parquet(PROC / "weeks2plus_graded.parquet", index=False)
    print("\nSaved data/processed/weeks2plus_graded.parquet")


if __name__ == "__main__":
    main()
