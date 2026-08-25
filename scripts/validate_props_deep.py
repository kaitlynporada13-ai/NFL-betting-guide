"""
DEEP prop validation (3 layers), all out-of-sample (train 2023-24, test 2025):

  Layer 1 — LINE INFLATION: for each prop, compute the player's baseline (prior-season
            per-game average for that stat). Does UNDER hit harder when the posted line
            is ABOVE the player's baseline? Does OVER hit when the line is BELOW baseline?
  Layer 2 — OVERS: are there any spots where OVER validates (not just unders)?
  Layer 3 — MARKET x WEEK-BUCKET x INFLATION interaction: where is the edge strongest?

Feeds the projection+confidence engine.
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
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    props = props[props["market"].isin(MARKET_STAT) & (props["outcome"] == "Over")].copy()
    props["pname"] = props["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()

    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[nc].str.lower().str.replace(".", "", regex=False).str.strip()

    # Prior-season per-game baseline for each player+stat
    rows = []
    for market, stat in MARKET_STAT.items():
        s = stats[["pname", "season", "week", stat]].dropna(subset=[stat])
        base = s.groupby(["pname", "season"])[stat].mean().reset_index()
        base = base.rename(columns={stat: "baseline", "season": "base_season"})
        base["market"] = market
        rows.append(base)
    baselines = pd.concat(rows, ignore_index=True)

    graded = []
    for market, stat in MARKET_STAT.items():
        pm = props[props["market"] == market]
        s = stats[["pname", "season", "week", stat]].dropna(subset=[stat]).copy()
        m = pm.merge(s, on=["pname", "season", "week"], how="inner")
        m["actual"] = m[stat]
        m = m[m["actual"] != m["line"]]
        m["under_win"] = m["actual"] < m["line"]
        # attach PRIOR season baseline
        b = baselines[baselines["market"] == market][["pname", "base_season", "baseline"]]
        m = m.merge(b, left_on=["pname"], right_on=["pname"], how="left")
        m = m[m["base_season"] == m["season"] - 1]  # baseline from prior season only
        graded.append(m[["season", "week", "market", "player_name", "line", "actual",
                          "under_win", "baseline"]])
    g = pd.concat(graded, ignore_index=True)
    g["inflation"] = g["line"] - g["baseline"]  # + = line set above player's norm
    g["split"] = np.where(g["season"] <= 2024, "train", "test")
    g["is_wk1"] = g["week"] == 1
    return g


def rate(df):
    n = len(df)
    return (df["under_win"].mean() * 100 if n else 0.0), n


def main():
    g = build()
    g.to_parquet(PROC / "props_deep_graded.parquet", index=False)
    print("=" * 90)
    print(f"DEEP PROP VALIDATION — {len(g)} props with prior-season baselines")
    print(f"Break-even {BREAKEVEN}%. Train 2023-24 / Test 2025.")
    print("=" * 90)

    # LAYER 1: inflation buckets (Week 1)
    print("\n[LAYER 1] LINE INFLATION vs baseline — Week 1 only")
    print("  (line ABOVE player norm should favor UNDER; BELOW should favor OVER)")
    print(f"  {'Inflation bucket':<24}{'Train U%':>9}{'n':>5}{'Test U%':>9}{'n':>5}  Verdict")
    print("  " + "-" * 74)
    w1 = g[g["is_wk1"]]
    buckets = [(-99, -3, "line well BELOW norm"), (-3, 0, "line just below"),
               (0, 3, "line just above"), (3, 8, "line ABOVE norm"),
               (8, 999, "line WELL above norm")]
    for lo, hi, lbl in buckets:
        sub = w1[(w1["inflation"] > lo) & (w1["inflation"] <= hi)]
        tr, trn = rate(sub[sub["split"] == "train"])
        te, ten = rate(sub[sub["split"] == "test"])
        v = ""
        if trn >= 12 and ten >= 6:
            if tr >= 55 and te >= 52.4:
                v = "*** UNDER holds"
            elif tr <= 45 and te <= 47.6:
                v = "*** OVER holds"
            else:
                v = "no edge"
        else:
            v = "(small)"
        print(f"  {lbl:<24}{tr:>8.1f}%{trn:>5}{te:>8.1f}%{ten:>5}  {v}")

    # LAYER 2: OVERS — where does OVER win? (under% < 47.6 in both)
    print("\n[LAYER 2] OVER opportunities (Week 1) — by market, line BELOW baseline")
    print(f"  {'Market':<18}{'Train O%':>9}{'n':>5}{'Test O%':>9}{'n':>5}  Verdict")
    print("  " + "-" * 66)
    for market in MARKET_STAT:
        sub = w1[(w1["market"] == market) & (w1["inflation"] < 0)]  # line below norm
        tr, trn = rate(sub[sub["split"] == "train"])
        te, ten = rate(sub[sub["split"] == "test"])
        # over% = 100 - under%
        tro, teo = 100 - tr, 100 - te
        v = "(small)" if (trn < 10 or ten < 5) else \
            "*** OVER holds" if (tro >= 54 and teo >= 52.4) else "no over edge"
        print(f"  {market.replace('player_',''):<18}{tro:>8.1f}%{trn:>5}{teo:>8.1f}%{ten:>5}  {v}")

    # LAYER 3: market x inflation (Week 1) — best combined spots
    print("\n[LAYER 3] BEST SPOTS — market + line-above-norm, Week 1, UNDER")
    print(f"  {'Market + condition':<34}{'Train U%':>9}{'n':>5}{'Test U%':>9}{'n':>5}")
    print("  " + "-" * 62)
    for market in MARKET_STAT:
        sub = w1[(w1["market"] == market) & (w1["inflation"] > 0)]  # line above norm
        tr, trn = rate(sub[sub["split"] == "train"])
        te, ten = rate(sub[sub["split"] == "test"])
        if trn >= 10:
            print(f"  {market.replace('player_','')+' (line>norm)':<34}"
                  f"{tr:>8.1f}%{trn:>5}{te:>8.1f}%{ten:>5}")
    print("\nSaved graded detail to data/processed/props_deep_graded.parquet")


if __name__ == "__main__":
    main()
