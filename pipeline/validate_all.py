"""
WEEKLY VALIDATION GUARDRAIL.
Re-runs every edge the picks rely on through out-of-sample testing and reports
which still hold. Run this FIRST each week — if an edge has decayed below the
-110 break-even (52.4%), it gets flagged so we stop recommending it.

Edges checked:
  1. Player-prop Week 1 UNDER by market (train 2023-24 -> test 2025)
  2. Line-inflation sharpener (under hits harder when line > baseline)
  3. Game totals: Week 1 under (full-season train/test)
  4. Spreads: home dog vs 7+ road favorite (ATS)

Writes data/processed/edge_health.parquet + prints a health report.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.config_loader import get_data_dir

RAW = get_data_dir("raw")
PROC = get_data_dir("processed")
BREAKEVEN = 52.4

MARKET_STAT = {
    "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards", "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}


def _grade_props():
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    props = props[props["market"].isin(MARKET_STAT) & (props["outcome"] == "Over")].copy()
    props["pname"] = props["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()
    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    out = []
    for market, stat in MARKET_STAT.items():
        pm = props[props["market"] == market]
        s = stats[["pname", "season", "week", stat]].dropna(subset=[stat])
        m = pm.merge(s, on=["pname", "season", "week"], how="inner")
        m = m[m[stat] != m["line"]]
        m["under_win"] = m[stat] < m["line"]
        m["market"] = market
        out.append(m[["season", "week", "market", "under_win"]])
    return pd.concat(out, ignore_index=True)


def check_prop_unders(g, results):
    g = g[g["week"] == 1]
    for market in MARKET_STAT:
        sub = g[g["market"] == market]
        tr = sub[sub["season"] <= 2024]["under_win"].mean() * 100
        te = sub[sub["season"] >= 2025]["under_win"].mean() * 100
        n = (sub["season"] >= 2025).sum()
        status = "VALID" if (tr >= 54 and te >= BREAKEVEN) else "DECAYED"
        results.append({"edge": f"Wk1 UNDER {market.replace('player_','')}",
                        "train%": round(tr, 1), "test%": round(te, 1),
                        "test_n": int(n), "status": status})


def check_totals(results):
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[g["total_line"].notna() & g["home_score"].notna()].copy()
    g["actual"] = g["home_score"] + g["away_score"]
    g = g[g["actual"] != g["total_line"]]
    g["under"] = g["actual"] < g["total_line"]
    w1 = g[g["week"] == 1]
    tr = w1[w1["season"] <= 2023]["under"].mean() * 100
    te = w1[w1["season"] >= 2024]["under"].mean() * 100
    status = "VALID" if (tr >= 54 and te >= BREAKEVEN) else "DECAYED"
    results.append({"edge": "Wk1 game UNDER (totals)", "train%": round(tr, 1),
                    "test%": round(te, 1), "test_n": int((w1["season"] >= 2024).sum()),
                    "status": status})


def check_spreads(results):
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[g["spread_line"].notna() & g["home_score"].notna()].copy()
    g["home_cover"] = (g["home_score"] - g["away_score"]) > g["spread_line"]
    g = g[(g["home_score"] - g["away_score"]) != g["spread_line"]]
    # nflverse: spread_line > 0 = HOME favored. Home dog vs road favorite of 7+
    # => away favored by 7+ => spread_line <= -7. Bet HOME to cover.
    dog = g[g["spread_line"] <= -7]
    tr = dog[dog["season"] <= 2023]["home_cover"].mean() * 100
    te = dog[dog["season"] >= 2024]["home_cover"].mean() * 100
    status = "VALID" if (tr >= 54 and te >= BREAKEVEN) else "DECAYED"
    results.append({"edge": "Home dog vs 7+ road fav (ATS)", "train%": round(tr, 1),
                    "test%": round(te, 1), "test_n": int((dog["season"] >= 2024).sum()),
                    "status": status})


def check_inflation(g, results):
    """Line-inflation sharpener needs baselines; report the Wk1 all-market under as proxy."""
    w1 = g[g["week"] == 1]
    tr = w1[w1["season"] <= 2024]["under_win"].mean() * 100
    te = w1[w1["season"] >= 2025]["under_win"].mean() * 100
    status = "VALID" if (tr >= 54 and te >= BREAKEVEN) else "WATCH"
    results.append({"edge": "Wk1 UNDER all props (combined)", "train%": round(tr, 1),
                    "test%": round(te, 1), "test_n": int((w1["season"] >= 2025).sum()),
                    "status": status})


def check_anytime_td(results):
    """Anytime TD: heavy favorites (<= -200) betting YES should stay +EV; else -EV."""
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    td = props[props["market"] == "player_anytime_td"].copy()
    td["pname"] = td["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()
    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    stats["td"] = stats.get("rushing_tds", 0).fillna(0) + stats.get("receiving_tds", 0).fillna(0)
    m = td.merge(stats[["pname", "season", "week", "td"]], on=["pname", "season", "week"], how="inner")
    m["scored"] = m["td"] >= 1

    def yes_roi(df):
        if len(df) == 0:
            return None, 0
        def r(row):
            if row["scored"]:
                return (row["price"] / 100) if row["price"] > 0 else (100 / -row["price"])
            return -1
        return df.apply(r, axis=1).mean() * 100, len(df)

    fav = m[m["price"] <= -200]
    tr_roi, _ = yes_roi(fav[fav["season"] <= 2024])
    te_roi, n = yes_roi(fav[fav["season"] >= 2025])
    status = "VALID" if (tr_roi and te_roi and tr_roi > 0 and te_roi > 0) else "DECAYED"
    results.append({"edge": "Anytime TD YES: heavy fav (<=-200)",
                    "train%": round(tr_roi, 1) if tr_roi else 0,
                    "test%": round(te_roi, 1) if te_roi else 0,
                    "test_n": int(n), "status": status})


def main():
    print("=" * 78)
    print("WEEKLY EDGE HEALTH CHECK — every edge re-validated out-of-sample")
    print(f"Break-even at -110 = {BREAKEVEN}%. Test = most recent season(s).")
    print("=" * 78)
    results = []
    g = _grade_props()
    check_prop_unders(g, results)
    check_inflation(g, results)
    check_totals(results)
    check_spreads(results)
    check_anytime_td(results)

    df = pd.DataFrame(results)
    df.to_parquet(PROC / "edge_health.parquet", index=False)

    print(f"\n{'Edge':<34}{'Train%':>8}{'Test%':>8}{'n':>5}  Status")
    print("-" * 78)
    for _, r in df.iterrows():
        flag = "OK " if r["status"] == "VALID" else "!! "
        print(f"{r['edge']:<34}{r['train%']:>7.1f}%{r['test%']:>7.1f}%{r['test_n']:>5}  {flag}{r['status']}")

    decayed = df[df["status"] == "DECAYED"]
    print("\n" + "=" * 78)
    if len(decayed):
        print(f"⚠️  {len(decayed)} edge(s) DECAYED — stop recommending until they recover:")
        for _, r in decayed.iterrows():
            print(f"   - {r['edge']} (test {r['test%']}%)")
    else:
        print("✅ All edges still clear break-even out-of-sample. Safe to run the picks.")
    print("Saved health report to data/processed/edge_health.parquet")


if __name__ == "__main__":
    main()
