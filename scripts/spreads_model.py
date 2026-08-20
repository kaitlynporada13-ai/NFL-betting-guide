"""
SPREADS (ATS) MODEL VALIDATION — same rigor as the totals model.

All games 2021-2025, graded against the closing spread.
TRAIN = 2021-2023, TEST = 2024-2025.
A rule survives only if a side clears the -110 break-even (52.4%) in BOTH periods.

Convention (nflverse): spread_line > 0 => HOME favored by that many.
Home covers if (home_score - away_score) > spread_line.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"
BREAKEVEN = 52.4


def load():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[g["spread_line"].notna() & g["home_score"].notna()].copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["ats_diff"] = g["home_margin"] - g["spread_line"]  # >0 home covers
    g = g[g["ats_diff"] != 0]  # drop pushes
    g["home_cover"] = g["ats_diff"] > 0
    g["abs_spread"] = g["spread_line"].abs()
    g["home_fav"] = g["spread_line"] > 0
    g["hour"] = pd.to_datetime(g["gametime"], format="%H:%M", errors="coerce").dt.hour
    g["is_dome"] = g["roof"].isin(["dome", "closed"])
    return g


def grade(df, mask, min_n=30):
    sub = df[mask]
    if len(sub) < min_n:
        return None, len(sub)
    # We report HOME cover %. A rule can be a home-side or away-side edge.
    return sub["home_cover"].mean() * 100, len(sub)


def test_rule(train, test, fn, label):
    tr, tr_n = grade(train, fn(train))
    te, te_n = grade(test, fn(test))
    if tr is None or te is None:
        return None
    # Home edge if both >54 & >52.4; Away edge if both <46 & <47.6
    survives = (tr >= 54 and te >= 52.4) or (tr <= 46 and te <= 47.6)
    return {"rule": label, "train_home%": round(tr, 1), "train_n": tr_n,
            "test_home%": round(te, 1), "test_n": te_n,
            "side": "HOME" if tr > 50 else "AWAY", "survives": survives}


def main():
    g = load()
    train = g[g["season"] <= 2023]
    test = g[g["season"] >= 2024]
    print("=" * 88)
    print(f"SPREADS (ATS) VALIDATION — train {len(train)} (2021-23), test {len(test)} (2024-25)")
    print(f"Break-even at -110 = {BREAKEVEN}%. Reporting HOME cover %.")
    print("=" * 88)
    print(f"\nBaseline home cover: train {train['home_cover'].mean()*100:.1f}% | "
          f"test {test['home_cover'].mean()*100:.1f}%")

    rules = [
        ("ALL home teams", lambda d: d["home_cover"] | ~d["home_cover"]),  # all games
        ("home favorite", lambda d: d["home_fav"]),
        ("home underdog", lambda d: ~d["home_fav"]),
        ("home fav >=7", lambda d: d["home_fav"] & (d["abs_spread"] >= 7)),
        ("home dog >=3", lambda d: (~d["home_fav"]) & (d["abs_spread"] >= 3)),
        ("away favorite >=7", lambda d: (~d["home_fav"]) & (d["abs_spread"] >= 7)),
        ("big fav any 10.5+", lambda d: d["abs_spread"] > 10),
        ("small spread <=3", lambda d: d["abs_spread"] <= 3),
        ("division game", lambda d: d["div_game"] == True),
        ("non-division", lambda d: d["div_game"] == False),
        ("Week 1", lambda d: d["week"] == 1),
        ("Weeks 14-18", lambda d: (d["week"] >= 14) & (d["week"] <= 18)),
        ("playoffs", lambda d: d["week"] > 18),
        ("primetime (18+)", lambda d: d["hour"] >= 18),
        ("early (<=13)", lambda d: d["hour"] <= 13),
        ("Thursday", lambda d: d["weekday"] == "Thursday"),
        ("Monday", lambda d: d["weekday"] == "Monday"),
        ("dome host", lambda d: d["is_dome"]),
        ("outdoor host", lambda d: ~d["is_dome"]),
        ("home dog primetime", lambda d: (~d["home_fav"]) & (d["hour"] >= 18)),
        ("home fav division", lambda d: d["home_fav"] & (d["div_game"] == True)),
        ("big road fav (away -7+)", lambda d: (~d["home_fav"]) & (d["abs_spread"] >= 7)),
    ]

    results = []
    for label, fn in rules:
        r = test_rule(train, test, fn, label)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    df["edge"] = (df["test_home%"] - 50).abs()
    df = df.sort_values(["survives", "edge"], ascending=[False, False])

    print(f"\n{'Rule':<26} {'Side':<6} {'TrainHome%':>10} {'n':>4} {'TestHome%':>10} {'n':>4} {'Survives'}")
    print("-" * 88)
    for _, r in df.iterrows():
        star = " *** SURVIVES" if r["survives"] else ""
        print(f"{r['rule']:<26} {r['side']:<6} {r['train_home%']:>9.1f}% {r['train_n']:>4} "
              f"{r['test_home%']:>9.1f}% {r['test_n']:>4}{star}")

    survivors = df[df["survives"]]
    print("\n" + "=" * 88)
    print(f"SURVIVING ATS RULES: {len(survivors)}")
    if len(survivors):
        for _, r in survivors.iterrows():
            print(f"  Bet {r['side']} on: {r['rule']} (train {r['train_home%']}%, test {r['test_home%']}%)")
    else:
        print("  NONE. The ATS market is efficient — no simple side rule beats the vig OOS.")


if __name__ == "__main__":
    main()
