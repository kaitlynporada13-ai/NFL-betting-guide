"""
FULL-SEASON TOTALS MODEL — built and validated on historical data.

Method:
  - All games 2021-2025 (~1400), graded over/under vs closing total_line.
  - TRAIN = 2021-2023, TEST = 2024-2025 (chronological, mimics real betting).
  - Test many single-factor and interaction rules.
  - A rule "survives" only if it clears the -110 break-even (52.4%) in BOTH
    train and test, in the SAME direction, with adequate sample.
  - Report survivors -> those become the model.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"

BREAKEVEN = 52.4  # -110 vig


def load():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[g["total_line"].notna() & g["home_score"].notna()].copy()
    g["actual_total"] = g["home_score"] + g["away_score"]
    g = g[g["actual_total"] != g["total_line"]]  # drop pushes
    g["under"] = g["actual_total"] < g["total_line"]
    g["abs_spread"] = g["spread_line"].abs()
    g["hour"] = pd.to_datetime(g["gametime"], format="%H:%M", errors="coerce").dt.hour
    g["is_dome"] = g["roof"].isin(["dome", "closed"])
    g["is_playoff"] = g["week"] > 18
    return g


def grade(df, mask, label, min_n=30):
    """Return under% for the subset, or None if too small."""
    sub = df[mask]
    if len(sub) < min_n:
        return None, len(sub)
    return sub["under"].mean() * 100, len(sub)


def test_rule(train, test, mask_fn, label):
    """Evaluate a rule on train and test. Return dict if it shows a consistent edge."""
    tr_under, tr_n = grade(train, mask_fn(train), label)
    te_under, te_n = grade(test, mask_fn(test), label)
    if tr_under is None or te_under is None:
        return None
    # Determine direction from train
    tr_edge_under = tr_under - 50
    # Under edge if both >break-even; Over edge if both < (100-break-even)
    survives_under = tr_under >= 54 and te_under >= 52.4
    survives_over = tr_under <= 46 and te_under <= 47.6
    return {
        "rule": label, "train_under%": round(tr_under, 1), "train_n": tr_n,
        "test_under%": round(te_under, 1), "test_n": te_n,
        "direction": "UNDER" if tr_under > 50 else "OVER",
        "survives": survives_under or survives_over,
    }


def main():
    g = load()
    train = g[g["season"] <= 2023]
    test = g[g["season"] >= 2024]
    print("=" * 90)
    print(f"TOTALS MODEL VALIDATION — train {len(train)} (2021-23), test {len(test)} (2024-25)")
    print(f"Break-even at -110 = {BREAKEVEN}%. A rule must clear it in BOTH train and test.")
    print("=" * 90)

    base_tr = train["under"].mean() * 100
    base_te = test["under"].mean() * 100
    print(f"\nBaseline under rate: train {base_tr:.1f}% | test {base_te:.1f}%  (market is ~efficient)")

    # Define candidate rules
    rules = [
        # Total line level
        ("total <=38", lambda d: d["total_line"] <= 38),
        ("total 38.5-42", lambda d: (d["total_line"] > 38) & (d["total_line"] <= 42)),
        ("total 42.5-45", lambda d: (d["total_line"] > 42) & (d["total_line"] <= 45)),
        ("total 45.5-48", lambda d: (d["total_line"] > 45) & (d["total_line"] <= 48)),
        ("total 48.5-51", lambda d: (d["total_line"] > 48) & (d["total_line"] <= 51)),
        ("total 51+", lambda d: d["total_line"] > 51),
        # Environment
        ("dome/closed", lambda d: d["is_dome"]),
        ("outdoors", lambda d: ~d["is_dome"]),
        ("cold <40F (outdoor)", lambda d: (~d["is_dome"]) & (d["temp"] < 40)),
        ("cool 40-55F (outdoor)", lambda d: (~d["is_dome"]) & (d["temp"] >= 40) & (d["temp"] <= 55)),
        ("hot >78F (outdoor)", lambda d: (~d["is_dome"]) & (d["temp"] > 78)),
        ("wind >=15mph", lambda d: (~d["is_dome"]) & (d["wind"] >= 15)),
        ("wind >=20mph", lambda d: (~d["is_dome"]) & (d["wind"] >= 20)),
        # Matchup
        ("division game", lambda d: d["div_game"] == True),
        ("non-division", lambda d: d["div_game"] == False),
        ("spread <=3", lambda d: d["abs_spread"] <= 3),
        ("spread 3.5-6.5", lambda d: (d["abs_spread"] > 3) & (d["abs_spread"] <= 6.5)),
        ("spread 7-10", lambda d: (d["abs_spread"] >= 7) & (d["abs_spread"] <= 10)),
        ("spread 10.5+", lambda d: d["abs_spread"] > 10),
        # Timing
        ("Week 1", lambda d: d["week"] == 1),
        ("Weeks 2-4", lambda d: (d["week"] >= 2) & (d["week"] <= 4)),
        ("Weeks 14-18", lambda d: (d["week"] >= 14) & (d["week"] <= 18)),
        ("playoffs", lambda d: d["is_playoff"]),
        ("primetime (18+)", lambda d: d["hour"] >= 18),
        ("early (<=13)", lambda d: d["hour"] <= 13),
        ("Thursday", lambda d: d["weekday"] == "Thursday"),
        ("Monday", lambda d: d["weekday"] == "Monday"),
        # Interactions (the promising ones)
        ("cold + wind15 (outdoor)", lambda d: (~d["is_dome"]) & (d["temp"] < 45) & (d["wind"] >= 15)),
        ("div + total<=43", lambda d: (d["div_game"] == True) & (d["total_line"] <= 43)),
        ("wind15 + total47+", lambda d: (~d["is_dome"]) & (d["wind"] >= 15) & (d["total_line"] >= 47)),
        ("dome + total50+", lambda d: d["is_dome"] & (d["total_line"] >= 50)),
        ("primetime + total48+", lambda d: (d["hour"] >= 18) & (d["total_line"] >= 48)),
        ("big fav 10.5+ (blowout)", lambda d: d["abs_spread"] > 10),
    ]

    results = []
    for label, fn in rules:
        r = test_rule(train, test, fn, label)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    # Sort: survivors first, then by test edge magnitude
    df["edge"] = (df["test_under%"] - 50).abs()
    df = df.sort_values(["survives", "edge"], ascending=[False, False])

    print(f"\n{'Rule':<26} {'Dir':<6} {'Train%':>7} {'n':>4} {'Test%':>7} {'n':>4} {'Survives'}")
    print("-" * 90)
    for _, r in df.iterrows():
        star = " *** SURVIVES" if r["survives"] else ""
        print(f"{r['rule']:<26} {r['direction']:<6} {r['train_under%']:>6.1f}% {r['train_n']:>4} "
              f"{r['test_under%']:>6.1f}% {r['test_n']:>4}{star}")

    survivors = df[df["survives"]]
    print("\n" + "=" * 90)
    print(f"SURVIVING RULES: {len(survivors)}")
    if len(survivors):
        for _, r in survivors.iterrows():
            print(f"  {r['direction']} on: {r['rule']}  (train {r['train_under%']}%, test {r['test_under%']}%)")
    else:
        print("  NONE cleared break-even in both train and test.")
        print("  -> The closing totals market is efficient; no simple rule beats it reliably.")

    out = Path(__file__).parent.parent / "data" / "processed" / "totals_model_rules.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved all rule results to {out}")


if __name__ == "__main__":
    main()
