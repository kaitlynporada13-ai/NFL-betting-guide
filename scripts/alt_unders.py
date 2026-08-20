"""
ALTERNATE UNDERS test: if we buy the total UP by X points and bet under,
does the higher hit rate beat the extra juice we pay?

Uses Week 1 games (the only validated under edge).
Compares empirical hit rate at line+X to the approximate break-even for
typical alternate-under pricing.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"

# Approx alternate-under break-even by points bought UP (typical NFL juice ~25c/pt)
# price -> break-even %. These are ballpark; real books vary.
BUYUP_BREAKEVEN = {
    0: 52.4,   # -110
    1: 55.6,   # ~-125
    2: 59.2,   # ~-145
    3: 63.0,   # ~-170
    4: 66.7,   # ~-200
    5: 70.6,   # ~-240
    6: 73.7,   # ~-280
}


def load_wk1():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[(g["week"] == 1) & g["total_line"].notna() & g["home_score"].notna()].copy()
    g["actual_total"] = g["home_score"] + g["away_score"]
    return g


def main():
    g = load_wk1()
    train = g[g["season"] <= 2023]
    test = g[g["season"] >= 2024]
    print("=" * 78)
    print(f"ALTERNATE UNDERS — Week 1 ({len(g)} games; train {len(train)}, test {len(test)})")
    print("Buy the total UP by X, bet under. Does hit rate beat the extra juice?")
    print("=" * 78)
    print(f"\n{'BuyUp':>5} {'Line+X':>7} {'Train hit%':>11} {'Test hit%':>10} {'BreakEven':>10} {'Test verdict':>14}")
    print("-" * 78)

    for x in range(0, 7):
        # hit if actual < line + x (bought-up under)
        tr_hit = (train["actual_total"] < train["total_line"] + x).mean() * 100
        te_hit = (test["actual_total"] < test["total_line"] + x).mean() * 100
        be = BUYUP_BREAKEVEN[x]
        verdict = "+EV" if te_hit > be else "-EV" if te_hit < be - 2 else "~break-even"
        label = f"+{x}"
        print(f"{label:>5} {'line+'+str(x):>7} {tr_hit:>10.1f}% {te_hit:>9.1f}% {be:>9.1f}% {verdict:>14}")

    # Distribution of margin (how far under do they land?)
    g["margin"] = g["actual_total"] - g["total_line"]
    print("\nMargin distribution (actual - line), Week 1:")
    print(f"  mean {g['margin'].mean():+.1f}, median {g['margin'].median():+.1f}, "
          f"std {g['margin'].std():.1f}")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  {pct}th percentile: {np.percentile(g['margin'], pct):+.1f}")

    print("\nInterpretation:")
    print("  If test hit% stays ABOVE break-even as you buy up, alt-unders add value.")
    print("  If it falls below, you're paying more juice than the extra cushion is worth.")


if __name__ == "__main__":
    main()
