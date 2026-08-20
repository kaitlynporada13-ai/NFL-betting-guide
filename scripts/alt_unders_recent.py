"""
Alt-unders + over-rescue, restricted to the LAST 2 YEARS (2024-2025) Week 1.
Small sample (~32 games) — directional only.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"

BUYUP_BREAKEVEN = {0: 52.4, 1: 55.6, 2: 59.2, 3: 63.0, 4: 66.7, 5: 70.6, 6: 73.7}


def main():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[(g["week"] == 1) & (g["season"] >= 2024) &
          g["total_line"].notna() & g["home_score"].notna()].copy()
    g["actual_total"] = g["home_score"] + g["away_score"]
    g["over_by"] = g["actual_total"] - g["total_line"]

    print("=" * 74)
    print(f"LAST 2 YEARS (2024-2025) WEEK 1 — {len(g)} games")
    print("=" * 74)

    unders = (g["over_by"] < 0).sum()
    overs = (g["over_by"] > 0).sum()
    pushes = (g["over_by"] == 0).sum()
    print(f"\nUnder {unders} / Over {overs} / Push {pushes}  "
          f"({unders/(unders+overs)*100:.1f}% under)")

    # By season
    for s in [2024, 2025]:
        ss = g[g["season"] == s]
        u = (ss["over_by"] < 0).sum(); o = (ss["over_by"] > 0).sum()
        print(f"  {s}: {u}-{o}  ({u/(u+o)*100:.0f}% under)")

    # Alt-under hit rates (buying up)
    print(f"\n{'BuyUp':>6} {'Hit%':>7} {'BreakEven':>10} {'Verdict':>12}")
    print("  " + "-" * 38)
    for x in range(0, 7):
        hit = (g["actual_total"] < g["total_line"] + x).mean() * 100
        be = BUYUP_BREAKEVEN[x]
        verdict = "+EV" if hit > be else "-EV" if hit < be - 2 else "~even"
        print(f"  +{x:<4} {hit:>6.1f}% {be:>9.1f}% {verdict:>12}")

    # Over-rescue
    ov = g[g["over_by"] > 0]
    print(f"\nWhen it went OVER ({len(ov)} games), by how much:")
    if len(ov):
        print(f"  mean +{ov['over_by'].mean():.1f}, median +{ov['over_by'].median():.1f}")
        for x in [3, 5, 7]:
            r = (ov["over_by"] <= x).sum()
            print(f"  buy +{x} would rescue {r}/{len(ov)} ({r/len(ov)*100:.0f}%)")

    print("\n  CAVEAT: ~13 over games. Too few to conclude much — directional only.")


if __name__ == "__main__":
    main()
