"""
When a Week 1 UNDER lost (game went OVER), how far over did it go?
And how many points of buy-up would have been needed to rescue it?

This tells us whether alt-unders (buying the line up) actually saves losses,
or whether overs blow way past the number (making buy-up futile).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"


def load_wk1():
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[(g["week"] == 1) & g["total_line"].notna() & g["home_score"].notna()].copy()
    g["actual_total"] = g["home_score"] + g["away_score"]
    g["over_by"] = g["actual_total"] - g["total_line"]  # + = went over
    return g


def main():
    g = load_wk1()
    overs = g[g["over_by"] > 0].copy()  # games where the under LOST
    unders = g[g["over_by"] < 0]
    print("=" * 74)
    print(f"WEEK 1 UNDER LOSSES — how far over did they go? ({len(g)} games total)")
    print(f"  Under WON: {len(unders)} | Under LOST (over): {len(overs)}")
    print("=" * 74)

    print(f"\nWhen the game went OVER, by how much (points past the line):")
    print(f"  mean +{overs['over_by'].mean():.1f}, median +{overs['over_by'].median():.1f}")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  {pct}th pctile: +{np.percentile(overs['over_by'], pct):.1f}")

    print(f"\nOf the {len(overs)} losses, how many would BUYING UP have rescued?")
    print(f"  {'BuyUp':>6} {'Rescued':>9} {'% of losses saved':>18}")
    print("  " + "-" * 38)
    for x in range(1, 8):
        rescued = (overs["over_by"] <= x).sum()  # would be under if line were +x higher
        pct = rescued / len(overs) * 100
        print(f"  +{x:<5} {rescued:>9} {pct:>17.1f}%")

    # The punchline: overs that are "narrow" vs "blowout"
    narrow = (overs["over_by"] <= 3).sum()
    blowout = (overs["over_by"] > 10).sum()
    print(f"\n  Narrow overs (within 3 pts, rescuable): {narrow}/{len(overs)} "
          f"({narrow/len(overs)*100:.0f}%)")
    print(f"  Blowout overs (10+ past line, unrescuable): {blowout}/{len(overs)} "
          f"({blowout/len(overs)*100:.0f}%)")

    print("\nInterpretation:")
    print("  If most overs are BLOWOUTS (10+ past the line), buying up a few points")
    print("  rescues very few losses — so alt-unders pay extra juice for little protection.")
    print("  If many overs are NARROW, buying up meaningfully converts losses to wins.")


if __name__ == "__main__":
    main()
