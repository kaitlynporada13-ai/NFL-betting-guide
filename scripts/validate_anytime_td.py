"""
VALIDATE Anytime TD out-of-sample (train 2023-24, test 2025).
It's a Yes/No market in American odds. Method:
  - Convert price -> implied probability.
  - Grade: did the player actually score a TD (rush or receiving)?
  - Compare actual hit rate to implied, by price bucket and by Week 1 vs rest.
  - A bettable edge = actual materially beats implied (bet YES) or falls short (bet NO/fade),
    consistently in BOTH train and test.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"


def american_to_prob(o):
    if pd.isna(o):
        return np.nan
    return (-o) / (-o + 100) if o < 0 else 100 / (o + 100)


def grade():
    props = pd.read_parquet(RAW / "historical_props_all.parquet")
    td = props[props["market"] == "player_anytime_td"].copy()
    td["pname"] = td["player_name"].str.lower().str.replace(".", "", regex=False).str.strip()

    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    nc = "player_display_name" if "player_display_name" in stats.columns else "player_name"
    stats["pname"] = stats[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    stats["td"] = (stats.get("rushing_tds", 0).fillna(0) + stats.get("receiving_tds", 0).fillna(0))
    s = stats[["pname", "season", "week", "td"]]

    m = td.merge(s, on=["pname", "season", "week"], how="inner")
    m["scored"] = m["td"] >= 1
    m["implied"] = m["price"].apply(american_to_prob)
    m = m[m["implied"].notna()]
    m["edge"] = m["scored"].astype(int) - m["implied"]  # + = YES beats market
    m["split"] = np.where(m["season"] <= 2024, "train", "test")
    return m


def summarize(df, label):
    n = len(df)
    if n < 20:
        return None
    hit = df["scored"].mean() * 100
    imp = df["implied"].mean() * 100
    # ROI of betting YES on all at their price
    def yes_roi(row):
        if row["scored"]:
            o = row["price"]
            return (o / 100) if o > 0 else (100 / -o)
        return -1
    roi = df.apply(yes_roi, axis=1).mean() * 100
    return {"scope": label, "n": n, "actual%": round(hit, 1),
            "implied%": round(imp, 1), "yes_roi%": round(roi, 1)}


def main():
    m = grade()
    print("=" * 82)
    print(f"ANYTIME TD VALIDATION — {len(m)} graded (2023-25). YES ROI>0 = betting YES is +EV.")
    print("=" * 82)

    # By price bucket, Week 1, train vs test
    print("\n[WEEK 1] by price bucket (YES side):")
    print(f"  {'Bucket':<18}{'Split':<7}{'n':>5}{'Actual%':>9}{'Implied%':>10}{'YES ROI%':>10}")
    print("  " + "-" * 62)
    w1 = m[m["week"] == 1]
    buckets = [(-10000, -200, "big fav <-200"), (-200, -110, "fav -200..-110"),
               (-110, 120, "pick -110..+120"), (120, 200, "dog +120..+200"),
               (200, 10000, "longshot +200+")]
    for lo, hi, lbl in buckets:
        b = w1[(w1["price"] > lo) & (w1["price"] <= hi)]
        for split in ["train", "test"]:
            r = summarize(b[b["split"] == split], f"{lbl}")
            if r:
                print(f"  {lbl:<18}{split:<7}{r['n']:>5}{r['actual%']:>8.1f}%{r['implied%']:>9.1f}%{r['yes_roi%']:>+9.1f}%")

    # Overall Week 1: YES ROI vs NO ROI
    print("\n[WEEK 1] overall betting-side test:")
    for split in ["train", "test"]:
        w = w1[w1["split"] == split]
        r = summarize(w, f"all Wk1 {split}")
        if r:
            no_roi = -r["yes_roi%"]  # rough inverse (fading), ignoring NO-side vig differences
            print(f"  {split}: actual {r['actual%']}% vs implied {r['implied%']}% | "
                  f"betting YES ROI {r['yes_roi%']:+.1f}%")

    # Also all-weeks for context
    print("\n[ALL WEEKS] YES ROI by price bucket (test 2025):")
    te = m[m["split"] == "test"]
    for lo, hi, lbl in buckets:
        b = te[(te["price"] > lo) & (te["price"] <= hi)]
        r = summarize(b, lbl)
        if r:
            print(f"  {lbl:<18}n={r['n']:<5} actual {r['actual%']:>5.1f}% "
                  f"implied {r['implied%']:>5.1f}% YES ROI {r['yes_roi%']:>+6.1f}%")

    m.to_parquet(Path(__file__).parent.parent / "data" / "processed" / "anytime_td_graded.parquet", index=False)
    print("\nSaved graded data to data/processed/anytime_td_graded.parquet")


if __name__ == "__main__":
    main()
