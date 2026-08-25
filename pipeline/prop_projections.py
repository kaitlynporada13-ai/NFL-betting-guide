"""
PROP PROJECTION + CONFIDENCE ENGINE.
For every posted Week 1 player prop, output:
  - the line
  - a projection (expected number)
  - over/under call
  - confidence score (anchored to OUT-OF-SAMPLE validated hit rates)
  - a short "why"

Confidence is grounded in scripts/validate_props_deep.py results (train 2023-24 -> test 2025).
Week 1 is UNDER-only (no validated overs). Line-above-baseline sharpens the under.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from pipeline.config_loader import get_data_dir
from pipeline.ingest_odds import pull_all_props_for_week

RAW = get_data_dir("raw")
PROC = get_data_dir("processed")

MARKET_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}
MARKET_LABEL = {
    "player_pass_yds": "Pass Yds", "player_pass_tds": "Pass TDs",
    "player_rush_yds": "Rush Yds", "player_receptions": "Receptions",
    "player_reception_yds": "Rec Yds",
}

# Validated OUT-OF-SAMPLE Week 1 UNDER hit rates (test 2025). Base by market.
BASE_UNDER = {
    "player_pass_tds": 0.667, "player_rush_yds": 0.600, "player_pass_yds": 0.567,
    "player_receptions": 0.538, "player_reception_yds": 0.514,
}
# Inflation bonus: line meaningfully above baseline pushed under to ~65-70% OOS.
def inflation_hit(market, infl_pct):
    base = BASE_UNDER.get(market, 0.50)
    if infl_pct is None:
        return base
    if infl_pct >= 0.15:      # line well above norm
        return min(base + 0.10, 0.72)
    if infl_pct >= 0.05:      # line above norm
        return min(base + 0.06, 0.70)
    if infl_pct <= -0.10:     # line below norm — edge weakens (but no over edge exists)
        return max(base - 0.06, 0.50)
    return base


def load_baselines():
    """Per-player 2025 per-game average for each stat (the projection base)."""
    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    s25 = stats[stats["season"] == 2025].copy()
    nc = "player_display_name" if "player_display_name" in s25.columns else "player_name"
    s25["pname"] = s25[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    base = {}
    for market, stat in MARKET_STAT.items():
        if stat in s25.columns:
            avg = s25.groupby("pname")[stat].mean()
            base[market] = avg.to_dict()
    return base


# Week 1 rust discount applied to the baseline to form the PROJECTION.
# Offenses score below their full-season talent in openers (validated: unders win).
RUST_DISCOUNT = {
    "player_pass_tds": 0.82, "player_pass_yds": 0.90, "player_rush_yds": 0.88,
    "player_receptions": 0.92, "player_reception_yds": 0.90,
}


def confidence_label(hit):
    if hit >= 0.65:
        return "HIGH"
    if hit >= 0.58:
        return "MEDIUM-HIGH"
    if hit >= 0.54:
        return "MEDIUM"
    if hit >= 0.51:
        return "LOW"
    return "PASS"


def build_projections():
    props = pull_all_props_for_week()
    if props.empty:
        print("No props available yet.")
        return pd.DataFrame()

    props = props[props["market"].isin(MARKET_STAT) & (props["outcome_name"] == "Over")].copy()
    baselines = load_baselines()

    rows = []
    for _, p in props.iterrows():
        market = p["market"]
        line = p.get("outcome_point")
        if line is None:
            continue
        name = p.get("player_name", "")
        pkey = name.lower().replace(".", "").strip()
        baseline = baselines.get(market, {}).get(pkey)

        # Projection = rust-discounted baseline (what we expect in a Week 1 opener)
        if baseline is not None and baseline > 0:
            projection = baseline * RUST_DISCOUNT.get(market, 0.9)
            infl_pct = (line - baseline) / baseline
        else:
            projection = None
            infl_pct = None

        hit = inflation_hit(market, infl_pct)
        # Week 1 is under-only; call is UNDER, confidence = validated hit rate
        call = "UNDER"
        conf = confidence_label(hit)

        # Build the "why"
        mk = MARKET_LABEL[market]
        if baseline is not None:
            base_str = f"2025 avg {baseline:.1f}"
            if infl_pct is not None and infl_pct >= 0.05:
                why = (f"Line {line} is {infl_pct:+.0%} above his {base_str}; "
                       f"Week 1 {mk} unders hit ~{hit:.0%} when the line is set above a player's norm.")
            elif infl_pct is not None and infl_pct <= -0.10:
                why = (f"Line {line} is below his {base_str}, but Week 1 rust still favors under "
                       f"(no validated overs exist); modest edge ~{hit:.0%}.")
            else:
                why = (f"Line {line} ~ his {base_str}; Week 1 {mk} unders hit ~{hit:.0%} (rust effect).")
        else:
            why = f"No 2025 baseline (new/rookie); default Week 1 {mk} under lean ~{hit:.0%}."

        rows.append({
            "player": name, "market": mk, "market_key": market, "line": line,
            "projection": round(projection, 1) if projection is not None else None,
            "baseline_2025": round(baseline, 1) if baseline is not None else None,
            "call": call, "confidence": conf, "hit_est": round(hit, 3),
            "why": why,
            "home_team": p.get("home_team", ""), "away_team": p.get("away_team", ""),
        })

    df = pd.DataFrame(rows).sort_values("hit_est", ascending=False)
    df.to_parquet(PROC / "prop_projections_latest.parquet", index=False)
    return df


def main():
    df = build_projections()
    if df.empty:
        return
    print("=" * 100)
    print(f"PROP PROJECTIONS — {len(df)} props | line / projection / call / confidence / why")
    print("=" * 100)
    print(f"{'Player':<20}{'Prop':<11}{'Line':>6}{'Proj':>6}{'Call':<7}{'Conf':<12}Why")
    print("-" * 100)
    for _, r in df.iterrows():
        proj = f"{r['projection']:.1f}" if r['projection'] is not None else "n/a"
        print(f"{r['player']:<20}{r['market']:<11}{r['line']:>6.1f}{proj:>6}"
              f"  {r['call']:<5}{r['confidence']:<12}{r['why'][:60]}")
    print(f"\nSaved to data/processed/prop_projections_latest.parquet")


if __name__ == "__main__":
    main()
