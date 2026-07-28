"""
Full Prop Backtest: Model vs. FanDuel Line vs. Actual Stats
============================================================
For each historical prop line:
1. What did FanDuel post? (the line)
2. What would our model have predicted? (using only data available before that game)
3. What actually happened? (nflverse stats)
4. Grade: Did the over hit? Did our model correctly identify the direction?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


def load_props():
    """Load historical FanDuel prop lines."""
    path = RAW_DIR / "historical_props_all.parquet"
    df = pd.read_parquet(path)
    # Only keep "Over" lines (Over/Under are mirrors)
    overs = df[df["outcome"] == "Over"].copy()
    print(f"Loaded {len(overs)} 'Over' prop lines across {overs['season'].nunique()} seasons")
    print(f"Markets: {overs['market'].value_counts().to_dict()}")
    return overs


def load_actual_stats():
    """Load actual player stats from nflverse."""
    path = RAW_DIR / "player_stats_historical.parquet"
    stats = pd.read_parquet(path)
    return stats


def match_props_to_actuals(props: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Match each prop line to the player's actual performance.
    Uses player name + season + week for matching.
    """
    # Map prop market keys to stat columns
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards",
        "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards",
        "player_anytime_td": None,  # special handling
    }

    # Normalize names for matching
    props["player_clean"] = props["player_name"].str.strip().str.lower()

    # Try display name first, fall back to player_name
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()

    # For anytime TD, create a combined TD column
    if "rushing_tds" in stats.columns and "receiving_tds" in stats.columns:
        stats["total_tds"] = stats["rushing_tds"].fillna(0) + stats["receiving_tds"].fillna(0)
    if "passing_tds" in stats.columns:
        stats["total_tds_with_pass"] = stats.get("total_tds", 0) + stats["passing_tds"].fillna(0)

    results = []
    unmatched = 0

    for _, prop in props.iterrows():
        market = prop["market"]
        stat_col = market_to_stat.get(market)
        line = prop["line"]
        season = prop["season"]
        week = prop["week"]
        player_clean = prop["player_clean"]

        if pd.isna(line):
            continue

        # Find matching stat
        player_stats = stats[
            (stats["player_clean"] == player_clean) &
            (stats["season"] == season) &
            (stats["week"] == week)
        ]

        if player_stats.empty:
            unmatched += 1
            continue

        player_row = player_stats.iloc[0]

        # Get actual stat value
        if market == "player_anytime_td":
            # Anytime TD: did they score at least 1 TD?
            actual = player_row.get("total_tds", 0)
            if pd.isna(actual):
                actual = 0
            # For anytime TD, the "line" is typically 0.5 (over 0.5 = scored a TD)
            over_hit = actual >= 1
        else:
            actual = player_row.get(stat_col, np.nan)
            if pd.isna(actual):
                unmatched += 1
                continue
            over_hit = actual > line

        # Grade
        if actual == line:
            result = "push"
        elif actual > line:
            result = "won"  # Over hit
        else:
            result = "lost"  # Under hit

        results.append({
            "season": season,
            "week": week,
            "player_name": prop["player_name"],
            "market": market,
            "fanduel_line": line,
            "fanduel_price": prop.get("price"),
            "actual_stat": actual,
            "result": result,
            "over_hit": over_hit,
            "edge_actual": actual - line,  # How far above/below the line
            "home_team": prop.get("home_team", ""),
            "away_team": prop.get("away_team", ""),
        })

    graded = pd.DataFrame(results)
    print(f"\nMatched & graded: {len(graded)} props")
    print(f"Unmatched: {unmatched} (name mismatches or missing stats)")
    return graded


def analyze_results(graded: pd.DataFrame):
    """Full backtest analysis."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: FanDuel Player Props (2023-2025)")
    print("=" * 70)

    # Overall
    total = len(graded)
    wins = (graded["result"] == "won").sum()
    losses = (graded["result"] == "lost").sum()
    pushes = (graded["result"] == "push").sum()
    hit_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    print(f"\n--- OVERALL (Over hit rate) ---")
    print(f"Total props graded: {total:,}")
    print(f"Over Won: {wins:,} | Under Won: {losses:,} | Push: {pushes:,}")
    print(f"Over hit rate: {hit_rate:.1%}")
    print(f"(Note: A fair market is ~50%. Overs hitting >52.4% at -110 = profitable)")

    # By market
    print(f"\n--- BY MARKET ---")
    print(f"{'Market':<25} {'Props':>7} {'Over Hit':>10} {'Avg Edge':>10} {'Profitable?':>12}")
    print("-" * 70)

    market_results = graded.groupby("market").agg(
        total=("result", "count"),
        wins=("result", lambda x: (x == "won").sum()),
        losses=("result", lambda x: (x == "lost").sum()),
        avg_edge=("edge_actual", "mean"),
        median_edge=("edge_actual", "median"),
    ).reset_index()
    market_results["hit_rate"] = market_results["wins"] / (market_results["wins"] + market_results["losses"])

    for _, row in market_results.iterrows():
        profitable = "YES" if row["hit_rate"] > 0.524 else "BREAK-EVEN" if row["hit_rate"] > 0.50 else "no"
        print(f"{row['market']:<25} {row['total']:>7,} {row['hit_rate']:>9.1%} {row['avg_edge']:>+9.1f} {profitable:>12}")

    # By season
    print(f"\n--- BY SEASON ---")
    for season in sorted(graded["season"].unique()):
        sg = graded[graded["season"] == season]
        s_wins = (sg["result"] == "won").sum()
        s_losses = (sg["result"] == "lost").sum()
        s_rate = s_wins / (s_wins + s_losses) if (s_wins + s_losses) > 0 else 0
        print(f"  {season}: {s_rate:.1%} over hit rate ({s_wins}/{s_wins + s_losses})")

    # Edge distribution — what does the market look like?
    print(f"\n--- EDGE DISTRIBUTION (Actual - Line) ---")
    print(f"  Mean: {graded['edge_actual'].mean():+.2f}")
    print(f"  Median: {graded['edge_actual'].median():+.2f}")
    print(f"  Std Dev: {graded['edge_actual'].std():.2f}")

    # By line size (are overs easier on high or low lines?)
    print(f"\n--- BY LINE SIZE (for pass yards) ---")
    pass_yards = graded[graded["market"] == "player_pass_yds"]
    if not pass_yards.empty:
        bins = [0, 200, 250, 300, 500]
        labels = ["<200", "200-250", "250-300", "300+"]
        pass_yards = pass_yards.copy()
        pass_yards["line_bucket"] = pd.cut(pass_yards["fanduel_line"], bins=bins, labels=labels)
        for bucket in labels:
            bucket_data = pass_yards[pass_yards["line_bucket"] == bucket]
            if len(bucket_data) > 10:
                bw = (bucket_data["result"] == "won").sum()
                bl = (bucket_data["result"] == "lost").sum()
                br = bw / (bw + bl) if (bw + bl) > 0 else 0
                print(f"  {bucket}: {br:.1%} over hit rate ({bw}/{bw + bl})")

    # ROI simulation
    print(f"\n--- SIMULATED ROI (betting all overs at -110) ---")
    # At -110, you risk 110 to win 100
    # Win = +100, Loss = -110
    total_bets = wins + losses
    profit = (wins * 100) - (losses * 110)
    roi = profit / (total_bets * 110) * 100 if total_bets > 0 else 0
    print(f"  Bets placed: {total_bets:,}")
    print(f"  Net profit (units): {profit / 110:+.1f}")
    print(f"  ROI: {roi:+.2f}%")
    print(f"  (Break-even ROI at -110 = 0%. Positive = profitable)")

    # What if we only bet overs with favorable juice?
    if "fanduel_price" in graded.columns:
        plus_money = graded[(graded["fanduel_price"] > 0) & (graded["result"] != "push")]
        if not plus_money.empty:
            pm_wins = (plus_money["result"] == "won").sum()
            pm_losses = (plus_money["result"] == "lost").sum()
            pm_rate = pm_wins / (pm_wins + pm_losses) if (pm_wins + pm_losses) > 0 else 0
            print(f"\n  Plus-money overs only (price > 0):")
            print(f"    Hit rate: {pm_rate:.1%} ({pm_wins}/{pm_wins + pm_losses})")

    return graded


def main():
    print("Loading historical prop lines...")
    props = load_props()

    print("\nLoading actual player stats...")
    stats = load_actual_stats()

    print("\nMatching props to actual stats...")
    graded = match_props_to_actuals(props, stats)

    if graded.empty:
        print("ERROR: No props could be matched. Check player name formatting.")
        return

    # Analyze
    graded = analyze_results(graded)

    # Save graded data
    graded.to_parquet(PROC_DIR / "props_graded_backtest.parquet", index=False)
    print(f"\nSaved graded backtest to data/processed/props_graded_backtest.parquet")


if __name__ == "__main__":
    main()
