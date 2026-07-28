"""
Final Research Round:
1. Opening vs Closing Line Movement (do lines move toward correct outcome?)
2. Player-Specific Mean Reversion (which players are most predictable?)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# ================================================================
# PART 1: LINE MOVEMENT ANALYSIS
# ================================================================
# We have snapshots ~2hr before kickoff. We can compare that to
# earlier snapshots to detect line movement direction.
# Theory: Lines that move AWAY from opening = sharp money.
# Fading the public (betting where the line moved FROM) may have edge.
# But also: if FanDuel moves a line, they had reason. Following the
# move could also work. Let's test BOTH hypotheses.

def analyze_line_movement():
    """
    Analyze whether the relationship between a player's rolling average
    and the FanDuel line tells us about line movement/sharpness.
    
    We don't have opening lines directly, but we can use:
    - The player's season average as a "pre-season opening line" proxy
    - The actual FanDuel line as the "closing" (moved) line
    - When they diverge significantly, that's equivalent to "the line moved"
    
    Question: When FanDuel's line is FAR from the season average in a
    specific direction, does that mean they know something (follow)
    or are they wrong (fade)?
    """
    print("=" * 70)
    print("PART 1: LINE MOVEMENT / MARKET SHARPNESS ANALYSIS")
    print("When FanDuel moves a line away from the player's average,")
    print("are they right or wrong?")
    print("=" * 70)
    
    # Load all graded props
    props = pd.read_parquet(RAW_DIR / "historical_props_all.parquet")
    overs = props[props["outcome"] == "Over"].copy()
    
    # Load stats
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    overs["player_clean"] = overs["player_name"].str.strip().str.lower()
    
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards",
        "player_receptions": "receptions",
        "player_pass_tds": "passing_tds",
    }
    
    # Build rolling averages
    all_parts = []
    for market, stat_col in market_to_stat.items():
        if stat_col not in stats.columns:
            continue
        
        # Player rolling average (walk-forward)
        player_avgs = stats.sort_values(["player_clean", "season", "week"]).copy()
        player_avgs[f"{stat_col}_avg5"] = (
            player_avgs.groupby("player_clean")[stat_col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
        )
        
        mkt = overs[overs["market"] == market].copy()
        merged = mkt.merge(
            player_avgs[["player_clean", "season", "week", stat_col, f"{stat_col}_avg5"]].dropna().drop_duplicates(),
            on=["player_clean", "season", "week"], how="inner",
        )
        
        merged["actual"] = merged[stat_col]
        merged["player_avg"] = merged[f"{stat_col}_avg5"]
        merged["line_deviation"] = merged["line"] - merged["player_avg"]
        merged["line_deviation_pct"] = merged["line_deviation"] / merged["player_avg"].replace(0, np.nan)
        merged["over_hit"] = merged["actual"] > merged["line"]
        merged["market_label"] = market
        
        all_parts.append(merged[["player_clean", "season", "week", "market_label",
                                  "line", "player_avg", "actual", "line_deviation",
                                  "line_deviation_pct", "over_hit", "price"]])
    
    df = pd.concat(all_parts, ignore_index=True)
    print(f"\nTotal props with averages: {len(df):,}")
    
    # Analysis: When FanDuel moves the line UP (above average), are they right?
    print(f"\n--- WHEN FANDUEL SETS LINE ABOVE PLAYER'S AVERAGE ---")
    print(f"(FanDuel thinks the player will do BETTER than usual)")
    print(f"{'Deviation Range':<25} {'Over Hit%':>10} {'N':>7} {'FD Right?':>10}")
    print("-" * 55)
    
    ranges = [
        ("+5% to +10%", 0.05, 0.10),
        ("+10% to +15%", 0.10, 0.15),
        ("+15% to +25%", 0.15, 0.25),
        ("+25% to +40%", 0.25, 0.40),
        ("+40%+", 0.40, 2.0),
    ]
    
    for label, low, high in ranges:
        subset = df[df["line_deviation_pct"].between(low, high)]
        if len(subset) < 50:
            continue
        over_rate = subset["over_hit"].mean()
        # If FanDuel moves line UP and over hits >50%, they were right
        fd_right = "YES" if over_rate > 0.50 else "NO (fade)"
        print(f"  {label:<23} {over_rate:>9.1%} {len(subset):>6,} {fd_right:>10}")
    
    print(f"\n--- WHEN FANDUEL SETS LINE BELOW PLAYER'S AVERAGE ---")
    print(f"(FanDuel thinks the player will do WORSE than usual)")
    print(f"{'Deviation Range':<25} {'Over Hit%':>10} {'N':>7} {'FD Right?':>10}")
    print("-" * 55)
    
    neg_ranges = [
        ("-5% to -10%", -0.10, -0.05),
        ("-10% to -15%", -0.15, -0.10),
        ("-15% to -25%", -0.25, -0.15),
        ("-25% to -40%", -0.40, -0.25),
        ("-40%+", -2.0, -0.40),
    ]
    
    for label, low, high in neg_ranges:
        subset = df[df["line_deviation_pct"].between(low, high)]
        if len(subset) < 50:
            continue
        over_rate = subset["over_hit"].mean()
        # If FanDuel moves line DOWN and over hits <50%, they were right
        fd_right = "YES" if over_rate < 0.50 else "NO (fade)"
        print(f"  {label:<23} {over_rate:>9.1%} {len(subset):>6,} {fd_right:>10}")
    
    # KEY QUESTION: When do we FOLLOW FanDuel vs FADE them?
    print(f"\n--- THE VERDICT: FOLLOW or FADE? ---")
    
    # FanDuel line ABOVE avg by a lot → does the player actually perform above avg?
    high_line = df[df["line_deviation_pct"] > 0.15]
    low_line = df[df["line_deviation_pct"] < -0.15]
    
    if len(high_line) > 100:
        # Did actual performance also exceed average?
        high_line_actual_above = (high_line["actual"] > high_line["player_avg"]).mean()
        print(f"\n  Line >15% above avg: Did player actually beat their average? {high_line_actual_above:.1%}")
        print(f"    → If YES >50%: FanDuel knows something (don't fade blindly)")
        print(f"    → If NO <50%: FanDuel is wrong (fade = under)")
    
    if len(low_line) > 100:
        low_line_actual_below = (low_line["actual"] < low_line["player_avg"]).mean()
        print(f"\n  Line >15% below avg: Did player actually miss their average? {low_line_actual_below:.1%}")
        print(f"    → If YES >50%: FanDuel knows something (don't chase over)")
        print(f"    → If NO <50%: FanDuel is wrong (over is value)")
    
    return df

# ================================================================
# PART 2: PLAYER-SPECIFIC MEAN REVERSION
# ================================================================

def analyze_player_predictability():
    """
    Which specific players are most predictable/exploitable?
    
    A "predictable" player is one who:
    1. Consistently reverts to their mean (low variance)
    2. Has props set by FanDuel that deviate from their rolling avg
    3. Our mean reversion strategy works well on them
    
    We want to find "bankable" players — ones we can bet on every week
    with confidence that they'll hit their average.
    """
    print(f"\n\n{'='*70}")
    print("PART 2: PLAYER-SPECIFIC PREDICTABILITY")
    print("Which players can we bank on week after week?")
    print("="*70)
    
    # Load enriched strategy results
    bets = pd.read_parquet(PROC_DIR / "bets_fully_enriched.parquet")
    bets["bet_won"] = (
        ((bets["signal"] == "bet_over") & (bets["result"] == "won")) |
        ((bets["signal"] == "bet_under") & (bets["result"] == "lost"))
    )
    
    # Load stats for volatility analysis
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    # Per-player performance on our strategy
    player_perf = bets.groupby("player_clean").agg(
        total_bets=("bet_won", "count"),
        wins=("bet_won", "sum"),
        avg_deviation=("line_deviation_pct", "mean"),
        markets=("market", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
    ).reset_index()
    player_perf["hit_rate"] = player_perf["wins"] / player_perf["total_bets"]
    player_perf["roi"] = ((player_perf["wins"] * 100) - ((player_perf["total_bets"] - player_perf["wins"]) * 110)) / (player_perf["total_bets"] * 110) * 100
    
    # Only players with enough history (30+ bets)
    reliable = player_perf[player_perf["total_bets"] >= 30].copy()
    
    # Add volatility from stats
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards",
        "player_receptions": "receptions",
    }
    
    # Get coefficient of variation per player
    volatility = {}
    for stat_col in market_to_stat.values():
        if stat_col in stats.columns:
            vol = stats.groupby("player_clean")[stat_col].agg(["mean", "std"])
            vol["cv"] = vol["std"] / vol["mean"].replace(0, np.nan)
            for player, row in vol.iterrows():
                if player not in volatility:
                    volatility[player] = {}
                volatility[player][stat_col] = row["cv"]
    
    vol_df = pd.DataFrame([{"player_clean": k, "avg_cv": np.nanmean(list(v.values()))} 
                           for k, v in volatility.items()])
    
    reliable = reliable.merge(vol_df, on="player_clean", how="left")
    
    # =============================
    # TOP BANKABLE PLAYERS
    # =============================
    print(f"\n{'='*60}")
    print("MOST BANKABLE PLAYERS (30+ bets, profitable)")
    print("These are players our strategy consistently profits on")
    print("="*60)
    
    profitable_players = reliable[reliable["roi"] > 5].sort_values("roi", ascending=False)
    
    print(f"\n{'Player':<28} {'Bets':>5} {'Hit%':>6} {'ROI':>7} {'CV':>6} {'Primary Market':>20}")
    print("-" * 80)
    
    for _, row in profitable_players.head(25).iterrows():
        cv_str = f"{row['avg_cv']:.2f}" if pd.notna(row['avg_cv']) else "N/A"
        print(f"  {row['player_clean']:<26} {row['total_bets']:>4} {row['hit_rate']:>5.1%} "
              f"{row['roi']:>+6.1f}% {cv_str:>5} {row['markets']:>20}")
    
    # =============================
    # WORST PLAYERS (AVOID)
    # =============================
    print(f"\n{'='*60}")
    print("PLAYERS TO AVOID (30+ bets, consistently unprofitable)")
    print("FanDuel is TOO SHARP on these players")
    print("="*60)
    
    avoid_players = reliable[reliable["roi"] < -10].sort_values("roi")
    
    print(f"\n{'Player':<28} {'Bets':>5} {'Hit%':>6} {'ROI':>7} {'CV':>6} {'Primary Market':>20}")
    print("-" * 80)
    
    for _, row in avoid_players.head(20).iterrows():
        cv_str = f"{row['avg_cv']:.2f}" if pd.notna(row['avg_cv']) else "N/A"
        print(f"  {row['player_clean']:<26} {row['total_bets']:>4} {row['hit_rate']:>5.1%} "
              f"{row['roi']:>+6.1f}% {cv_str:>5} {row['markets']:>20}")
    
    # =============================
    # VOLATILITY vs PROFITABILITY
    # =============================
    print(f"\n{'='*60}")
    print("VOLATILITY vs PROFITABILITY")
    print("Are low-volatility (consistent) players more profitable to bet on?")
    print("="*60)
    
    vol_valid = reliable[reliable["avg_cv"].notna()].copy()
    
    # Bin by volatility
    vol_valid["vol_tier"] = pd.qcut(vol_valid["avg_cv"], q=4, labels=["Low", "Med-Low", "Med-High", "High"])
    
    print(f"\n{'Volatility Tier':<15} {'Players':>8} {'Avg Hit%':>10} {'Avg ROI':>10} {'% Profitable':>13}")
    print("-" * 60)
    
    for tier in ["Low", "Med-Low", "Med-High", "High"]:
        tier_data = vol_valid[vol_valid["vol_tier"] == tier]
        if tier_data.empty:
            continue
        print(f"  {tier:<13} {len(tier_data):>7} {tier_data['hit_rate'].mean():>9.1%} "
              f"{tier_data['roi'].mean():>+9.1f}% {(tier_data['roi'] > 0).mean():>12.0%}")
    
    # =============================
    # SEASON-OVER-SEASON CONSISTENCY
    # =============================
    print(f"\n{'='*60}")
    print("MOST CONSISTENT PLAYERS (profitable in MULTIPLE seasons)")
    print("="*60)
    
    # Per-player per-season
    player_season = bets.groupby(["player_clean", "season"]).agg(
        total=("bet_won", "count"),
        wins=("bet_won", "sum"),
    ).reset_index()
    player_season["hit"] = player_season["wins"] / player_season["total"]
    player_season["profitable"] = player_season["hit"] > 0.524
    
    # Players profitable in 2+ seasons with 10+ bets each
    player_consistency = player_season[player_season["total"] >= 10].groupby("player_clean").agg(
        seasons_played=("season", "count"),
        seasons_profitable=("profitable", "sum"),
        total_bets=("total", "sum"),
        avg_hit=("hit", "mean"),
    ).reset_index()
    
    multi_season = player_consistency[
        (player_consistency["seasons_played"] >= 2) & 
        (player_consistency["seasons_profitable"] >= 2)
    ].sort_values("avg_hit", ascending=False)
    
    print(f"\n  Players profitable in 2+ seasons (min 10 bets/season):")
    print(f"  {'Player':<28} {'Seasons':>8} {'Profitable':>11} {'Avg Hit%':>10} {'Total Bets':>11}")
    print("  " + "-" * 70)
    
    for _, row in multi_season.head(20).iterrows():
        print(f"  {row['player_clean']:<26} {row['seasons_played']:>7} "
              f"{int(row['seasons_profitable']):>10} {row['avg_hit']:>9.1%} {row['total_bets']:>10}")
    
    # Save bankable players list
    profitable_players[["player_clean", "total_bets", "hit_rate", "roi", "markets"]].to_parquet(
        PROC_DIR / "bankable_players.parquet", index=False
    )
    avoid_players[["player_clean", "total_bets", "hit_rate", "roi", "markets"]].to_parquet(
        PROC_DIR / "avoid_players.parquet", index=False
    )
    
    print(f"\nSaved: bankable_players.parquet ({len(profitable_players)} players)")
    print(f"Saved: avoid_players.parquet ({len(avoid_players)} players)")


def main():
    line_data = analyze_line_movement()
    analyze_player_predictability()
    
    print(f"\n\n{'='*70}")
    print("RESEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
