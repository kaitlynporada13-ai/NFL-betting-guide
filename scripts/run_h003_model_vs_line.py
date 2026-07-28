"""
H003: Model Prediction vs FanDuel Line — The Key Question
==========================================================
For each historical prop:
1. Generate what our model WOULD have predicted (using only pre-game data)
2. Compare to FanDuel's line
3. When our model disagrees, who was right?
4. Does selective betting (only where model sees edge) produce profit?

This is a walk-forward backtest: for each game, the model only uses
data from PRIOR weeks (no lookahead bias).
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


def build_rolling_prediction(player_stats: pd.DataFrame, stat_col: str, window: int = 5) -> pd.DataFrame:
    """
    Build a simple rolling-average prediction for each player-game.
    Uses only data available BEFORE that game (shifted by 1).
    This simulates what our model would have predicted.
    """
    df = player_stats.sort_values(["player_clean", "season", "week"]).copy()
    
    # Shifted rolling mean = prediction using only prior games
    df[f"pred_{stat_col}"] = (
        df.groupby("player_clean")[stat_col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    
    # Also build a weighted recent prediction (more weight to recent games)
    df[f"pred_{stat_col}_w3"] = (
        df.groupby("player_clean")[stat_col]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    )
    
    # Blended prediction (60% last-3, 40% last-5)
    df[f"pred_{stat_col}_blend"] = (
        0.6 * df[f"pred_{stat_col}_w3"] + 0.4 * df[f"pred_{stat_col}"]
    )
    
    return df


def run_h003():
    """Main H003 backtest."""
    print("=" * 70)
    print("H003: MODEL PREDICTION vs FANDUEL LINE")
    print("Does our model identify profitable edge against FanDuel?")
    print("=" * 70)

    # Load graded props
    graded_path = PROC_DIR / "props_graded_backtest.parquet"
    if not graded_path.exists():
        print("ERROR: Run run_backtest.py first")
        return
    graded = pd.read_parquet(graded_path)

    # Load player stats for building predictions
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    
    # Normalize names
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()

    graded["player_clean"] = graded["player_name"].str.strip().str.lower()

    # Map markets to stat columns
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards",
        "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards",
    }

    # Build rolling predictions for each stat
    print("\nBuilding walk-forward predictions...")
    for market, stat_col in market_to_stat.items():
        if stat_col in stats.columns:
            stats = build_rolling_prediction(stats, stat_col)

    # Merge predictions onto graded props
    print("Matching predictions to prop lines...")
    
    results_all = []
    
    for market, stat_col in market_to_stat.items():
        market_props = graded[graded["market"] == market].copy()
        if market_props.empty:
            continue

        pred_col = f"pred_{stat_col}_blend"
        if pred_col not in stats.columns:
            continue

        # Merge on player + season + week
        merged = market_props.merge(
            stats[["player_clean", "season", "week", pred_col]].dropna(subset=[pred_col]),
            on=["player_clean", "season", "week"],
            how="inner",
        )

        if merged.empty:
            continue

        # Calculate model's edge vs FanDuel line
        merged["model_prediction"] = merged[pred_col]
        merged["model_edge"] = merged["model_prediction"] - merged["fanduel_line"]
        merged["model_says_over"] = merged["model_edge"] > 0
        merged["model_says_under"] = merged["model_edge"] < 0

        results_all.append(merged)

    if not results_all:
        print("ERROR: No predictions could be matched")
        return

    all_results = pd.concat(results_all, ignore_index=True)
    print(f"\nTotal props with model predictions: {len(all_results):,}")

    # =====================================================
    # ANALYSIS
    # =====================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Overall: When model says Over, how often does over hit?
    model_overs = all_results[all_results["model_says_over"]]
    model_unders = all_results[all_results["model_says_under"]]

    over_correct = (model_overs["over_hit"] == True).sum()
    over_total = len(model_overs)
    over_rate = over_correct / over_total if over_total > 0 else 0

    under_correct = (model_unders["over_hit"] == False).sum()
    under_total = len(model_unders)
    under_rate = under_correct / under_total if under_total > 0 else 0

    print(f"\n--- OVERALL DIRECTIONAL ACCURACY ---")
    print(f"When model says OVER: {over_rate:.1%} hit rate ({over_correct:,}/{over_total:,})")
    print(f"When model says UNDER: {under_rate:.1%} hit rate ({under_correct:,}/{under_total:,})")
    print(f"Combined accuracy: {(over_correct + under_correct) / len(all_results):.1%}")
    print(f"(Need >52.4% for profit at -110)")

    # By edge size — THE KEY ANALYSIS
    print(f"\n--- BY MODEL EDGE SIZE (the money question) ---")
    print(f"{'Edge Threshold':<20} {'Bet Over':>12} {'Bet Under':>12} {'Combined':>12} {'Bets':>8} {'ROI':>8}")
    print("-" * 75)

    for min_edge in [0, 2, 5, 8, 10, 15, 20, 25, 30]:
        # Overs where model is min_edge above line
        big_overs = all_results[all_results["model_edge"] >= min_edge]
        big_unders = all_results[all_results["model_edge"] <= -min_edge]

        if len(big_overs) > 20:
            ov_hit = (big_overs["over_hit"] == True).sum()
            ov_total = len(big_overs)
            ov_rate = ov_hit / ov_total
        else:
            ov_rate = 0
            ov_total = 0

        if len(big_unders) > 20:
            un_hit = (big_unders["over_hit"] == False).sum()
            un_total = len(big_unders)
            un_rate = un_hit / un_total
        else:
            un_rate = 0
            un_total = 0

        total_bets = ov_total + un_total
        total_correct = (ov_hit if ov_total > 0 else 0) + (un_hit if un_total > 0 else 0)
        combined = total_correct / total_bets if total_bets > 0 else 0

        # ROI at -110
        wins = total_correct
        losses = total_bets - total_correct
        roi = ((wins * 100) - (losses * 110)) / (total_bets * 110) * 100 if total_bets > 0 else 0

        profitable = " ✓" if roi > 0 else ""
        print(f"Edge >= {min_edge:<12} {ov_rate:>11.1%} {un_rate:>11.1%} {combined:>11.1%} {total_bets:>7,} {roi:>+7.1f}%{profitable}")

    # By market
    print(f"\n--- BY MARKET (edge >= 5) ---")
    print(f"{'Market':<25} {'Over Hit':>10} {'Under Hit':>10} {'Bets':>8} {'ROI':>8}")
    print("-" * 65)

    for market in all_results["market"].unique():
        mkt = all_results[all_results["market"] == market]
        big_edge = mkt[mkt["model_edge"].abs() >= 5]
        if len(big_edge) < 20:
            continue

        overs = big_edge[big_edge["model_says_over"]]
        unders = big_edge[big_edge["model_says_under"]]

        ov_hit = (overs["over_hit"] == True).sum() if len(overs) > 0 else 0
        un_hit = (unders["over_hit"] == False).sum() if len(unders) > 0 else 0

        total = len(big_edge)
        wins = ov_hit + un_hit
        losses = total - wins
        roi = ((wins * 100) - (losses * 110)) / (total * 110) * 100 if total > 0 else 0

        ov_rate = ov_hit / len(overs) if len(overs) > 0 else 0
        un_rate = un_hit / len(unders) if len(unders) > 0 else 0

        print(f"{market:<25} {ov_rate:>9.1%} {un_rate:>9.1%} {total:>7,} {roi:>+7.1f}%")

    # By season
    print(f"\n--- BY SEASON (edge >= 5) ---")
    for season in sorted(all_results["season"].unique()):
        s = all_results[(all_results["season"] == season) & (all_results["model_edge"].abs() >= 5)]
        if len(s) < 20:
            continue
        overs = s[s["model_says_over"]]
        unders = s[s["model_says_under"]]
        wins = (overs["over_hit"] == True).sum() + (unders["over_hit"] == False).sum()
        total = len(s)
        rate = wins / total
        roi = ((wins * 100) - ((total - wins) * 110)) / (total * 110) * 100
        print(f"  {season}: {rate:.1%} hit rate, {roi:+.1f}% ROI ({total} bets)")

    # Confidence tiers
    print(f"\n--- CONFIDENCE TIERS ---")
    print(f"(Simulating our live betting strategy)")
    
    tiers = [
        ("High (edge >= 15)", 15),
        ("Medium (edge 8-15)", 8),
        ("Low (edge 5-8)", 5),
        ("No bet (edge < 5)", 0),
    ]

    total_profit = 0
    total_units = 0
    
    for tier_name, min_e in tiers:
        if min_e == 15:
            tier = all_results[all_results["model_edge"].abs() >= 15]
        elif min_e == 8:
            tier = all_results[(all_results["model_edge"].abs() >= 8) & (all_results["model_edge"].abs() < 15)]
        elif min_e == 5:
            tier = all_results[(all_results["model_edge"].abs() >= 5) & (all_results["model_edge"].abs() < 8)]
        else:
            tier = all_results[all_results["model_edge"].abs() < 5]

        if len(tier) < 10:
            print(f"  {tier_name}: insufficient data")
            continue

        overs = tier[tier["model_says_over"]]
        unders = tier[tier["model_says_under"]]
        wins = (overs["over_hit"] == True).sum() + (unders["over_hit"] == False).sum()
        total = len(tier)
        rate = wins / total
        losses = total - wins
        profit = (wins * 100) - (losses * 110)
        roi = profit / (total * 110) * 100

        if min_e >= 5:
            total_profit += profit
            total_units += total

        print(f"  {tier_name}: {rate:.1%} ({wins}/{total}) | ROI: {roi:+.1f}% | Profit: {profit/110:+.0f} units")

    if total_units > 0:
        overall_roi = total_profit / (total_units * 110) * 100
        print(f"\n  TOTAL (all edge >= 5): ROI {overall_roi:+.1f}% over {total_units} bets, {total_profit/110:+.0f} units")

    # Save detailed results
    all_results.to_parquet(PROC_DIR / "h003_model_vs_line.parquet", index=False)
    print(f"\nDetailed results saved to data/processed/h003_model_vs_line.parquet")


if __name__ == "__main__":
    run_h003()
