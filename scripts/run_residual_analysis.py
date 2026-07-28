"""
Residual Analysis: Back Into The WHY
=====================================
For every prop bet that our strategy triggered:
- What factors were present in BIG WINS?
- What factors were present in BIG LOSSES?
- Which factors distinguish wins from losses?

Goal: Find combination filters that improve hit rate from ~50% to >53%+
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


def load_enriched_data():
    """Load strategy results and enrich with all available context."""
    
    # Load H013 strategy results
    results = pd.read_parquet(PROC_DIR / "h013_strategy_results.parquet")
    bets = results[results["signal"] != "no_bet"].copy()
    
    # Load game context
    games = pd.read_parquet(RAW_DIR / "games_historical.parquet")
    
    # Merge game features
    game_cols = ["season", "week", "home_team", "away_team", "home_rest", "away_rest",
                 "div_game", "weekday", "roof", "surface", "temp", "wind",
                 "home_score", "away_score", "spread_line", "total_line"]
    available = [c for c in game_cols if c in games.columns]
    
    bets = bets.merge(
        games[available].drop_duplicates(subset=["season", "week", "home_team"]),
        on=["season", "week", "home_team"],
        how="left",
        suffixes=("", "_g"),
    )
    
    # Compute game total points
    if "home_score" in bets.columns and "away_score" in bets.columns:
        bets["game_total"] = bets["home_score"] + bets["away_score"]
        bets["high_scoring"] = bets["game_total"] > 48
        bets["low_scoring"] = bets["game_total"] < 34
    
    # Load injuries
    inj_path = RAW_DIR / "injuries_historical.parquet"
    if inj_path.exists():
        injuries = pd.read_parquet(inj_path)
        # Count injured starters per team per week
        out_counts = injuries[injuries["report_status"] == "Out"].groupby(
            ["season", "week", "team"]
        ).size().reset_index(name="team_players_out")
        
        bets = bets.merge(
            out_counts.rename(columns={"team": "home_team"}),
            on=["season", "week", "home_team"],
            how="left",
        )
        bets["team_players_out"] = bets["team_players_out"].fillna(0)
    
    # Add derived features
    bets["is_primetime"] = bets["weekday"].isin(["Thursday", "Monday", "Sunday Night"]) if "weekday" in bets.columns else False
    bets["is_dome"] = bets["roof"].isin(["dome", "closed"]) if "roof" in bets.columns else False
    bets["is_cold"] = bets["temp"].fillna(60) <= 35 if "temp" in bets.columns else False
    bets["is_windy"] = bets["wind"].fillna(0) >= 15 if "wind" in bets.columns else False
    bets["is_division"] = bets["div_game"] == 1 if "div_game" in bets.columns else False
    bets["is_early_season"] = bets["week"] <= 4
    bets["is_late_season"] = bets["week"] >= 14
    
    # Deviation magnitude buckets
    bets["deviation_bucket"] = pd.cut(
        bets["line_deviation_pct"].abs(),
        bins=[0, 0.10, 0.15, 0.20, 0.30, 1.0],
        labels=["10-15%", "15-20%", "20-30%", "30-50%", "50%+"],
    )
    
    return bets


def compare_wins_vs_losses(bets: pd.DataFrame):
    """Compare characteristics of wins vs losses."""
    
    wins = bets[bets["bet_won_10pct"] == True]
    losses = bets[bets["bet_won_10pct"] == False]
    
    print(f"\n{'='*70}")
    print(f"WINS vs LOSSES COMPARISON")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"{'='*70}")
    
    # Compare binary factors
    factors = [
        ("is_dome", "Dome game"),
        ("is_cold", "Cold game (<=35F)"),
        ("is_windy", "Windy (>=15mph)"),
        ("is_division", "Division game"),
        ("is_primetime", "Primetime game"),
        ("is_early_season", "Weeks 1-4"),
        ("is_late_season", "Weeks 14+"),
    ]
    
    print(f"\n{'Factor':<25} {'Win %':>8} {'Loss %':>8} {'Diff':>8} {'Signal?':>8}")
    print("-" * 60)
    
    signals = []
    
    for col, label in factors:
        if col not in bets.columns:
            continue
        win_pct = wins[col].mean() if col in wins.columns else 0
        loss_pct = losses[col].mean() if col in losses.columns else 0
        diff = win_pct - loss_pct
        significant = abs(diff) > 0.02
        
        marker = "→" if significant else " "
        print(f"{marker} {label:<23} {win_pct:>7.1%} {loss_pct:>7.1%} {diff:>+7.1%} {'YES' if significant else '':>8}")
        
        if significant:
            signals.append({"factor": label, "col": col, "diff": diff, "higher_in": "wins" if diff > 0 else "losses"})
    
    # Compare continuous factors
    print(f"\n{'Continuous Factor':<25} {'Win Mean':>10} {'Loss Mean':>10} {'Diff':>10}")
    print("-" * 60)
    
    continuous = [
        ("line_deviation_pct", "Line deviation %"),
        ("team_players_out", "Team players out"),
        ("week", "Week number"),
    ]
    
    for col, label in continuous:
        if col not in bets.columns:
            continue
        win_mean = wins[col].mean()
        loss_mean = losses[col].mean()
        diff = win_mean - loss_mean
        print(f"  {label:<23} {win_mean:>9.2f} {loss_mean:>9.2f} {diff:>+9.2f}")
    
    return signals


def find_profitable_filters(bets: pd.DataFrame):
    """Test each factor as a filter to improve hit rate."""
    
    print(f"\n{'='*70}")
    print(f"FILTER TESTING: Which factors improve hit rate when ADDED?")
    print(f"{'='*70}")
    
    base_hit = bets["bet_won_10pct"].mean()
    base_n = len(bets)
    print(f"\nBaseline: {base_hit:.1%} hit rate ({base_n} bets)")
    print(f"Need: >52.4% for profit at -110\n")
    
    # Test each filter
    filters = [
        # Exclude conditions
        ("~is_cold", "Exclude cold games", ~bets.get("is_cold", pd.Series(False))),
        ("~is_windy", "Exclude windy games", ~bets.get("is_windy", pd.Series(False))),
        ("~is_late_season", "Exclude weeks 14+", ~bets.get("is_late_season", pd.Series(False))),
        ("early_season_only", "Only weeks 1-6", bets["week"] <= 6),
        ("mid_season", "Only weeks 5-12", bets["week"].between(5, 12)),
        ("dome_only", "Dome games only", bets.get("is_dome", pd.Series(False))),
        ("not_division", "Exclude division games", ~bets.get("is_division", pd.Series(False))),
        ("under_only", "Only UNDER bets", bets["signal"] == "bet_under"),
        ("over_only", "Only OVER bets", bets["signal"] == "bet_over"),
        ("moderate_dev", "Deviation 10-20% only", bets["line_deviation_pct"].abs().between(0.10, 0.20)),
        ("small_dev", "Deviation 10-15% only", bets["line_deviation_pct"].abs().between(0.10, 0.15)),
        ("few_injuries", "Team <3 players out", bets.get("team_players_out", pd.Series(0)) < 3),
        ("pass_tds_only", "Only passing TD props", bets["market"] == "player_pass_tds"),
        ("no_rec_yards", "Exclude receiving yards", bets["market"] != "player_reception_yds"),
        ("receptions_under", "Receptions + UNDER only", (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under")),
    ]
    
    print(f"{'Filter':<30} {'Hit%':>7} {'Δ':>7} {'Bets':>7} {'ROI':>8} {'Profitable':>11}")
    print("-" * 75)
    
    profitable_filters = []
    
    for name, label, mask in filters:
        if not isinstance(mask, pd.Series):
            continue
        subset = bets[mask]
        if len(subset) < 50:
            continue
        
        hit = subset["bet_won_10pct"].mean()
        delta = hit - base_hit
        n = len(subset)
        wins = subset["bet_won_10pct"].sum()
        losses = n - wins
        roi = ((wins * 100) - (losses * 110)) / (n * 110) * 100
        is_profitable = roi > 0
        
        marker = "✓" if is_profitable else " "
        print(f"{marker} {label:<28} {hit:>6.1%} {delta:>+6.1%} {n:>6,} {roi:>+7.1f}% {'PROFIT' if is_profitable else '':>10}")
        
        if is_profitable:
            profitable_filters.append({"filter": label, "hit_rate": hit, "roi": roi, "bets": n})
    
    # Test COMBINATIONS of top filters
    print(f"\n{'='*70}")
    print(f"COMBINATION FILTERS")
    print(f"{'='*70}\n")
    
    combos = [
        ("Pass TDs + Under only",
         (bets["market"] == "player_pass_tds") & (bets["signal"] == "bet_under")),
        ("Pass TDs + Over only",
         (bets["market"] == "player_pass_tds") & (bets["signal"] == "bet_over")),
        ("Receptions Under + not cold",
         (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets.get("is_cold", pd.Series(False)))),
        ("Under + early season (wk 1-6)",
         (bets["signal"] == "bet_under") & (bets["week"] <= 6)),
        ("Under + moderate deviation (10-20%)",
         (bets["signal"] == "bet_under") & (bets["line_deviation_pct"].abs().between(0.10, 0.20))),
        ("No rec yards + Under",
         (bets["market"] != "player_reception_yds") & (bets["signal"] == "bet_under")),
        ("Rush yards + Under + early",
         (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_under") & (bets["week"] <= 8)),
        ("Pass TDs + early season",
         (bets["market"] == "player_pass_tds") & (bets["week"] <= 8)),
        ("All props + Under + week 1-4",
         (bets["signal"] == "bet_under") & (bets["week"] <= 4)),
        ("Dome + Pass TDs",
         (bets["market"] == "player_pass_tds") & (bets.get("is_dome", pd.Series(False)))),
        ("Not division + Under + 10-15% dev",
         (~bets.get("is_division", pd.Series(False))) & (bets["signal"] == "bet_under") & (bets["line_deviation_pct"].abs().between(0.10, 0.15))),
    ]
    
    print(f"{'Combo':<45} {'Hit%':>7} {'Bets':>7} {'ROI':>8}")
    print("-" * 70)
    
    for label, mask in combos:
        if not isinstance(mask, pd.Series):
            continue
        subset = bets[mask]
        if len(subset) < 30:
            continue
        
        hit = subset["bet_won_10pct"].mean()
        n = len(subset)
        wins = subset["bet_won_10pct"].sum()
        losses = n - wins
        roi = ((wins * 100) - (losses * 110)) / (n * 110) * 100
        
        marker = "✓" if roi > 0 else " "
        print(f"{marker} {label:<43} {hit:>6.1%} {n:>6,} {roi:>+7.1f}%")
    
    return profitable_filters


def main():
    print("=" * 70)
    print("RESIDUAL ANALYSIS + COMBINATION FILTER DISCOVERY")
    print("=" * 70)
    
    bets = load_enriched_data()
    print(f"Loaded {len(bets):,} bets with full context")
    
    # Compare what wins and losses look like
    signals = compare_wins_vs_losses(bets)
    
    # Find profitable filters
    profitable = find_profitable_filters(bets)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"ACTIONABLE FINDINGS")
    print(f"{'='*70}")
    
    if profitable:
        print(f"\n{len(profitable)} profitable filter(s) found:")
        for p in sorted(profitable, key=lambda x: x["roi"], reverse=True):
            print(f"  {p['filter']}: {p['hit_rate']:.1%} hit, {p['roi']:+.1f}% ROI, {p['bets']} bets")
    else:
        print("\nNo single filter produced profit. Need combinations or different approach.")


if __name__ == "__main__":
    main()
