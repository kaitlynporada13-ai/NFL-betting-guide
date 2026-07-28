"""
H013 Strategy Backtest: Mean Reversion Edge
============================================
Strategy: When FanDuel's line deviates significantly from the player's
rolling average, bet toward the average (mean reversion).

- Line >10% above average → BET UNDER
- Line >10% below average → BET OVER
- Otherwise → NO BET

Then: residual analysis on the biggest misses to find combination effects.
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


def build_player_rolling_averages(stats: pd.DataFrame) -> pd.DataFrame:
    """Build rolling averages using only prior games (no lookahead)."""
    stat_cols = ["passing_yards", "passing_tds", "rushing_yards", "receptions", "receiving_yards"]
    available = [c for c in stat_cols if c in stats.columns]
    
    df = stats.sort_values(["player_clean", "season", "week"]).copy()
    
    for col in available:
        # 5-game rolling average shifted (only uses prior data)
        df[f"{col}_avg5"] = (
            df.groupby("player_clean")[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
        )
        # 3-game rolling
        df[f"{col}_avg3"] = (
            df.groupby("player_clean")[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
        )
        # Season average (all prior games)
        df[f"{col}_season_avg"] = (
            df.groupby(["player_clean", "season"])[col]
            .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
        )
    
    return df


def run_strategy_backtest():
    """Run the H013 mean reversion strategy."""
    print("=" * 70)
    print("H013 STRATEGY BACKTEST: MEAN REVERSION vs FANDUEL")
    print("=" * 70)

    # Load props
    graded = pd.read_parquet(PROC_DIR / "props_graded_backtest.parquet")
    graded["player_clean"] = graded["player_name"].str.strip().str.lower()
    
    # Load stats and build rolling averages
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    print("Building player rolling averages (walk-forward)...")
    stats = build_player_rolling_averages(stats)
    
    # Map markets to stat columns
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards",
        "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards",
    }
    
    # Merge rolling averages onto props
    print("Matching rolling averages to prop lines...")
    all_bets = []
    
    for market, stat_col in market_to_stat.items():
        avg_col = f"{stat_col}_avg5"
        avg3_col = f"{stat_col}_avg3"
        season_avg_col = f"{stat_col}_season_avg"
        
        if avg_col not in stats.columns:
            continue
        
        mkt_props = graded[graded["market"] == market].copy()
        if mkt_props.empty:
            continue
        
        # Merge
        merged = mkt_props.merge(
            stats[["player_clean", "season", "week", avg_col, avg3_col, season_avg_col]].dropna(subset=[avg_col]).drop_duplicates(),
            on=["player_clean", "season", "week"],
            how="inner",
        )
        
        if merged.empty:
            continue
        
        # Calculate line deviation from player's rolling average
        merged["player_avg"] = merged[avg_col]
        merged["player_avg3"] = merged[avg3_col]
        merged["line_deviation"] = merged["fanduel_line"] - merged["player_avg"]
        merged["line_deviation_pct"] = merged["line_deviation"] / merged["player_avg"].replace(0, np.nan)
        
        # Strategy signals
        merged["signal"] = "no_bet"
        merged.loc[merged["line_deviation_pct"] > 0.10, "signal"] = "bet_under"  # Line too high
        merged.loc[merged["line_deviation_pct"] < -0.10, "signal"] = "bet_over"  # Line too low
        
        # More aggressive thresholds
        merged["signal_5pct"] = "no_bet"
        merged.loc[merged["line_deviation_pct"] > 0.05, "signal_5pct"] = "bet_under"
        merged.loc[merged["line_deviation_pct"] < -0.05, "signal_5pct"] = "bet_over"
        
        merged["signal_15pct"] = "no_bet"
        merged.loc[merged["line_deviation_pct"] > 0.15, "signal_15pct"] = "bet_under"
        merged.loc[merged["line_deviation_pct"] < -0.15, "signal_15pct"] = "bet_over"
        
        # Grade: did the bet win?
        merged["bet_won_10pct"] = (
            ((merged["signal"] == "bet_over") & (merged["result"] == "won")) |
            ((merged["signal"] == "bet_under") & (merged["result"] == "lost"))  # under = over didn't hit
        )
        merged["bet_won_5pct"] = (
            ((merged["signal_5pct"] == "bet_over") & (merged["result"] == "won")) |
            ((merged["signal_5pct"] == "bet_under") & (merged["result"] == "lost"))
        )
        merged["bet_won_15pct"] = (
            ((merged["signal_15pct"] == "bet_over") & (merged["result"] == "won")) |
            ((merged["signal_15pct"] == "bet_under") & (merged["result"] == "lost"))
        )
        
        all_bets.append(merged)
    
    combined = pd.concat(all_bets, ignore_index=True)
    print(f"Total props with rolling averages: {len(combined):,}")
    
    # =========================================
    # RESULTS BY THRESHOLD
    # =========================================
    print("\n" + "=" * 70)
    print("STRATEGY PERFORMANCE BY THRESHOLD")
    print("=" * 70)
    
    for threshold, signal_col, won_col in [
        ("5%", "signal_5pct", "bet_won_5pct"),
        ("10%", "signal", "bet_won_10pct"),
        ("15%", "signal_15pct", "bet_won_15pct"),
    ]:
        bets = combined[combined[signal_col] != "no_bet"]
        if bets.empty:
            continue
        
        wins = bets[won_col].sum()
        total = len(bets)
        hit_rate = wins / total
        losses = total - wins
        profit = (wins * 100) - (losses * 110)
        roi = profit / (total * 110) * 100
        
        print(f"\n  Threshold: Line deviates >= {threshold} from rolling avg")
        print(f"  Bets: {total:,} | Wins: {wins:,} | Hit Rate: {hit_rate:.1%}")
        print(f"  ROI: {roi:+.1f}% | Profit: {profit/110:+.1f} units")
        
        # Breakdown by direction
        overs = bets[bets[signal_col] == "bet_over"]
        unders = bets[bets[signal_col] == "bet_under"]
        
        if len(overs) > 0:
            ov_wins = overs[won_col].sum()
            ov_rate = ov_wins / len(overs)
            print(f"    OVERS: {ov_rate:.1%} ({ov_wins}/{len(overs)})")
        if len(unders) > 0:
            un_wins = unders[won_col].sum()
            un_rate = un_wins / len(unders)
            print(f"    UNDERS: {un_rate:.1%} ({un_wins}/{len(unders)})")
    
    # =========================================
    # BY MARKET (10% threshold)
    # =========================================
    print("\n" + "=" * 70)
    print("BY MARKET (10% threshold)")
    print("=" * 70)
    
    bets_10 = combined[combined["signal"] != "no_bet"]
    
    print(f"\n{'Market':<28} {'Bets':>6} {'Wins':>6} {'Hit%':>7} {'ROI':>8} {'Units':>8}")
    print("-" * 65)
    
    for market in bets_10["market"].unique():
        mkt = bets_10[bets_10["market"] == market]
        wins = mkt["bet_won_10pct"].sum()
        total = len(mkt)
        hit = wins / total
        profit = (wins * 100) - ((total - wins) * 110)
        roi = profit / (total * 110) * 100
        print(f"  {market:<26} {total:>5,} {wins:>5,} {hit:>6.1%} {roi:>+7.1f}% {profit/110:>+7.1f}")
    
    # =========================================
    # BY SEASON
    # =========================================
    print(f"\n--- BY SEASON (10% threshold) ---")
    for season in sorted(bets_10["season"].unique()):
        s = bets_10[bets_10["season"] == season]
        wins = s["bet_won_10pct"].sum()
        total = len(s)
        hit = wins / total
        roi = ((wins * 100) - ((total - wins) * 110)) / (total * 110) * 100
        print(f"  {season}: {hit:.1%} hit rate, {roi:+.1f}% ROI ({total} bets)")
    
    # =========================================
    # COMBINED WITH H011 (injury return)
    # =========================================
    print("\n" + "=" * 70)
    print("H013 + H011 COMBINATION (injury return + line deviation)")
    print("=" * 70)
    
    inj_path = RAW_DIR / "injuries_historical.parquet"
    if inj_path.exists():
        injuries = pd.read_parquet(inj_path)
        out_players = injuries[injuries["report_status"] == "Out"][["season", "week", "full_name"]].copy()
        out_players["player_clean"] = out_players["full_name"].str.strip().str.lower()
        out_players["return_week"] = out_players["week"] + 1
        
        return_flags = out_players[["season", "return_week", "player_clean"]].rename(
            columns={"return_week": "week"}
        ).drop_duplicates()
        return_flags["is_return"] = True
        
        combined_inj = combined.merge(return_flags, on=["season", "week", "player_clean"], how="left")
        combined_inj["is_return"] = combined_inj["is_return"].fillna(False)
        
        # Injury return + line above average = STRONG under signal
        strong_under = combined_inj[
            (combined_inj["is_return"] == True) & 
            (combined_inj["signal"] == "bet_under")
        ]
        if len(strong_under) > 10:
            wins = strong_under["bet_won_10pct"].sum()
            total = len(strong_under)
            hit = wins / total
            roi = ((wins * 100) - ((total - wins) * 110)) / (total * 110) * 100
            print(f"  Injury return + line above avg (UNDER): {hit:.1%} hit rate ({wins}/{total}), ROI: {roi:+.1f}%")
    
    # =========================================
    # SAVE RESULTS
    # =========================================
    combined.to_parquet(PROC_DIR / "h013_strategy_results.parquet", index=False)
    print(f"\nDetailed results saved to data/processed/h013_strategy_results.parquet")
    
    # =========================================
    # RESIDUAL ANALYSIS — FIND THE WHY
    # =========================================
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS: BIGGEST MISSES (backing into the WHY)")
    print("=" * 70)
    
    # Large positive residuals: player massively exceeded line AND our strategy
    combined["residual"] = combined["actual_stat"] - combined["fanduel_line"]
    combined["abs_residual"] = combined["residual"].abs()
    
    # Focus on bets we placed that LOST by a lot
    bets_placed = combined[combined["signal"] != "no_bet"].copy()
    big_losses = bets_placed[
        (bets_placed["bet_won_10pct"] == False) & 
        (bets_placed["abs_residual"] > bets_placed["abs_residual"].quantile(0.75))
    ]
    
    print(f"\n  Big losses to analyze: {len(big_losses)} bets")
    print(f"  Average residual on big losses: {big_losses['residual'].mean():+.1f}")
    
    # What do big losses have in common?
    print(f"\n  --- Profile of BIG LOSSES (when our strategy failed badly) ---")
    
    # By market
    print(f"\n  By market:")
    for market, count in big_losses["market"].value_counts().items():
        pct = count / len(big_losses)
        print(f"    {market}: {count} ({pct:.0%})")
    
    # By signal direction
    print(f"\n  By signal direction:")
    for sig, count in big_losses["signal"].value_counts().items():
        print(f"    {sig}: {count}")
    
    # Average line deviation on losses
    print(f"\n  Avg line deviation (pct) on big losses: {big_losses['line_deviation_pct'].mean():+.1%}")
    print(f"  Avg line deviation (pct) on all wins: {bets_placed[bets_placed['bet_won_10pct']==True]['line_deviation_pct'].mean():+.1%}")
    
    # Season distribution of losses
    print(f"\n  By season (big losses):")
    for season, count in big_losses["season"].value_counts().sort_index().items():
        total_in_season = len(bets_placed[bets_placed["season"] == season])
        print(f"    {season}: {count} big losses / {total_in_season} total bets ({count/total_in_season:.0%})")
    
    # Week distribution — are early/late season different?
    print(f"\n  By week range (big losses):")
    early = big_losses[big_losses["week"] <= 4]
    mid = big_losses[(big_losses["week"] > 4) & (big_losses["week"] <= 12)]
    late = big_losses[big_losses["week"] > 12]
    total_early = len(bets_placed[bets_placed["week"] <= 4])
    total_mid = len(bets_placed[(bets_placed["week"] > 4) & (bets_placed["week"] <= 12)])
    total_late = len(bets_placed[bets_placed["week"] > 12])
    print(f"    Weeks 1-4:  {len(early)} losses / {total_early} bets ({len(early)/total_early:.0%} loss rate)")
    print(f"    Weeks 5-12: {len(mid)} losses / {total_mid} bets ({len(mid)/total_mid:.0%} loss rate)")
    print(f"    Weeks 13+:  {len(late)} losses / {total_late} bets ({len(late)/total_late:.0%} loss rate)")
    
    # FLIP SIDE: What do our BIGGEST WINS have in common?
    big_wins = bets_placed[
        (bets_placed["bet_won_10pct"] == True) & 
        (bets_placed["abs_residual"] > bets_placed["abs_residual"].quantile(0.75))
    ]
    
    print(f"\n  --- Profile of BIG WINS ---")
    print(f"  Big wins: {len(big_wins)} bets")
    print(f"\n  By week range (big wins):")
    early_w = big_wins[big_wins["week"] <= 4]
    mid_w = big_wins[(big_wins["week"] > 4) & (big_wins["week"] <= 12)]
    late_w = big_wins[big_wins["week"] > 12]
    print(f"    Weeks 1-4:  {len(early_w)}")
    print(f"    Weeks 5-12: {len(mid_w)}")
    print(f"    Weeks 13+:  {len(late_w)}")
    
    # Key insight: are losses concentrated early season (when averages are unstable)?
    print(f"\n  --- KEY INSIGHT ---")
    early_hit = bets_placed[bets_placed["week"] <= 4]["bet_won_10pct"].mean()
    mid_hit = bets_placed[(bets_placed["week"] > 4) & (bets_placed["week"] <= 12)]["bet_won_10pct"].mean()
    late_hit = bets_placed[bets_placed["week"] > 12]["bet_won_10pct"].mean()
    print(f"  Strategy hit rate by season phase:")
    print(f"    Weeks 1-4:  {early_hit:.1%} (rolling avg unstable)")
    print(f"    Weeks 5-12: {mid_hit:.1%} (averages stabilize)")
    print(f"    Weeks 13+:  {late_hit:.1%} (most data, most stable)")


if __name__ == "__main__":
    run_strategy_backtest()
