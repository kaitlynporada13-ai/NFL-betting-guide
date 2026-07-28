"""
Deeper Hypothesis Testing: Game Script, Pace, Opponent, Usage Patterns
======================================================================
Focus: What factors correlate with props going OVER vs UNDER?
Testing: game script (blowout/close), pace, opponent defense quality,
         player-specific patterns, time slots, spread context, etc.
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


def load_enriched():
    """Load fully enriched bets dataset."""
    path = PROC_DIR / "bets_fully_enriched.parquet"
    return pd.read_parquet(path)


def calc_roi(subset):
    """Quick ROI calc for a subset of bets."""
    if len(subset) < 25:
        return None
    wins = subset["bet_won_10pct"].sum()
    total = len(subset)
    hit = wins / total
    roi = ((wins * 100) - ((total - wins) * 110)) / (total * 110) * 100
    return {"hit": hit, "roi": roi, "n": total, "wins": wins}


def print_result(label, result, baseline_hit=0.499):
    """Pretty print a test result."""
    if result is None:
        return
    marker = "✓" if result["roi"] > 0 else " "
    direction = "OVER" if result["hit"] > 0.52 else "UNDER" if result["hit"] < 0.48 else ""
    print(f"{marker} {label:<50} {result['hit']:>6.1%} {result['n']:>5,} {result['roi']:>+7.1f}%")


def test_game_script(bets):
    """Test: Does game script (blowout vs close game) affect prop outcomes?"""
    print(f"\n{'='*70}")
    print("GAME SCRIPT: Blowouts vs Close Games")
    print("Theory: Blowouts change play-calling (more run, garbage time)")
    print("="*70 + "\n")
    
    if "spread_line" not in bets.columns or bets["spread_line"].isna().all():
        print("  No spread data available for game script analysis")
        return
    
    # Use spread as proxy for expected game script
    # Big favorites → expected blowout → run more, pass less in 2nd half
    bets["big_favorite"] = bets["spread_line"].fillna(0).abs() >= 7
    bets["close_game_expected"] = bets["spread_line"].fillna(0).abs() <= 3
    bets["medium_spread"] = bets["spread_line"].fillna(0).abs().between(3.5, 6.5)
    
    # Actual game closeness (final margin)
    if "home_score" in bets.columns and "away_score" in bets.columns:
        bets["actual_margin"] = (bets["home_score"] - bets["away_score"]).abs()
        bets["was_blowout"] = bets["actual_margin"] >= 17
        bets["was_close"] = bets["actual_margin"] <= 7
    
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    tests = [
        # Expected game script (pre-game)
        ("Big favorite game (spread >= 7) + OVER", bets[(bets["big_favorite"]) & (bets["signal"] == "bet_over")]),
        ("Big favorite game (spread >= 7) + UNDER", bets[(bets["big_favorite"]) & (bets["signal"] == "bet_under")]),
        ("Close game expected (spread <= 3) + OVER", bets[(bets["close_game_expected"]) & (bets["signal"] == "bet_over")]),
        ("Close game expected (spread <= 3) + UNDER", bets[(bets["close_game_expected"]) & (bets["signal"] == "bet_under")]),
        
        # Expected script + specific markets
        ("Big fav + Rush OVER", bets[(bets["big_favorite"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")]),
        ("Big fav + Pass UNDER", bets[(bets["big_favorite"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")]),
        ("Close game + Pass TDs", bets[(bets["close_game_expected"]) & (bets["market"] == "player_pass_tds")]),
        ("Close game + Receptions OVER", bets[(bets["close_game_expected"]) & (bets["market"] == "player_receptions") & (bets["signal"] == "bet_over")]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_totals_context(bets):
    """Test: Does the expected game total affect individual props?"""
    print(f"\n{'='*70}")
    print("GAME TOTAL CONTEXT: High/Low Expected Scoring")
    print("Theory: High total = more volume/yards for everyone")
    print("="*70 + "\n")
    
    if "total_line" not in bets.columns or bets["total_line"].isna().all():
        print("  No total line data available")
        return
    
    bets["high_total"] = bets["total_line"].fillna(44) >= 48
    bets["low_total"] = bets["total_line"].fillna(44) <= 40
    bets["normal_total"] = bets["total_line"].fillna(44).between(41, 47)
    
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    tests = [
        ("High total (>=48) + all OVER", bets[(bets["high_total"]) & (bets["signal"] == "bet_over")]),
        ("High total (>=48) + all UNDER", bets[(bets["high_total"]) & (bets["signal"] == "bet_under")]),
        ("Low total (<=40) + all OVER", bets[(bets["low_total"]) & (bets["signal"] == "bet_over")]),
        ("Low total (<=40) + all UNDER", bets[(bets["low_total"]) & (bets["signal"] == "bet_under")]),
        ("High total + Pass TDs", bets[(bets["high_total"]) & (bets["market"] == "player_pass_tds")]),
        ("Low total + Pass UNDER", bets[(bets["low_total"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")]),
        ("High total + Rec yards OVER", bets[(bets["high_total"]) & (bets["market"] == "player_reception_yds") & (bets["signal"] == "bet_over")]),
        ("Low total + Rush OVER", bets[(bets["low_total"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")]),
        ("Normal total + Pass TDs", bets[(bets["normal_total"]) & (bets["market"] == "player_pass_tds")]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_weekly_patterns(bets):
    """Test: Are certain weeks/days systematically different?"""
    print(f"\n{'='*70}")
    print("WEEKLY & DAY-OF-WEEK PATTERNS")
    print("Theory: Certain weeks (post-bye, pre-playoff) have patterns")
    print("="*70 + "\n")
    
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    tests = [
        # Day of week
        ("Sunday only + OVER", bets[(bets.get("weekday","") == "Sunday") & (bets["signal"] == "bet_over")]),
        ("Sunday only + UNDER", bets[(bets.get("weekday","") == "Sunday") & (bets["signal"] == "bet_under")]),
        ("Thursday + all", bets[bets["is_thursday"]]),
        ("Monday + OVER", bets[(bets["is_monday"]) & (bets["signal"] == "bet_over")]),
        ("Monday + UNDER", bets[(bets["is_monday"]) & (bets["signal"] == "bet_under")]),
        
        # Specific weeks
        ("Week 1 only + UNDER", bets[(bets["week"] == 1) & (bets["signal"] == "bet_under")]),
        ("Week 1 only + OVER", bets[(bets["week"] == 1) & (bets["signal"] == "bet_over")]),
        ("Week 17-18 (late) + UNDER", bets[(bets["week"] >= 17) & (bets["signal"] == "bet_under")]),
        ("Week 17-18 (late) + OVER", bets[(bets["week"] >= 17) & (bets["signal"] == "bet_over")]),
        ("Weeks 2-4 + UNDER", bets[(bets["week"].between(2, 4)) & (bets["signal"] == "bet_under")]),
        ("Weeks 5-8 + Pass TDs", bets[(bets["week"].between(5, 8)) & (bets["market"] == "player_pass_tds")]),
        ("Weeks 9-12 + Pass TDs", bets[(bets["week"].between(9, 12)) & (bets["market"] == "player_pass_tds")]),
        ("Weeks 13-18 + Pass TDs", bets[(bets["week"] >= 13) & (bets["market"] == "player_pass_tds")]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_player_consistency(bets):
    """Test: Do consistent vs volatile players behave differently vs props?"""
    print(f"\n{'='*70}")
    print("PLAYER CONSISTENCY: Reliable vs Volatile players")
    print("Theory: Volatile players create over/under opportunities")
    print("="*70 + "\n")
    
    # Load player features for std dev
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    # Calculate career volatility for each player-stat combo
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards",
        "player_receptions": "receptions",
    }
    
    enriched_parts = []
    for market, stat_col in market_to_stat.items():
        if stat_col not in stats.columns:
            continue
        
        # Player volatility (std / mean = coefficient of variation)
        player_vol = stats.groupby("player_clean")[stat_col].agg(["mean", "std"]).reset_index()
        player_vol["cv"] = player_vol["std"] / player_vol["mean"].replace(0, np.nan)
        player_vol["high_volatility"] = player_vol["cv"] > player_vol["cv"].quantile(0.70)
        player_vol["low_volatility"] = player_vol["cv"] < player_vol["cv"].quantile(0.30)
        
        mkt_bets = bets[bets["market"] == market].copy()
        mkt_bets = mkt_bets.merge(
            player_vol[["player_clean", "cv", "high_volatility", "low_volatility"]],
            on="player_clean", how="left"
        )
        enriched_parts.append(mkt_bets)
    
    if not enriched_parts:
        print("  No volatility data available")
        return
    
    vol_bets = pd.concat(enriched_parts, ignore_index=True)
    
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    tests = [
        ("High volatility players + OVER", vol_bets[(vol_bets["high_volatility"] == True) & (vol_bets["signal"] == "bet_over")]),
        ("High volatility players + UNDER", vol_bets[(vol_bets["high_volatility"] == True) & (vol_bets["signal"] == "bet_under")]),
        ("Low volatility players + OVER", vol_bets[(vol_bets["low_volatility"] == True) & (vol_bets["signal"] == "bet_over")]),
        ("Low volatility players + UNDER", vol_bets[(vol_bets["low_volatility"] == True) & (vol_bets["signal"] == "bet_under")]),
        ("High vol + Rec yards OVER", vol_bets[(vol_bets["high_volatility"] == True) & (vol_bets["market"] == "player_reception_yds") & (vol_bets["signal"] == "bet_over")]),
        ("Low vol + Receptions UNDER", vol_bets[(vol_bets["low_volatility"] == True) & (vol_bets["market"] == "player_receptions") & (vol_bets["signal"] == "bet_under")]),
        ("High vol + Rush UNDER", vol_bets[(vol_bets["high_volatility"] == True) & (vol_bets["market"] == "player_rush_yds") & (vol_bets["signal"] == "bet_under")]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_line_movement_proxy(bets):
    """Test: Does the line-to-average ratio tell us about sharp money?"""
    print(f"\n{'='*70}")
    print("LINE DEVIATION DEPTH: Sweet spots in deviation size")
    print("Theory: Small deviations = noise. Large = FanDuel knows something.")
    print("="*70 + "\n")
    
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    tests = [
        # Exact deviation ranges
        ("UNDER + deviation 10-12%", bets[(bets["signal"] == "bet_under") & (bets["line_deviation_pct"].between(0.10, 0.12))]),
        ("UNDER + deviation 12-15%", bets[(bets["signal"] == "bet_under") & (bets["line_deviation_pct"].between(0.12, 0.15))]),
        ("UNDER + deviation 15-20%", bets[(bets["signal"] == "bet_under") & (bets["line_deviation_pct"].between(0.15, 0.20))]),
        ("UNDER + deviation 20-30%", bets[(bets["signal"] == "bet_under") & (bets["line_deviation_pct"].between(0.20, 0.30))]),
        ("UNDER + deviation 30%+", bets[(bets["signal"] == "bet_under") & (bets["line_deviation_pct"] > 0.30)]),
        ("OVER + deviation 10-12%", bets[(bets["signal"] == "bet_over") & (bets["line_deviation_pct"].between(-0.12, -0.10))]),
        ("OVER + deviation 12-15%", bets[(bets["signal"] == "bet_over") & (bets["line_deviation_pct"].between(-0.15, -0.12))]),
        ("OVER + deviation 15-20%", bets[(bets["signal"] == "bet_over") & (bets["line_deviation_pct"].between(-0.20, -0.15))]),
        ("OVER + deviation 20-30%", bets[(bets["signal"] == "bet_over") & (bets["line_deviation_pct"].between(-0.30, -0.20))]),
        ("OVER + deviation 30%+", bets[(bets["signal"] == "bet_over") & (bets["line_deviation_pct"] < -0.30)]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_home_away(bets):
    """Test: Home vs away player performance vs props."""
    print(f"\n{'='*70}")
    print("HOME vs AWAY: Player performance context")
    print("Theory: Home players in comfortable environments outperform")
    print("="*70 + "\n")
    
    # We don't have which team the player is on directly, but we can 
    # look at dome for home team vs away
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    # Home team in dome (double comfort factor)
    tests = [
        ("Dome + all UNDER", bets[(bets["is_dome"]) & (bets["signal"] == "bet_under")]),
        ("Dome + Receptions UNDER", bets[(bets["is_dome"]) & (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under")]),
        ("Outdoor + Receptions UNDER", bets[(~bets["is_dome"]) & (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under")]),
        ("Dome + Rush OVER", bets[(bets["is_dome"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")]),
        ("Outdoor + Rush UNDER", bets[(~bets["is_dome"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_under")]),
        ("Dome + Pass TDs + OVER", bets[(bets["is_dome"]) & (bets["market"] == "player_pass_tds") & (bets["signal"] == "bet_over")]),
        ("Dome + Pass TDs + UNDER", bets[(bets["is_dome"]) & (bets["market"] == "player_pass_tds") & (bets["signal"] == "bet_under")]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_injury_context(bets):
    """Test: Team injury load and opponent injury load effects."""
    print(f"\n{'='*70}")
    print("TEAM INJURY LOAD: How do injuries affect props?")
    print("Theory: Opponent injuries = easier matchup = overs hit")
    print("="*70 + "\n")
    
    if "home_players_out" not in bets.columns:
        print("  No injury data merged")
        return
    
    if "away_players_out" not in bets.columns:
        bets["away_players_out"] = 0
    
    bets["many_home_out"] = bets["home_players_out"] >= 4
    bets["many_away_out"] = bets["away_players_out"] >= 4
    bets["few_injuries"] = (bets["home_players_out"] + bets["away_players_out"]) <= 4
    bets["high_injury_game"] = (bets["home_players_out"] + bets["away_players_out"]) >= 8
    
    print(f"{'Filter':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 75)
    
    tests = [
        ("Many home injuries (4+) + all", bets[bets["many_home_out"]]),
        ("Many home injuries + UNDER", bets[(bets["many_home_out"]) & (bets["signal"] == "bet_under")]),
        ("Many away injuries (4+) + OVER", bets[(bets["many_away_out"]) & (bets["signal"] == "bet_over")]),
        ("Few total injuries + OVER", bets[(bets["few_injuries"]) & (bets["signal"] == "bet_over")]),
        ("High injury game + UNDER", bets[(bets["high_injury_game"]) & (bets["signal"] == "bet_under")]),
        ("Many home out + Rush props", bets[(bets["many_home_out"]) & (bets["market"] == "player_rush_yds")]),
        ("Many away out + Pass TDs", bets[(bets["many_away_out"]) & (bets["market"] == "player_pass_tds")]),
    ]
    
    for label, subset in tests:
        r = calc_roi(subset)
        print_result(label, r)


def test_advanced_combos(bets):
    """Test: Multi-factor combinations based on all findings."""
    print(f"\n{'='*70}")
    print("ADVANCED MULTI-FACTOR COMBINATIONS")
    print("Stacking confirmed signals together")
    print("="*70 + "\n")
    
    if "spread_line" in bets.columns:
        bets["big_fav"] = bets["spread_line"].fillna(0).abs() >= 7
        bets["close_expected"] = bets["spread_line"].fillna(0).abs() <= 3
    if "total_line" in bets.columns:
        bets["high_total"] = bets["total_line"].fillna(44) >= 48
        bets["low_total"] = bets["total_line"].fillna(44) <= 40
    
    print(f"{'Filter':<55} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 80)
    
    tests = [
        # Weather + market + direction stacks
        ("Pass TDs + not windy + not cold + not division",
         bets[(bets["market"] == "player_pass_tds") & (~bets["is_windy"]) & (~bets["is_cold"]) & (~bets["is_division"])]),
        ("Pass TDs + dome + not division + not cold",
         bets[(bets["market"] == "player_pass_tds") & (bets["is_dome"]) & (~bets["is_division"]) & (~bets["is_cold"])]),
        ("Rec UNDER + outdoor + not primetime",
         bets[(bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets["is_dome"]) & (~bets["is_primetime"])]),
        ("Rec UNDER + outdoor + cold",
         bets[(bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets["is_dome"]) & (bets["is_cold"])]),
        
        # Game context stacks
        ("UNDER + week 1-4 + not primetime + not dome",
         bets[(bets["signal"] == "bet_under") & (bets["is_early"]) & (~bets["is_primetime"]) & (~bets["is_dome"])]),
        ("Pass TDs + high total expected",
         bets[(bets["market"] == "player_pass_tds") & (bets.get("high_total", pd.Series(False)))]),
        ("Rush OVER + low total expected",
         bets[(bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over") & (bets.get("low_total", pd.Series(False)))]),
        ("UNDER + division + Thursday",
         bets[(bets["signal"] == "bet_under") & (bets["is_division"]) & (bets["is_thursday"])]),
        
        # Deviation + context
        ("UNDER + deviation 12-20% + not dome",
         bets[(bets["signal"] == "bet_under") & (bets["line_deviation_pct"].between(0.12, 0.20)) & (~bets["is_dome"])]),
        ("OVER + deviation 12-20% + dome",
         bets[(bets["signal"] == "bet_over") & (bets["line_deviation_pct"].between(-0.20, -0.12)) & (bets["is_dome"])]),
        
        # Spread context combos
        ("Pass TDs + close game expected (spread <= 3)",
         bets[(bets["market"] == "player_pass_tds") & (bets.get("close_expected", pd.Series(False)))]),
        ("Rush OVER + big favorite game",
         bets[(bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over") & (bets.get("big_fav", pd.Series(False)))]),
        
        # Injury context combos
        ("OVER + many away injuries + not cold",
         bets[(bets["signal"] == "bet_over") & (bets.get("many_away_out", pd.Series(False))) & (~bets["is_cold"])]),
        ("Pass TDs + many away injuries",
         bets[(bets["market"] == "player_pass_tds") & (bets.get("many_away_out", pd.Series(False)))]),
        
        # Triple stacks
        ("Pass TDs + dome + high total",
         bets[(bets["market"] == "player_pass_tds") & (bets["is_dome"]) & (bets.get("high_total", pd.Series(False)))]),
        ("Cold + windy + Pass UNDER",
         bets[(bets["is_cold"]) & (bets["is_windy"]) & (bets["market"].isin(["player_pass_yds", "player_reception_yds"])) & (bets["signal"] == "bet_under")]),
        ("Rec UNDER + division + outdoor + not early",
         bets[(bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (bets["is_division"]) & (~bets["is_dome"]) & (~bets["is_early"])]),
    ]
    
    profitable = []
    for label, subset in tests:
        r = calc_roi(subset)
        if r and r["roi"] > 0:
            profitable.append({"combo": label, **r})
        print_result(label, r)
    
    # Summary
    print(f"\n{'='*70}")
    print("ALL PROFITABLE FINDINGS THIS ROUND (sorted by ROI)")
    print("="*70 + "\n")
    
    for p in sorted(profitable, key=lambda x: x["roi"], reverse=True):
        print(f"  {p['combo']}")
        print(f"    Hit: {p['hit']:.1%} | ROI: {p['roi']:+.1f}% | Bets: {p['n']} (~{p['n']//3}/season)")
        print()


def main():
    print("=" * 70)
    print("DEEPER HYPOTHESIS TESTING — Round 2")
    print("Finding correlates to props going OVER vs UNDER")
    print("=" * 70)
    
    bets = load_enriched()
    print(f"Loaded {len(bets):,} enriched bets\n")
    
    test_game_script(bets)
    test_totals_context(bets)
    test_weekly_patterns(bets)
    test_player_consistency(bets)
    test_line_movement_proxy(bets)
    test_home_away(bets)
    test_injury_context(bets)
    test_advanced_combos(bets)


if __name__ == "__main__":
    main()
