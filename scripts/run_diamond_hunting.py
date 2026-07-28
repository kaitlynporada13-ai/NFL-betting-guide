"""
Diamond Hunting: Find +200 line (underdog) props that hit historically.
======================================================================
Focus: PLUS MONEY props (+150, +200, +300 etc) that ACTUALLY HIT.
These are the high-value, low-probability plays that FanDuel underprices.

Questions:
1. Which plus-money props hit more than their implied probability suggests?
2. What factors were present when longshot props hit?
3. Are there repeatable patterns in plus-money winners?
4. Weekly variance patterns by week of season.
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

FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def american_to_implied_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def american_to_payout(odds, stake=100):
    """Convert American odds to payout on $100 bet."""
    if pd.isna(odds):
        return 0
    if odds > 0:
        return odds  # profit on $100
    else:
        return 100 / (-odds / 100)  # profit on enough to win $100


def load_all_props():
    """Load all historical props (not just 'signal' bets)."""
    props = pd.read_parquet(RAW_DIR / "historical_props_all.parquet")
    # Keep overs (we'll analyze both sides)
    return props


def load_and_enrich_all():
    """Load ALL props with game context (not just those where we'd bet)."""
    props = load_all_props()
    
    # Load actual stats
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    props["player_clean"] = props["player_name"].str.strip().str.lower()
    
    # Map markets to stat columns
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_pass_tds": "passing_tds",
        "player_rush_yds": "rushing_yards",
        "player_receptions": "receptions",
        "player_reception_yds": "receiving_yards",
    }
    
    # Grade all props
    graded_parts = []
    for market, stat_col in market_to_stat.items():
        if stat_col not in stats.columns:
            continue
        mkt_props = props[props["market"] == market].copy()
        if mkt_props.empty:
            continue
        
        merged = mkt_props.merge(
            stats[["player_clean", "season", "week", stat_col]].dropna(subset=[stat_col]).drop_duplicates(),
            on=["player_clean", "season", "week"],
            how="inner",
        )
        merged["actual_stat"] = merged[stat_col]
        merged["over_hit"] = merged["actual_stat"] > merged["line"]
        merged["residual"] = merged["actual_stat"] - merged["line"]
        graded_parts.append(merged)
    
    graded = pd.concat(graded_parts, ignore_index=True)
    
    # Add game context
    games = pd.read_parquet(RAW_DIR / "games_historical.parquet")
    graded["home_abbr"] = graded["home_team"].map(FULL_TO_ABBR)
    graded["away_abbr"] = graded["away_team"].map(FULL_TO_ABBR)
    
    game_cols = ["season", "week", "home_team", "away_team", "home_rest", "away_rest",
                 "div_game", "weekday", "roof", "surface", "temp", "wind",
                 "spread_line", "total_line"]
    available = [c for c in game_cols if c in games.columns]
    
    graded = graded.merge(
        games[available].drop_duplicates(subset=["season", "week", "home_team"]),
        left_on=["season", "week", "home_abbr"],
        right_on=["season", "week", "home_team"],
        how="left", suffixes=("", "_g"),
    )
    
    # Derived
    graded["is_dome"] = graded["roof"].isin(["dome", "closed"]) if "roof" in graded.columns else False
    graded["is_cold"] = graded["temp"].fillna(65) <= 35 if "temp" in graded.columns else False
    graded["is_windy"] = graded["wind"].fillna(0) >= 15 if "wind" in graded.columns else False
    graded["is_division"] = graded["div_game"] == 1 if "div_game" in graded.columns else False
    graded["is_thursday"] = graded["weekday"].str.contains("Thursday", na=False) if "weekday" in graded.columns else False
    
    # Implied probability from odds
    graded["implied_prob"] = graded["price"].apply(american_to_implied_prob)
    graded["payout"] = graded["price"].apply(american_to_payout)
    
    # Plus money classification
    graded["is_plus_money"] = graded["price"] > 0
    graded["is_big_plus"] = graded["price"] >= 150
    graded["is_massive_plus"] = graded["price"] >= 200
    
    print(f"Total graded props: {len(graded):,}")
    print(f"Plus money (>100): {graded['is_plus_money'].sum():,}")
    print(f"+150 or more: {graded['is_big_plus'].sum():,}")
    print(f"+200 or more: {graded['is_massive_plus'].sum():,}")
    
    return graded


def analyze_plus_money_winners(graded):
    """Find which plus-money props hit more than implied probability says they should."""
    
    print(f"\n{'='*70}")
    print("PLUS MONEY PROP ANALYSIS")
    print("Question: Which +150/+200/+300 lines hit MORE than expected?")
    print("="*70)
    
    # Filter to plus money on the OVER side
    plus = graded[graded["is_plus_money"] & (graded["outcome"] == "Over")].copy()
    
    print(f"\nPlus-money OVERS: {len(plus):,}")
    
    # By price bucket
    print(f"\n--- HIT RATE vs IMPLIED PROBABILITY BY ODDS RANGE ---")
    print(f"{'Odds Range':<15} {'Bets':>6} {'Hit%':>7} {'Implied%':>9} {'Edge':>7} {'EV':>10}")
    print("-" * 60)
    
    buckets = [
        ("+100 to +130", 100, 130),
        ("+131 to +160", 131, 160),
        ("+161 to +200", 161, 200),
        ("+201 to +250", 201, 250),
        ("+251 to +300", 251, 300),
        ("+301 to +400", 301, 400),
        ("+401+", 401, 9999),
    ]
    
    for label, low, high in buckets:
        bucket = plus[(plus["price"] >= low) & (plus["price"] <= high)]
        if len(bucket) < 20:
            continue
        
        hit_rate = bucket["over_hit"].mean()
        implied = bucket["implied_prob"].mean()
        edge = hit_rate - implied
        
        # Expected value: hit_rate * payout - (1-hit_rate) * stake
        avg_payout = bucket["payout"].mean()
        ev = (hit_rate * avg_payout) - ((1 - hit_rate) * 100)
        
        marker = "✓ +EV" if ev > 0 else ""
        print(f"  {label:<13} {len(bucket):>5,} {hit_rate:>6.1%} {implied:>8.1%} {edge:>+6.1%} ${ev:>+8.1f} {marker}")
    
    # Now: analyze the PLUS MONEY WINNERS specifically
    plus_winners = plus[plus["over_hit"] == True].copy()
    plus_losers = plus[plus["over_hit"] == False].copy()
    
    print(f"\n{'='*70}")
    print(f"PROFILE OF PLUS-MONEY WINNERS ({len(plus_winners):,} hits)")
    print(f"vs PLUS-MONEY LOSERS ({len(plus_losers):,} misses)")
    print(f"{'='*70}")
    
    factors = [
        ("is_dome", "Dome game"),
        ("is_cold", "Cold (<=35F)"),
        ("is_windy", "Windy (>=15mph)"),
        ("is_division", "Division game"),
        ("is_thursday", "Thursday game"),
    ]
    
    print(f"\n{'Factor':<20} {'Winners':>10} {'Losers':>10} {'Diff':>8}")
    print("-" * 50)
    
    for col, label in factors:
        if col not in plus.columns:
            continue
        win_rate = plus_winners[col].mean()
        lose_rate = plus_losers[col].mean()
        diff = win_rate - lose_rate
        sig = " ← SIGNAL" if abs(diff) > 0.03 else ""
        print(f"  {label:<18} {win_rate:>9.1%} {lose_rate:>9.1%} {diff:>+7.1%}{sig}")
    
    # By market: where do plus money overs actually hit?
    print(f"\n--- PLUS MONEY HIT RATE BY MARKET ---")
    for market in plus["market"].unique():
        mkt = plus[plus["market"] == market]
        if len(mkt) < 20:
            continue
        hit = mkt["over_hit"].mean()
        implied = mkt["implied_prob"].mean()
        edge = hit - implied
        print(f"  {market:<25} {hit:.1%} actual vs {implied:.1%} implied | edge: {edge:+.1%}")
    
    # By week
    print(f"\n--- PLUS MONEY HIT RATE BY WEEK ---")
    week_hits = plus.groupby("week").agg(
        hit_rate=("over_hit", "mean"),
        implied=("implied_prob", "mean"),
        n=("over_hit", "count"),
    ).reset_index()
    week_hits["edge"] = week_hits["hit_rate"] - week_hits["implied"]
    
    for _, row in week_hits.iterrows():
        marker = " ←" if row["edge"] > 0.05 else ""
        print(f"  Week {int(row['week']):>2}: {row['hit_rate']:.1%} actual vs {row['implied']:.1%} implied "
              f"(n={int(row['n'])}) edge: {row['edge']:+.1%}{marker}")


def find_diamonds(graded):
    """Find the specific situations where +200 props hit at exploitable rates."""
    
    print(f"\n{'='*70}")
    print("DIAMOND HUNTING: Where do +200 (and higher) props ACTUALLY HIT?")
    print("="*70)
    
    # Focus on +150 and above (bigger payouts)
    big_plus = graded[(graded["price"] >= 150) & (graded["outcome"] == "Over")].copy()
    
    print(f"\nProps at +150 or higher: {len(big_plus):,}")
    print(f"Overall hit rate: {big_plus['over_hit'].mean():.1%}")
    print(f"Overall implied prob: {big_plus['implied_prob'].mean():.1%}")
    
    # Test conditions that boost plus-money hit rate
    print(f"\n--- CONDITIONS THAT BOOST PLUS-MONEY OVERS ---")
    print(f"{'Condition':<55} {'Hit%':>7} {'Impl%':>7} {'Edge':>7} {'N':>5} {'EV/bet':>8}")
    print("-" * 95)
    
    conditions = [
        # Basic context
        ("Dome game", big_plus["is_dome"]),
        ("Outdoor game", ~big_plus["is_dome"]),
        ("Cold game", big_plus["is_cold"]),
        ("Not cold", ~big_plus["is_cold"]),
        ("Division game", big_plus["is_division"]),
        ("Non-division", ~big_plus["is_division"]),
        ("Thursday", big_plus["is_thursday"]),
        ("Windy", big_plus["is_windy"]),
        
        # Week ranges
        ("Week 1", big_plus["week"] == 1),
        ("Weeks 1-4", big_plus["week"] <= 4),
        ("Weeks 5-8", big_plus["week"].between(5, 8)),
        ("Weeks 9-12", big_plus["week"].between(9, 12)),
        ("Weeks 13+", big_plus["week"] >= 13),
        
        # By market
        ("Passing yards", big_plus["market"] == "player_pass_yds"),
        ("Passing TDs", big_plus["market"] == "player_pass_tds"),
        ("Rushing yards", big_plus["market"] == "player_rush_yds"),
        ("Receiving yards", big_plus["market"] == "player_reception_yds"),
        ("Receptions", big_plus["market"] == "player_receptions"),
        
        # Combos
        ("Dome + Pass TDs", (big_plus["is_dome"]) & (big_plus["market"] == "player_pass_tds")),
        ("Dome + Rec yards", (big_plus["is_dome"]) & (big_plus["market"] == "player_reception_yds")),
        ("Cold + Rush yards", (big_plus["is_cold"]) & (big_plus["market"] == "player_rush_yds")),
        ("Week 1 + all", big_plus["week"] == 1),
        ("Division + Pass TDs", (big_plus["is_division"]) & (big_plus["market"] == "player_pass_tds")),
        ("Non-div + Rec yards", (~big_plus["is_division"]) & (big_plus["market"] == "player_reception_yds")),
        ("Weeks 13+ + Pass TDs", (big_plus["week"] >= 13) & (big_plus["market"] == "player_pass_tds")),
        ("Thursday + Rush yards", (big_plus["is_thursday"]) & (big_plus["market"] == "player_rush_yds")),
    ]
    
    # Add spread context
    if "spread_line" in big_plus.columns:
        big_plus["close_game"] = big_plus["spread_line"].fillna(0).abs() <= 3
        big_plus["big_fav"] = big_plus["spread_line"].fillna(0).abs() >= 7
        conditions.extend([
            ("Close game (spread <= 3)", big_plus["close_game"]),
            ("Big favorite game (spread >= 7)", big_plus["big_fav"]),
            ("Close game + Pass TDs", (big_plus["close_game"]) & (big_plus["market"] == "player_pass_tds")),
            ("Close game + Dome", (big_plus["close_game"]) & (big_plus["is_dome"])),
        ])
    
    if "total_line" in big_plus.columns:
        big_plus["high_total"] = big_plus["total_line"].fillna(44) >= 48
        big_plus["low_total"] = big_plus["total_line"].fillna(44) <= 40
        conditions.extend([
            ("High total expected (>=48)", big_plus["high_total"]),
            ("Low total expected (<=40)", big_plus["low_total"]),
            ("High total + Pass TDs", (big_plus["high_total"]) & (big_plus["market"] == "player_pass_tds")),
            ("High total + Dome + Pass TDs", (big_plus["high_total"]) & (big_plus["is_dome"]) & (big_plus["market"] == "player_pass_tds")),
        ])
    
    diamonds = []
    
    for label, mask in conditions:
        subset = big_plus[mask]
        if len(subset) < 20:
            continue
        
        hit = subset["over_hit"].mean()
        implied = subset["implied_prob"].mean()
        edge = hit - implied
        avg_payout = subset["payout"].mean()
        ev = (hit * avg_payout) - ((1 - hit) * 100)
        
        marker = "✓ +EV" if ev > 0 else ""
        print(f"{'✓' if ev > 0 else ' '} {label:<53} {hit:>6.1%} {implied:>6.1%} {edge:>+6.1%} {len(subset):>4} ${ev:>+6.0f} {marker}")
        
        if ev > 0:
            diamonds.append({"condition": label, "hit_rate": hit, "implied": implied, 
                           "edge": edge, "ev_per_bet": ev, "n": len(subset)})
    
    # Summary of diamonds
    if diamonds:
        print(f"\n{'='*70}")
        print(f"DIAMONDS FOUND: {len(diamonds)} +EV conditions at +150 or higher")
        print(f"{'='*70}\n")
        
        for d in sorted(diamonds, key=lambda x: x["ev_per_bet"], reverse=True):
            print(f"  💎 {d['condition']}")
            print(f"     Hit: {d['hit_rate']:.1%} vs {d['implied']:.1%} implied")
            print(f"     EV per $100 bet: ${d['ev_per_bet']:+.0f} | Edge: {d['edge']:+.1%} | Sample: {d['n']}")
            print()
    
    return diamonds


def analyze_weekly_variance(graded):
    """Catalog weekly patterns for dashboard weekly notes."""
    
    print(f"\n{'='*70}")
    print("WEEKLY VARIANCE PATTERNS")
    print("Which weeks have the highest/lowest over rates by market?")
    print("="*70)
    
    overs = graded[graded["outcome"] == "Over"].copy()
    
    # By week and market
    weekly = overs.groupby(["week", "market"]).agg(
        over_rate=("over_hit", "mean"),
        avg_residual=("residual", "mean"),
        n=("over_hit", "count"),
    ).reset_index()
    
    # Find extreme weeks by market
    print(f"\n--- EXTREME WEEKS (highest/lowest over rates by market) ---\n")
    
    for market in ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions", "player_pass_tds"]:
        mkt = weekly[(weekly["market"] == market) & (weekly["n"] >= 30)]
        if mkt.empty:
            continue
        
        avg = mkt["over_rate"].mean()
        best_week = mkt.loc[mkt["over_rate"].idxmax()]
        worst_week = mkt.loc[mkt["over_rate"].idxmin()]
        
        print(f"  {market}:")
        print(f"    Average over rate: {avg:.1%}")
        print(f"    BEST week for overs:  Week {int(best_week['week'])} ({best_week['over_rate']:.1%}, n={int(best_week['n'])})")
        print(f"    WORST week for overs: Week {int(worst_week['week'])} ({worst_week['over_rate']:.1%}, n={int(worst_week['n'])})")
        print()
    
    # Overall weekly pattern
    print(f"\n--- OVERALL OVER RATE BY WEEK (all markets) ---\n")
    overall_weekly = overs.groupby("week").agg(
        over_rate=("over_hit", "mean"),
        n=("over_hit", "count"),
    ).reset_index()
    
    print(f"{'Week':>4} {'Over Rate':>10} {'N':>6} {'Strategy Note':>40}")
    print("-" * 65)
    
    for _, row in overall_weekly.iterrows():
        week = int(row["week"])
        rate = row["over_rate"]
        n = int(row["n"])
        
        if rate < 0.45:
            note = "STRONG UNDER WEEK → fade overs heavily"
        elif rate < 0.48:
            note = "Under-leaning week → slight under edge"
        elif rate > 0.55:
            note = "STRONG OVER WEEK → target overs"
        elif rate > 0.52:
            note = "Over-leaning week → slight over edge"
        else:
            note = "Neutral"
        
        print(f"  {week:>2}   {rate:>9.1%} {n:>5,}   {note}")
    
    # Save for dashboard
    weekly_notes = []
    for _, row in overall_weekly.iterrows():
        week = int(row["week"])
        rate = row["over_rate"]
        weekly_notes.append({
            "week": week,
            "overall_over_rate": rate,
            "lean": "under" if rate < 0.48 else "over" if rate > 0.52 else "neutral",
            "strength": abs(rate - 0.50),
        })
    
    notes_df = pd.DataFrame(weekly_notes)
    notes_df.to_parquet(PROC_DIR / "weekly_lean_patterns.parquet", index=False)
    print(f"\nSaved weekly patterns to data/processed/weekly_lean_patterns.parquet")
    
    return notes_df


def main():
    print("=" * 70)
    print("DIAMOND HUNTING: Plus-Money Props That Hit")
    print("+ Weekly Variance Analysis")
    print("=" * 70)
    
    graded = load_and_enrich_all()
    
    # Analyze plus money props
    analyze_plus_money_winners(graded)
    
    # Find specific diamond conditions
    diamonds = find_diamonds(graded)
    
    # Weekly variance patterns
    weekly = analyze_weekly_variance(graded)


if __name__ == "__main__":
    main()
