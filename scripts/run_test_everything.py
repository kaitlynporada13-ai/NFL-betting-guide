"""
TEST EVERYTHING: Comprehensive hypothesis testing.
Line movement, revenge games, player patterns, game script,
streaks, blowouts, positional matchups, and more.
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

def calc_roi(subset):
    if len(subset) < 25:
        return None
    wins = subset["bet_won"].sum()
    total = len(subset)
    hit = wins / total
    roi = ((wins * 100) - ((total - wins) * 110)) / (total * 110) * 100
    return {"hit": hit, "roi": roi, "n": total, "wins": wins}

def pr(label, result):
    if result is None:
        return
    m = "✓" if result["roi"] > 0 else " "
    print(f"{m} {label:<55} {result['hit']:>6.1%} {result['n']:>5,} {result['roi']:>+7.1f}%")

def load_master_dataset():
    """Build the most comprehensive dataset possible."""
    # Load enriched bets
    bets = pd.read_parquet(PROC_DIR / "bets_fully_enriched.parquet")
    
    # Load player stats for additional features
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    # Build streak features: how many consecutive overs/unders did this player have?
    stats_sorted = stats.sort_values(["player_clean", "season", "week"])
    
    # Player's prior week performance relative to their average
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards",
        "player_receptions": "receptions",
        "player_pass_tds": "passing_tds",
    }
    
    for market, stat_col in market_to_stat.items():
        if stat_col not in stats.columns:
            continue
        # Previous game performance
        stats[f"{stat_col}_prev"] = stats.groupby("player_clean")[stat_col].shift(1)
        # 3-game avg prior
        stats[f"{stat_col}_avg3"] = (
            stats.groupby("player_clean")[stat_col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
        )
        # Was prev game a "boom" (>30% above avg)?
        stats[f"{stat_col}_prev_boom"] = (
            stats[f"{stat_col}_prev"] > stats[f"{stat_col}_avg3"] * 1.3
        )
        # Was prev game a "bust" (<30% below avg)?
        stats[f"{stat_col}_prev_bust"] = (
            stats[f"{stat_col}_prev"] < stats[f"{stat_col}_avg3"] * 0.7
        )
    
    # Merge streak features onto bets
    for market, stat_col in market_to_stat.items():
        prev_col = f"{stat_col}_prev"
        boom_col = f"{stat_col}_prev_boom"
        bust_col = f"{stat_col}_prev_bust"
        
        cols_to_merge = ["player_clean", "season", "week"]
        merge_cols = []
        for c in [prev_col, boom_col, bust_col, f"{stat_col}_avg3"]:
            if c in stats.columns:
                merge_cols.append(c)
        
        if merge_cols:
            mkt_mask = bets["market"] == market
            mkt_stats = stats[cols_to_merge + merge_cols].dropna(subset=merge_cols[:1]).drop_duplicates()
            bets = bets.merge(mkt_stats, on=cols_to_merge, how="left", suffixes=("", f"_{market}"))
    
    # Determine "bet_won" for our mean reversion strategy
    bets["bet_won"] = (
        ((bets["signal"] == "bet_over") & (bets["result"] == "won")) |
        ((bets["signal"] == "bet_under") & (bets["result"] == "lost"))
    )
    
    return bets, stats

def test_streaks_and_momentum(bets):
    """After a boom game, does regression happen? After a bust, bounce back?"""
    print(f"\n{'='*70}")
    print("STREAKS & MOMENTUM: Boom/Bust Regression")
    print("Theory: After a big game, FanDuel raises the line. Player regresses.")
    print("After a bad game, FanDuel lowers. Player bounces back.")
    print("="*70 + "\n")
    
    print(f"{'Filter':<55} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 80)
    
    # Find boom/bust columns dynamically
    boom_cols = [c for c in bets.columns if "prev_boom" in c]
    bust_cols = [c for c in bets.columns if "prev_bust" in c]
    
    # General: after ANY boom, bet under (regression)
    if boom_cols:
        any_boom = bets[boom_cols].any(axis=1).fillna(False)
        pr("After boom game → UNDER (all markets)", calc_roi(bets[any_boom & (bets["signal"] == "bet_under")]))
        pr("After boom game → all bets", calc_roi(bets[any_boom]))
    
    if bust_cols:
        any_bust = bets[bust_cols].any(axis=1).fillna(False)
        pr("After bust game → OVER (all markets)", calc_roi(bets[any_bust & (bets["signal"] == "bet_over")]))
        pr("After bust game → all bets", calc_roi(bets[any_bust]))
    
    # By specific market
    for market, stat in [("player_pass_yds", "passing_yards"), ("player_rush_yds", "rushing_yards"),
                         ("player_reception_yds", "receiving_yards"), ("player_receptions", "receptions")]:
        boom_col = f"{stat}_prev_boom"
        bust_col = f"{stat}_prev_bust"
        mkt = bets[bets["market"] == market]
        
        if boom_col in mkt.columns:
            boom_mask = mkt[boom_col].fillna(False)
            pr(f"{market}: after boom → UNDER", calc_roi(mkt[boom_mask & (mkt["signal"] == "bet_under")]))
        if bust_col in mkt.columns:
            bust_mask = mkt[bust_col].fillna(False)
            pr(f"{market}: after bust → OVER", calc_roi(mkt[bust_mask & (mkt["signal"] == "bet_over")]))


def test_game_script_deep(bets):
    """Blowout effects: how does margin affect 2nd half props?"""
    print(f"\n{'='*70}")
    print("GAME SCRIPT (DEEP): Spread & Blowout Interactions")
    print("Theory: Big underdogs behind = garbage time passing stats")
    print("        Big favorites ahead = run out the clock")
    print("="*70 + "\n")
    
    if "spread_line" not in bets.columns:
        print("  No spread data")
        return
    
    # Spread buckets
    bets["is_big_dog"] = bets["spread_line"].fillna(0) >= 6  # team is big underdog (positive spread)
    bets["is_big_fav"] = bets["spread_line"].fillna(0) <= -6
    bets["is_slight_dog"] = bets["spread_line"].fillna(0).between(2.5, 5.5)
    bets["is_slight_fav"] = bets["spread_line"].fillna(0).between(-5.5, -2.5)
    bets["is_pick"] = bets["spread_line"].fillna(0).abs() <= 2.5
    
    print(f"{'Filter':<55} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 80)
    
    tests = [
        ("Big underdog + Pass yards OVER", bets[(bets["is_big_dog"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_over")]),
        ("Big underdog + Pass yards UNDER", bets[(bets["is_big_dog"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")]),
        ("Big underdog + Rec yards OVER", bets[(bets["is_big_dog"]) & (bets["market"] == "player_reception_yds") & (bets["signal"] == "bet_over")]),
        ("Big underdog + Rush UNDER", bets[(bets["is_big_dog"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_under")]),
        ("Big favorite + Rush OVER", bets[(bets["is_big_fav"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")]),
        ("Big favorite + Pass UNDER", bets[(bets["is_big_fav"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")]),
        ("Big favorite + Receptions UNDER", bets[(bets["is_big_fav"]) & (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under")]),
        ("Pick'em game + Pass TDs", bets[(bets["is_pick"]) & (bets["market"] == "player_pass_tds")]),
        ("Pick'em + Rec yards OVER", bets[(bets["is_pick"]) & (bets["market"] == "player_reception_yds") & (bets["signal"] == "bet_over")]),
        ("Slight dog + Pass TDs", bets[(bets["is_slight_dog"]) & (bets["market"] == "player_pass_tds")]),
        ("Slight fav + Rush OVER", bets[(bets["is_slight_fav"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")]),
    ]
    
    for label, subset in tests:
        pr(label, calc_roi(subset))

def test_player_specific_patterns(bets, stats):
    """Which types of players are most predictable/exploitable?"""
    print(f"\n{'='*70}")
    print("PLAYER-SPECIFIC PATTERNS")
    print("Theory: Some player types are more predictable than others")
    print("="*70 + "\n")
    
    # Players who have the most props in our dataset (high volume = high confidence)
    player_counts = bets.groupby("player_clean").size().reset_index(name="prop_count")
    high_volume_players = player_counts[player_counts["prop_count"] >= 30]["player_clean"].tolist()
    
    # Per-player hit rates on our strategy
    player_results = bets[bets["player_clean"].isin(high_volume_players)].groupby("player_clean").agg(
        total=("bet_won", "count"),
        wins=("bet_won", "sum"),
    ).reset_index()
    player_results["hit_rate"] = player_results["wins"] / player_results["total"]
    player_results["roi"] = ((player_results["wins"] * 100) - ((player_results["total"] - player_results["wins"]) * 110)) / (player_results["total"] * 110) * 100
    
    # Most profitable players to bet on
    profitable_players = player_results[player_results["roi"] > 5].sort_values("roi", ascending=False)
    worst_players = player_results[player_results["roi"] < -15].sort_values("roi")
    
    print(f"  High-volume players analyzed: {len(player_results)}")
    print(f"  Profitable (ROI > +5%): {len(profitable_players)}")
    print(f"  Money pits (ROI < -15%): {len(worst_players)}")
    
    print(f"\n  TOP 15 MOST PROFITABLE PLAYERS (our strategy):")
    print(f"  {'Player':<25} {'Bets':>5} {'Hit%':>6} {'ROI':>7}")
    print("  " + "-" * 50)
    for _, row in profitable_players.head(15).iterrows():
        print(f"  {row['player_clean']:<25} {row['total']:>4} {row['hit_rate']:>5.1%} {row['roi']:>+6.1f}%")
    
    print(f"\n  BOTTOM 10 WORST PLAYERS (avoid these):")
    for _, row in worst_players.head(10).iterrows():
        print(f"  {row['player_clean']:<25} {row['total']:>4} {row['hit_rate']:>5.1%} {row['roi']:>+6.1f}%")
    
    # Player position analysis
    if "position" in stats.columns:
        # Get positions for players
        pos_map = stats.drop_duplicates("player_clean")[["player_clean", "position"]]
        player_pos = player_results.merge(pos_map, on="player_clean", how="left")
        
        print(f"\n  BY POSITION:")
        pos_summary = player_pos.groupby("position").agg(
            players=("player_clean", "count"),
            avg_roi=("roi", "mean"),
            profitable_pct=("roi", lambda x: (x > 0).mean()),
        ).reset_index()
        for _, row in pos_summary.iterrows():
            print(f"    {row['position']}: {row['players']} players | Avg ROI: {row['avg_roi']:+.1f}% | {row['profitable_pct']:.0%} profitable")


def test_revenge_and_narrative(bets, stats):
    """Test revenge games (former team) by looking at roster changes."""
    print(f"\n{'='*70}")
    print("REVENGE GAMES & NARRATIVES")
    print("Theory: Players facing former teams outperform")
    print("(Approximated: player changed teams between seasons)")
    print("="*70 + "\n")
    
    # Detect team changes: player was on team X last season, now on team Y
    if "recent_team" not in stats.columns:
        print("  No team data for revenge analysis")
        return
    
    # Get each player's team by season
    player_teams = stats.groupby(["player_clean", "season"])["recent_team"].first().reset_index()
    player_teams_prev = player_teams.copy()
    player_teams_prev["season"] = player_teams_prev["season"] + 1
    player_teams_prev.rename(columns={"recent_team": "prev_team"}, inplace=True)
    
    # Merge to find who changed teams
    team_changes = player_teams.merge(player_teams_prev, on=["player_clean", "season"], how="left")
    team_changes["changed_team"] = (
        team_changes["recent_team"] != team_changes["prev_team"]
    ) & team_changes["prev_team"].notna()
    
    changers = team_changes[team_changes["changed_team"]][["player_clean", "season", "prev_team"]]
    
    # Merge onto bets
    bets_with_teams = bets.merge(changers, on=["player_clean", "season"], how="left")
    bets_with_teams["is_new_team"] = bets_with_teams["prev_team"].notna()
    
    # Revenge game: playing AGAINST their former team
    bets_with_teams["is_revenge"] = (
        bets_with_teams["is_new_team"] &
        ((bets_with_teams["home_abbr"] == bets_with_teams["prev_team"]) |
         (bets_with_teams["away_abbr"] == bets_with_teams["prev_team"]))
    )
    
    print(f"  Team changers found: {changers['player_clean'].nunique()} players")
    print(f"  Revenge game props: {bets_with_teams['is_revenge'].sum()}")
    print(f"  New team props (all): {bets_with_teams['is_new_team'].sum()}")
    
    print(f"\n{'Filter':<55} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 80)
    
    tests = [
        ("Revenge game → all bets", bets_with_teams[bets_with_teams["is_revenge"]]),
        ("Revenge game → OVER", bets_with_teams[(bets_with_teams["is_revenge"]) & (bets_with_teams["signal"] == "bet_over")]),
        ("Revenge game → UNDER", bets_with_teams[(bets_with_teams["is_revenge"]) & (bets_with_teams["signal"] == "bet_under")]),
        ("New team (all games) → OVER", bets_with_teams[(bets_with_teams["is_new_team"]) & (bets_with_teams["signal"] == "bet_over")]),
        ("New team (all games) → UNDER", bets_with_teams[(bets_with_teams["is_new_team"]) & (bets_with_teams["signal"] == "bet_under")]),
        ("New team + weeks 1-4 → UNDER", bets_with_teams[(bets_with_teams["is_new_team"]) & (bets_with_teams["week"] <= 4) & (bets_with_teams["signal"] == "bet_under")]),
        ("New team + weeks 1-4 → OVER", bets_with_teams[(bets_with_teams["is_new_team"]) & (bets_with_teams["week"] <= 4) & (bets_with_teams["signal"] == "bet_over")]),
    ]
    
    for label, subset in tests:
        pr(label, calc_roi(subset))

def test_line_size_patterns(bets):
    """Test if absolute line size matters (high lines vs low lines)."""
    print(f"\n{'='*70}")
    print("LINE SIZE ANALYSIS")
    print("Theory: High-line players (stars) vs low-line (role players) behave differently")
    print("="*70 + "\n")
    
    print(f"{'Filter':<55} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 80)
    
    # Passing yards line buckets
    pass_yds = bets[bets["market"] == "player_pass_yds"]
    tests = [
        ("Pass yds: line < 220 (backup/low vol QB)", pass_yds[pass_yds["fanduel_line"] < 220]),
        ("Pass yds: line 220-260 (mid QB)", pass_yds[pass_yds["fanduel_line"].between(220, 260)]),
        ("Pass yds: line > 260 (star QB)", pass_yds[pass_yds["fanduel_line"] > 260]),
        ("Pass yds: line > 280 UNDER", pass_yds[(pass_yds["fanduel_line"] > 280) & (pass_yds["signal"] == "bet_under")]),
        ("Pass yds: line < 220 OVER", pass_yds[(pass_yds["fanduel_line"] < 220) & (pass_yds["signal"] == "bet_over")]),
    ]
    for label, subset in tests:
        pr(label, calc_roi(subset))
    
    # Rushing yards
    rush_yds = bets[bets["market"] == "player_rush_yds"]
    tests = [
        ("Rush yds: line < 40 (backup/committee)", rush_yds[rush_yds["fanduel_line"] < 40]),
        ("Rush yds: line 40-70 (starter)", rush_yds[rush_yds["fanduel_line"].between(40, 70)]),
        ("Rush yds: line > 70 (bellcow)", rush_yds[rush_yds["fanduel_line"] > 70]),
        ("Rush yds: bellcow (>70) UNDER", rush_yds[(rush_yds["fanduel_line"] > 70) & (rush_yds["signal"] == "bet_under")]),
        ("Rush yds: committee (<40) OVER", rush_yds[(rush_yds["fanduel_line"] < 40) & (rush_yds["signal"] == "bet_over")]),
    ]
    for label, subset in tests:
        pr(label, calc_roi(subset))
    
    # Receiving yards
    rec_yds = bets[bets["market"] == "player_reception_yds"]
    tests = [
        ("Rec yds: line < 40 (depth WR/TE)", rec_yds[rec_yds["fanduel_line"] < 40]),
        ("Rec yds: line 40-70 (solid starter)", rec_yds[rec_yds["fanduel_line"].between(40, 70)]),
        ("Rec yds: line > 70 (alpha WR)", rec_yds[rec_yds["fanduel_line"] > 70]),
        ("Rec yds: alpha (>70) UNDER", rec_yds[(rec_yds["fanduel_line"] > 70) & (rec_yds["signal"] == "bet_under")]),
        ("Rec yds: depth (<40) OVER", rec_yds[(rec_yds["fanduel_line"] < 40) & (rec_yds["signal"] == "bet_over")]),
    ]
    for label, subset in tests:
        pr(label, calc_roi(subset))
    
    # Receptions
    recs = bets[bets["market"] == "player_receptions"]
    tests = [
        ("Receptions: line < 3.5 (low target)", recs[recs["fanduel_line"] < 3.5]),
        ("Receptions: line 3.5-5.5 (mid target)", recs[recs["fanduel_line"].between(3.5, 5.5)]),
        ("Receptions: line > 5.5 (high target)", recs[recs["fanduel_line"] > 5.5]),
        ("Receptions: high target (>5.5) UNDER", recs[(recs["fanduel_line"] > 5.5) & (recs["signal"] == "bet_under")]),
        ("Receptions: low target (<3.5) OVER", recs[(recs["fanduel_line"] < 3.5) & (recs["signal"] == "bet_over")]),
    ]
    for label, subset in tests:
        pr(label, calc_roi(subset))


def test_combined_stacks(bets):
    """Test highest-potential multi-factor combinations."""
    print(f"\n{'='*70}")
    print("MEGA STACKS: Best multi-factor combinations")
    print("="*70 + "\n")
    
    # Build all needed columns
    if "spread_line" in bets.columns:
        bets["is_pick"] = bets["spread_line"].fillna(0).abs() <= 2.5
        bets["big_dog"] = bets["spread_line"].fillna(0) >= 6
        bets["big_fav"] = bets["spread_line"].fillna(0) <= -6
    if "total_line" in bets.columns:
        bets["high_total"] = bets["total_line"].fillna(44) >= 48
        bets["low_total"] = bets["total_line"].fillna(44) <= 40
    
    boom_cols = [c for c in bets.columns if "prev_boom" in c]
    bust_cols = [c for c in bets.columns if "prev_bust" in c]
    any_boom = bets[boom_cols].any(axis=1).fillna(False) if boom_cols else pd.Series(False, index=bets.index)
    any_bust = bets[bust_cols].any(axis=1).fillna(False) if bust_cols else pd.Series(False, index=bets.index)
    
    print(f"{'Filter':<60} {'Hit%':>7} {'Bets':>6} {'ROI':>8}")
    print("-" * 85)
    
    tests = [
        # Boom regression + context
        ("After boom + UNDER + outdoor",
         bets[any_boom & (bets["signal"] == "bet_under") & (~bets["is_dome"])]),
        ("After boom + UNDER + division",
         bets[any_boom & (bets["signal"] == "bet_under") & (bets["is_division"])]),
        ("After bust + OVER + dome",
         bets[any_bust & (bets["signal"] == "bet_over") & (bets["is_dome"])]),
        ("After bust + OVER + not division",
         bets[any_bust & (bets["signal"] == "bet_over") & (~bets["is_division"])]),
        
        # Spread + weather + market
        ("Pick'em + dome + Pass TDs",
         bets[(bets.get("is_pick", pd.Series(False))) & (bets["is_dome"]) & (bets["market"] == "player_pass_tds")]),
        ("Big dog + Pass yards OVER + not cold",
         bets[(bets.get("big_dog", pd.Series(False))) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_over") & (~bets["is_cold"])]),
        ("Big fav + Rec UNDER + outdoor",
         bets[(bets.get("big_fav", pd.Series(False))) & (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets["is_dome"])]),
        
        # Week + context combos
        ("Week 1 + UNDER + Pass TDs",
         bets[(bets["week"] == 1) & (bets["signal"] == "bet_under") & (bets["market"] == "player_pass_tds")]),
        ("Week 1 + UNDER + Receptions",
         bets[(bets["week"] == 1) & (bets["signal"] == "bet_under") & (bets["market"] == "player_receptions")]),
        ("Week 1 + UNDER + Pass yards",
         bets[(bets["week"] == 1) & (bets["signal"] == "bet_under") & (bets["market"] == "player_pass_yds")]),
        ("Weeks 5-8 + Pass TDs + dome",
         bets[(bets["week"].between(5, 8)) & (bets["market"] == "player_pass_tds") & (bets["is_dome"])]),
        ("Weeks 13+ + Pass TDs + not cold + OVER",
         bets[(bets["week"] >= 13) & (bets["market"] == "player_pass_tds") & (~bets["is_cold"]) & (bets["signal"] == "bet_over")]),
        
        # High injury + specific contexts
        ("High injury + UNDER + outdoor",
         bets[(bets.get("home_players_out", 0) + bets.get("away_players_out", 0) >= 8) & (bets["signal"] == "bet_under") & (~bets["is_dome"])] if "home_players_out" in bets.columns else pd.DataFrame()),
        
        # Line size + context
        ("Rec yds alpha (>70) UNDER + division + outdoor",
         bets[(bets["market"] == "player_reception_yds") & (bets["fanduel_line"] > 70) & (bets["signal"] == "bet_under") & (bets["is_division"]) & (~bets["is_dome"])]),
        ("Receptions high (>5.5) UNDER + outdoor + not primetime",
         bets[(bets["market"] == "player_receptions") & (bets["fanduel_line"] > 5.5) & (bets["signal"] == "bet_under") & (~bets["is_dome"]) & (~bets["is_primetime"])]),
        
        # The ultimate stack: boom regression + outdoor + division + under
        ("Boom regression + outdoor + division + UNDER",
         bets[any_boom & (~bets["is_dome"]) & (bets["is_division"]) & (bets["signal"] == "bet_under")]),
    ]
    
    profitable = []
    for label, subset in tests:
        if isinstance(subset, pd.DataFrame) and not subset.empty:
            r = calc_roi(subset)
            pr(label, r)
            if r and r["roi"] > 0:
                profitable.append({"combo": label, **r})
        else:
            pass
    
    return profitable

def main():
    print("=" * 70)
    print("COMPREHENSIVE TESTING: ALL REMAINING HYPOTHESES")
    print("=" * 70)
    
    bets, stats = load_master_dataset()
    print(f"Master dataset: {len(bets):,} bets")
    
    test_streaks_and_momentum(bets)
    test_game_script_deep(bets)
    test_player_specific_patterns(bets, stats)
    test_revenge_and_narrative(bets, stats)
    test_line_size_patterns(bets)
    profitable = test_combined_stacks(bets)
    
    # Final summary of ALL profitable findings
    print(f"\n{'='*70}")
    print("COMPLETE PROFITABLE FINDINGS THIS SESSION")
    print("="*70 + "\n")
    
    if profitable:
        for p in sorted(profitable, key=lambda x: x["roi"], reverse=True):
            print(f"  ✓ {p['combo']}")
            print(f"    Hit: {p['hit']:.1%} | ROI: {p['roi']:+.1f}% | Bets: {p['n']} (~{p['n']//3}/season)")
            print()
    
    print("\nDone. All results above. Check for ✓ marks for profitable strategies.")


if __name__ == "__main__":
    main()
