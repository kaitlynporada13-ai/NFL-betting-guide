"""
Deep Residual Analysis: WHY did props over/underperform?
=========================================================
1. Fix the game context merge (team name mapping)
2. Enrich every prop bet with ALL known factors
3. Analyze overperformers: what did they have in common?
4. Analyze underperformers: what did they have in common?
5. Find combination filters that predict when to bet
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

# Full name → abbreviation mapping
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


def load_and_enrich():
    """Load strategy results and properly merge ALL game context."""
    
    results = pd.read_parquet(PROC_DIR / "h013_strategy_results.parquet")
    bets = results[results["signal"] != "no_bet"].copy()
    
    # Convert full team names to abbreviations for merge
    bets["home_abbr"] = bets["home_team"].map(FULL_TO_ABBR)
    bets["away_abbr"] = bets["away_team"].map(FULL_TO_ABBR)
    
    # Load games
    games = pd.read_parquet(RAW_DIR / "games_historical.parquet")
    
    # Merge on abbreviation
    game_cols = ["season", "week", "home_team", "away_team", "home_rest", "away_rest",
                 "div_game", "weekday", "roof", "surface", "temp", "wind",
                 "home_score", "away_score", "spread_line", "total_line"]
    available = [c for c in game_cols if c in games.columns]
    
    bets = bets.merge(
        games[available].drop_duplicates(subset=["season", "week", "home_team"]),
        left_on=["season", "week", "home_abbr"],
        right_on=["season", "week", "home_team"],
        how="left",
        suffixes=("", "_g"),
    )
    
    # Load injuries and count per team
    inj_path = RAW_DIR / "injuries_historical.parquet"
    if inj_path.exists():
        injuries = pd.read_parquet(inj_path)
        out_counts = injuries[injuries["report_status"] == "Out"].groupby(
            ["season", "week", "team"]
        ).size().reset_index(name="home_players_out")
        
        bets = bets.merge(out_counts, left_on=["season", "week", "home_abbr"],
                         right_on=["season", "week", "team"], how="left")
        bets["home_players_out"] = bets["home_players_out"].fillna(0)
        
        # Away team injuries
        away_out = out_counts.rename(columns={"team": "away_team_inj", "home_players_out": "away_players_out"})
        bets = bets.merge(away_out, left_on=["season", "week", "away_abbr"],
                         right_on=["season", "week", "away_team_inj"], how="left")
        bets["away_players_out"] = bets["away_players_out"].fillna(0)
    
    # Load referee assignments
    officials_path = RAW_DIR / "officials_historical.parquet"
    ref_tend_path = RAW_DIR / "referee_tendencies.parquet"
    if officials_path.exists() and ref_tend_path.exists():
        officials = pd.read_parquet(officials_path)
        ref_tend = pd.read_parquet(ref_tend_path)
        
        # Build game_id to match
        head_refs = officials[officials["off_pos"] == "R"][["game_id", "name"]].rename(columns={"name": "referee"})
        
        # Build game_id for props
        bets["game_id_build"] = (
            bets["season"].astype(str) + "_" +
            bets["week"].astype(str).str.zfill(2) + "_" +
            bets["away_abbr"].fillna("") + "_" +
            bets["home_abbr"].fillna("")
        )
        bets = bets.merge(head_refs, left_on="game_id_build", right_on="game_id", how="left")
        
        # Classify refs
        median_pen = ref_tend["avg_penalties"].median()
        bets["flag_heavy_ref"] = bets["referee"].isin(
            ref_tend[ref_tend["avg_penalties"] > median_pen + 0.3]["referee"].tolist()
        )
        bets["flag_light_ref"] = bets["referee"].isin(
            ref_tend[ref_tend["avg_penalties"] < median_pen - 0.3]["referee"].tolist()
        )
    
    # Derived features
    bets["is_dome"] = bets["roof"].isin(["dome", "closed"]) if "roof" in bets.columns else False
    bets["is_outdoor"] = bets["roof"] == "outdoors" if "roof" in bets.columns else False
    bets["is_cold"] = bets["temp"].fillna(65) <= 35 if "temp" in bets.columns else False
    bets["is_hot"] = bets["temp"].fillna(65) >= 85 if "temp" in bets.columns else False
    bets["is_windy"] = bets["wind"].fillna(0) >= 15 if "wind" in bets.columns else False
    bets["is_very_windy"] = bets["wind"].fillna(0) >= 20 if "wind" in bets.columns else False
    bets["is_division"] = bets["div_game"] == 1 if "div_game" in bets.columns else False
    bets["is_thursday"] = bets["weekday"].str.contains("Thursday", na=False) if "weekday" in bets.columns else False
    bets["is_monday"] = bets["weekday"].str.contains("Monday", na=False) if "weekday" in bets.columns else False
    bets["is_primetime"] = bets["is_thursday"] | bets["is_monday"]
    bets["is_grass"] = bets["surface"].str.contains("grass", case=False, na=False) if "surface" in bets.columns else False
    bets["is_turf"] = ~bets["is_grass"]
    bets["is_early"] = bets["week"] <= 4
    bets["is_mid"] = bets["week"].between(5, 12)
    bets["is_late"] = bets["week"] >= 13
    bets["home_short_rest"] = bets["home_rest"].fillna(7) <= 6 if "home_rest" in bets.columns else False
    bets["away_short_rest"] = bets["away_rest"].fillna(7) <= 6 if "away_rest" in bets.columns else False
    
    # Game total (actual scoring environment)
    if "home_score" in bets.columns and "away_score" in bets.columns:
        bets["game_total"] = bets["home_score"].fillna(0) + bets["away_score"].fillna(0)
        bets["high_scoring_game"] = bets["game_total"] > 50
        bets["low_scoring_game"] = bets["game_total"] < 30
    
    # Residual (actual - line)
    bets["residual"] = bets["actual_stat"] - bets["fanduel_line"]
    bets["abs_residual"] = bets["residual"].abs()
    
    # Verify merge worked
    print(f"Context merge check:")
    for c in ["is_dome", "is_cold", "is_windy", "is_division", "is_primetime", "flag_heavy_ref"]:
        if c in bets.columns:
            print(f"  {c}: {bets[c].sum()} True / {bets[c].notna().sum()} total")
    
    return bets


def analyze_overperformers(bets: pd.DataFrame):
    """Find what factors are present when players CRUSH their lines."""
    
    print(f"\n{'='*70}")
    print("OVERPERFORMERS: When did players CRUSH their props?")
    print("(Top 20% of positive residuals — player way above the line)")
    print("="*70)
    
    # Top 20% overperformers (actual >> line)
    threshold = bets["residual"].quantile(0.80)
    crushers = bets[bets["residual"] >= threshold].copy()
    normal = bets[(bets["residual"] > bets["residual"].quantile(0.30)) & 
                  (bets["residual"] < bets["residual"].quantile(0.70))].copy()
    
    print(f"\nOverperformers: {len(crushers)} props (residual >= {threshold:.1f})")
    print(f"Normal range: {len(normal)} props (for comparison)")
    
    factors = [
        ("is_dome", "Dome game"),
        ("is_outdoor", "Outdoor game"),
        ("is_cold", "Cold (<=35F)"),
        ("is_windy", "Windy (>=15mph)"),
        ("is_division", "Division game"),
        ("is_primetime", "Primetime (Thu/Mon)"),
        ("is_thursday", "Thursday game"),
        ("is_grass", "Grass surface"),
        ("is_turf", "Turf surface"),
        ("is_early", "Weeks 1-4"),
        ("is_mid", "Weeks 5-12"),
        ("is_late", "Weeks 13+"),
        ("home_short_rest", "Home short rest"),
        ("away_short_rest", "Away short rest"),
        ("flag_heavy_ref", "Flag-heavy referee"),
        ("flag_light_ref", "Flag-light referee"),
    ]
    
    print(f"\n{'Factor':<25} {'Crushers':>10} {'Normal':>10} {'Diff':>8} {'Signal':>8}")
    print("-" * 65)
    
    signals = []
    for col, label in factors:
        if col not in bets.columns:
            continue
        crush_rate = crushers[col].mean() if crushers[col].dtype == bool else 0
        normal_rate = normal[col].mean() if normal[col].dtype == bool else 0
        diff = crush_rate - normal_rate
        sig = "→ YES" if abs(diff) > 0.03 else ""
        print(f"  {label:<23} {crush_rate:>9.1%} {normal_rate:>9.1%} {diff:>+7.1%} {sig:>8}")
        if abs(diff) > 0.03:
            signals.append({"factor": label, "direction": "over" if diff > 0 else "avoid", "diff": diff})
    
    # By market
    print(f"\n  Market distribution of crushers:")
    for market, count in crushers["market"].value_counts().items():
        normal_pct = (normal["market"] == market).mean()
        crush_pct = count / len(crushers)
        print(f"    {market}: {crush_pct:.1%} of crushers vs {normal_pct:.1%} normal")
    
    # Continuous factors
    print(f"\n  Continuous factors:")
    print(f"    Avg line_deviation_pct: Crushers {crushers['line_deviation_pct'].mean():+.1%} vs Normal {normal['line_deviation_pct'].mean():+.1%}")
    if "home_players_out" in bets.columns:
        print(f"    Avg home_players_out: Crushers {crushers['home_players_out'].mean():.1f} vs Normal {normal['home_players_out'].mean():.1f}")
    if "game_total" in bets.columns:
        print(f"    Avg game_total: Crushers {crushers['game_total'].mean():.1f} vs Normal {normal['game_total'].mean():.1f}")
    
    return signals


def analyze_underperformers(bets: pd.DataFrame):
    """Find what factors are present when players BUST their props."""
    
    print(f"\n{'='*70}")
    print("UNDERPERFORMERS: When did players BUST their props?")
    print("(Bottom 20% of residuals — player way below the line)")
    print("="*70)
    
    threshold = bets["residual"].quantile(0.20)
    busters = bets[bets["residual"] <= threshold].copy()
    normal = bets[(bets["residual"] > bets["residual"].quantile(0.30)) & 
                  (bets["residual"] < bets["residual"].quantile(0.70))].copy()
    
    print(f"\nUnderperformers: {len(busters)} props (residual <= {threshold:.1f})")
    
    factors = [
        ("is_dome", "Dome game"),
        ("is_outdoor", "Outdoor game"),
        ("is_cold", "Cold (<=35F)"),
        ("is_windy", "Windy (>=15mph)"),
        ("is_division", "Division game"),
        ("is_primetime", "Primetime (Thu/Mon)"),
        ("is_thursday", "Thursday game"),
        ("is_grass", "Grass surface"),
        ("is_turf", "Turf surface"),
        ("is_early", "Weeks 1-4"),
        ("is_mid", "Weeks 5-12"),
        ("is_late", "Weeks 13+"),
        ("home_short_rest", "Home short rest"),
        ("away_short_rest", "Away short rest"),
        ("flag_heavy_ref", "Flag-heavy referee"),
        ("flag_light_ref", "Flag-light referee"),
    ]
    
    print(f"\n{'Factor':<25} {'Busters':>10} {'Normal':>10} {'Diff':>8} {'Signal':>8}")
    print("-" * 65)
    
    signals = []
    for col, label in factors:
        if col not in bets.columns:
            continue
        bust_rate = busters[col].mean() if busters[col].dtype == bool else 0
        normal_rate = normal[col].mean() if normal[col].dtype == bool else 0
        diff = bust_rate - normal_rate
        sig = "→ YES" if abs(diff) > 0.03 else ""
        print(f"  {label:<23} {bust_rate:>9.1%} {normal_rate:>9.1%} {diff:>+7.1%} {sig:>8}")
        if abs(diff) > 0.03:
            signals.append({"factor": label, "direction": "under" if diff > 0 else "avoid", "diff": diff})
    
    # By market
    print(f"\n  Market distribution of busters:")
    for market, count in busters["market"].value_counts().items():
        normal_pct = (normal["market"] == market).mean()
        bust_pct = count / len(busters)
        print(f"    {market}: {bust_pct:.1%} of busters vs {normal_pct:.1%} normal")
    
    # Continuous factors
    print(f"\n  Continuous factors:")
    print(f"    Avg line_deviation_pct: Busters {busters['line_deviation_pct'].mean():+.1%} vs Normal {normal['line_deviation_pct'].mean():+.1%}")
    if "home_players_out" in bets.columns:
        print(f"    Avg home_players_out: Busters {busters['home_players_out'].mean():.1f} vs Normal {normal['home_players_out'].mean():.1f}")
    if "game_total" in bets.columns:
        print(f"    Avg game_total: Busters {busters['game_total'].mean():.1f} vs Normal {normal['game_total'].mean():.1f}")
    
    return signals


def test_discovered_combos(bets: pd.DataFrame, over_signals, under_signals):
    """Test combinations discovered from residual analysis."""
    
    print(f"\n{'='*70}")
    print("TESTING DISCOVERED COMBINATIONS")
    print("Applying residual insights as betting filters")
    print("="*70)
    
    base_hit = bets["bet_won_10pct"].mean()
    print(f"\nBaseline: {base_hit:.1%} ({len(bets)} bets)")
    
    # Build dynamic combos based on signals found
    combos = []
    
    # Standard combos with now-working factors
    combo_defs = [
        # Dome effects
        ("Dome + Pass TDs", (bets["is_dome"]) & (bets["market"] == "player_pass_tds")),
        ("Dome + Receiving yards OVER", (bets["is_dome"]) & (bets["market"] == "player_reception_yds") & (bets["signal"] == "bet_over")),
        ("Dome + all OVER", (bets["is_dome"]) & (bets["signal"] == "bet_over")),
        ("Outdoor + UNDER (pass)", (bets["is_outdoor"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")),
        
        # Weather
        ("Cold + Rush OVER", (bets["is_cold"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")),
        ("Cold + Pass UNDER", (bets["is_cold"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")),
        ("Windy + Pass UNDER", (bets["is_windy"]) & (bets["market"].isin(["player_pass_yds", "player_reception_yds"])) & (bets["signal"] == "bet_under")),
        ("Not windy + Pass TDs", (~bets["is_windy"]) & (bets["market"] == "player_pass_tds")),
        
        # Rest/schedule
        ("Thursday + UNDER", (bets["is_thursday"]) & (bets["signal"] == "bet_under")),
        ("Short rest (home) + UNDER", (bets["home_short_rest"]) & (bets["signal"] == "bet_under")),
        ("Monday night + OVER", (bets["is_monday"]) & (bets["signal"] == "bet_over")),
        
        # Division
        ("Division + UNDER", (bets["is_division"]) & (bets["signal"] == "bet_under")),
        ("Non-division + Pass TDs", (~bets["is_division"]) & (bets["market"] == "player_pass_tds")),
        
        # Referee
        ("Flag-heavy ref + Pass props", (bets["flag_heavy_ref"]) & (bets["market"].isin(["player_pass_yds", "player_reception_yds", "player_receptions"]))),
        ("Flag-heavy ref + OVER", (bets["flag_heavy_ref"]) & (bets["signal"] == "bet_over")),
        ("Flag-light ref + UNDER", (bets["flag_light_ref"]) & (bets["signal"] == "bet_under")),
        
        # Surface
        ("Grass + Rush OVER", (bets["is_grass"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")),
        ("Turf + Receiving OVER", (bets["is_turf"]) & (bets["market"] == "player_reception_yds") & (bets["signal"] == "bet_over")),
        
        # Multi-factor
        ("Dome + early season + OVER", (bets["is_dome"]) & (bets["is_early"]) & (bets["signal"] == "bet_over")),
        ("Outdoor + cold + Pass UNDER", (bets["is_outdoor"]) & (bets["is_cold"]) & (bets["market"] == "player_pass_yds") & (bets["signal"] == "bet_under")),
        ("Thursday + Division + UNDER", (bets["is_thursday"]) & (bets["is_division"]) & (bets["signal"] == "bet_under")),
        ("Pass TDs + Dome + not division", (bets["market"] == "player_pass_tds") & (bets["is_dome"]) & (~bets["is_division"])),
        ("Receptions UNDER + not primetime", (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets["is_primetime"])),
        ("Flag-heavy + Dome + Pass", (bets["flag_heavy_ref"]) & (bets["is_dome"]) & (bets["market"].isin(["player_pass_yds", "player_pass_tds"]))),
        
        # Proven winners with new context
        ("Pass TDs + not windy + not cold", (bets["market"] == "player_pass_tds") & (~bets["is_windy"]) & (~bets["is_cold"])),
        ("Rec UNDER + not dome", (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets["is_dome"])),
        ("All UNDER + week 1-4 + not primetime", (bets["signal"] == "bet_under") & (bets["is_early"]) & (~bets["is_primetime"])),
    ]
    
    print(f"\n{'Combination':<50} {'Hit%':>7} {'Bets':>6} {'ROI':>8} {'Status':>8}")
    print("-" * 82)
    
    profitable = []
    
    for label, mask in combo_defs:
        subset = bets[mask]
        if len(subset) < 30:
            continue
        
        wins = subset["bet_won_10pct"].sum()
        total = len(subset)
        hit = wins / total
        losses = total - wins
        roi = ((wins * 100) - (losses * 110)) / (total * 110) * 100
        
        marker = "✓ PROFIT" if roi > 0 else ""
        print(f"{'✓' if roi > 0 else ' '} {label:<48} {hit:>6.1%} {total:>5,} {roi:>+7.1f}% {marker}")
        
        if roi > 0:
            profitable.append({"combo": label, "hit_rate": hit, "roi": roi, "bets": total})
    
    # Final summary
    print(f"\n{'='*70}")
    print("ALL PROFITABLE STRATEGIES (sorted by ROI)")
    print("="*70)
    
    if profitable:
        for p in sorted(profitable, key=lambda x: x["roi"], reverse=True):
            print(f"  {p['combo']}")
            print(f"    Hit: {p['hit_rate']:.1%} | ROI: {p['roi']:+.1f}% | Bets/season: ~{p['bets']//3}")
            print()
    
    return profitable


def main():
    print("=" * 70)
    print("DEEP RESIDUAL ANALYSIS + COMBINATION DISCOVERY")
    print("=" * 70)
    
    bets = load_and_enrich()
    print(f"\nTotal bets enriched: {len(bets):,}")
    
    # Analyze overperformers
    over_signals = analyze_overperformers(bets)
    
    # Analyze underperformers
    under_signals = analyze_underperformers(bets)
    
    # Test combinations
    profitable = test_discovered_combos(bets, over_signals, under_signals)
    
    # Save enriched dataset
    bets.to_parquet(PROC_DIR / "bets_fully_enriched.parquet", index=False)
    print(f"\nEnriched dataset saved: data/processed/bets_fully_enriched.parquet")


if __name__ == "__main__":
    main()
