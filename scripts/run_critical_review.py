"""
Critical Review: Apply Operating Principles to Our Findings
============================================================
For each "profitable" strategy, ask:
1. Does it survive EACH season independently? (not just pooled)
2. Is the sample large enough to trust? (n>100 minimum for confidence)
3. Is FanDuel likely already aware of this? (is it obvious?)
4. Does the edge DECAY over time? (2023 vs 2024 vs 2025)
5. Could this be overfitting to noise?

Principle 2: Every hypothesis starts false. Attempt to DISPROVE.
Principle 12: Must survive multiple seasons independently.
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


def load_data():
    """Load the enriched bets + build strategy flags."""
    bets = pd.read_parquet(PROC_DIR / "bets_fully_enriched.parquet")
    
    # Rebuild needed columns
    bets["home_abbr"] = bets["home_team"].map(FULL_TO_ABBR)
    bets["away_abbr"] = bets["away_team"].map(FULL_TO_ABBR)
    
    # Load games for context
    games = pd.read_parquet(RAW_DIR / "games_historical.parquet")
    game_cols = ["season", "week", "home_team", "div_game", "weekday", "roof",
                 "surface", "temp", "wind", "spread_line", "total_line"]
    available = [c for c in game_cols if c in games.columns]
    
    bets = bets.merge(
        games[available].drop_duplicates(subset=["season", "week", "home_team"]),
        left_on=["season", "week", "home_abbr"],
        right_on=["season", "week", "home_team"],
        how="left", suffixes=("", "_gm"),
    )
    
    bets["is_dome"] = bets["roof"].isin(["dome", "closed"]) if "roof" in bets.columns else False
    bets["is_cold"] = bets["temp"].fillna(65) <= 35 if "temp" in bets.columns else False
    bets["is_windy"] = bets["wind"].fillna(0) >= 15 if "wind" in bets.columns else False
    bets["is_division"] = bets["div_game"] == 1 if "div_game" in bets.columns else False
    bets["is_primetime"] = bets["weekday"].str.contains("Thursday|Monday", na=False) if "weekday" in bets.columns else False
    
    bets["bet_won"] = (
        ((bets["signal"] == "bet_over") & (bets["result"] == "won")) |
        ((bets["signal"] == "bet_under") & (bets["result"] == "lost"))
    )
    
    return bets


def season_stability_test(bets, mask, label):
    """
    THE CRITICAL TEST: Does this strategy profit in EACH season independently?
    If it only works in 1 of 3 seasons, it's likely noise.
    """
    subset = bets[mask]
    if len(subset) < 30:
        return None
    
    overall_wins = subset["bet_won"].sum()
    overall_total = len(subset)
    overall_hit = overall_wins / overall_total
    overall_roi = ((overall_wins * 100) - ((overall_total - overall_wins) * 110)) / (overall_total * 110) * 100
    
    # Per-season breakdown
    seasons_profitable = 0
    season_data = []
    
    for season in sorted(subset["season"].unique()):
        s = subset[subset["season"] == season]
        if len(s) < 10:
            continue
        s_wins = s["bet_won"].sum()
        s_total = len(s)
        s_hit = s_wins / s_total
        s_roi = ((s_wins * 100) - ((s_total - s_wins) * 110)) / (s_total * 110) * 100
        season_data.append({"season": season, "n": s_total, "hit": s_hit, "roi": s_roi})
        if s_roi > 0:
            seasons_profitable += 1
    
    total_seasons = len(season_data)
    consistency = seasons_profitable / total_seasons if total_seasons > 0 else 0
    
    # Verdict
    if overall_roi > 0 and consistency >= 0.67:  # profitable in 2/3+ seasons
        verdict = "CONFIRMED"
    elif overall_roi > 0 and consistency >= 0.5:
        verdict = "WEAK (inconsistent)"
    elif overall_roi <= 0:
        verdict = "REJECTED (not profitable)"
    else:
        verdict = "SUSPECT (1-season wonder)"
    
    return {
        "label": label,
        "overall_hit": overall_hit,
        "overall_roi": overall_roi,
        "overall_n": overall_total,
        "seasons_profitable": seasons_profitable,
        "total_seasons": total_seasons,
        "consistency": consistency,
        "season_data": season_data,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("CRITICAL REVIEW: Applying Research Principles")
    print("Testing season-by-season stability of ALL claimed edges")
    print("=" * 70)
    print("\nPrinciple 2: Every hypothesis starts FALSE.")
    print("Principle 12: Must survive MULTIPLE SEASONS independently.")
    print("Requirement: Profitable in at least 2 of 3 seasons to be CONFIRMED.\n")
    
    bets = load_data()
    print(f"Total bets: {len(bets):,}")
    print(f"Seasons: {sorted(bets['season'].unique())}\n")
    
    # Define all strategies to test
    strategies = [
        ("Week 1 + all UNDER",
         (bets["week"] == 1) & (bets["signal"] == "bet_under")),
        ("Week 1 + Pass TDs UNDER",
         (bets["week"] == 1) & (bets["signal"] == "bet_under") & (bets["market"] == "player_pass_tds")),
        ("Week 1 + Pass yards UNDER",
         (bets["week"] == 1) & (bets["signal"] == "bet_under") & (bets["market"] == "player_pass_yds")),
        ("Week 1 + Receptions UNDER",
         (bets["week"] == 1) & (bets["signal"] == "bet_under") & (bets["market"] == "player_receptions")),
        ("Pass TDs (not windy, not cold, not division)",
         (bets["market"] == "player_pass_tds") & (~bets["is_windy"]) & (~bets["is_cold"]) & (~bets["is_division"])),
        ("Receptions UNDER + outdoor + not primetime",
         (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (~bets["is_dome"]) & (~bets["is_primetime"])),
        ("Rec UNDER + division + outdoor",
         (bets["market"] == "player_receptions") & (bets["signal"] == "bet_under") & (bets["is_division"]) & (~bets["is_dome"])),
        ("UNDER + weeks 1-4 + not primetime + not dome",
         (bets["signal"] == "bet_under") & (bets["week"] <= 4) & (~bets["is_primetime"]) & (~bets["is_dome"])),
        ("Dome + Pass TDs",
         (bets["is_dome"]) & (bets["market"] == "player_pass_tds")),
        ("Cold + Rush OVER",
         (bets["is_cold"]) & (bets["market"] == "player_rush_yds") & (bets["signal"] == "bet_over")),
        ("Windy + Pass UNDER",
         (bets["is_windy"]) & (bets["market"].isin(["player_pass_yds", "player_reception_yds"])) & (bets["signal"] == "bet_under")),
        ("Weeks 13+ + Pass TDs OVER + not cold",
         (bets["week"] >= 13) & (bets["market"] == "player_pass_tds") & (bets["signal"] == "bet_over") & (~bets["is_cold"])),
        ("Division + all UNDER",
         (bets["is_division"]) & (bets["signal"] == "bet_under")),
        ("Monday + UNDER",
         (bets["is_primetime"]) & (bets["weekday"].str.contains("Monday", na=False)) & (bets["signal"] == "bet_under")),
        ("All UNDER (no filter)",
         bets["signal"] == "bet_under"),
        ("All OVER (no filter)",
         bets["signal"] == "bet_over"),
    ]
    
    # Run stability tests
    print(f"{'Strategy':<55} {'Hit%':>6} {'ROI':>7} {'N':>6} {'Seasons+':>9} {'Verdict':>15}")
    print("=" * 105)
    
    confirmed = []
    weak = []
    rejected = []
    
    for label, mask in strategies:
        result = season_stability_test(bets, mask, label)
        if result is None:
            continue
        
        print(f"  {result['label']:<53} {result['overall_hit']:>5.1%} {result['overall_roi']:>+6.1f}% "
              f"{result['overall_n']:>5,} {result['seasons_profitable']}/{result['total_seasons']:>7} "
              f"{result['verdict']:>15}")
        
        # Print per-season detail
        for sd in result["season_data"]:
            marker = "+" if sd["roi"] > 0 else "-"
            print(f"      {sd['season']}: {sd['hit']:.1%} hit, {sd['roi']:+.1f}% ROI (n={sd['n']}) [{marker}]")
        print()
        
        if "CONFIRMED" in result["verdict"]:
            confirmed.append(result)
        elif "WEAK" in result["verdict"]:
            weak.append(result)
        else:
            rejected.append(result)
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICTS (after season-stability testing)")
    print("=" * 70)
    
    print(f"\n✅ CONFIRMED ({len(confirmed)} strategies survive scrutiny):")
    for r in sorted(confirmed, key=lambda x: x["overall_roi"], reverse=True):
        print(f"   {r['label']}: {r['overall_hit']:.1%} | {r['overall_roi']:+.1f}% ROI | {r['overall_n']} bets | {r['seasons_profitable']}/{r['total_seasons']} seasons profitable")
    
    print(f"\n⚠️ WEAK ({len(weak)} — use with caution, inconsistent across seasons):")
    for r in sorted(weak, key=lambda x: x["overall_roi"], reverse=True):
        print(f"   {r['label']}: {r['overall_hit']:.1%} | {r['overall_roi']:+.1f}% ROI | {r['seasons_profitable']}/{r['total_seasons']} seasons")
    
    print(f"\n❌ REJECTED ({len(rejected)} — did not survive multi-season test):")
    for r in rejected:
        print(f"   {r['label']}: {r['overall_roi']:+.1f}% ROI | {r['seasons_profitable']}/{r['total_seasons']} seasons")


if __name__ == "__main__":
    main()
