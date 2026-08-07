"""
Production Strategy Engine.
Implements all confirmed profitable strategies as betting signals.
Filters props through multi-factor rules to surface only +EV bets.

CONFIRMED STRATEGIES (survived 3/3 seasons):
1. Week 1 ALL UNDER (+21.7% ROI)
2. Week 1 Pass TDs UNDER (+44.6% ROI)
3. Week 1 Pass yards UNDER (+34.8% ROI)
4. Weeks 1-4 UNDER + outdoor + not primetime (+8.1% ROI)
5. Weeks 13+ Pass TDs OVER + not cold (+10.9% ROI)
6. Monday UNDER (+2.1% ROI)
7. Rec UNDER + division + outdoor + not early (+9.5% ROI)
8. Big underdog + Rush UNDER (+10.3% ROI)
9. Dome + Pass TDs OVER (+8.7% ROI)
10. High volatility players + UNDER (+8.7% ROI)
11. Bellcow RB (line >70) UNDER (+4.3% ROI)
12. Big fav + Receptions UNDER + outdoor (+5.5% ROI)
13. Big underdog + Pass yards OVER + not cold (+6.1% ROI)
14. Low total game (<=40) + UNDER (+2.5% ROI)
15. Close game expected + Pass TDs OVER (+5.3% ROI)
16. Cold outdoor Rec UNDER (+7.1% ROI)
17. Outdoor Rec UNDER not primetime (+4.4% ROI, volume)
18. Dome + Pass TD over at plus money (+25% EV, diamond)
19. Contract year OVER boost (estimated +3-7.5% ROI)

WEAK BUT DIRECTIONAL (2/3 seasons):
20. Pass TDs (not windy/cold/division) - USE WITH DECLINING CONFIDENCE
21. Rec UNDER + division + outdoor (weaker without not-early filter)
22. Windy + Pass UNDER
23. Cold + Rush OVER
24. New team + early season UNDER
25. Boom regression + division + outdoor
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from pipeline.config_loader import load_settings, get_data_dir, load_stadiums
from pipeline.ingest_contracts import is_contract_year_player


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

ABBR_TO_FULL = {v: k for k, v in FULL_TO_ABBR.items()}


def evaluate_prop(
    player_name: str,
    market: str,
    fanduel_line: float,
    fanduel_price: float,
    player_rolling_avg: float,
    week: int,
    home_team: str,
    away_team: str,
    is_dome: bool = False,
    is_cold: bool = False,
    is_windy: bool = False,
    is_division: bool = False,
    is_primetime: bool = False,
    is_monday: bool = False,
    is_new_team: bool = False,
    prev_game_boom: bool = False,
    opponent_injuries_out: int = 0,
    home_injuries_out: int = 0,
    is_big_underdog: bool = False,
    is_big_favorite: bool = False,
    game_total: float = 45.0,
    is_high_volatility_player: bool = False,
    is_bellcow_rb: bool = False,
    is_close_game_expected: bool = False,
    player_position: str = "",
    season: int = 2026,
) -> dict:
    """
    Evaluate a single prop bet through all confirmed strategy filters.
    Returns a signal with confidence, reasoning, and recommended action.
    """
    signals = []
    total_confidence = 0
    reasoning_parts = []

    # Line deviation from player average
    if player_rolling_avg > 0:
        line_deviation_pct = (fanduel_line - player_rolling_avg) / player_rolling_avg
    else:
        line_deviation_pct = 0

    # ================================================
    # STRATEGY 1: Week 1 ALL UNDER (CONFIRMED 3/3, +21.7%)
    # ================================================
    if week == 1:
        signals.append({
            "strategy": "Week 1 UNDER",
            "direction": "under",
            "confidence": 0.9,
            "roi_historical": 21.7,
            "reasoning": "Week 1: Players rusty, no current-season data. Confirmed 3/3 seasons (63.8% hit)."
        })
        # Extra boost for specific markets
        if market == "player_pass_tds":
            signals[-1]["confidence"] = 0.95
            signals[-1]["roi_historical"] = 44.6
            signals[-1]["reasoning"] += " Pass TDs especially strong (75.8% hit)."
        elif market == "player_pass_yds":
            signals[-1]["confidence"] = 0.92
            signals[-1]["roi_historical"] = 34.8
            signals[-1]["reasoning"] += " Pass yards very strong (70.6% hit)."

    # ================================================
    # STRATEGY 4: Weeks 1-4 UNDER + outdoor + not primetime (CONFIRMED 3/3, +8.1%)
    # ================================================
    if week <= 4 and not is_dome and not is_primetime:
        if line_deviation_pct > 0.05:  # line above average
            signals.append({
                "strategy": "Early season outdoor UNDER",
                "direction": "under",
                "confidence": 0.7,
                "roi_historical": 8.1,
                "reasoning": f"Weeks 1-4 + outdoor + not primetime. Line {line_deviation_pct:+.0%} above avg. Confirmed 3/3 (56.6% hit)."
            })

    # ================================================
    # STRATEGY 5: Weeks 13+ Pass TDs OVER + not cold (CONFIRMED 3/3, +10.9%)
    # ================================================
    if week >= 13 and market == "player_pass_tds" and not is_cold:
        if line_deviation_pct < -0.05:  # line below average (book expects fewer TDs)
            signals.append({
                "strategy": "Late season Pass TDs OVER",
                "direction": "over",
                "confidence": 0.75,
                "roi_historical": 10.9,
                "reasoning": "Weeks 13+: Playoff push = aggressive play calling. Not cold. Confirmed 3/3 (58.1% hit)."
            })
        else:
            signals.append({
                "strategy": "Late season Pass TDs OVER",
                "direction": "over",
                "confidence": 0.6,
                "roi_historical": 10.9,
                "reasoning": "Weeks 13+ Pass TDs lean OVER regardless. Confirmed 3/3."
            })

    # ================================================
    # STRATEGY 6: Monday UNDER (CONFIRMED 3/3, +2.1%)
    # ================================================
    if is_monday and line_deviation_pct > 0.05:
        signals.append({
            "strategy": "Monday Night UNDER",
            "direction": "under",
            "confidence": 0.55,
            "roi_historical": 2.1,
            "reasoning": "Monday Night: Historically lower scoring. Confirmed 3/3 (53.5% hit). Small but consistent."
        })

    # ================================================
    # WEAK STRATEGIES (2/3 seasons, use with lower confidence)
    # ================================================

    # New team + early season UNDER
    if is_new_team and week <= 4 and line_deviation_pct > 0:
        signals.append({
            "strategy": "New team early UNDER",
            "direction": "under",
            "confidence": 0.6,
            "roi_historical": 5.3,
            "reasoning": f"Player on new team, early season. No chemistry yet. (55.2% hit, 2/3 seasons)."
        })

    # Boom regression + division + outdoor + UNDER
    if prev_game_boom and is_division and not is_dome and line_deviation_pct > 0:
        signals.append({
            "strategy": "Boom regression + division",
            "direction": "under",
            "confidence": 0.6,
            "roi_historical": 5.7,
            "reasoning": "Coming off boom game + division opponent + outdoor. Regression likely. (55.3% hit)."
        })

    # Windy + pass/rec UNDER
    if is_windy and market in ["player_pass_yds", "player_reception_yds"] and line_deviation_pct > 0:
        signals.append({
            "strategy": "Windy pass UNDER",
            "direction": "under",
            "confidence": 0.55,
            "roi_historical": 6.3,
            "reasoning": "Wind >=15mph suppresses passing. (55.7% hit, 2/3 seasons)."
        })

    # Cold + rush OVER
    if is_cold and market == "player_rush_yds" and line_deviation_pct < -0.05:
        signals.append({
            "strategy": "Cold rush OVER",
            "direction": "over",
            "confidence": 0.55,
            "roi_historical": 4.1,
            "reasoning": "Cold game shifts to run-heavy. Rush over. (54.5% hit, 2/3 seasons)."
        })

    # High injury game UNDER
    total_injuries = opponent_injuries_out + home_injuries_out
    if total_injuries >= 8 and not is_dome:
        signals.append({
            "strategy": "High injury UNDER",
            "direction": "under",
            "confidence": 0.7,
            "roi_historical": 20.6,
            "reasoning": f"High injury game ({total_injuries} players out) + outdoor. Everyone underperforms. (63.2% hit)."
        })

    # ================================================
    # NEW STRATEGIES (from all_profitable_strategies.md)
    # ================================================

    # ================================================
    # STRATEGY: Rec UNDER + division + outdoor + not early season (CONFIRMED, +9.5%)
    # ================================================
    if (market in ["player_reception_yds", "player_receptions"]
            and is_division and not is_dome and week > 4 and line_deviation_pct > 0):
        signals.append({
            "strategy": "Division outdoor Rec UNDER",
            "direction": "under",
            "confidence": 0.7,
            "roi_historical": 9.5,
            "reasoning": "Division familiarity + outdoor elements = tight coverage. (57.4% hit, ~106 bets/yr)."
        })

    # ================================================
    # STRATEGY: Big underdog + Rush UNDER (CONFIRMED, +10.3%)
    # ================================================
    if is_big_underdog and market == "player_rush_yds" and line_deviation_pct > 0:
        signals.append({
            "strategy": "Big underdog Rush UNDER",
            "direction": "under",
            "confidence": 0.7,
            "roi_historical": 10.3,
            "reasoning": "Big underdogs fall behind, abandon the run. Rush UNDER. (57.8% hit, ~75 bets/yr)."
        })

    # ================================================
    # STRATEGY: Dome + Pass TDs OVER all prices (CONFIRMED, +8.7%)
    # ================================================
    if is_dome and market == "player_pass_tds" and week > 4:
        signals.append({
            "strategy": "Dome Pass TDs OVER",
            "direction": "over",
            "confidence": 0.65,
            "roi_historical": 8.7,
            "reasoning": "Dome = perfect passing conditions. Pass TDs OVER. (57.0% hit, ~53 bets/yr)."
        })

    # ================================================
    # STRATEGY: High volatility players + UNDER (CONFIRMED, +8.7%)
    # ================================================
    if is_high_volatility_player and line_deviation_pct > 0.05:
        signals.append({
            "strategy": "High volatility UNDER",
            "direction": "under",
            "confidence": 0.65,
            "roi_historical": 8.7,
            "reasoning": "Boom/bust player — FanDuel anchors to boom games but bust rate is higher. (57.0% hit)."
        })

    # ================================================
    # STRATEGY: Bellcow RB (line >70) UNDER (CONFIRMED, +4.3%)
    # ================================================
    if is_bellcow_rb and market == "player_rush_yds" and fanduel_line > 70:
        signals.append({
            "strategy": "Bellcow RB UNDER",
            "direction": "under",
            "confidence": 0.58,
            "roi_historical": 4.3,
            "reasoning": "Star RB lines inflated by name value. UNDER at high lines. (54.7% hit, ~28 bets/yr)."
        })

    # ================================================
    # STRATEGY: Big favorite + Receptions UNDER + outdoor (CONFIRMED, +5.5%)
    # ================================================
    if (is_big_favorite and market == "player_receptions"
            and not is_dome and line_deviation_pct > 0):
        signals.append({
            "strategy": "Big fav Rec UNDER",
            "direction": "under",
            "confidence": 0.6,
            "roi_historical": 5.5,
            "reasoning": "Favorites run the clock = fewer pass attempts = fewer catches. (55.3% hit, ~50 bets/yr)."
        })

    # ================================================
    # STRATEGY: Big underdog + Pass yards OVER + not cold (CONFIRMED, +6.1%)
    # ================================================
    if is_big_underdog and market == "player_pass_yds" and not is_cold and line_deviation_pct < -0.05:
        signals.append({
            "strategy": "Big dog Pass yards OVER",
            "direction": "over",
            "confidence": 0.6,
            "roi_historical": 6.1,
            "reasoning": "Big underdogs throw to catch up. Pass yards OVER when line is below avg. (55.6% hit)."
        })

    # ================================================
    # STRATEGY: Low total game (<=40) + UNDER (CONFIRMED, +2.5%)
    # ================================================
    if game_total <= 40 and line_deviation_pct > 0.05:
        signals.append({
            "strategy": "Low total game UNDER",
            "direction": "under",
            "confidence": 0.55,
            "roi_historical": 2.5,
            "reasoning": f"Game total only {game_total}. Vegas knows it's low-scoring but props don't adjust enough. (53.7% hit)."
        })

    # ================================================
    # STRATEGY: Close game expected + Pass TDs (CONFIRMED, +5.3%)
    # ================================================
    if is_close_game_expected and market == "player_pass_tds":
        signals.append({
            "strategy": "Close game Pass TDs",
            "direction": "over",
            "confidence": 0.6,
            "roi_historical": 5.3,
            "reasoning": "Tight games = red zone aggression = more TDs. (55.1% hit, ~142 bets/yr)."
        })

    # ================================================
    # STRATEGY: Rec UNDER + outdoor + cold (CONFIRMED, +7.1%)
    # ================================================
    if (market in ["player_reception_yds", "player_receptions"]
            and is_cold and not is_dome and line_deviation_pct > 0):
        signals.append({
            "strategy": "Cold outdoor Rec UNDER",
            "direction": "under",
            "confidence": 0.65,
            "roi_historical": 7.1,
            "reasoning": "Cold hands = dropped passes. Reception/yards UNDER. (56.1% hit, ~49 bets/yr)."
        })

    # ================================================
    # STRATEGY: Receptions UNDER + outdoor + not primetime (volume play, +4.4%)
    # ================================================
    if (market == "player_receptions" and not is_dome and not is_primetime
            and line_deviation_pct > 0.05 and week > 4):
        signals.append({
            "strategy": "Outdoor Rec UNDER (volume)",
            "direction": "under",
            "confidence": 0.57,
            "roi_historical": 4.4,
            "reasoning": "Outdoor + not primetime = slight reception suppression. Volume play. (54.7% hit, ~351 bets/yr)."
        })

    # ================================================
    # DIAMOND PLAY: Dome + Pass TD OVER at plus money (+150+)
    # ================================================
    if is_dome and market == "player_pass_tds" and fanduel_price >= 150:
        signals.append({
            "strategy": "Diamond: Dome Pass TD +money",
            "direction": "over",
            "confidence": 0.7,
            "roi_historical": 25.0,
            "reasoning": "Dome + plus money pass TD over. 45.9% hit vs 36.9% implied = +$25 EV per $100. DIAMOND."
        })

    # ================================================
    # STRATEGY: Contract Year OVER boost
    # ================================================
    contract_info = is_contract_year_player(player_name, season)
    if contract_info and player_position in ["WR", "TE", "RB"]:
        # Contract year is a BOOST, not a standalone bet
        # Only fires when line is at or below rolling avg (book hasn't priced in motivation)
        if line_deviation_pct <= 0.02:  # line not inflated
            boost = contract_info["contract_boost"]
            tier = contract_info["contract_tier"]
            signals.append({
                "strategy": "Contract year OVER",
                "direction": "over",
                "confidence": 0.5 + boost,  # base 0.5 + tier boost
                "roi_historical": 3.0 + (4 - tier) * 1.5,  # estimated: tier 1 = 7.5%, tier 3 = 4.5%
                "reasoning": f"Contract year player (Tier {tier}). Financial motivation to outperform. Line not inflated ({line_deviation_pct:+.0%} vs avg)."
            })

    # ================================================
    # DETERMINE FINAL SIGNAL
    # ================================================
    if not signals:
        return {
            "action": "no_bet",
            "confidence": 0,
            "direction": None,
            "reasoning": "No confirmed strategy applies.",
            "strategies_triggered": [],
        }

    # Take highest confidence signal
    best_signal = max(signals, key=lambda s: s["confidence"])
    
    # Boost confidence if multiple signals agree
    agreeing = [s for s in signals if s["direction"] == best_signal["direction"]]
    if len(agreeing) > 1:
        best_signal["confidence"] = min(best_signal["confidence"] + 0.1, 0.95)
        best_signal["reasoning"] += f" [{len(agreeing)} confirming signals]"

    # Unit sizing based on confidence
    if best_signal["confidence"] >= 0.85:
        units = 3.0
        tier = "HIGH"
    elif best_signal["confidence"] >= 0.7:
        units = 2.0
        tier = "MEDIUM"
    elif best_signal["confidence"] >= 0.55:
        units = 1.0
        tier = "LOW"
    else:
        units = 0.5
        tier = "SPECULATIVE"

    return {
        "action": f"bet_{best_signal['direction']}",
        "direction": best_signal["direction"],
        "confidence": best_signal["confidence"],
        "confidence_tier": tier,
        "units": units,
        "strategy": best_signal["strategy"],
        "roi_historical": best_signal["roi_historical"],
        "reasoning": best_signal["reasoning"],
        "strategies_triggered": [s["strategy"] for s in signals],
        "line_deviation_pct": line_deviation_pct,
    }
