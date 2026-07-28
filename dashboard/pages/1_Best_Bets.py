"""
Best Bets Page — Top picks ranked by EV and confidence.
Shows team props and player props combined, sorted by expected value.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Best Bets | NFL Analytics", page_icon="💰", layout="wide")

st.title("💰 Best Bets")
st.caption("Ranked by Expected Value (EV) — only showing +EV opportunities")
st.markdown("---")


def american_to_decimal(odds):
    """Convert American odds to decimal."""
    if pd.isna(odds): return np.nan
    if odds > 0: return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def break_even_pct(american_odds):
    """
    Calculate break-even win percentage from American odds.
    This is the minimum win rate needed to not lose money.
    Source: BettingPros standard formula.
    
    Negative odds: 100 / ((100/odds) + 1)
    Positive odds: 100 / ((odds/100) + 1)
    """
    if pd.isna(american_odds): return np.nan
    if american_odds < 0:
        return 100 / ((100 / abs(american_odds)) + 1) / 100  # as decimal
    else:
        return 100 / ((american_odds / 100) + 1) / 100  # as decimal


def calc_ev(win_prob, american_odds, stake=100):
    """
    Calculate Expected Value per $100 bet.
    EV = (win_prob × profit) - (lose_prob × stake)
    
    Positive EV = good bet. Our model's probability exceeds break-even.
    """
    if pd.isna(american_odds) or pd.isna(win_prob):
        return 0
    decimal = american_to_decimal(american_odds)
    profit = (decimal - 1) * stake
    ev = (win_prob * profit) - ((1 - win_prob) * stake)
    return ev


def is_good_bet(win_prob, american_odds, min_edge=0.025):
    """
    The decision rule: Is this a +EV bet with sufficient edge?
    
    Returns True if our win probability exceeds break-even by at least min_edge (2.5%).
    This ensures we clear the vig and have real expected profit.
    """
    be = break_even_pct(american_odds)
    if pd.isna(be): return False
    return win_prob > (be + min_edge)


def implied_prob(american_odds):
    """Convert American odds to implied probability (what the book thinks)."""
    return break_even_pct(american_odds)  # Same formula, different name


# ===== EV FORMULA EXPLANATION =====
with st.expander("📐 How EV is Calculated"):
    st.markdown("""
    **Expected Value (EV)** = (Our Win Probability × Profit if Win) - (Our Loss Probability × Stake)
    
    **The Decision Rule:**
    1. Calculate **break-even win %** from the odds
    2. Compare to **our model's win probability**
    3. If our probability > break-even → **BET** (positive EV)
    4. If our probability < break-even → **SKIP** (negative EV)
    
    **Break-even formula:**
    - Negative odds: `100 / ((100/odds) + 1)` → e.g., -110 = 52.4%
    - Positive odds: `100 / ((odds/100) + 1)` → e.g., +150 = 40.0%
    
    **Example:** Odds are -110 (break-even = 52.4%). Our model says 56% win probability.
    - Profit if win: $90.91 (on $100 bet at -110)
    - EV = (0.56 × $90.91) - (0.44 × $100) = **+$6.91 per $100 bet**
    - 56% > 52.4% → **Good bet ✓**
    
    **Quick Reference:**
    | Odds | Break-Even % | Need Our Model At | To Clear By |
    |:---:|:---:|:---:|:---:|
    | -110 | 52.4% | 55%+ | 2.6%+ edge |
    | -115 | 53.5% | 56%+ | 2.5%+ edge |
    | -120 | 54.5% | 57%+ | 2.5%+ edge |
    | -130 | 56.5% | 59%+ | 2.5%+ edge |
    | +100 | 50.0% | 53%+ | 3%+ edge |
    | +110 | 47.6% | 50%+ | 2.4%+ edge |
    | +120 | 45.5% | 48%+ | 2.5%+ edge |
    | +140 | 41.7% | 44%+ | 2.3%+ edge |
    | +150 | 40.0% | 43%+ | 3%+ edge |
    | +200 | 33.3% | 36%+ | 2.7%+ edge |
    
    **Our rule:** We only bet when our edge > 2.5% above break-even.
    This ensures we clear the vig and have real expected profit.
    """)

st.markdown("---")

# ===== LOAD BEST BETS =====
# In-season: this would pull from the strategy engine output
# Pre-season: show example format + confirmed strategy overview

st.info(
    "**Preseason Mode** — Live best bets populate once FanDuel posts lines. "
    "Below is the format you'll see during the season."
)

# Example data structure
example_bets = pd.DataFrame([
    {"Player": "Example QB", "Market": "Pass TDs", "FD Line": "Over 1.5", "Odds": -130,
     "Model Proj": "2.1 TDs", "Our Win Prob": 0.62, "Implied Prob": 0.565, "EV/$100": 7.2,
     "Strategy": "Week 1 Under", "Confidence": "HIGH", "Units": 2.0},
    {"Player": "Example WR", "Market": "Rec Yards", "FD Line": "Under 62.5", "Odds": -110,
     "Model Proj": "48.3 yds", "Our Win Prob": 0.57, "Implied Prob": 0.524, "EV/$100": 5.8,
     "Strategy": "New team early UNDER", "Confidence": "HIGH", "Units": 2.0},
    {"Player": "Example RB", "Market": "Rush Yards", "FD Line": "Over 58.5", "Odds": +100,
     "Model Proj": "71.2 yds", "Our Win Prob": 0.55, "Implied Prob": 0.500, "EV/$100": 10.0,
     "Strategy": "Cold rush OVER", "Confidence": "MEDIUM", "Units": 1.0},
])

st.markdown("### Best Bets Format (Season Preview)")
st.dataframe(
    example_bets,
    use_container_width=True,
    hide_index=True,
    column_config={
        "EV/$100": st.column_config.NumberColumn("EV/$100", format="$%.1f"),
        "Our Win Prob": st.column_config.NumberColumn("Win Prob", format="%.0%%"),
        "Implied Prob": st.column_config.NumberColumn("Implied", format="%.0%%"),
        "Units": st.column_config.NumberColumn("Units", format="%.1f"),
    },
)

st.markdown("---")
st.markdown("### What Triggers a Best Bet")
st.markdown("""
A prop appears here when:
1. **Confirmed strategy fires** (one of our 6 proven strategies applies)
2. **Positive EV** (our win probability > implied probability from odds)
3. **Player not on AVOID list** (67 players FanDuel is too sharp on)
4. **Bonus:** Player on BANKABLE list gets extra confidence

Sorted by EV descending — highest value at the top.
""")
