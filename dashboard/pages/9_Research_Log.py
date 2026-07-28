"""
Research Log — All confirmed hypotheses, rejected ideas, and findings.
Reference this anytime to understand WHY the model makes certain picks.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Research Log | NFL Analytics", page_icon="🔬", layout="wide")

st.title("🔬 Research Log")
st.caption("All confirmed findings, rejected hypotheses, and operating principles")
st.markdown("---")

# ===== OPERATING PRINCIPLES =====
with st.expander("📏 Operating Principles (How We Think)"):
    st.markdown("""
    1. **The Market Is Smart** — If it's obvious, it's priced in. Only exploit what FanDuel undervalues.
    2. **Every Hypothesis Starts False** — We disprove, not prove. 3/3 seasons minimum.
    3. **Don't Learn From Individual Games** — Large samples only.
    4. **Separate Signal From Story** — Narratives ≠ evidence.
    5. **Discover Market Mistakes** — What assumption is FanDuel making? Is it correct?
    6. **Interaction Effects > Single Variables** — Combinations are where edge lives.
    7. **Predict Roles Before Statistics** — Usage changes before box scores do.
    8. **Optimize for EV, Not Accuracy** — We don't need to predict stats. We need to find mispriced bets.
    """)

st.markdown("---")

# ===== CONFIRMED STRATEGIES =====
st.header("✅ Confirmed Strategies (Survived 3/3 Seasons)")

confirmed = [
    {"Strategy": "Week 1 ALL UNDER", "Hit Rate": "63.8%", "ROI": "+21.7%", "Bets/Yr": "~90",
     "Why It Works": "FanDuel uses last-season averages. Players are rusty Week 1. No current data to calibrate.",
     "Seasons": "2023 ✓ 2024 ✓ 2025 ✓"},
    {"Strategy": "Week 1 Pass TDs UNDER", "Hit Rate": "75.8%", "ROI": "+44.6%", "Bets/Yr": "~11",
     "Why It Works": "QBs never in rhythm game 1. Timing off, new plays, limited reps.",
     "Seasons": "2023 ✓ 2024 ✓"},
    {"Strategy": "Week 1 Pass Yards UNDER", "Hit Rate": "70.6%", "ROI": "+34.8%", "Bets/Yr": "~11",
     "Why It Works": "Same as above — passing volume suppressed Week 1.",
     "Seasons": "2023 ✓ 2024 ✓ 2025 ✓"},
    {"Strategy": "Weeks 1-4 UNDER (outdoor, not primetime)", "Hit Rate": "56.6%", "ROI": "+8.1%", "Bets/Yr": "~213",
     "Why It Works": "Early season + outdoor = universal suppression. FanDuel slow to adjust from preseason projections.",
     "Seasons": "2023 ✓ 2024 ✓ 2025 ✓"},
    {"Strategy": "Weeks 13+ Pass TDs OVER (not cold)", "Hit Rate": "58.1%", "ROI": "+10.9%", "Bets/Yr": "~55",
     "Why It Works": "Playoff push = aggressive play-calling. Desperate teams throw more TDs. Edge STRENGTHENING each year.",
     "Seasons": "2023 ✓ 2024 ✓ 2025 ✓"},
    {"Strategy": "Monday Night UNDER", "Hit Rate": "53.5%", "ROI": "+2.1%", "Bets/Yr": "~138",
     "Why It Works": "MNF historically lower-scoring. Small but rock-steady edge.",
     "Seasons": "2023 ✓ 2024 ✓ 2025 ✓"},
]

st.dataframe(pd.DataFrame(confirmed), use_container_width=True, hide_index=True)

# ===== DIRECTIONAL (2/3 SEASONS) =====
st.markdown("---")
st.header("⚠️ Directional Findings (2/3 Seasons — Use With Caution)")

directional = [
    {"Strategy": "Pass TDs (not windy/cold/division)", "Hit Rate": "55.5%", "ROI": "+5.9%",
     "Concern": "DECAYING: 17.8% → 2.5% → -2.7% over 3 seasons. FanDuel may be learning."},
    {"Strategy": "Rec UNDER + division + outdoor", "Hit Rate": "56.8%", "ROI": "+8.4%",
     "Concern": "Failed in 2024. Only 2023 + 2025."},
    {"Strategy": "Windy + Pass UNDER", "Hit Rate": "55.7%", "ROI": "+6.3%",
     "Concern": "Small per-season samples. Directionally sound but noisy."},
    {"Strategy": "Cold + Rush OVER", "Hit Rate": "54.5%", "ROI": "+4.1%",
     "Concern": "Near zero in 2025. May be fading."},
    {"Strategy": "New team + weeks 1-4 UNDER", "Hit Rate": "55.2%", "ROI": "+5.3%",
     "Concern": "Strong logic (no chemistry). 2/3 seasons."},
    {"Strategy": "Boom regression + division + outdoor UNDER", "Hit Rate": "55.3%", "ROI": "+5.7%",
     "Concern": "High volume (209/yr). Logic sound. 2/3 seasons."},
]

st.dataframe(pd.DataFrame(directional), use_container_width=True, hide_index=True)

# ===== KEY MARKET INSIGHTS =====
st.markdown("---")
st.header("🧠 Key Market Insights")

st.markdown("""
**1. FanDuel's structural weakness:**
They predict the DIRECTION of performance correctly (64-68% accuracy on trend) 
but OVERSHOOT the magnitude. Lines move too far. Mean reversion exploits this.

**2. Pass TDs is our most exploitable market.**
High variance + context-dependent (game script, red zone, dome) = FanDuel misprices.

**3. Receiving yards is FanDuel's sharpest market.**
Only 34% of WRs are profitable. Avoid unless specific conditions stack (division + outdoor + cold).

**4. TEs are the most exploitable position.**
53% of TEs in our dataset are profitable. FanDuel underestimates TE floor consistency.

**5. The market is most vulnerable in Week 1 and Weeks 13-18.**
Week 1: No data to calibrate. Late season: Motivation/aggression not fully priced.

**6. Underdogs abandon the run.**
Big dogs (spread +7) rush UNDER hits 57.8%. Game script predictably shifts to passing.
""")

# ===== PLUS MONEY DIAMONDS =====
st.markdown("---")
st.header("💎 Plus Money Diamonds (+EV at Long Odds)")

diamonds = [
    {"Situation": "Dome + Pass TD Over (+150+)", "Hit Rate": "45.9%", "Implied": "36.9%", "EV/$100": "+$25"},
    {"Situation": "Any dome prop Over (+150+)", "Hit Rate": "44.2%", "Implied": "37.1%", "EV/$100": "+$21"},
    {"Situation": "Weeks 13+ Pass TD (+150+)", "Hit Rate": "40.0%", "Implied": "36.4%", "EV/$100": "+$10"},
]

st.dataframe(pd.DataFrame(diamonds), use_container_width=True, hide_index=True)

# ===== REJECTED =====
st.markdown("---")
st.header("❌ Rejected Hypotheses (Do Not Revisit)")

rejected = [
    {"Hypothesis": "Revenge games = OVER", "Result": "48.7% over rate. Players try too hard, defense game-plans.", "Sample": "3,317"},
    {"Hypothesis": "Flag-heavy refs = more scoring", "Result": "No consistent edge. Already priced in.", "Sample": "3,774"},
    {"Hypothesis": "Target share momentum (standalone)", "Result": "Not predictive alone. May work in combination.", "Sample": "Large"},
    {"Hypothesis": "High total games = overs hit", "Result": "Already priced in by FanDuel.", "Sample": "1,651"},
    {"Hypothesis": "After bust game = OVER", "Result": "49.2% hit. Regression UP is slower than DOWN.", "Sample": "4,770"},
    {"Hypothesis": "Home/away (standalone)", "Result": "No signal found.", "Sample": "Full dataset"},
    {"Hypothesis": "Division games UNDER (standalone)", "Result": "Only 1/3 seasons profitable. Noise.", "Sample": "1,644"},
    {"Hypothesis": "Receptions UNDER outdoor (standalone)", "Result": "Only 1/3 seasons. 2025 only.", "Sample": "1,055"},
    {"Hypothesis": "All UNDER (blanket)", "Result": "Not profitable. Needs filters.", "Sample": "5,252"},
    {"Hypothesis": "All OVER (blanket)", "Result": "Negative ROI all 3 seasons.", "Sample": "8,553"},
]

st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)

# ===== WEEKLY PATTERNS =====
st.markdown("---")
st.header("📅 Weekly Over Rate Patterns (Historical)")

weekly_data = [
    {"Week": 1, "Over Rate": "42.9%", "Note": "SLAM UNDER"},
    {"Week": 2, "Over Rate": "49.3%", "Note": "Neutral"},
    {"Week": 3, "Over Rate": "48.2%", "Note": "Slight under"},
    {"Week": 4, "Over Rate": "49.9%", "Note": "Neutral"},
    {"Week": 5, "Over Rate": "52.3%", "Note": "Slight over"},
    {"Week": 6, "Over Rate": "49.1%", "Note": "Neutral"},
    {"Week": 7, "Over Rate": "49.5%", "Note": "Neutral"},
    {"Week": 8, "Over Rate": "51.0%", "Note": "Neutral"},
    {"Week": 9, "Over Rate": "47.2%", "Note": "Under lean"},
    {"Week": 10, "Over Rate": "47.8%", "Note": "Under lean"},
    {"Week": 11, "Over Rate": "47.3%", "Note": "Under lean"},
    {"Week": 12, "Over Rate": "51.9%", "Note": "Slight over"},
    {"Week": 13, "Over Rate": "51.2%", "Note": "Neutral → Pass TD OVER starts"},
    {"Week": 14, "Over Rate": "44.9%", "Note": "STRONG UNDER (general) + Pass TD OVER"},
    {"Week": 15, "Over Rate": "47.1%", "Note": "Under lean"},
    {"Week": 16, "Over Rate": "47.7%", "Note": "Under lean"},
    {"Week": 17, "Over Rate": "46.8%", "Note": "Under lean"},
    {"Week": 18, "Over Rate": "50.2%", "Note": "Neutral (rest risk)"},
    {"Week": "19-21", "Over Rate": "38.6-49.2%", "Note": "Playoffs: HEAVY UNDER"},
]

st.dataframe(pd.DataFrame(weekly_data), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Last updated: July 28, 2026 | 63,441 props backtested | 3 NFL seasons")
