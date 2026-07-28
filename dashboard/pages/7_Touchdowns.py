"""
Touchdown Props Page — Passing TDs + Anytime TD
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Touchdowns | NFL Analytics", page_icon="🎯", layout="wide")

st.title("🎯 Touchdowns")
st.caption("Passing TDs and Anytime TD scorer props")
st.markdown("---")

td_type = st.radio("TD Type", ["Passing TDs", "Anytime TD"], horizontal=True)
period = st.radio("Period", ["Full Game", "1st Half", "2nd Half"], horizontal=True, key="td_period")
st.markdown("---")

st.info("**Preseason Mode** — Props populate once FanDuel posts lines.")

if td_type == "Passing TDs":
    st.markdown("### 🏆 PASSING TDs — Our BEST Market")
    st.markdown("""
    **This is our single most profitable prop market across all conditions.**
    
    **Confirmed findings:**
    - **Pass TDs (not windy/cold/division): 55.5% hit, +5.9% ROI** (declining — was 17.8% in 2023)
    - **Week 1 Pass TDs UNDER: 75.8%** hit rate (SLAM)
    - **Weeks 13-18 Pass TDs OVER (not cold): 58.1%, +10.9% ROI** — STRENGTHENING each year
    - **Dome + Pass TDs (plus money): 45.9% vs 36.9% implied = +$25 EV**
    - **Close game expected (spread ≤3) + Pass TDs: 55.1%** hit
    
    **Why Pass TDs are exploitable:**
    TD scoring is high-variance. A QB can throw 0, 1, 2, or 4 TDs on any given day.
    FanDuel uses recent averages but TDs are heavily context-dependent:
    - Red zone opportunities (team-dependent)
    - Game script (close games = more pass TDs)
    - Indoor vs outdoor
    - Defensive matchup
    
    The market consistently underestimates TD variance in favorable conditions
    and overestimates it in Week 1 / early season.
    """)

else:
    st.markdown("### Anytime TD Scorer")
    st.markdown("""
    **Diamond play territory — plus money with edge:**
    
    - **Dome + Anytime TD (plus money): +$25 EV per $100**
    - Red zone target share is the key predictor
    - Players with high goal-line usage are underpriced as TD scorers
    
    **Key factors for anytime TD:**
    - Inside-5 opportunities per game
    - Red zone target share
    - Goal-line carry share
    - Dome (more scoring = more TDs available)
    """)
