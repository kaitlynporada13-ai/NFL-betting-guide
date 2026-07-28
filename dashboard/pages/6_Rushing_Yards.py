"""
Rushing Yards Props Page
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Rushing Yards | NFL Analytics", page_icon="🎯", layout="wide")

st.title("🎯 Rushing Yards")
st.caption("RB/QB rushing yard props ranked by confidence")
st.markdown("---")

period = st.radio("Period", ["Full Game", "1st Half", "2nd Half"], horizontal=True, key="rush_period")
st.markdown("---")

st.info("**Preseason Mode** — Props populate once FanDuel posts lines.")

st.markdown("### Key Rushing Yards Insights")
st.markdown("""
**Confirmed findings:**
- **Cold games + Rush OVER: 54.5%** hit (+4.1% ROI, 2/3 seasons)
- **Big underdogs + Rush UNDER: 57.8%** hit (+10.3% ROI) — they abandon the run
- **Boom regression on rushing: 54.8%** UNDER after a big rushing game
- **Bellcow RBs (line >70) lean UNDER**: 54.7% hit

**Game script is KEY for rushing:**
- Big favorite expected → might run MORE (clock management)
- Big underdog expected → will run LESS (playing from behind)
- But: FanDuel usually prices expected game script correctly
- Edge only exists in EXTREME spreads (7+)

**Bankable:**
- Zack Moss: 62.5% hit, 32 bets
- James Conner: 62.3% hit, 69 bets
- J.K. Dobbins: 59.0% hit, 39 bets
- Chase Brown: 58.4% hit, 77 bets
- Baker Mayfield (rush): 59.4% hit, 101 bets (most profitable!)

**Avoid:**
- Devin Singletary (39.6%), Justice Hill (39.7%)
""")
