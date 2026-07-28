"""
Passing Yards Props Page
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Passing Yards | NFL Analytics", page_icon="🎯", layout="wide")

st.title("🎯 Passing Yards")
st.caption("QB passing yard props ranked by confidence score")
st.markdown("---")

period = st.radio("Period", ["Full Game", "1st Half", "2nd Half"], horizontal=True, key="pass_period")

st.markdown("---")

st.info("**Preseason Mode** — Props populate once FanDuel posts lines.")

# Column structure for in-season
st.markdown("### Column Format (In-Season)")
st.markdown("""
| Player | Team | Opp | Line | Projection | Direction | Odds | EV/$100 | Confidence |
|--------|------|-----|------|-----------|-----------|------|---------|-----------|
| Jalen Hurts | PHI | vs DAL | 268.5 | 241.3 | UNDER | -110 | +$8.20 | HIGH |
""")

st.markdown("### Key Passing Yards Insights")
st.markdown("""
**Confirmed findings:**
- **Week 1 Pass Yards UNDER: 70.6% hit rate** (+34.8% ROI, 3/3 seasons)
- Star QBs (line >260) slightly more exploitable on OVER (53.1%)
- Dome games boost passing but FanDuel already prices this in
- **Windy games (15+ mph): Pass UNDER** confirmed edge (55.7%, 2/3 seasons)

**Bankable QBs (our strategy works well on):**
- Baker Mayfield: 59.4% hit, 101 bets
- Gardner Minshew: 59.5% hit, 42 bets
- Kyler Murray: 58.0% hit, 38 bets

**Avoid (FanDuel too sharp):**
- Derek Carr, Mac Jones
""")
