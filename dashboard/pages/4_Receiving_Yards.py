"""
Receiving Yards Props Page
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Receiving Yards | NFL Analytics", page_icon="🎯", layout="wide")

st.title("🎯 Receiving Yards")
st.caption("WR/TE/RB receiving yard props ranked by confidence")
st.markdown("---")

period = st.radio("Period", ["Full Game", "1st Half", "2nd Half"], horizontal=True, key="rec_period")
st.markdown("---")

st.info("**Preseason Mode** — Props populate once FanDuel posts lines.")

st.markdown("### Key Receiving Yards Insights")
st.markdown("""
**Confirmed findings:**
- **Rec yards is FanDuel's SHARPEST market** — hardest to exploit
- **UNDER in outdoor + cold** is confirmed (56.1% hit)
- **Division + outdoor + not early: UNDER 57.4%** hit (+9.5% ROI)
- **Alpha WRs (line >70) trend toward under** — name inflation

**Bankable Players (our strategy works):**
- Jonnu Smith (TE): 66.7% hit, 72 bets, 3/3 seasons ✓
- Pat Freiermuth (TE): 61.9% hit, 63 bets, 3/3 seasons ✓
- James Conner (RB): 62.3% hit, 69 bets
- Jakobi Meyers: 62.2% hit, 45 bets, 3/3 seasons ✓
- Cole Kmet (TE): 58.5% hit, 65 bets
- Nico Collins: 57.7% hit, 52 bets, 3/3 seasons ✓

**Avoid:**
- Chris Olave (31.6%), DJ Moore (39.6%), Keenan Allen (37.3%), Calvin Ridley (39.0%)
- Michael Wilson, Rashid Shaheed, Tank Dell

**Pattern:** TEs are the most exploitable position for receiving yards.
""")
