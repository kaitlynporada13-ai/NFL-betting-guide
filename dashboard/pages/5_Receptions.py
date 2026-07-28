"""
Receptions Props Page
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Receptions | NFL Analytics", page_icon="🎯", layout="wide")

st.title("🎯 Receptions")
st.caption("Reception props ranked by confidence")
st.markdown("---")

period = st.radio("Period", ["Full Game", "1st Half", "2nd Half"], horizontal=True, key="rec_count_period")
st.markdown("---")

st.info("**Preseason Mode** — Props populate once FanDuel posts lines.")

st.markdown("### Key Receptions Insights")
st.markdown("""
**Confirmed findings:**
- **Receptions UNDER + outdoor + not primetime: 54.7%** (+4.4% ROI) — but only 1/3 seasons
- **High-target WRs (>5.5 rec line) UNDER + outdoor: 71.9%** (small sample but massive)
- **Boom regression on receptions: 54.3%** hit on UNDER after a boom game

**The meta-insight:** FanDuel consistently sets reception lines slightly too high.
The overall under hit rate on receptions is 52.7% — the highest of any market.
This is our "steady drip" market — small edge on many bets.

**Bankable:**
- Quentin Johnston: 58.5% hit, 65 bets
- David Njoku: 57.7% hit, 52 bets

**Avoid:**
- Cade Otton (36.7%), Olamide Zaccheaus (34.1%)
""")
