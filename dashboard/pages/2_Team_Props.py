"""
Team Props Page — Spreads, Totals, Moneylines
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Team Props | NFL Analytics", page_icon="🏟️", layout="wide")

st.title("🏟️ Team Props")
st.caption("Spreads, Totals, Moneylines — ranked by confidence")
st.markdown("---")

# Period selector
period = st.radio("Period", ["Full Game", "1st Half", "1st Quarter", "2nd Half", "3rd Quarter"], horizontal=True)

st.info("**Preseason Mode** — Team prop picks populate when games are scheduled and lines are posted.")

st.markdown("### How Team Props Work")
st.markdown("""
**Spreads:** Our model predicts the margin. If model says Home -4.5 but FanDuel says Home -2.5, we have edge on Home.

**Totals:** Model predicts combined points. If model says 47 and FanDuel posts 44.5, we bet Over.

**Quarter/Half breakdowns** use our PBP data:
- Teams that score 60%+ of points in 1st half → 1st Half Over
- Teams that allow most points in 3rd quarter → opponent Q3 props

Confirmed edges for team props:
- **Week 1 game totals UNDER** (part of our Week 1 slam)
- **Monday Night UNDER** (confirmed 3/3 seasons)
- **Division games UNDER** (tighter, lower scoring — 2/3 seasons)
""")
