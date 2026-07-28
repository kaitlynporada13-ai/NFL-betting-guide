"""
NFL Betting Analytics Dashboard — Main App (Home/Weekly page)
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="NFL Betting Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding: 1rem 1.5rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

import yaml
import pandas as pd
from datetime import datetime

DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "human_notes"
RESEARCH_DIR = PROJECT_ROOT / "research"
PROC_DIR = DATA_DIR / "processed"

def load_preseason():
    path = NOTES_DIR / "preseason_2026_intel.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return None

# ===== HEADER =====
st.title("🏈 NFL Betting Analytics")
st.caption("2026 Season | Strategy-driven prop betting")
st.markdown("---")

# ===== CURRENT WEEK SUMMARY =====
st.header("📋 This Week")

# Preseason mode
preseason = load_preseason()
st.info(
    "**Preseason Mode** — The 2026 season hasn't started yet. "
    "Showing pre-season analysis and Week 1 strategy preparation."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Week 1 ROI", "+21.7%", delta="UNDER everything")
with col2:
    st.metric("Pass TDs Under", "75.8% hit", delta="+44.6% ROI")
with col3:
    st.metric("Confirmed Edges", "6 strategies")
with col4:
    st.metric("Projected Season", "~46 units")

st.markdown("---")

# ===== PRESEASON ANALYSIS =====
st.header("🔍 Preseason Analysis")

st.markdown("""
**Key Findings for 2026:**

1. **Week 1 is our biggest edge of the year.** Slam unders across the board — especially passing TDs (75.8% hit) and passing yards (70.6% hit). This is confirmed across all 3 backtested seasons.

2. **Unusually high roster turnover this offseason** — A.J. Brown, DJ Moore, Jaylen Waddle, Mike Evans, Kyler Murray all on new teams. Our "new team + early season = UNDER" finding (55.2% hit) should be even stronger in 2026.

3. **Late-season Pass TD OVERs** are our second-strongest confirmed edge (+10.9% ROI, strengthening each year). Mark weeks 13-18 for aggressive Pass TD over bets in non-cold games.

4. **FanDuel systematically overshoots** when they move lines away from player averages. They predict the right direction but overestimate the magnitude. This is the structural market inefficiency we exploit.
""")

st.markdown("---")

# ===== WEEKLY OUTLOOK (Expandable) =====
st.header("📅 Weekly Playbook")

weeks = {
    1: {"lean": "🔴 SLAM UNDER", "strat": "All props UNDER. Pass TDs especially (75.8%). This is our biggest single-week edge.", "confidence": "VERY HIGH"},
    2: {"lean": "🟠 Under lean", "strat": "Continue early-season under strategy. Target outdoor, non-primetime games. New team players.", "confidence": "HIGH"},
    3: {"lean": "🟠 Under lean", "strat": "Same as Week 2. Focus on players who haven't established rhythm yet.", "confidence": "HIGH"},
    4: {"lean": "🟡 Under lean (fading)", "strat": "Last week of confirmed early-season edge. Be more selective.", "confidence": "MEDIUM"},
    5: {"lean": "🟢 Pass TD OVER begins", "strat": "Offenses clicking. Target dome games for pass TD overs. Weeks 5-8 historically over-lean.", "confidence": "MEDIUM"},
    6: {"lean": "⚪ Neutral", "strat": "Monitor boom regression opportunities. Division games = under lean.", "confidence": "LOW"},
    7: {"lean": "⚪ Neutral", "strat": "Standard mean reversion strategy. Player-specific plays only.", "confidence": "LOW"},
    8: {"lean": "🟢 Pass TD OVER", "strat": "Peak mid-season. Dome + Pass TDs historically strongest.", "confidence": "MEDIUM"},
    9: {"lean": "🟡 Under lean", "strat": "Historical under week. Selective.", "confidence": "LOW-MED"},
    10: {"lean": "🟡 Under lean", "strat": "Under lean continues through mid-season.", "confidence": "LOW-MED"},
    11: {"lean": "🟡 Under lean", "strat": "Under lean. Monitor for injury stacking opportunities.", "confidence": "LOW-MED"},
    12: {"lean": "🟢 Slight over", "strat": "Thanksgiving week. Historically slightly over-leaning.", "confidence": "LOW"},
    13: {"lean": "🟢 Pass TD OVER begins (late push)", "strat": "Playoff push = aggressive play-calling. Confirmed 3/3 seasons.", "confidence": "HIGH"},
    14: {"lean": "🔴 UNDER + 🟢 Pass TD OVER", "strat": "Split strategy: General unders BUT Pass TD overs (not cold). Historically strongest under week.", "confidence": "HIGH"},
    15: {"lean": "🟢 Pass TD OVER", "strat": "Late season aggression continues. Non-cold games only.", "confidence": "HIGH"},
    16: {"lean": "🟢 Pass TD OVER", "strat": "Same. Target desperate teams fighting for playoffs.", "confidence": "HIGH"},
    17: {"lean": "🟢 Pass TD OVER + ⚠️ Rest risk", "strat": "Watch for teams resting starters. Check inactive reports.", "confidence": "MEDIUM"},
    18: {"lean": "⚠️ Caution", "strat": "Many starters rest. Reduced volume. Only bet if lineup confirmed.", "confidence": "LOW"},
}

for wk, info in weeks.items():
    with st.expander(f"Week {wk}: {info['lean']}"):
        st.markdown(f"**Strategy:** {info['strat']}")
        st.markdown(f"**Confidence:** {info['confidence']}")

st.markdown("---")

# ===== KEY MOVERS =====
st.header("🔄 2026 Key Player Moves")
st.caption("New team = UNDER early season (55.2% confirmed)")

if preseason:
    players = preseason.get("new_team_players", [])
    for p in players:
        name = p.get("player", "")
        new = p.get("new_team", "")
        old = p.get("old_team", "")
        pos = p.get("position", "")
        notes = p.get("notes", "").strip()
        with st.expander(f"**{name}** ({pos}) → {new} *(from {old})*"):
            st.markdown(notes)

# ===== INJURY/NEWS =====
st.markdown("---")
st.header("🏥 News & Injuries")
st.caption("Pull latest during the season. Currently in preseason mode.")
st.info("Injury reports will populate once training camps produce reports. Check back closer to Week 1.")

st.markdown("---")
st.caption("Built on 63,441 backtested FanDuel props (2023-2025) | 6 confirmed profitable strategies")
