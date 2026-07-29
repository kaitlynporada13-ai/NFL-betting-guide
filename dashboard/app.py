"""
NFL Prop Analytics — Main App
"""
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="NFL Props",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide default sidebar nav + streamlit branding for cleaner mobile feel
st.markdown("""
<style>
    /* Push content below Streamlit's header */
    .block-container { padding: 2.5rem 1rem 4rem 1rem; max-width: 100%; }
    
    /* Smaller headers for mobile */
    h1 { font-size: 1.5rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    
    /* Hide streamlit footer */
    footer { visibility: hidden; }
    
    /* Hide the top-right deploy/share buttons */
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    header [data-testid="stDecoration"] { display: none; }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
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
st.markdown("## 🏈 NFL Props")

# Season countdown
season_start = date(2026, 9, 5)  # Approximate Week 1
today = date.today()
days_until = (season_start - today).days

if days_until > 0:
    st.markdown(f"**{days_until} days until Week 1** — Prep mode")
else:
    current_week = max(1, (today - season_start).days // 7 + 1)
    st.markdown(f"**Week {current_week}** — Season active")

st.markdown("---")

# ===== WHAT TO DO RIGHT NOW =====
st.markdown("### What to Focus On")

if days_until > 0:
    st.markdown("""
    **Before the season:**
    - Review the [Weekly Playbook](#weekly-playbook) below for Week 1 strategy
    - Study key player moves (new team = UNDER early)
    - Monitor training camp injuries via the Injuries tab
    - Week 1 is our BIGGEST edge — be ready to bet UNDER across the board
    """)
else:
    st.markdown("""
    **This week:**
    - Check **Best Bets** tab for this week's top plays
    - Review player-specific matchups on prop pages
    - Add any human notes (injury news, narratives) before Sunday
    """)

st.markdown("---")

# ===== ASK FUNCTION =====
st.markdown("### 💬 Quick Lookup")

query = st.text_input(
    "Ask about a player or factor",
    placeholder="e.g., 'A.J. Brown' or 'week 1 strategy' or 'dome games'",
    label_visibility="collapsed",
)

st.caption("Ask anything: 'Mahomes vs Bills' • 'AJ Brown dome games' • 'Kelce man zone' • 'Henry last 5' • 'Hurts without Brown' • 'Cowboys defense' • Any natural language sports question")

if query:
    from dashboard.query_engine import process_query_with_fallback
    with st.spinner("Looking up..."):
        result, source = process_query_with_fallback(query)
    if result:
        st.markdown(result)
        if source == "statmuse":
            st.caption("📡 Answer from StatMuse (1 credit used)")
        else:
            st.caption("💾 Answer from local data")

st.markdown("---")

# ===== WEEKLY PLAYBOOK =====
st.markdown("### 📅 Weekly Playbook")

weeks = {
    1: ("🔴 SLAM UNDER", "All props UNDER. Pass TDs (75.8%), Pass yds (70.6%). Biggest edge of the year."),
    2: ("🟠 Under lean", "Outdoor + not primetime. New team players."),
    3: ("🟠 Under lean", "Same as Week 2. Focus new team players."),
    4: ("🟡 Fading under", "Last week of confirmed early edge."),
    5: ("🟢 Pass TD Over", "Offenses clicking. Dome games especially."),
    6: ("⚪ Neutral", "Boom regression plays. Division = under."),
    7: ("⚪ Neutral", "Player-specific plays only."),
    8: ("🟢 Pass TD Over", "Peak mid-season. Dome Pass TDs."),
    9: ("🟡 Under lean", "Historical under week."),
    10: ("🟡 Under lean", "Under lean continues."),
    11: ("🟡 Under lean", "Monitor injury stacking."),
    12: ("🟢 Slight over", "Thanksgiving week."),
    13: ("🟢 Pass TD OVER", "Playoff push begins. Confirmed 3/3."),
    14: ("🔴 UNDER + Pass TD OVER", "Split: general under BUT Pass TD over."),
    15: ("🟢 Pass TD OVER", "Not cold games only."),
    16: ("🟢 Pass TD OVER", "Target desperate playoff teams."),
    17: ("⚠️ Check inactives", "Pass TD over but starters may rest."),
    18: ("⚠️ Caution", "Many starters rest. Reduced volume."),
}

for wk, (lean, notes) in weeks.items():
    with st.expander(f"Week {wk}: {lean}"):
        st.markdown(notes)

st.markdown("---")

# ===== KEY MOVERS =====
st.markdown("### 🔄 New Team Players (Fade Early)")

preseason = load_preseason()
if preseason:
    for p in preseason.get("new_team_players", [])[:5]:
        name = p.get("player", "")
        team = p.get("new_team", "")
        pos = p.get("position", "")
        st.markdown(f"- **{name}** ({pos}) → {team}")

    with st.expander("See all movers"):
        for p in preseason.get("new_team_players", [])[5:]:
            name = p.get("player", "")
            team = p.get("new_team", "")
            old = p.get("old_team", "")
            st.markdown(f"- **{name}** → {team} (from {old})")

st.markdown("---")
st.caption("☰ Open sidebar from the top-left menu to navigate pages")
st.caption("Pages: Best Bets • Team Props • Pass Yds • Rec Yds • Receptions • Rush Yds • TDs • Tracker • Research • Player Intel")
