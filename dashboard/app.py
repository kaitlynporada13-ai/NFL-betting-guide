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
    /* Hide sidebar toggle */
    [data-testid="collapsedControl"] { display: none; }
    
    /* Tighter padding for mobile */
    .block-container { padding: 0.5rem 1rem 4rem 1rem; max-width: 100%; }
    
    /* Nav bar styling */
    .nav-bar {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: #f8f9fa;
        border-bottom: 1px solid #e0e0e0;
        padding: 8px 0;
        margin: -0.5rem -1rem 1rem -1rem;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .nav-item {
        text-align: center;
        font-size: 0.7rem;
        color: #555;
        text-decoration: none;
        padding: 4px 8px;
    }
    .nav-item.active {
        color: #1f77b4;
        font-weight: bold;
    }
    
    /* Smaller headers for mobile */
    h1 { font-size: 1.5rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    
    /* Hide streamlit footer */
    footer { visibility: hidden; }
    
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


# ===== NAVIGATION =====
st.markdown("""
<div class="nav-bar">
    <span class="nav-item active">🏠 Home</span>
    <span class="nav-item">💰 Bets</span>
    <span class="nav-item">🏟️ Teams</span>
    <span class="nav-item">🎯 Props</span>
    <span class="nav-item">📊 Track</span>
    <span class="nav-item">🔬 Intel</span>
</div>
""", unsafe_allow_html=True)

st.caption("Use the sidebar (swipe right) to navigate between pages")

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

if query:
    query_lower = query.lower()
    
    # Search bankable/avoid lists
    bankable_path = PROC_DIR / "bankable_players.parquet"
    avoid_path = PROC_DIR / "avoid_players.parquet"
    
    found = False
    
    if bankable_path.exists():
        bankable = pd.read_parquet(bankable_path)
        match = bankable[bankable["player_clean"].str.contains(query_lower, na=False)]
        if not match.empty:
            row = match.iloc[0]
            st.success(f"✅ **{row['player_clean'].title()}** is on our BANKABLE list")
            st.markdown(f"- Hit rate: **{row['hit_rate']:.1%}** | ROI: **{row['roi']:+.1f}%** | Bets: {row['total_bets']}")
            st.markdown(f"- Primary market: {row['markets']}")
            st.markdown("- **Action:** Size up when our strategy triggers on this player")
            found = True
    
    if not found and avoid_path.exists():
        avoid = pd.read_parquet(avoid_path)
        match = avoid[avoid["player_clean"].str.contains(query_lower, na=False)]
        if not match.empty:
            row = match.iloc[0]
            st.error(f"🚫 **{row['player_clean'].title()}** is on our AVOID list")
            st.markdown(f"- Hit rate: **{row['hit_rate']:.1%}** | ROI: **{row['roi']:+.1f}%** | Bets: {row['total_bets']}")
            st.markdown("- **Action:** Skip or reduce size. FanDuel prices them accurately.")
            found = True
    
    # Check preseason notes
    if not found:
        preseason = load_preseason()
        if preseason:
            for player in preseason.get("new_team_players", []):
                if query_lower in player.get("player", "").lower():
                    st.info(f"🔄 **{player['player']}** — New team player ({player['old_team']} → {player['new_team']})")
                    st.markdown(player.get("notes", ""))
                    st.markdown("**Our finding:** New team + weeks 1-4 = UNDER (55.2% hit)")
                    found = True
                    break
    
    # Strategy lookups
    if not found:
        strategies = {
            "week 1": "🔴 **Week 1 Strategy:** SLAM UNDER on everything. Pass TDs UNDER hits 75.8%. Pass yards UNDER hits 70.6%. This is our single biggest edge of the entire year. Confirmed 3/3 seasons.",
            "dome": "🏟️ **Dome Games:** Pass TDs OVER at plus money (+150+) is +EV ($25 per $100 bet). FanDuel underestimates indoor TD rates. Hit rate: 45.9% vs 36.9% implied.",
            "cold": "🥶 **Cold Games:** Rush OVER (54.5% hit). Cold suppresses passing → game scripts go run-heavy. Also: Rec UNDER + outdoor + cold hits 56.1%.",
            "wind": "💨 **Windy Games:** Pass props UNDER (55.7% hit). Wind >=15mph suppresses passing. Confirmed 2/3 seasons.",
            "monday": "🌙 **Monday Night:** UNDER lean (53.5% hit, 3/3 seasons). MNF historically lower-scoring. Small but rock-steady edge.",
            "division": "🏈 **Division Games:** General UNDER lean. Familiarity = tighter games. Rec UNDER + division + outdoor hits 57.4%.",
            "injury": "🏥 **High Injury Games:** When both teams have 4+ players out + outdoor: UNDER hits 63.2% (+20.6% ROI). Massive edge.",
            "new team": "🔄 **New Team Players:** UNDER weeks 1-4 hits 55.2%. No chemistry yet. Confirmed finding.",
            "boom": "📈 **After Boom Game:** Rush UNDER hits 54.8%, Receptions UNDER hits 54.3%. FanDuel raises the line after a big game but regression is more powerful.",
        }
        
        for keyword, response in strategies.items():
            if keyword in query_lower:
                st.markdown(response)
                found = True
                break
    
    if not found:
        st.caption(f"No specific data found for '{query}'. Try a player name or keyword like 'dome', 'cold', 'week 1', 'injury'.")

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
st.caption("Swipe right for full navigation • Built on 63K backtested FanDuel props")
