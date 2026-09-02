"""
NFL Props — Home. Best bets for the week + strategy summary + ask.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DASH = Path(__file__).parent
sys.path.insert(0, str(DASH))
PROC = PROJECT_ROOT / "data" / "processed"

st.set_page_config(page_title="NFL Props", page_icon="🏈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding: 2.5rem 1rem 4rem 1rem; max-width: 100%; }
    h1 { font-size: 1.6rem !important; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

CONF_RANK = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4, "ROLE-CHANGE": 5}


def load_projections():
    path = PROC / "prop_projections_latest.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["crank"] = df["confidence"].map(CONF_RANK).fillna(9)
    return df


st.markdown("## 🏈 NFL Props — This Week")

# Season countdown / week label
season_start = date(2026, 9, 10)
today = date.today()
days = (season_start - today).days
if days > 0:
    st.caption(f"{days} days until Week 1 kickoff (Sep 10). Lines populate as FanDuel posts them.")
else:
    wk = max(1, (today - season_start).days // 7 + 1)
    st.caption(f"Week {wk} — season active.")

# ===== EXECUTIVE SUMMARY =====
st.markdown("### 📋 This Week's Strategy")
st.markdown("""
**Validated edge (tested out-of-sample, 2023-24 → 2025):** Week 1 player props lean **UNDER** —
offenses underperform their talent in openers (rust, new schemes, unsettled rosters).

- **Pass TD unders** are the strongest play (~67% historical hit rate)
- **Rush yard unders** (~60%) and **pass yard unders** (~57%) next
- **Receptions** are a thinner edge (~54%); **receiving yards** is weakest — bet sparingly
- **No validated OVERS** in Week 1 — the rust effect swamps them
- **The under hits harder when the line sits above a player's real baseline** (inflated lines)
- **Avoid injury role-change players** — when a starter is hurt, the backup's line jumps and their
  baseline is stale (that's priced-in volume, not a real under). These are flagged AVOID.

**Game totals:** the only validated totals edge is flat Week 1 unders — no in-season or matchup edge
survived testing, so props are the sharper play. Bet flat, size modestly (2024 openers were a down year).
""")

st.markdown("---")


# ===== BEST BETS =====
st.markdown("### 🔥 Best Bets This Week")
df = load_projections()
if df is None:
    st.info("No props posted yet. Best bets appear once FanDuel releases Week 1 player props.")
else:
    best = df[df["confidence"].isin(["HIGH", "MEDIUM-HIGH"])].sort_values(
        ["crank", "hit_est"], ascending=[True, False], na_position="last")
    if best.empty:
        st.info("No high-confidence plays on the board yet — check back as more lines post.")
    else:
        show = best[["player", "market", "line", "projection", "call", "confidence", "why"]].copy()
        show.columns = ["Player", "Prop", "Line", "Projection", "O/U", "Confidence", "Why"]
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            column_config={
                "Line": st.column_config.NumberColumn(format="%.1f"),
                "Projection": st.column_config.NumberColumn(format="%.1f"),
                "Why": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption(f"{len(best)} high-confidence plays. Full lists by market in the sidebar. "
                   "Bet flat units, keep parlays short.")

st.markdown("---")

# ===== ASK / CHATBOT =====
st.markdown("### 💬 Ask")
query = st.text_input("Ask about a player, matchup, or the strategy",
                      placeholder="e.g., 'Drake Maye', 'why under on pass TDs', 'Mahomes vs Bills'",
                      label_visibility="collapsed")
if query:
    try:
        from dashboard.query_engine import process_query_with_fallback
        with st.spinner("Looking up..."):
            result, source = process_query_with_fallback(query)
        if result:
            st.markdown(result)
            st.caption("📡 StatMuse" if source == "statmuse" else "💾 Local data")
    except Exception as e:
        st.error(f"Lookup unavailable: {e}")

st.markdown("---")
st.caption("Sidebar — Props: Pass Yards · Pass TDs · Rush Yards · Receptions · Receiving Yards · Anytime TD  |  "
           "Games: Game Totals · Spreads · Game Specials")
