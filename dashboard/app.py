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

CONF_RANK = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4,
             "ROLE-CHANGE": 5, "NO-EDGE": 6}


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
    nfl_week = 1
    st.caption(f"{days} days until Week 1 kickoff (Sep 10). Lines populate as FanDuel posts them.")
else:
    nfl_week = min(18, (today - season_start).days // 7 + 1)
    st.caption(f"Week {nfl_week} — season active.")

# ===== EXECUTIVE SUMMARY =====
st.markdown("### 📋 This Week's Strategy")
st.markdown("""
**How the picks are made (every week, every prop):** a blended model projects each player's
number from recent form, opponent, pace, role/target trend, and game context — then compares that
projection to the line. Projection above the line is an **OVER**, below is an **UNDER**. No bias
toward either side; the data picks.

**Confidence is earned, not assumed.** Each market was validated out-of-sample (train 2023-24 →
test 2025). Confidence reflects how often the model's side actually won at that projection-vs-line gap:

- **Pass TDs — the strongest edge (~60-63% out-of-sample), and it holds all season.** The model
  reliably separates which QBs beat their TD line and which don't, on both overs and unders.
- **Receptions — a thinner edge (~53-55%)**, best when the projection sits close to the line.
- **Pass / Rush / Receiving yards — mostly efficient.** The model still projects them, but the
  line is sharp, so these are flagged low-confidence / no-play unless a spot clears break-even.
- **Injury role-change players are flagged AVOID** — recent form understates a promoted backup's
  new role, so the projection isn't trustworthy.

**Best bets below are the HIGH / MEDIUM-HIGH plays** — where the validated hit rate earns it.
Everything else is on its market page, ranked, so you can see the full board.
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
        if nfl_week != 1:
            st.info(f"No best bets in Week {nfl_week} — no validated prop edge exists past Week 1 "
                    "(see strategy note above). Sitting out props is the correct call this week.")
        else:
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
