"""
Season Props — FanDuel 2026-27 Regular Season Totals Analysis
Shows best bets, leans, and unders for season-long player prop futures.
"""
import streamlit as st
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
NOTES_DIR = PROJECT_ROOT / "data" / "human_notes"

st.set_page_config(page_title="Season Props | NFL Analytics", page_icon="📊", layout="wide")

st.title("📊 Season Props 2026-27")
st.caption("FanDuel Regular Season Totals — Analyzed 2026-08-07")
st.markdown("---")

# Load data
props_path = NOTES_DIR / "season_props_2026.yaml"
if not props_path.exists():
    st.error("Season props file not found. Expected at data/human_notes/season_props_2026.yaml")
    st.stop()

with open(props_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

best_bets = data.get("best_bets", [])
strong_leans = data.get("strong_leans", [])
under_plays = data.get("under_plays", [])
skips = data.get("skips", [])
notes = data.get("notes", [])


# ===== BEST BETS =====
st.markdown("### 🔥 Best Bets (Highest Conviction)")
st.markdown("These are the strongest edges on the board. Bet with confidence.")
st.markdown("")

for bet in best_bets:
    direction_emoji = "🟢" if bet["direction"] == "OVER" else "🔴"
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"**{direction_emoji} {bet['direction']}** — "
            f"**{bet['player']}** {bet['market'].replace('_', ' ').title()} "
            f"**{bet['line']}**"
        )
        st.caption(f"Edge: {bet.get('edge_estimate', '?')} | {bet.get('historical', '')}")
    with col2:
        st.markdown(f"**{bet['confidence']}**")

    with st.expander(f"Full reasoning — {bet['player']}"):
        st.markdown(bet["reasoning"])
    st.markdown("")

st.markdown("---")

# ===== STRONG LEANS =====
st.markdown("### 📈 Strong Leans (Good Value)")
st.markdown("High-confidence plays with solid reasoning. Consider sizing at 1-2 units.")
st.markdown("")

for bet in strong_leans:
    direction_emoji = "🟢" if bet["direction"] == "OVER" else "🔴"
    price_note = ""
    if bet.get("over_price"):
        price_note = f" (Over {bet['over_price']}/Under {bet['under_price']})"

    st.markdown(
        f"**{direction_emoji} {bet['direction']}** — "
        f"**{bet['player']}** {bet['market'].replace('_', ' ').title()} "
        f"**{bet['line']}**{price_note}"
    )
    st.caption(f"Edge: {bet.get('edge_estimate', '?')} | {bet.get('historical', '')}")

    with st.expander(f"Reasoning — {bet['player']}"):
        st.markdown(bet["reasoning"])
    st.markdown("")

st.markdown("---")

# ===== UNDER PLAYS =====
st.markdown("### 🔻 Under Plays")
st.markdown("Lines set too high. UNDER benefits from any missed games (built-in injury edge).")
st.markdown("")

for bet in under_plays:
    conf = bet.get("confidence", "LEAN")
    st.markdown(
        f"**🔴 UNDER** — **{bet['player']}** "
        f"{bet['market'].replace('_', ' ').title()} **{bet['line']}** "
        f"({conf})"
    )
    st.caption(f"Edge: {bet.get('edge_estimate', '?')} | {bet.get('historical', '')}")

    with st.expander(f"Reasoning — {bet['player']}"):
        st.markdown(bet["reasoning"])
    st.markdown("")

st.markdown("---")

# ===== SKIPS =====
st.markdown("### ⚪ Skips (Fair Lines / Too Uncertain)")
st.markdown("These lines are well-priced by FanDuel or have too many unknowns.")
st.markdown("")

for s in skips:
    st.markdown(f"- **{s['player']}** {s['market'].replace('_', ' ').title()} {s['line']} — _{s['reason']}_")

st.markdown("---")

# ===== NOTES =====
st.markdown("### 📝 Key Notes")
# Only show betting-relevant notes (first 7), not backend roster verification data
betting_notes = [n for n in notes if not n.startswith("KEY 2026") and not n.startswith("2026 ROOKIES") and not n.startswith("YEAR-2")]
for note in betting_notes:
    st.markdown(f"- {note}")

st.markdown("---")
st.caption("Analysis date: 2026-08-07 | Source: FanDuel NJ | All lines -114/-114 unless noted")
