"""Shared renderer for a single player-prop market page."""
import streamlit as st
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent / "data" / "processed"
CONF_RANK = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4,
             "ROLE-CHANGE": 5, "NO-EDGE": 6}


def load_projections():
    path = PROC / "prop_projections_latest.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["crank"] = df["confidence"].map(CONF_RANK).fillna(9)
    return df


def render_market(market_label: str, icon: str = "🏈"):
    st.set_page_config(page_title=f"{market_label} | NFL Props", page_icon=icon, layout="wide")
    st.title(f"{icon} {market_label}")

    df = load_projections()
    if df is None:
        st.warning("No projections yet. Props may not be posted. "
                   "Run the projection engine when lines are live.")
        st.stop()

    sub = df[df["market"] == market_label].sort_values(["crank", "hit_est"],
                                                       ascending=[True, False], na_position="last")
    if sub.empty:
        st.info(f"No {market_label} props posted yet for this week.")
        st.stop()

    # Week-awareness: past Week 1 there is no validated prop edge.
    if (sub["confidence"] == "NO-EDGE").all():
        st.error("**No validated edge this week.** The only prop edge that survived "
                 "out-of-sample testing is the Week 1 rust under. Weeks 2-18 the prop market "
                 "is efficient, so these projections are informational only — not bets.")

    # Role-change warning
    if "role_change" in sub.columns and sub["role_change"].any():
        rc = sub[sub["role_change"]]
        st.warning("⚠️ Injury role-change — do NOT bet the under (baseline stale): "
                   + ", ".join(rc["player"].tolist()))

    show = sub[["player", "line", "projection", "call", "confidence", "why"]].copy()
    show.columns = ["Player", "Line", "Projection", "O/U", "Confidence", "Why"]
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "Line": st.column_config.NumberColumn(format="%.1f"),
            "Projection": st.column_config.NumberColumn(format="%.1f"),
            "Why": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption("O/U call and confidence are anchored to out-of-sample validated Week 1 rates. "
               "Projection = 2025 per-game baseline adjusted for Week 1 rust.")
