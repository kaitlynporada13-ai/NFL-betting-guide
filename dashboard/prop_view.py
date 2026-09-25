"""Shared renderer for a single player-prop market page."""
import streamlit as st
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent / "data" / "processed"
CONF_RANK = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4,
             "ROLE-CHANGE": 5, "NO-EDGE": 6}

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _abbr(team: str) -> str:
    return TEAM_ABBR.get(team, team)


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

    # --- Game filter pills ---
    if "home_team" in sub.columns and "away_team" in sub.columns:
        sub = sub.copy()
        sub["matchup"] = sub["away_team"].map(_abbr) + " @ " + sub["home_team"].map(_abbr)
        matchups = sorted(sub["matchup"].dropna().unique())
        if matchups:
            selected = st.pills(
                "Filter by game",
                matchups,
                selection_mode="multi",
                key=f"game_pills_{market_label}",
            )
            if selected:
                sub = sub[sub["matchup"].isin(selected)]
            if sub.empty:
                st.info("No props for the selected game(s).")
                st.stop()

    # If the whole market is low-confidence, say so plainly (efficient market).
    playable = sub[sub["confidence"].isin(["HIGH", "MEDIUM-HIGH", "MEDIUM"])]
    if playable.empty:
        st.info("**No green-light plays in this market this week.** The model projects every "
                "player below, but the line is efficient here (validated out-of-sample), so these "
                "are informational — not bets. The projection vs line and reasoning are shown for context.")
    else:
        st.success(f"**{len(playable)} playable spot(s)** where the model's edge validated "
                   "out-of-sample. Higher confidence = bigger validated hit rate at that projection gap.")

    # Role-change warning
    if "role_change" in sub.columns and sub["role_change"].any():
        rc = sub[sub["role_change"]]
        st.warning("⚠️ Injury role-change — recent form understates the new role, so the "
                   "projection isn't reliable. No play: " + ", ".join(rc["player"].tolist()))

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
    st.caption("Projection = blended model (recent form, opponent, pace, role, context). "
               "O/U call follows the projection vs the line; confidence = the model's validated "
               "out-of-sample hit rate for this market at this projection-vs-line gap.")
