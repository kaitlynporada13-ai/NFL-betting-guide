import streamlit as st
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent.parent / "data" / "processed"
st.set_page_config(page_title="Spreads | NFL", page_icon="📐", layout="wide")
st.title("📐 Spreads")

path = PROC / "spreads_picks_latest.parquet"
if not path.exists():
    st.warning("No spreads posted yet. Run: `python -m pipeline.game_projections`")
    st.stop()

df = pd.read_parquet(path)
rank = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4}
df["r"] = df["confidence"].map(rank).fillna(9)
df = df.sort_values("r")

st.info("**Validated edge:** The only ATS edge that survived testing is home underdog vs a road "
        "favorite of 7+ points (~60-67% out-of-sample). Every other game is PASS — the spread "
        "market is otherwise efficient.")

quals = df[df["confidence"] != "PASS"]
if quals.empty:
    st.markdown("**No qualifying spread plays this week** — no road team is favored by 7+.")

show = df[["matchup", "line", "call", "confidence", "why"]].copy()
show.columns = ["Matchup", "Spread (home)", "Call", "Confidence", "Why"]
st.dataframe(show, use_container_width=True, hide_index=True,
             column_config={"Why": st.column_config.TextColumn(width="large")})
