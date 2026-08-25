import streamlit as st
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent.parent / "data" / "processed"
st.set_page_config(page_title="Game Totals | NFL", page_icon="📊", layout="wide")
st.title("📊 Game Totals")

path = PROC / "totals_picks_latest.parquet"
if not path.exists():
    st.warning("No totals posted yet. Run: `python -m pipeline.game_projections`")
    st.stop()

df = pd.read_parquet(path)
rank = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4}
df["r"] = df["confidence"].map(rank).fillna(9)
df = df.sort_values("r")

st.info("**Validated edge:** Only Week 1 game unders survived out-of-sample testing (~59%). "
        "No in-season or matchup edge held up, so Weeks 2+ are PASS. Bet flat, size modestly.")

show = df[["matchup", "line", "call", "confidence", "why"]].copy()
show.columns = ["Matchup", "Total", "Call", "Confidence", "Why"]
st.dataframe(show, use_container_width=True, hide_index=True,
             column_config={"Total": st.column_config.NumberColumn(format="%.1f"),
                            "Why": st.column_config.TextColumn(width="large")})
