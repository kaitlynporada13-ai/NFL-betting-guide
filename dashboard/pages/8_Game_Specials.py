import streamlit as st
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent.parent / "data" / "processed"
st.set_page_config(page_title="Game Specials | NFL", page_icon="🎁", layout="wide")
st.title("🎁 Game Specials")

path = PROC / "specials_picks_latest.parquet"
if not path.exists():
    st.warning("No specials built yet. Run: `python -m pipeline.specials_projections`")
    st.stop()

df = pd.read_parquet(path)

st.info("**How to read this:** FanDuel's bespoke specials/boosts aren't in our odds feed, and we "
        "have no *validated* edge on them. These are historical **base-rate probabilities** for common "
        "special structures. Compare our probability to FanDuel's offered odds — bet only if the "
        "payout beats the implied odds. 'Confidence' = how likely the event is, not a proven edge.")

show = df[["special", "scope", "probability", "confidence", "why"]].copy()
show.columns = ["Special", "Scope", "Probability %", "Likelihood", "Why"]
st.dataframe(show, use_container_width=True, hide_index=True,
             column_config={"Probability %": st.column_config.NumberColumn(format="%.1f%%"),
                            "Why": st.column_config.TextColumn(width="large")})

st.caption("To assess a specific FanDuel special not listed here, share it and I'll compute its "
           "base-rate probability from historical data.")
