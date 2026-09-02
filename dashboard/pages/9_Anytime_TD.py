import streamlit as st
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent.parent / "data" / "processed"
st.set_page_config(page_title="Anytime TD | NFL", page_icon="🏆", layout="wide")
st.title("🏆 Anytime TD")

path = PROC / "anytime_td_picks_latest.parquet"
if not path.exists():
    st.warning("No anytime TD props built yet. Run: `python -m pipeline.anytime_td`")
    st.stop()

df = pd.read_parquet(path)

st.info("**Validated (2023-25, out-of-sample):** In Week 1, players score FEWER TDs than the "
        "market implies (rust) — so betting YES is broadly **-EV**. Longshot TD darts (+200) lose "
        "~14% ROI — fun but a trap. The ONLY +EV angle is **heavy favorites** (priced -200 or "
        "shorter — bellcow RBs, red-zone monsters), which beat their number ~+8%. Bet those, fade the rest.")

greens = df[df["call"] == "YES"]
if len(greens):
    st.success(f"✅ {len(greens)} validated YES play(s) — the near-locks that beat their implied price.")
else:
    st.markdown("**No green-light YES plays** — no heavy favorite clears the projection threshold this week.")

show = df[["player", "price", "projection_pct", "implied_pct", "call", "confidence", "why"]].copy()
show.columns = ["Player", "Price", "Proj TD%", "Implied%", "Call", "Confidence", "Why"]
st.dataframe(show, use_container_width=True, hide_index=True,
             column_config={
                 "Proj TD%": st.column_config.NumberColumn(format="%.0f%%"),
                 "Implied%": st.column_config.NumberColumn(format="%.0f%%"),
                 "Why": st.column_config.TextColumn(width="large"),
             })
st.caption("Proj TD% = 2025 per-game TD rate adjusted for Week 1 rust. Bet YES only where "
           "projection clearly beats the implied price (heavy favorites).")
