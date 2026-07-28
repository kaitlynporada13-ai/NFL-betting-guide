"""
Player Intel — Bankable players, avoid list, and per-player stats.
Shows which players our strategy consistently profits on vs. loses on.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Player Intel | NFL Analytics", page_icon="👤", layout="wide")

st.title("👤 Player Intel")
st.caption("Bankable players (bet confidently) vs Avoid list (FanDuel too sharp)")
st.markdown("---")

PROC_DIR = PROJECT_ROOT / "data" / "processed"

# Load player data
bankable_path = PROC_DIR / "bankable_players.parquet"
avoid_path = PROC_DIR / "avoid_players.parquet"

# ===== HOW THIS WORKS =====
st.markdown("""
**How to use this page:**
- 🟢 **Bankable Players:** When our strategy fires on these players, SIZE UP. History shows we consistently profit.
- 🔴 **Avoid Players:** When our strategy fires on these players, SKIP or reduce size. FanDuel prices them too well.
- The confidence on each prop page factors this in automatically.
""")

st.markdown("---")

# ===== BANKABLE PLAYERS =====
st.header("🟢 Bankable Players (50 players)")
st.caption("Our strategy is historically profitable on these players. Sorted by ROI.")

if bankable_path.exists():
    bankable = pd.read_parquet(bankable_path)
    bankable = bankable.sort_values("roi", ascending=False)
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        market_filter = st.multiselect(
            "Filter by market",
            bankable["markets"].unique().tolist(),
            default=[],
            key="bank_mkt"
        )
    with col2:
        min_bets = st.slider("Minimum bets", 30, 100, 30, key="bank_bets")
    
    filtered = bankable[bankable["total_bets"] >= min_bets]
    if market_filter:
        filtered = filtered[filtered["markets"].isin(market_filter)]
    
    st.dataframe(
        filtered.rename(columns={
            "player_clean": "Player",
            "total_bets": "Bets",
            "hit_rate": "Hit Rate",
            "roi": "ROI %",
            "markets": "Primary Market",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "ROI %": st.column_config.NumberColumn(format="+%.1f%%"),
        },
    )
    
    # Highlight multi-season consistent players
    st.markdown("---")
    st.subheader("⭐ Multi-Season Consistent (2+ profitable seasons)")
    st.markdown("*These players are profitable REPEATEDLY — not just one hot streak.*")
    
    multi_season = [
        {"Player": "Jonnu Smith", "Seasons Profitable": "3/3", "Hit Rate": "65.6%", "Market": "Receiving Yards"},
        {"Player": "Pat Freiermuth", "Seasons Profitable": "3/3", "Hit Rate": "62.6%", "Market": "Receiving Yards"},
        {"Player": "Jakobi Meyers", "Seasons Profitable": "3/3", "Hit Rate": "61.9%", "Market": "Receiving Yards"},
        {"Player": "Sam LaPorta", "Seasons Profitable": "3/3", "Hit Rate": "58.4%", "Market": "Receiving Yards"},
        {"Player": "Nico Collins", "Seasons Profitable": "3/3", "Hit Rate": "57.6%", "Market": "Receiving Yards"},
        {"Player": "Baker Mayfield", "Seasons Profitable": "3 played/2 prof", "Hit Rate": "59.2%", "Market": "Rushing Yards"},
        {"Player": "Kyler Murray", "Seasons Profitable": "2/2", "Hit Rate": "58.0%", "Market": "Mixed"},
        {"Player": "James Conner", "Seasons Profitable": "2/2", "Hit Rate": "62.5%", "Market": "Receiving Yards"},
        {"Player": "Jerome Ford", "Seasons Profitable": "2/2", "Hit Rate": "61.7%", "Market": "Mixed"},
        {"Player": "Cole Kmet", "Seasons Profitable": "3 played/2 prof", "Hit Rate": "58.1%", "Market": "Receiving Yards"},
        {"Player": "Quentin Johnston", "Seasons Profitable": "3 played/2 prof", "Hit Rate": "58.1%", "Market": "Receptions"},
        {"Player": "Stefon Diggs", "Seasons Profitable": "3 played/2 prof", "Hit Rate": "58.3%", "Market": "Mixed"},
    ]
    
    st.dataframe(pd.DataFrame(multi_season), use_container_width=True, hide_index=True)

else:
    st.warning("Bankable players data not found. Run research scripts first.")

st.markdown("---")

# ===== AVOID LIST =====
st.header("🔴 Avoid List (67 players)")
st.caption("FanDuel is too accurate on these players. Skip or reduce size.")

if avoid_path.exists():
    avoid = pd.read_parquet(avoid_path)
    avoid = avoid.sort_values("roi")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        market_filter_a = st.multiselect(
            "Filter by market",
            avoid["markets"].unique().tolist(),
            default=[],
            key="avoid_mkt"
        )
    with col2:
        max_roi = st.slider("Show ROI worse than", -50, 0, -10, key="avoid_roi")
    
    filtered_a = avoid[avoid["roi"] <= max_roi]
    if market_filter_a:
        filtered_a = filtered_a[filtered_a["markets"].isin(market_filter_a)]
    
    st.dataframe(
        filtered_a.rename(columns={
            "player_clean": "Player",
            "total_bets": "Bets",
            "hit_rate": "Hit Rate",
            "roi": "ROI %",
            "markets": "Primary Market",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "ROI %": st.column_config.NumberColumn(format="+%.1f%%"),
        },
    )
    
    st.markdown("---")
    st.subheader("🚫 Worst of the Worst (Never Bet These)")
    st.markdown("""
    | Player | Hit Rate | ROI | Why |
    |--------|:---:|:---:|---|
    | Malik Washington | 29.0% | -44.6% | FanDuel has perfect read |
    | Chris Olave | 31.6% | -39.7% | Market knows his floor/ceiling |
    | Olamide Zaccheaus | 34.1% | -34.9% | Role player priced correctly |
    | Tre Tucker | 35.0% | -33.2% | Low volume, sharp pricing |
    | Derek Carr | 35.5% | -32.3% | QB with predictable ceiling |
    | Cade Otton | 36.7% | -30.0% | TE priced perfectly |
    | Keenan Allen | 37.3% | -28.9% | Veteran, well-understood |
    | DJ Moore | 39.6% | -24.4% | Now in BUF — still avoid |
    | Isaiah Likely | 39.2% | -25.1% | Despite "breakout" hype |
    """)

else:
    st.warning("Avoid list not found. Run research scripts first.")

st.markdown("---")

# ===== POSITION BREAKDOWN =====
st.header("📊 Position Exploitability")
st.markdown("""
| Position | % Profitable | Best Strategy | Notes |
|:---:|:---:|---|---|
| **TE** | **53%** | Mean reversion on receiving yards | Most exploitable position |
| RB | 43% | Rush UNDER after boom, cold game OVER | Context-dependent |
| QB | 38% | Pass TDs in specific situations | Market-by-market |
| WR | 34% | UNDER in outdoor/division/cold | Hardest position — FanDuel sharpest here |
""")

st.markdown("---")
st.caption("Player data based on 63,441 graded props (2023-2025) | 191 high-volume players analyzed")
