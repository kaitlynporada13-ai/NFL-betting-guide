"""
Bet Tracker Page — Track all bets, results, and P&L
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Bet Tracker | NFL Analytics", page_icon="📊", layout="wide")

st.title("📊 Bet Tracker")
st.caption("Track all bets, results, and season P&L")
st.markdown("---")

TRACKER_FILE = PROJECT_ROOT / "data" / "tracker" / "bets.json"
TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_bets():
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return []


def save_bets(bets):
    with open(TRACKER_FILE, "w") as f:
        json.dump(bets, f, indent=2, default=str)


bets = load_bets()

# ===== SUMMARY METRICS =====
if bets:
    df = pd.DataFrame(bets)
    
    total_bets = len(df)
    resolved = df[df["result"].isin(["won", "lost", "push"])]
    wins = len(resolved[resolved["result"] == "won"])
    losses = len(resolved[resolved["result"] == "lost"])
    pushes = len(resolved[resolved["result"] == "push"])
    pending = len(df[df["result"] == "pending"])
    
    # P&L calculation
    total_risked = 0
    total_profit = 0
    for _, bet in resolved.iterrows():
        units = bet.get("units", 1)
        odds = bet.get("odds", -110)
        if bet["result"] == "won":
            if odds > 0:
                total_profit += units * (odds / 100)
            else:
                total_profit += units * (100 / abs(odds))
        elif bet["result"] == "lost":
            total_profit -= units
        total_risked += units
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Bets", total_bets)
    with col2:
        hit_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        st.metric("Record", f"{wins}-{losses}-{pushes}", delta=f"{hit_rate:.1%}")
    with col3:
        st.metric("Pending", pending)
    with col4:
        st.metric("Net Units", f"{total_profit:+.1f}")
    with col5:
        roi = (total_profit / total_risked * 100) if total_risked > 0 else 0
        st.metric("ROI", f"{roi:+.1f}%")
    
    st.markdown("---")
    
    # Bet history table
    st.markdown("### Bet History")
    display_df = df[["date", "player", "market", "direction", "line", "odds", "units", "result", "strategy"]].copy()
    display_df = display_df.sort_values("date", ascending=False)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.markdown("### No bets tracked yet")
    st.caption("Add your first bet below to start tracking.")

st.markdown("---")

# ===== ADD NEW BET =====
st.header("➕ Add Bet")

with st.form("add_bet_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        date = st.date_input("Date", datetime.now())
        player = st.text_input("Player / Team", placeholder="e.g., Patrick Mahomes")
        market = st.selectbox("Market", [
            "Passing Yards", "Passing TDs", "Rushing Yards",
            "Receiving Yards", "Receptions", "Anytime TD",
            "Spread", "Total (O/U)", "Moneyline", "Other"
        ])
        direction = st.selectbox("Direction", ["Over", "Under", "Home", "Away", "Other"])
    
    with col2:
        line = st.number_input("Line", value=0.0, step=0.5)
        odds = st.number_input("American Odds", value=-110, step=5)
        units = st.number_input("Units", value=1.0, min_value=0.5, max_value=5.0, step=0.5)
        strategy = st.text_input("Strategy (optional)", placeholder="e.g., Week 1 UNDER")
        result = st.selectbox("Result", ["pending", "won", "lost", "push"])
    
    notes = st.text_area("Notes (optional)", placeholder="Why you made this bet...")
    
    submitted = st.form_submit_button("Add Bet")
    
    if submitted and player:
        new_bet = {
            "date": str(date),
            "player": player,
            "market": market,
            "direction": direction,
            "line": line,
            "odds": odds,
            "units": units,
            "strategy": strategy,
            "result": result,
            "notes": notes,
            "added_at": str(datetime.now()),
        }
        bets.append(new_bet)
        save_bets(bets)
        st.success(f"Added: {player} {market} {direction} {line}")
        st.rerun()

# ===== UPDATE RESULTS =====
if bets:
    st.markdown("---")
    st.header("✏️ Update Results")
    
    pending_bets = [b for b in bets if b.get("result") == "pending"]
    if pending_bets:
        st.markdown(f"**{len(pending_bets)} pending bets to grade:**")
        
        for i, bet in enumerate(bets):
            if bet.get("result") != "pending":
                continue
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{bet['player']}** {bet['market']} {bet['direction']} {bet['line']} ({bet['date']})")
            with col2:
                new_result = st.selectbox(
                    "Result", ["pending", "won", "lost", "push"],
                    key=f"result_{i}",
                    index=0,
                )
            with col3:
                if st.button("Update", key=f"update_{i}"):
                    bets[i]["result"] = new_result
                    save_bets(bets)
                    st.rerun()
    else:
        st.success("All bets graded!")
