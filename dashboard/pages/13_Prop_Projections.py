"""
Prop Projections — every posted player prop with line, projection, call, confidence, why.
Reads data/processed/prop_projections_latest.parquet (from pipeline.prop_projections).
Confidence is anchored to out-of-sample validated Week 1 under rates.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROC = PROJECT_ROOT / "data" / "processed"

st.set_page_config(page_title="Prop Projections | NFL", page_icon="🎯", layout="wide")

CONF_RANK = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW": 3, "PASS": 4, "ROLE-CHANGE": 5}

st.title("🎯 Prop Projections")
st.caption("Every posted prop: line, projection, over/under call, confidence, and why. "
           "Confidence anchored to out-of-sample validated Week 1 rates.")

path = PROC / "prop_projections_latest.parquet"
if not path.exists():
    st.warning("No projections yet. Run: `python -m pipeline.prop_projections`")
    st.stop()

df = pd.read_parquet(path)
df["crank"] = df["confidence"].map(CONF_RANK).fillna(9)
df = df.sort_values(["crank", "hit_est"], ascending=[True, False])

# Validation banner
st.info("**Validated edge (2023-24 → 2025 holdout):** Week 1 is UNDER-only — no overs survived. "
        "Pass TD unders 67% · Rush Yd unders 60% · Pass Yd unders 57% · Receptions 54% · Rec Yds weakest. "
        "Under hits harder when the line sits above the player's baseline.")


# Summary metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Props", len(df))
with c2:
    st.metric("HIGH conviction", int((df["confidence"] == "HIGH").sum()))
with c3:
    st.metric("Markets", df["market"].nunique())
with c4:
    backups = int(df["backup_line"].sum()) if "backup_line" in df.columns else 0
    st.metric("Backup-line flags", backups)

st.markdown("---")

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    mkt = st.multiselect("Market", sorted(df["market"].unique()), default=[])
with col2:
    conf_opts = ["HIGH", "MEDIUM-HIGH", "MEDIUM", "LOW", "PASS", "ROLE-CHANGE"]
    conf = st.multiselect("Confidence", conf_opts, default=["HIGH", "MEDIUM-HIGH", "MEDIUM"])
with col3:
    hide_backup = st.checkbox("Hide backup-line flags", value=True)

# Surface any role-change (injury) flags prominently
if "role_change" in df.columns and df["role_change"].any():
    rc = df[df["role_change"]]
    st.warning(f"⚠️ {len(rc)} prop(s) flagged ROLE-CHANGE (injury bumped their role — "
               f"baseline is stale, do NOT bet the under): "
               + ", ".join(f"{r['player']} {r['market']}" for _, r in rc.iterrows()))

view = df.copy()
if mkt:
    view = view[view["market"].isin(mkt)]
if conf:
    view = view[view["confidence"].isin(conf)]
if hide_backup and "backup_line" in view.columns:
    view = view[~view["backup_line"]]

# Main table
st.markdown("### Board")
show = view[["player", "market", "line", "projection", "call", "confidence", "why"]].copy()
show.columns = ["Player", "Prop", "Line", "Projection", "Call", "Confidence", "Why"]
st.dataframe(
    show, use_container_width=True, hide_index=True,
    column_config={
        "Line": st.column_config.NumberColumn(format="%.1f"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Why": st.column_config.TextColumn(width="large"),
    },
)

st.markdown("---")

# Top plays by tier
st.markdown("### Top Plays by Tier")
for tier in ["HIGH", "MEDIUM-HIGH", "MEDIUM"]:
    sub = view[view["confidence"] == tier]
    if len(sub):
        st.markdown(f"**{tier}** ({len(sub)})")
        for _, r in sub.iterrows():
            st.markdown(f"- 🔴 **UNDER {r['player']}** {r['line']:.1f} {r['market']} "
                        f"(proj {r['projection'] if pd.notna(r['projection']) else 'n/a'})")
            st.caption(r["why"])

st.markdown("---")
st.caption("Re-run `python -m pipeline.prop_projections` when new props post to refresh this board.")
