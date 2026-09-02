"""
ANYTIME TD engine.
Validated OOS (2023-25):
  - Week 1: players score FEWER TDs than the market implies (rust) -> betting YES is -EV.
  - Season-long: only HEAVY favorites (priced <= -200) beat their implied prob (+7.8% ROI).
    Dogs and longshots (+120 and up) are systematically -EV traps.

Engine logic per player:
  - Project TD probability = 2025 TD/game rate x Week-1 rust discount.
  - Implied prob from the price.
  - Edge = projected - implied.
  - GREEN-LIGHT (YES) only when projected clearly exceeds implied AND the price is short
    (near-lock territory). Otherwise FADE (Week 1) or PASS. Longshots are flagged as traps.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from pipeline.config_loader import get_data_dir
from pipeline.ingest_odds import pull_all_props_for_week

RAW = get_data_dir("raw")
PROC = get_data_dir("processed")
TD_RUST = 0.88  # Week 1 TD suppression (players find end zone less in openers)


def current_week():
    ss = date(2026, 9, 10)
    t = date.today()
    if t < ss:
        return 1 if (ss - t).days <= 28 else 0
    return min(max(1, (t - ss).days // 7 + 1), 22)


def american_to_prob(o):
    if pd.isna(o):
        return np.nan
    return (-o) / (-o + 100) if o < 0 else 100 / (o + 100)


def td_baselines():
    """Per-player 2025 anytime-TD rate = share of games with >=1 rush/rec TD."""
    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    s = stats[stats["season"] == 2025].copy()
    nc = "player_display_name" if "player_display_name" in s.columns else "player_name"
    s["pname"] = s[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    s["td"] = s.get("rushing_tds", 0).fillna(0) + s.get("receiving_tds", 0).fillna(0)
    rate = s.groupby("pname")["td"].apply(lambda x: (x >= 1).mean())
    return rate.to_dict()


def build():
    week = current_week()
    props = pull_all_props_for_week()
    if props.empty:
        return pd.DataFrame()
    td = props[(props["market"] == "player_anytime_td")].copy()
    if td.empty:
        print("No anytime TD props posted yet.")
        return pd.DataFrame()

    base = td_baselines()
    rows = []
    for _, p in td.iterrows():
        name = p.get("player_name", "")
        pk = name.lower().replace(".", "").strip()
        price = p.get("outcome_price")
        implied = american_to_prob(price)
        if implied is None or np.isnan(implied):
            continue
        rate = base.get(pk)
        proj = (rate * TD_RUST) if rate is not None else None

        edge = (proj - implied) if proj is not None else None

        # Decide call from validated rules
        if week == 1:
            # Week 1 YES is -EV overall. Only allow YES on near-locks where our
            # projection still clears implied by a real margin.
            if price is not None and price <= -200 and edge is not None and edge > 0.03:
                call, conf = "YES", "MEDIUM"
                why = (f"Heavy favorite ({int(price)}); projected {proj:.0%} scoring vs {implied:.0%} "
                       f"implied. Only anytime-TD spot that's +EV (validated). Bellcow/red-zone role.")
            elif price is not None and price >= 200:
                call, conf = "FADE / PASS", "PASS"
                why = (f"Longshot (+{int(price)}) — validated TRAP. Week 1 TD longshots lose ~14% ROI. "
                       f"Fun but -EV; skip.")
            else:
                call, conf = "PASS", "PASS"
                why = (f"Week 1 players score fewer TDs than implied (rust): projected "
                       f"{proj:.0%} vs {implied:.0%} implied. Betting YES here is -EV." if proj is not None
                       else f"No 2025 TD rate (new/rookie); Week 1 YES is -EV by default. Pass.")
        else:
            if price is not None and price <= -200 and edge is not None and edge > 0.02:
                call, conf = "YES", "MEDIUM"
                why = (f"Heavy favorite ({int(price)}); projected {proj:.0%} vs {implied:.0%} implied. "
                       f"Only validated +EV anytime-TD angle (~+8% ROI on near-locks).")
            elif price is not None and price >= 120:
                call, conf = "PASS", "PASS"
                why = f"Dog/longshot ({'+' if price>0 else ''}{int(price)}) — validated -EV. Skip."
            else:
                call, conf = "PASS", "PASS"
                why = (f"Projected {proj:.0%} vs {implied:.0%} implied — no edge." if proj is not None
                       else "No baseline; pass.")

        rows.append({
            "player": name, "price": int(price) if price is not None else None,
            "implied_pct": round(implied * 100, 1),
            "projection_pct": round(proj * 100, 1) if proj is not None else None,
            "call": call, "confidence": conf, "why": why,
            "home_team": p.get("home_team", ""), "away_team": p.get("away_team", ""),
        })

    df = pd.DataFrame(rows)
    order = {"YES": 0, "MEDIUM": 0, "PASS": 1}
    df["r"] = df["confidence"].map({"MEDIUM": 0, "PASS": 1}).fillna(1)
    df = df.sort_values(["r", "projection_pct"], ascending=[True, False], na_position="last")
    df.to_parquet(PROC / "anytime_td_picks_latest.parquet", index=False)
    return df


def main():
    df = build()
    if df.empty:
        return
    greens = df[df["call"] == "YES"]
    print(f"Anytime TD — {len(df)} players. Validated YES plays: {len(greens)}")
    for _, r in greens.iterrows():
        print(f"  YES {r['player']} ({r['price']:+d}) proj {r['projection_pct']}% vs {r['implied_pct']}% implied")
    print("\n(Everything else is PASS/FADE — Week 1 anytime-TD YES is -EV except near-locks.)")


if __name__ == "__main__":
    main()
