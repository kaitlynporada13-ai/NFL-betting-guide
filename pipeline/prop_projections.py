"""
PROP PROJECTION + CONFIDENCE ENGINE.
For every posted Week 1 player prop, output:
  - the line
  - a projection (expected number)
  - over/under call
  - confidence score (anchored to OUT-OF-SAMPLE validated hit rates)
  - a short "why"

Confidence is grounded in scripts/validate_props_deep.py results (train 2023-24 -> test 2025).
CALL follows the projection: projection > line -> OVER, projection < line -> UNDER.
Confidence is asymmetric because the data is: Week 1 UNDERS validate (54-67%), OVERS
do NOT (best market ~45-51% OOS). So overs are shown honestly but capped low-confidence.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from pipeline.config_loader import get_data_dir
from pipeline.ingest_odds import pull_all_props_for_week

RAW = get_data_dir("raw")
PROC = get_data_dir("processed")

# 2026 NFL season opener (Week 1 kickoff). Used to compute the current week so the
# engine only applies the validated Week-1 edge in Week 1.
SEASON_START = date(2026, 9, 10)


def get_nfl_week(today: date | None = None) -> int:
    """Current NFL week (1-18). <1 before the season = treat as Week 1 prep."""
    today = today or date.today()
    delta = (today - SEASON_START).days
    if delta < 0:
        return 1  # preseason / lines posting for the opener
    return min(18, delta // 7 + 1)

MARKET_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}
MARKET_LABEL = {
    "player_pass_yds": "Pass Yds", "player_pass_tds": "Pass TDs",
    "player_rush_yds": "Rush Yds", "player_receptions": "Receptions",
    "player_reception_yds": "Rec Yds",
}

# Validated OUT-OF-SAMPLE Week 1 UNDER hit rates (test 2025). Base by market.
BASE_UNDER = {
    "player_pass_tds": 0.667, "player_rush_yds": 0.600, "player_pass_yds": 0.567,
    "player_receptions": 0.538, "player_reception_yds": 0.514,
}
# Inflation bonus: line meaningfully above baseline pushed under to ~65-70% OOS.
def inflation_hit(market, infl_pct):
    base = BASE_UNDER.get(market, 0.50)
    if infl_pct is None:
        return base
    if infl_pct >= 0.15:      # line well above norm
        return min(base + 0.10, 0.72)
    if infl_pct >= 0.05:      # line above norm
        return min(base + 0.06, 0.70)
    if infl_pct <= -0.10:     # line below norm — edge weakens (but no over edge exists)
        return max(base - 0.06, 0.50)
    return base


def load_baselines():
    """Per-player 2025 per-game average for each stat (the projection base)."""
    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    s25 = stats[stats["season"] == 2025].copy()
    nc = "player_display_name" if "player_display_name" in s25.columns else "player_name"
    s25["pname"] = s25[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    base = {}
    for market, stat in MARKET_STAT.items():
        if stat in s25.columns:
            avg = s25.groupby("pname")[stat].mean()
            base[market] = avg.to_dict()
    return base


def load_role_changes():
    """Players whose role changed due to an injury ahead of them (baseline stale)."""
    import yaml
    path = Path(__file__).parent.parent / "data" / "human_notes" / "depth_chart_overrides_2026.yaml"
    role_up, reasons = set(), {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key in ("role_up", "role_up_receivers"):
            for entry in data.get(key, []):
                pk = entry["player"].lower().replace(".", "").strip()
                role_up.add(pk)
                reasons[pk] = entry.get("reason", "")
    return role_up, reasons


# Week 1 rust discount applied to the baseline to form the PROJECTION.
# Offenses score below their full-season talent in openers (validated: unders win).
RUST_DISCOUNT = {
    "player_pass_tds": 0.82, "player_pass_yds": 0.90, "player_rush_yds": 0.88,
    "player_receptions": 0.92, "player_reception_yds": 0.90,
}

# Minimum line for a prop to be a "meaningful volume" starter play. Below these,
# the player is a backup/low-usage role: the % inflation looks huge but the edge
# is thin and noisy (FanDuel prices low lines tightly). Flag + cap confidence.
STARTER_MIN_LINE = {
    "player_pass_yds": 175, "player_pass_tds": 0.5, "player_rush_yds": 30,
    "player_receptions": 3.5, "player_reception_yds": 35,
}


def is_backup_line(market, line):
    return line < STARTER_MIN_LINE.get(market, 0)


def confidence_label(hit):
    if hit >= 0.65:
        return "HIGH"
    if hit >= 0.58:
        return "MEDIUM-HIGH"
    if hit >= 0.54:
        return "MEDIUM"
    if hit >= 0.51:
        return "LOW"
    return "PASS"


# Validated OUT-OF-SAMPLE Week 1 OVER hit rates (test 2025, from validate_props_deep.py
# Layer 2: line-below-baseline). Overs are structurally weak in Week 1 (rust) — even
# deflated lines went under more often. Best market (rush) only ~45%. So OVER calls
# are shown (the projection says so) but capped LOW/PASS to reflect the real headwind.
BASE_OVER = {
    "player_pass_tds": 0.42, "player_rush_yds": 0.45, "player_pass_yds": 0.33,
    "player_receptions": 0.49, "player_reception_yds": 0.51,
}


def over_hit(market, infl_pct):
    """Projected-over hit estimate. The further projection sits above the line
    (more negative inflation), the better the over — but capped by the weak
    validated ceiling. None of these clear break-even, so overs stay low-conf."""
    base = BASE_OVER.get(market, 0.45)
    if infl_pct is None:
        return base
    # infl_pct = (line - baseline)/baseline; more negative => line well below norm => stronger over
    if infl_pct <= -0.15:
        return min(base + 0.06, 0.55)
    if infl_pct <= -0.05:
        return min(base + 0.03, 0.53)
    return base


def build_projections():
    props = pull_all_props_for_week()
    if props.empty:
        print("No props available yet.")
        return pd.DataFrame()

    props = props[props["market"].isin(MARKET_STAT) & (props["outcome_name"] == "Over")].copy()
    baselines = load_baselines()
    role_up, role_reasons = load_role_changes()

    nfl_week = get_nfl_week()
    week1 = nfl_week == 1
    if not week1:
        print(f"[prop_projections] NFL Week {nfl_week}: NO validated prop edge exists past "
              f"Week 1 (markets are efficient weeks 2-18, confirmed OOS). Projections shown "
              f"as informational only; all props flagged NO-EDGE / no play.")

    rows = []
    for _, p in props.iterrows():
        market = p["market"]
        line = p.get("outcome_point")
        if line is None:
            continue
        name = p.get("player_name", "")
        pkey = name.lower().replace(".", "").strip()
        baseline = baselines.get(market, {}).get(pkey)
        role_changed = pkey in role_up

        # Projection = rust-discounted baseline (what we expect in a Week 1 opener)
        if baseline is not None and baseline > 0:
            projection = baseline * RUST_DISCOUNT.get(market, 0.9)
            infl_pct = (line - baseline) / baseline
        else:
            projection = None
            infl_pct = None

        mk = MARKET_LABEL[market]
        backup = is_backup_line(market, line)

        # Role-change override: player promoted because someone ahead is hurt.
        # Their 2025 baseline understates their new role -> the projection is unreliable.
        if role_changed:
            call = "AVOID (role change)"
            conf = "ROLE-CHANGE"
            hit = None
            rsn = role_reasons.get(pkey, "role increased due to injury ahead")
            why = (f"ROLE CHANGE — {rsn}. His 2025 baseline understates the new role, so neither "
                   f"the projection nor the line is trustworthy here. No play.")

        # No baseline (rookie/new) -> can't project; default to the Week 1 under lean, low conf.
        elif projection is None:
            call = "UNDER"
            hit = BASE_UNDER.get(market, 0.50)
            conf = "LOW"
            why = f"No 2025 baseline (new/rookie) — can't project. Default Week 1 {mk} under lean ~{hit:.0%}."

        else:
            # CALL FOLLOWS THE PROJECTION. Projection is our number; if it clears the
            # line the honest call is OVER, if it's below the line it's UNDER.
            base_str = f"2025 avg {baseline:.1f} (proj {projection:.1f} w/ Wk1 rust)"
            if projection > line:
                call = "OVER"
                hit = over_hit(market, infl_pct)
                # Overs are structurally weak in Week 1; even a projection above the line
                # only earns real confidence when it's a big gap AND market isn't a known trap.
                conf = confidence_label(hit)
                gap = (projection - line)
                why = (f"Projection {projection:.1f} is above the {line} line ({base_str}) — "
                       f"call is OVER. But Week 1 overs are structurally weak (rust): validated "
                       f"~{hit:.0%} even for deflated lines. Lower-confidence; size down.")
                if backup:
                    conf = "PASS"
                    why = (f"Projection {projection:.1f} clears {line}, but this is a low-volume/backup "
                           f"line — the baseline is noisy and Week 1 overs don't validate. No play.")
            else:
                call = "UNDER"
                hit = inflation_hit(market, infl_pct)
                if backup:
                    hit = min(BASE_UNDER.get(market, 0.50), 0.55)  # strip inflation bonus, cap
                    why = (f"Low-volume/backup line ({line}) — % inflation is misleading on small "
                           f"lines; base Week 1 {mk} under only ~{hit:.0%}. Thin edge, size down.")
                elif infl_pct is not None and infl_pct >= 0.05:
                    why = (f"Projection {projection:.1f} is under the {line} line, which sits {infl_pct:+.0%} "
                           f"above his {base_str}; Week 1 {mk} unders hit ~{hit:.0%} when the line is inflated.")
                else:
                    why = (f"Projection {projection:.1f} is below the {line} line ({base_str}); "
                           f"Week 1 {mk} unders hit ~{hit:.0%} (rust effect).")
                conf = confidence_label(hit)

        # WEEK-AWARENESS: the ONLY validated prop edge is the Week 1 rust under.
        # Weeks 2-18 the market is efficient (OOS: ~50% train, no edge in ANY market
        # or inflation bucket). So past Week 1 we keep the projection + mechanical call
        # for reference but strip confidence to NO-EDGE — never green-light a prop.
        # (Role-change AVOIDs stand on their own regardless of week.)
        if not week1 and not role_changed:
            conf = "NO-EDGE"
            hit = None
            proj_str = f"{projection:.1f}" if projection is not None else "n/a"
            why = (f"Week {nfl_week}: no validated prop edge exists past Week 1 — the market is "
                   f"efficient (tested OOS). Projection {proj_str} vs line {line} is informational "
                   f"only; the {call.lower()} lean is NOT a bet. No play.")

        rows.append({
            "player": name, "market": mk, "market_key": market, "line": line,
            "projection": round(projection, 1) if projection is not None else None,
            "baseline_2025": round(baseline, 1) if baseline is not None else None,
            "call": call, "confidence": conf,
            "hit_est": round(hit, 3) if hit is not None else None,
            "backup_line": backup, "role_change": role_changed, "why": why,
            "home_team": p.get("home_team", ""), "away_team": p.get("away_team", ""),
        })

    df = pd.DataFrame(rows).sort_values("hit_est", ascending=False, na_position="last")
    df.to_parquet(PROC / "prop_projections_latest.parquet", index=False)
    return df


def main():
    df = build_projections()
    if df.empty:
        return
    print("=" * 100)
    print(f"PROP PROJECTIONS — {len(df)} props | line / projection / call / confidence / why")
    print("=" * 100)
    print(f"{'Player':<20}{'Prop':<11}{'Line':>6}{'Proj':>6}{'Call':<7}{'Conf':<12}Why")
    print("-" * 100)
    for _, r in df.iterrows():
        proj = f"{r['projection']:.1f}" if r['projection'] is not None else "n/a"
        print(f"{r['player']:<20}{r['market']:<11}{r['line']:>6.1f}{proj:>6}"
              f"  {r['call']:<5}{r['confidence']:<12}{r['why'][:60]}")
    print(f"\nSaved to data/processed/prop_projections_latest.parquet")


if __name__ == "__main__":
    main()
