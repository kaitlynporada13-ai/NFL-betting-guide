"""
Automated Bet Card Generator.

Reads the blended-model prop projections (data/processed/prop_projections_latest.parquet,
produced by pipeline.prop_projections) and turns the playable picks into a ranked bet
card + readable YAML. ONE source of truth: the same projections the app and CLV tracker use.

Playable = HIGH / MEDIUM-HIGH / MEDIUM (the tiers whose OOS hit rate clears break-even).
Units are sized off the validated confidence tier. Role-change / backup / efficient-market
props are excluded from the card (they show on the market pages as no-play).

Run: python -m pipeline.generate_bet_card   (after pipeline.prop_projections)
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config_loader import get_data_dir
from pipeline.prop_projections import get_nfl_week

PROC_DIR = get_data_dir("processed")

# Units by validated confidence tier (flat-ish, scaled to edge strength).
TIER_UNITS = {"HIGH": 3.0, "MEDIUM-HIGH": 2.0, "MEDIUM": 1.0}
PLAYABLE = ("HIGH", "MEDIUM-HIGH", "MEDIUM")
TIER_RANK = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2}


def generate_bet_card():
    week = get_nfl_week()
    print("=" * 70)
    print(f"BET CARD GENERATOR (blended model) — Week {week}")
    print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    path = PROC_DIR / "prop_projections_latest.parquet"
    if not path.exists():
        print("\n  No projections found. Run: python -m pipeline.prop_projections")
        return None

    proj = pd.read_parquet(path)
    card = proj[proj["confidence"].isin(PLAYABLE)].copy()
    if card.empty:
        print("\n  No playable (HIGH/MEDIUM-HIGH/MEDIUM) props this week — nothing to bet.")
        # still write an empty card so the app shows the honest state
        _write_yaml([], week, 0)
        return None

    card["units"] = card["confidence"].map(TIER_UNITS).fillna(1.0)
    card["trank"] = card["confidence"].map(TIER_RANK).fillna(9)
    card = card.sort_values(["trank", "hit_est"], ascending=[True, False], na_position="last")

    card.to_parquet(PROC_DIR / "bet_card_latest.parquet", index=False)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    card.to_parquet(PROC_DIR / f"bet_card_week{week}_{ts}.parquet", index=False)

    plays = []
    for _, r in card.iterrows():
        plays.append({
            "player": r["player"],
            "market": r["market"],
            "line": float(r["line"]),
            "projection": float(r["projection"]) if pd.notna(r["projection"]) else None,
            "direction": str(r["call"]).lower(),
            "units": float(r["units"]),
            "confidence_tier": r["confidence"],
            "hit_rate_oos": float(r["hit_est"]) if pd.notna(r["hit_est"]) else None,
            "reasoning": r["why"],
        })
    _write_yaml(plays, week, len(card))

    print(f"\n{'='*70}\nWEEK {week} BET CARD — {len(card)} playable plays\n{'='*70}")
    print(f"\n{'Tier':<12}{'Dir':<7}{'Player':<22}{'Market':<12}{'Line':>6}{'Proj':>7}{'U':>4}")
    print("-" * 74)
    for _, r in card.head(30).iterrows():
        proj = f"{r['projection']:.1f}" if pd.notna(r["projection"]) else "n/a"
        print(f"{r['confidence']:<12}{str(r['call']):<7}{r['player']:<22}{r['market']:<12}"
              f"{r['line']:>6.1f}{proj:>7}{r['units']:>4.0f}")
    print(f"\nSaved: data/processed/bet_card_latest.parquet (+ .yaml, + weekly snapshot)")
    return card


def _write_yaml(plays, week, total):
    card_yaml = {
        "week": week,
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "blended projection model (prop_projections_latest.parquet)",
        "total_plays": total,
        "top_plays": plays[:25],
    }
    with open(PROC_DIR / "bet_card_latest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(card_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    generate_bet_card()
