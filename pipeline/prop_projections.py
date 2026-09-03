"""
PROP PROJECTION + CONFIDENCE ENGINE  (blended ML model, no under bias).

For every posted player prop, every week, output:
  - the line
  - a PROJECTION from the blended gradient-boosted model (trained on pre-game
    features: rolling form, volatility, target-share trend, snap trend, opponent
    red-zone defense, game context). The model learns the signal weights; there is
    NO hand-set under bias.
  - over/under CALL = projection vs line (mechanical).
  - CONFIDENCE anchored to the model's OUT-OF-SAMPLE hit rate for that market at
    that projection-vs-line gap (data/processed/blended_model_report.parquet).
  - a short "why".

Only markets/gaps that validated OOS earn real confidence:
  pass TDs (~60%, both sides) is the strongest; receptions thin (~53-55% small gap).
  pass/rush/reception yards are ~efficient -> capped LOW/PASS honestly.

Models: models/trained/proj_<market>.pkl  (built by scripts/build_blended_model.py)
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date

from pipeline.config_loader import get_data_dir
from pipeline.ingest_odds import pull_all_props_for_week

RAW = get_data_dir("raw")
PROC = get_data_dir("processed")
MODELS = Path(__file__).parent.parent / "models" / "trained"

SEASON_START = date(2026, 9, 10)

MARKET_STAT = {
    "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards", "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}
MARKET_LABEL = {
    "player_pass_yds": "Pass Yds", "player_pass_tds": "Pass TDs",
    "player_rush_yds": "Rush Yds", "player_receptions": "Receptions",
    "player_reception_yds": "Rec Yds",
}
# Below these lines a player is a backup/low-volume role; lines are priced tight and
# the projection is noisy. Flag + cap confidence.
STARTER_MIN_LINE = {
    "player_pass_yds": 175, "player_pass_tds": 0.5, "player_rush_yds": 30,
    "player_receptions": 3.5, "player_reception_yds": 35,
}


def get_nfl_week(today: date | None = None) -> int:
    today = today or date.today()
    delta = (today - SEASON_START).days
    return 1 if delta < 0 else min(18, delta // 7 + 1)


def gap_bucket(gap_pct):
    if gap_pct < 0.05:
        return "tiny <5%"
    if gap_pct < 0.10:
        return "small 5-10%"
    if gap_pct < 0.20:
        return "med 10-20%"
    return "big >20%"


def confidence_from_hit(hit):
    """Confidence tier from a validated OOS hit rate (break-even 52.4%)."""
    if hit is None:
        return "LOW"
    if hit >= 0.60:
        return "HIGH"
    if hit >= 0.565:
        return "MEDIUM-HIGH"
    if hit >= 0.54:
        return "MEDIUM"
    if hit >= 0.524:
        return "LOW"
    return "PASS"


def load_models():
    models = {}
    for market in MARKET_STAT:
        p = MODELS / f"proj_{market}.pkl"
        if p.exists():
            models[market] = joblib.load(p)
    return models


def load_oos_report():
    """market -> {gap_bucket -> (hit_rate, n, over_share)} from the validation run."""
    path = PROC / "blended_model_report.parquet"
    table = {}
    if path.exists():
        rep = pd.read_parquet(path)
        for _, r in rep.iterrows():
            table.setdefault(r["market"], {})[r["gap_bucket"]] = (
                float(r["hit_rate"]), int(r["n"]), float(r["over_share"]))
    return table


def load_latest_features():
    """Most-recent pre-game feature row per player (basis for projecting next game)."""
    pf = pd.read_parquet(PROC / "player_features.parquet")
    pf = pf.drop_duplicates(subset=["player_id", "season", "week"])
    nc = "player_display_name" if "player_display_name" in pf.columns else "player_name"
    pf["pname"] = pf[nc].str.lower().str.replace(".", "", regex=False).str.strip()
    pf = pf.sort_values(["season", "week"])
    latest = pf.groupby("pname").tail(1).set_index("pname")
    return latest


def load_role_changes():
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


def is_backup_line(market, line):
    return line < STARTER_MIN_LINE.get(market, 0)


def build_projections():
    props = pull_all_props_for_week()
    if props.empty:
        print("No props available yet.")
        return pd.DataFrame()

    props = props[props["market"].isin(MARKET_STAT) & (props["outcome_name"] == "Over")].copy()
    models = load_models()
    oos = load_oos_report()
    feats = load_latest_features()
    role_up, role_reasons = load_role_changes()
    nfl_week = get_nfl_week()

    if not models:
        print("[prop_projections] No trained models found. Run scripts/build_blended_model.py")
        return pd.DataFrame()

    rows = []
    for _, p in props.iterrows():
        market = p["market"]
        line = p.get("outcome_point")
        if line is None or market not in models:
            continue
        name = p.get("player_name", "")
        pkey = name.lower().replace(".", "").strip()
        mk = MARKET_LABEL[market]
        role_changed = pkey in role_up
        backup = is_backup_line(market, line)

        # --- PROJECTION from the blended model ---
        projection = None
        if pkey in feats.index:
            bundle = models[market]
            model, mfeat = bundle["model"], bundle["features"]
            row = feats.loc[pkey]
            x = pd.DataFrame([[row.get(c, np.nan) for c in mfeat]], columns=mfeat).astype(float)
            try:
                projection = max(0.0, float(model.predict(x)[0]))  # stats can't be negative
            except Exception:
                projection = None

        # --- Role change: projection unreliable, do not bet ---
        if role_changed:
            rsn = role_reasons.get(pkey, "role increased due to injury ahead")
            rows.append(_row(name, mk, market, line, projection, "AVOID (role change)",
                             "ROLE-CHANGE", None, backup, True, p,
                             f"ROLE CHANGE — {rsn}. Recent form understates the new role, so the "
                             f"projection is unreliable. No play."))
            continue

        # --- No features (rookie/new): can't project ---
        if projection is None:
            rows.append(_row(name, mk, market, line, None, "NO PROJECTION", "LOW", None,
                             backup, False, p,
                             f"No recent data to project {mk} (new/rookie or name unmatched). "
                             f"Model can't form a confident number — no play."))
            continue

        # --- CALL follows projection; CONFIDENCE from validated OOS hit rate ---
        call = "OVER" if projection > line else "UNDER"
        gap_pct = abs(projection - line) / max(line, 0.5)
        bucket = gap_bucket(gap_pct)
        hit, n, over_share = oos.get(market, {}).get(bucket, (None, 0, None))

        conf = confidence_from_hit(hit)
        if backup:
            conf = "PASS" if conf in ("HIGH", "MEDIUM-HIGH", "MEDIUM") else conf

        # --- why ---
        proj_s = f"{projection:.1f}"
        if hit is not None and hit >= 0.524 and not backup:
            why = (f"Model projects {proj_s} vs line {line} ({gap_pct:.0%} gap) -> {call}. "
                   f"This market+gap hit {hit:.0%} out-of-sample (2025, n={n}). "
                   f"Projection blends recent form, opponent, pace, role & context.")
        elif backup:
            why = (f"Model projects {proj_s} vs {line}, but this is a low-volume/backup line — "
                   f"noisy and tightly priced. No play.")
        else:
            hr = f"{hit:.0%}" if hit is not None else "n/a"
            why = (f"Model projects {proj_s} vs line {line} ({call}), but this market is "
                   f"efficient here (OOS {hr}, below break-even). Informational only — no edge.")

        rows.append(_row(name, mk, market, line, projection, call, conf, hit,
                         backup, False, p, why))

    df = pd.DataFrame(rows).sort_values("hit_est", ascending=False, na_position="last")
    df.to_parquet(PROC / "prop_projections_latest.parquet", index=False)
    return df


def _row(name, mk, market, line, projection, call, conf, hit, backup, role_changed, p, why):
    return {
        "player": name, "market": mk, "market_key": market, "line": line,
        "projection": round(projection, 1) if projection is not None else None,
        "call": call, "confidence": conf,
        "hit_est": round(hit, 3) if hit is not None else None,
        "backup_line": backup, "role_change": role_changed, "why": why,
        "home_team": p.get("home_team", ""), "away_team": p.get("away_team", ""),
    }


def main():
    df = build_projections()
    if df.empty:
        return
    print("=" * 100)
    print(f"PROP PROJECTIONS (blended model) — {len(df)} props | line / proj / call / conf / why")
    print("=" * 100)
    print(f"{'Player':<20}{'Prop':<11}{'Line':>6}{'Proj':>6}  {'Call':<7}{'Conf':<12}Why")
    print("-" * 100)
    for _, r in df.iterrows():
        proj = f"{r['projection']:.1f}" if r["projection"] is not None else "n/a"
        print(f"{r['player']:<20}{r['market']:<11}{r['line']:>6.1f}{proj:>6}  "
              f"{r['call']:<7}{r['confidence']:<12}{r['why'][:58]}")
    print("\nSaved to data/processed/prop_projections_latest.parquet")


if __name__ == "__main__":
    main()
