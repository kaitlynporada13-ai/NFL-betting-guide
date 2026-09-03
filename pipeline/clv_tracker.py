"""
CLOSING-LINE-VALUE (CLV) TRACKER.

Closes the loop on line capture: links OUR picks to the line we recorded them at,
then to the closing line near kickoff, and scores CLV.

CLV = did the market move TOWARD our side after we logged the pick?
  - OVER pick: positive CLV if the closing line is LOWER than our line
    (we took the over at a cheaper number than it closed).
  - UNDER pick: positive CLV if the closing line is HIGHER than our line.
Consistently positive CLV is the #1 predictor of long-term profit — it tells us
we're sharp even before results come in, and on a much larger sample than W/L.

Flow each week:
  1. record_picks()        after generating picks -> logs pick + line + timestamp
  2. update_closing()      near kickoff (after a late capture_lines run) -> attaches
                           the closing line + computes CLV per pick
  3. clv_report()          summarize avg CLV + % positive by tier/market (sharpness)

Storage: data/clv/clv_log.parquet  (append-only, one row per pick per week)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from pipeline.config_loader import get_data_dir

PROC = get_data_dir("processed")
LINES = get_data_dir("lines")


def _clv_dir() -> Path:
    d = Path(get_data_dir("processed")).parent / "clv"
    d.mkdir(parents=True, exist_ok=True)
    return d


CLV_LOG = _clv_dir() / "clv_log.parquet"


def _pick_key(df):
    """Stable key for a pick: season/week not always present -> use player+market+line window."""
    return (df["player"].astype(str).str.lower().str.strip() + "|"
            + df["market_key"].astype(str) + "|" + df["captured_line"].astype(str))


def record_picks(week: int | None = None) -> pd.DataFrame:
    """
    Log the current model picks with the line we're recording them at (the 'bet' line)
    and a timestamp. Only logs playable picks (HIGH / MEDIUM-HIGH / MEDIUM) so the CLV
    scorecard measures the bets we'd actually make.
    """
    path = PROC / "prop_projections_latest.parquet"
    if not path.exists():
        print("[clv] No prop projections found — run pipeline.prop_projections first.")
        return pd.DataFrame()

    picks = pd.read_parquet(path)
    playable = picks[picks["confidence"].isin(["HIGH", "MEDIUM-HIGH", "MEDIUM"])].copy()
    if playable.empty:
        print("[clv] No playable picks to log this week.")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    playable["captured_line"] = playable["line"]
    playable["captured_utc"] = now.isoformat()
    playable["week"] = week if week is not None else _infer_week()
    playable["closing_line"] = np.nan
    playable["closing_utc"] = pd.NA
    playable["clv"] = np.nan
    playable["clv_recorded"] = False

    cols = ["week", "player", "market", "market_key", "captured_line", "call",
            "confidence", "projection", "captured_utc", "closing_line",
            "closing_utc", "clv", "clv_recorded", "home_team", "away_team"]
    cols = [c for c in cols if c in playable.columns]
    out = playable[cols].copy()

    # append to the log, de-duping on player+market+week+captured_line (idempotent per week)
    if CLV_LOG.exists():
        prev = pd.read_parquet(CLV_LOG)
        combined = pd.concat([prev, out], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["week", "player", "market_key", "captured_line"], keep="first")
    else:
        combined = out
    combined.to_parquet(CLV_LOG, index=False)
    print(f"[clv] Logged {len(out)} playable picks for week {out['week'].iloc[0]} "
          f"(log now {len(combined)} rows).")
    return out


def update_closing() -> pd.DataFrame:
    """
    Attach the latest captured (closing) line to logged picks that don't yet have one,
    and compute CLV. Run this near kickoff AFTER a late capture_lines.capture_all().
    """
    if not CLV_LOG.exists():
        print("[clv] No CLV log yet — run record_picks first.")
        return pd.DataFrame()
    latest_path = LINES / "prop_lines_latest.parquet"
    if not latest_path.exists():
        print("[clv] No captured closing lines — run pipeline.capture_lines first.")
        return pd.DataFrame()

    log = pd.read_parquet(CLV_LOG)
    close = pd.read_parquet(latest_path)
    close = close[close.get("outcome_name", "Over") == "Over"].copy()
    if "outcome_point" not in close.columns:
        print("[clv] Captured lines lack outcome_point.")
        return log
    close["pkey"] = close["player_name"].astype(str).str.lower().str.strip()
    close["mkey"] = close["market"].astype(str)
    close_map = close.groupby(["pkey", "mkey"])["outcome_point"].last().to_dict()

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    for i, r in log.iterrows():
        if r.get("clv_recorded"):
            continue
        k = (str(r["player"]).lower().strip(), str(r["market_key"]))
        cl = close_map.get(k)
        if cl is None:
            continue
        log.at[i, "closing_line"] = cl
        log.at[i, "closing_utc"] = now
        # CLV: OVER wants closing lower; UNDER wants closing higher
        if str(r["call"]).upper().startswith("OVER"):
            log.at[i, "clv"] = r["captured_line"] - cl
        else:
            log.at[i, "clv"] = cl - r["captured_line"]
        log.at[i, "clv_recorded"] = True
        updated += 1

    log.to_parquet(CLV_LOG, index=False)
    print(f"[clv] Updated closing lines + CLV for {updated} picks.")
    return log


def clv_report() -> pd.DataFrame:
    """Sharpness scorecard: avg CLV and % of picks with positive CLV, by tier + market."""
    if not CLV_LOG.exists():
        print("[clv] No CLV log yet.")
        return pd.DataFrame()
    log = pd.read_parquet(CLV_LOG)
    scored = log[log["clv_recorded"] == True].copy()
    if scored.empty:
        print("[clv] No picks have closing lines yet — run update_closing near kickoff.")
        return pd.DataFrame()

    scored["clv_positive"] = scored["clv"] > 0
    print("=" * 70)
    print(f"CLV SHARPNESS SCORECARD — {len(scored)} scored picks")
    print("Positive CLV = the market moved toward our side after we logged it.")
    print("Consistently positive avg CLV = we're beating the number (sharp).")
    print("=" * 70)

    def summarize(df, label):
        if len(df) == 0:
            return
        print(f"  {label:<22} n={len(df):>4}  avg CLV {df['clv'].mean():+.2f}  "
              f"% positive {df['clv_positive'].mean():.0%}")

    print("\nBy confidence tier:")
    for tier in ["HIGH", "MEDIUM-HIGH", "MEDIUM"]:
        summarize(scored[scored["confidence"] == tier], tier)
    print("\nBy market:")
    for mk in scored["market"].unique():
        summarize(scored[scored["market"] == mk], mk)
    print("\nOverall:")
    summarize(scored, "ALL")
    return scored


def _infer_week() -> int:
    from pipeline.prop_projections import get_nfl_week
    return get_nfl_week()


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "record"
    if cmd == "record":
        record_picks()
    elif cmd == "close":
        update_closing()
    elif cmd == "report":
        clv_report()
    else:
        print("Usage: python -m pipeline.clv_tracker [record|close|report]")
