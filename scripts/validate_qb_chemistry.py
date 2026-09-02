"""
VALIDATE the QB -> receiver-position chemistry hypothesis (out-of-sample).

Hypothesis: when a pass-catcher gets a NEW QB, does that QB's PRIOR tendency to
target the player's position group predict the player beating their own baseline?
(e.g., Cousins historically feeds TEs -> does that lift a TE he inherits?)

Build:
  1. From PBP, QB-season position lean = share of that QB's completions to TE/WR/RB.
  2. A QB's PRIOR lean = average of their earlier seasons (excludes current).
  3. New-QB player-seasons: player whose team's primary QB changed (or player changed team).
  4. Test: does the new QB's prior lean-to-that-position predict the player's receiving
     yards beating their prior-season baseline? Train (<=2023) / Test (2024-25).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nfl_data_py as nfl

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"
SEASONS = [2021, 2022, 2023, 2024, 2025]


def build_qb_lean():
    """Per QB-season: share of completions to each position group."""
    pos_by_id = {}
    for yr in SEASONS:
        ros = nfl.import_seasonal_rosters([yr])
        for _, r in ros.iterrows():
            pos_by_id[r["player_id"]] = r["position"]

    frames = []
    for yr in SEASONS:
        pbp = nfl.import_pbp_data([yr], downcast=True)
        p = pbp[(pbp["play_type"] == "pass") & (pbp["complete_pass"] == 1)
                & pbp["receiver_player_id"].notna()].copy()
        p["pos"] = p["receiver_player_id"].map(pos_by_id)
        p["season"] = yr
        p = p[p["pos"].isin(["TE", "WR", "RB"])]
        frames.append(p[["season", "passer_player_name", "pos", "pass_touchdown"]])
    allp = pd.concat(frames, ignore_index=True)

    # share of completions by position for each QB-season
    grp = allp.groupby(["passer_player_name", "season", "pos"]).size().reset_index(name="n")
    tot = grp.groupby(["passer_player_name", "season"])["n"].transform("sum")
    grp["share"] = grp["n"] / tot
    grp = grp[tot >= 100]  # QB with real volume
    lean = grp.pivot_table(index=["passer_player_name", "season"], columns="pos",
                           values="share", fill_value=0).reset_index()
    return lean


def main():
    cache = RAW / "qb_position_lean.parquet"
    if cache.exists():
        print("Loading cached QB position-lean profiles...")
        lean = pd.read_parquet(cache)
    else:
        print("Building QB position-lean profiles from PBP (this takes a minute)...")
        lean = build_qb_lean()
        lean.to_parquet(cache, index=False)
    print(f"  QB-seasons profiled: {len(lean)}")

    # Prior lean per QB (mean of earlier seasons)
    prior = {}
    for qb in lean["passer_player_name"].unique():
        sub = lean[lean["passer_player_name"] == qb].sort_values("season")
        for i, row in sub.iterrows():
            earlier = sub[sub["season"] < row["season"]]
            if len(earlier):
                prior[(qb, row["season"])] = {
                    "TE": earlier["TE"].mean() if "TE" in earlier else 0,
                    "WR": earlier["WR"].mean() if "WR" in earlier else 0,
                    "RB": earlier["RB"].mean() if "RB" in earlier else 0,
                }

    # Player-seasons: receiving yards per game + position + team's primary QB
    stats = pd.read_parquet(RAW / "player_stats_historical.parquet")
    stats = stats[stats["position"].isin(["TE", "WR", "RB"])].copy()
    # team primary QB per team-season (most attempts)
    qbs = pd.read_parquet(RAW / "player_stats_historical.parquet")
    qbs = qbs[qbs["position"] == "QB"]
    prim = qbs.groupby(["recent_team", "season"]).apply(
        lambda d: d.groupby("player_display_name")["attempts"].sum().idxmax()
        if d["attempts"].sum() > 0 else None).to_dict()

    # player-season receiving avg + baseline (prior season)
    rec = stats.groupby(["player_display_name", "position", "recent_team", "season"]).agg(
        games=("week", "count"), rec_yds=("receiving_yards", "sum")).reset_index()
    rec["ypg"] = rec["rec_yds"] / rec["games"]
    rec = rec.sort_values(["player_display_name", "season"])
    rec["prior_ypg"] = rec.groupby("player_display_name")["ypg"].shift(1)
    rec["qb"] = rec.apply(lambda r: prim.get((r["recent_team"], r["season"])), axis=1)
    rec["prior_qb"] = rec.groupby("player_display_name")["qb"].shift(1)
    rec = rec[rec["prior_ypg"].notna() & rec["qb"].notna()]
    rec["new_qb"] = rec["qb"] != rec["prior_qb"]

    # Normalize full QB names ("Kyler Murray") to PBP format ("K.Murray")
    SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
    def full_to_pbp(name):
        if not isinstance(name, str) or not name.strip():
            return None
        parts = [p for p in name.replace(".", "").split() if p.lower().rstrip(".") not in SUFFIX]
        if len(parts) < 2:
            return None
        return f"{parts[0][0]}.{parts[-1]}"

    # For new-QB players, attach the new QB's PRIOR lean to the player's position
    def qb_lean_for(row):
        pl = prior.get((full_to_pbp(row["qb"]), row["season"]))
        if pl is None:
            return np.nan
        return pl.get(row["position"], np.nan)
    nq = rec[rec["new_qb"]].copy()
    nq["qb_prior_lean_to_pos"] = nq.apply(qb_lean_for, axis=1)
    nq = nq[nq["qb_prior_lean_to_pos"].notna()]
    nq["beat_baseline"] = nq["ypg"] > nq["prior_ypg"]
    nq["split"] = np.where(nq["season"] <= 2023, "train", "test")

    print("\n" + "=" * 80)
    print(f"QB CHEMISTRY TEST — {len(nq)} new-QB player-seasons with QB prior lean")
    print("Does a new QB's prior lean to the player's position predict beating baseline?")
    print("=" * 80)

    # Split by whether QB leaned HIGH to that position (top tercile) vs not
    for pos in ["TE", "WR", "RB"]:
        sub = nq[nq["position"] == pos]
        if len(sub) < 20:
            print(f"\n{pos}: only {len(sub)} cases — too few to test.")
            continue
        hi_thresh = sub["qb_prior_lean_to_pos"].quantile(0.66)
        print(f"\n{pos} (n={len(sub)}, high-lean threshold = {hi_thresh:.0%} of completions):")
        for split in ["train", "test"]:
            s = sub[sub["split"] == split]
            hi = s[s["qb_prior_lean_to_pos"] >= hi_thresh]
            lo = s[s["qb_prior_lean_to_pos"] < hi_thresh]
            if len(hi) >= 5 and len(lo) >= 5:
                print(f"  {split}: QB high-lean beat-baseline {hi['beat_baseline'].mean():.0%} (n={len(hi)}) "
                      f"vs low-lean {lo['beat_baseline'].mean():.0%} (n={len(lo)})")

    nq.to_parquet(Path(__file__).parent.parent / "data" / "processed" / "qb_chemistry_graded.parquet", index=False)
    print("\nSaved to data/processed/qb_chemistry_graded.parquet")


if __name__ == "__main__":
    main()
