"""
WALK-FORWARD VALIDATION of Week 1 totals layer weights.

Leave-one-season-out CV: for each season, compute weights on the OTHER 4 seasons,
then predict the held-out season. Aggregates 80 out-of-sample predictions.

Tells us:
  1. Does the weighted model beat the naive "always under" baseline (63.7%) out-of-sample?
  2. Are the layer weight SIGNS stable across folds (real) or do they flip (noise)?
  3. Is conviction calibrated (bigger |net| = higher accuracy)?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
RAW = Path(__file__).parent.parent / "data" / "raw"


def build_coach_change(games):
    rows = []
    for _, g in games.iterrows():
        rows.append({"team": g["home_team"], "season": g["season"], "coach": g.get("home_coach")})
        rows.append({"team": g["away_team"], "season": g["season"], "coach": g.get("away_coach")})
    tc = pd.DataFrame(rows).dropna(subset=["coach"])
    tsc = tc.groupby(["team", "season"])["coach"].agg(lambda x: x.mode().iloc[0])
    changed = {}
    for (team, season), coach in tsc.items():
        prev = tsc.get((team, season - 1))
        changed[(team, season)] = (prev is not None and prev != coach)
    return changed


def featurize():
    games = pd.read_parquet(RAW / "games_historical.parquet")
    cc = build_coach_change(games)
    w = games[games["week"] == 1].copy()
    w["actual_total"] = w["home_score"] + w["away_score"]
    w = w[w["total_line"].notna()].copy()
    w["result_under"] = w["actual_total"] < w["total_line"]
    w = w[w["actual_total"] != w["total_line"]]  # drop pushes
    w["abs_spread"] = w["spread_line"].abs()
    w["hour"] = pd.to_datetime(w["gametime"], format="%H:%M", errors="coerce").dt.hour

    def feats(r):
        tl = r["total_line"]
        total = "low" if tl <= 42 else "mid" if tl <= 47 else "high" if tl <= 49.5 else "very_high"
        roof = "outdoors" if r["roof"] == "outdoors" else "indoor"
        div = "yes" if r["div_game"] else "no"
        sp = "big" if r["abs_spread"] >= 7 else "moderate" if r["abs_spread"] > 3 else "close"
        nc = "yes" if (cc.get((r["home_team"], r["season"])) or cc.get((r["away_team"], r["season"]))) else "no"
        h = r["hour"]
        slot = "early" if h <= 13 else "primetime" if h >= 18 else "afternoon"
        fav = "home" if r["spread_line"] > 0 else "away"
        return pd.Series({"f_total": total, "f_roof": roof, "f_div": div,
                          "f_spread": sp, "f_coach": nc, "f_slot": slot, "f_fav": fav})

    w = pd.concat([w, w.apply(feats, axis=1)], axis=1)
    return w


LAYERS = ["f_total", "f_roof", "f_div", "f_spread", "f_coach", "f_slot", "f_fav"]


def compute_weights(train):
    base = train["result_under"].mean() * 100
    weights = {}
    for layer in LAYERS:
        weights[layer] = {}
        for val, sub in train.groupby(layer):
            n = len(sub)
            up = sub["result_under"].mean() * 100
            rel = min(n / 20.0, 1.0)
            weights[layer][val] = (up - base) * rel
    return weights, base


def score_game(row, weights):
    return sum(weights.get(l, {}).get(row[l], 0.0) for l in LAYERS)


def main():
    w = featurize()
    seasons = sorted(w["season"].unique())
    print("=" * 84)
    print(f"LEAVE-ONE-SEASON-OUT VALIDATION ({len(w)} Week 1 games, {seasons})")
    print("=" * 84)

    # LOSO predictions
    preds = []
    fold_weights = {}
    for s in seasons:
        train = w[w["season"] != s]
        test = w[w["season"] == s]
        weights, base = compute_weights(train)
        fold_weights[s] = weights
        for _, r in test.iterrows():
            net = score_game(r, weights)
            pred_under = net > 0
            preds.append({
                "season": s, "matchup": f"{r['away_team']}@{r['home_team']}",
                "net": net, "pred_under": pred_under,
                "actual_under": r["result_under"],
                "correct": pred_under == r["result_under"],
            })
    p = pd.DataFrame(preds)

    # 1. Overall OOS accuracy vs naive baseline
    model_acc = p["correct"].mean() * 100
    always_under_acc = p["actual_under"].mean() * 100
    print(f"\n[1] OUT-OF-SAMPLE ACCURACY")
    print(f"  Weighted model:      {model_acc:.1f}%  ({p['correct'].sum()}/{len(p)})")
    print(f"  Always-bet-under:    {always_under_acc:.1f}%  (the bar to beat)")
    edge = model_acc - always_under_acc
    print(f"  Model edge over naive: {edge:+.1f} pts", 
          "-> ADDS VALUE" if edge > 2 else "-> NO REAL VALUE" if edge < 1 else "-> marginal")

    # 2. When model says OVER, does it find real overs?
    overs = p[~p["pred_under"]]
    if len(overs):
        over_hit = (~overs["actual_under"]).mean() * 100
        print(f"\n[2] MODEL'S 'OVER' CALLS")
        print(f"  Model predicted OVER on {len(overs)} games; actual OVER rate: {over_hit:.1f}%")
        print(f"  (Needs to beat ~36% — the base over rate — to be useful)")

    # 3. Conviction calibration: accuracy by |net| bucket
    print(f"\n[3] CONVICTION CALIBRATION (does bigger |net| = more accurate?)")
    p["abs_net"] = p["net"].abs()
    for lo, hi, lbl in [(0, 4, "weak (0-4)"), (4, 8, "moderate (4-8)"),
                        (8, 15, "strong (8-15)"), (15, 999, "very strong (15+)")]:
        sub = p[(p["abs_net"] >= lo) & (p["abs_net"] < hi)]
        if len(sub):
            print(f"  |net| {lbl:<18} n={len(sub):<3} accuracy {sub['correct'].mean()*100:.1f}%")

    # 4. Layer weight SIGN stability across folds
    print(f"\n[4] LAYER WEIGHT STABILITY (sign consistent across all 5 folds = trustworthy)")
    all_vals = {}
    for s, weights in fold_weights.items():
        for layer, vals in weights.items():
            for val, wt in vals.items():
                all_vals.setdefault((layer, val), []).append(wt)
    print(f"  {'Layer.value':<22} {'signs':<12} {'avg wt':>7} {'verdict'}")
    print("  " + "-" * 60)
    for (layer, val), wts in sorted(all_vals.items()):
        if len(wts) < 5:
            continue
        pos = sum(1 for x in wts if x > 0)
        neg = sum(1 for x in wts if x < 0)
        avg = np.mean(wts)
        stable = (pos == 5 or neg == 5)
        verdict = "STABLE" if stable else "flips (noise)" if abs(avg) < 3 else "mostly"
        signs = f"{pos}+/{neg}-"
        print(f"  {layer[2:]+'.'+val:<22} {signs:<12} {avg:>+7.1f} {verdict}")

    out = Path(__file__).parent.parent / "data" / "processed" / "layer_weight_validation.parquet"
    p.to_parquet(out, index=False)
    print(f"\nSaved OOS predictions to {out}")


if __name__ == "__main__":
    main()
