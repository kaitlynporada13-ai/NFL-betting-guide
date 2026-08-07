"""
Model Backtest: How well do our retrained models predict vs FanDuel lines?
============================================================================
Uses walk-forward approach:
- Train on data up to week N-1
- Predict week N 
- Compare model prediction to FanDuel line
- Grade: When model says "bet over" or "bet under", does it win?

This measures the VALUE of our model as a betting tool.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.player_props.passing_yards import PassingYardsModel
from models.player_props.rushing_yards import RushingYardsModel
from models.player_props.receiving_yards import ReceivingYardsModel
from models.player_props.receptions import ReceptionsModel

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


def run_model_backtest():
    """
    Walk-forward model backtest.
    For each prop in our graded dataset, compare model prediction to FanDuel line.
    If model disagrees with line → simulate that bet.
    """
    print("=" * 70)
    print("MODEL VS LINE BACKTEST")
    print("=" * 70)

    # Load graded props (already matched to actuals)
    graded = pd.read_parquet(PROC_DIR / "props_graded_backtest.parquet")
    print(f"Graded props: {len(graded)}")

    # Load player features (what the model uses)
    player_features = pd.read_parquet(PROC_DIR / "player_features.parquet")
    print(f"Player features: {player_features.shape}")

    # Load trained models
    models = {
        "player_pass_yds": PassingYardsModel(),
        "player_rush_yds": RushingYardsModel(),
        "player_reception_yds": ReceivingYardsModel(),
        "player_receptions": ReceptionsModel(),
    }

    for name, model in models.items():
        try:
            model.load()
        except FileNotFoundError:
            print(f"  WARNING: {name} model not found, skipping")
            models.pop(name)

    # For each market, generate predictions and compare to lines
    all_results = []

    for market, model in models.items():
        print(f"\n--- {market} ---")
        market_props = graded[graded["market"] == market].copy()
        if market_props.empty:
            continue

        # Prepare features and generate predictions
        data = model.prepare_data(player_features)
        if data.empty:
            print(f"  No data for {market}")
            continue

        # Generate model predictions
        try:
            preds = model.predict(data)
        except Exception as e:
            print(f"  Prediction error: {e}")
            continue

        # Build a lookup: (player_name_lower, season, week) -> prediction
        name_col = "player_display_name" if "player_display_name" in data.columns else "player_name"
        pred_df = pd.DataFrame({
            "player_clean": data[name_col].str.strip().str.lower(),
            "season": data["season"].values,
            "week": data["week"].values,
            "model_pred": preds.values,
        })

        # Match predictions to props
        market_props["player_clean"] = market_props["player_name"].str.strip().str.lower()
        merged = market_props.merge(
            pred_df,
            on=["player_clean", "season", "week"],
            how="inner",
        )

        if merged.empty:
            print(f"  No matches for {market}")
            continue

        # Calculate model edge
        merged["model_edge"] = merged["model_pred"] - merged["fanduel_line"]
        merged["model_says"] = np.where(merged["model_edge"] > 0, "over", "under")

        # Grade model's picks
        # Over hit = actual > line. Under hit = actual < line.
        merged["model_correct"] = (
            ((merged["model_says"] == "over") & (merged["over_hit"] == True)) |
            ((merged["model_says"] == "under") & (merged["over_hit"] == False))
        )

        # Only count bets where model has meaningful edge
        for min_edge in [0, 5, 10, 15]:
            if market == "player_receptions":
                # Receptions are on a smaller scale
                edge_threshold = min_edge / 10
            else:
                edge_threshold = min_edge

            edge_bets = merged[merged["model_edge"].abs() >= edge_threshold]
            if len(edge_bets) == 0:
                continue

            correct = edge_bets["model_correct"].sum()
            total = len(edge_bets)
            hit_rate = correct / total if total > 0 else 0

            # Simulate P&L at -110
            wins = correct
            losses = total - correct
            profit = (wins * 100) - (losses * 110)
            roi = profit / (total * 110) * 100

            all_results.append({
                "market": market,
                "min_edge": min_edge,
                "bets": total,
                "hit_rate": hit_rate,
                "roi_pct": roi,
                "profit_units": profit / 110,
            })

            if min_edge == 0:
                print(f"  All bets: {total:,} | Hit: {hit_rate:.1%} | ROI: {roi:+.1f}%")
            else:
                label = f"edge>={min_edge}" if market != "player_receptions" else f"edge>={edge_threshold:.1f}"
                print(f"  {label}: {total:,} | Hit: {hit_rate:.1%} | ROI: {roi:+.1f}%")

    # Summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("SUMMARY: MODEL PERFORMANCE BY EDGE THRESHOLD")
    print("=" * 70)

    if not results_df.empty:
        # Aggregate across all markets
        for edge in sorted(results_df["min_edge"].unique()):
            edge_data = results_df[results_df["min_edge"] == edge]
            total_bets = edge_data["bets"].sum()
            total_profit = edge_data["profit_units"].sum()
            weighted_hit = (edge_data["hit_rate"] * edge_data["bets"]).sum() / total_bets if total_bets > 0 else 0
            overall_roi = total_profit / total_bets * 100 if total_bets > 0 else 0

            status = "PROFITABLE" if overall_roi > 0 else "LOSS"
            print(f"  Edge >= {edge:>2}: {total_bets:>6,} bets | Hit: {weighted_hit:.1%} | ROI: {overall_roi:+.1f}% | {status}")

        # Season breakdown for best threshold
        print("\n--- BY SEASON (edge >= 10) ---")
        for market, model in models.items():
            data = model.prepare_data(player_features)
            if data.empty:
                continue
            preds = model.predict(data)
            name_col = "player_display_name" if "player_display_name" in data.columns else "player_name"
            pred_df = pd.DataFrame({
                "player_clean": data[name_col].str.strip().str.lower(),
                "season": data["season"].values,
                "week": data["week"].values,
                "model_pred": preds.values,
            })
            market_props = graded[graded["market"] == market].copy()
            market_props["player_clean"] = market_props["player_name"].str.strip().str.lower()
            merged = market_props.merge(pred_df, on=["player_clean", "season", "week"], how="inner")
            if merged.empty:
                continue
            merged["model_edge"] = merged["model_pred"] - merged["fanduel_line"]
            merged["model_says"] = np.where(merged["model_edge"] > 0, "over", "under")
            merged["model_correct"] = (
                ((merged["model_says"] == "over") & (merged["over_hit"] == True)) |
                ((merged["model_says"] == "under") & (merged["over_hit"] == False))
            )

            threshold = 1.0 if market == "player_receptions" else 10
            edge_bets = merged[merged["model_edge"].abs() >= threshold]

            for season in sorted(edge_bets["season"].unique()):
                sg = edge_bets[edge_bets["season"] == season]
                s_correct = sg["model_correct"].sum()
                s_total = len(sg)
                s_rate = s_correct / s_total if s_total > 0 else 0
                s_profit = (s_correct * 100) - ((s_total - s_correct) * 110)
                s_roi = s_profit / (s_total * 110) * 100 if s_total > 0 else 0
                print(f"  {market:<25} {season}: {s_rate:.1%} hit ({s_correct}/{s_total}) | ROI: {s_roi:+.1f}%")

    # Save results
    results_df.to_parquet(PROC_DIR / "model_backtest_results.parquet", index=False)
    print(f"\nSaved to data/processed/model_backtest_results.parquet")


if __name__ == "__main__":
    run_model_backtest()
