"""
Pipeline Orchestrator.
Ties together: data ingestion → feature engineering → model predictions → output.
Designed to be run on a schedule (GitHub Actions) or manually.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

from pipeline.config_loader import load_settings, get_data_dir
from pipeline.ingest_stats import pull_game_data, pull_player_stats, pull_team_stats
from pipeline.ingest_odds import pull_game_odds, pull_all_props_for_week
from pipeline.ingest_injuries import pull_all_injuries, get_injury_impact_features, pull_all_news
from pipeline.ingest_hierarchy import pull_depth_charts, build_coaching_features
from pipeline.feature_engineering import build_game_features, build_player_prop_features
from pipeline.human_notes import load_weekly_notes, apply_notes_to_predictions

from models.spread_model import SpreadModel, SpreadCoverModel
from models.totals_model import TotalsModel
from models.moneyline_model import MoneylineModel
from models.player_props import ALL_PROP_MODELS


def train_all_models():
    """
    Train all models on historical data.
    Run this once before the season, then periodically retrain.
    """
    settings = load_settings()
    processed_dir = get_data_dir("processed")

    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS")
    print("=" * 70)

    # Load processed features
    game_features_path = processed_dir / "game_features.parquet"
    player_features_path = processed_dir / "player_features.parquet"

    if not game_features_path.exists() or not player_features_path.exists():
        print("ERROR: Processed features not found. Run feature_engineering.save_features() first.")
        print("  Run: python -m pipeline.feature_engineering")
        return

    game_features = pd.read_parquet(game_features_path)
    player_features = pd.read_parquet(player_features_path)

    # ---- Game-level models ----
    print("\n--- GAME-LEVEL MODELS ---")

    # Spread model
    spread = SpreadModel()
    spread.train(game_features)
    spread.save()

    # Spread cover classifier
    spread_cover = SpreadCoverModel()
    spread_cover.train(game_features)
    spread_cover.save()

    # Totals model
    totals = TotalsModel()
    totals.train(game_features)
    totals.save()

    # Moneyline model
    ml = MoneylineModel()
    ml.train(game_features)
    ml.save()

    # ---- Player prop models ----
    print("\n--- PLAYER PROP MODELS ---")

    for prop_name, ModelClass in ALL_PROP_MODELS.items():
        model = ModelClass()
        model.train(player_features)
        model.save()

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED AND SAVED")
    print("=" * 70)


def run_weekly_predictions():
    """
    Run the full prediction pipeline for the current week.
    1. Pull fresh odds from FanDuel
    2. Pull latest stats
    3. Build features for upcoming games
    4. Generate predictions
    5. Compare to lines → find edges
    6. Apply human notes
    7. Save output for dashboard
    """
    settings = load_settings()
    raw_dir = get_data_dir("raw")
    output_dir = get_data_dir("processed")
    timestamp = datetime.utcnow().strftime("%Y%m%d")

    print("\n" + "=" * 70)
    print(f"WEEKLY PREDICTION RUN - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # --- Step 1: Pull fresh data ---
    print("\n[1/6] Pulling fresh data...")

    try:
        current_odds = pull_game_odds()
        current_odds.to_parquet(raw_dir / "odds_games_latest.parquet", index=False)
        print(f"  ✓ Game odds: {current_odds['game_id'].nunique()} games")
    except Exception as e:
        print(f"  ✗ Could not pull odds: {e}")
        current_odds = pd.DataFrame()

    # Pull injury reports and news
    try:
        injuries = pull_all_injuries()
        if not injuries.empty:
            injuries.to_parquet(raw_dir / "injuries_current.parquet", index=False)
            injury_impact = get_injury_impact_features(injuries)
            injury_impact.to_parquet(raw_dir / "injury_impact_features.parquet", index=False)
            print(f"  ✓ Injuries: {len(injuries)} players across {injuries['team'].nunique()} teams")
    except Exception as e:
        print(f"  ✗ Could not pull injuries: {e}")

    try:
        news = pull_all_news()
        if not news.empty:
            news.to_parquet(raw_dir / "news_latest.parquet", index=False)
            print(f"  ✓ News: {len(news)} articles")
    except Exception as e:
        print(f"  ✗ Could not pull news: {e}")

    # Pull latest season stats
    current_season = settings["data"]["current_season"]
    try:
        current_games = pull_game_data([current_season])
        current_team_stats = pull_team_stats([current_season])
        current_player_stats = pull_player_stats([current_season])
        print(f"  ✓ Current season stats pulled")
    except Exception as e:
        print(f"  ✗ Could not pull current stats: {e}")
        return

    # --- Step 2: Build features for current week ---
    print("\n[2/6] Building features...")
    game_features = build_game_features(current_games, current_team_stats)
    player_features = build_player_prop_features(
        current_player_stats, current_games, current_team_stats
    )

    # --- Step 3: Load trained models ---
    print("\n[3/6] Loading trained models...")
    spread = SpreadModel()
    totals = TotalsModel()
    moneyline = MoneylineModel()

    try:
        spread.load()
        totals.load()
        moneyline.load()
    except FileNotFoundError:
        print("  ERROR: Models not trained yet. Run train_all_models() first.")
        return

    prop_models = {}
    for prop_name, ModelClass in ALL_PROP_MODELS.items():
        model = ModelClass()
        try:
            model.load()
            prop_models[prop_name] = model
        except FileNotFoundError:
            print(f"  WARNING: {prop_name} model not found, skipping")

    # --- Step 4: Generate predictions ---
    print("\n[4/6] Generating predictions...")

    # Get the most recent week's data for predictions
    # (In production, this would be the upcoming week)
    results = {}

    # Spread predictions
    try:
        spread_preds = spread.predict_with_confidence(game_features)
        spread_preds["game_id"] = game_features.loc[spread_preds.index, "game_id"].values
        results["spreads"] = spread_preds
        print(f"  ✓ Spread predictions: {len(spread_preds)} games")
    except Exception as e:
        print(f"  ✗ Spread predictions failed: {e}")

    # Totals predictions
    try:
        totals_preds = totals.predict_with_confidence(game_features)
        totals_preds["game_id"] = game_features.loc[totals_preds.index, "game_id"].values
        results["totals"] = totals_preds
        print(f"  ✓ Totals predictions: {len(totals_preds)} games")
    except Exception as e:
        print(f"  ✗ Totals predictions failed: {e}")

    # Player prop predictions
    for prop_name, model in prop_models.items():
        try:
            prop_preds = model.predict_with_confidence(player_features)
            if not player_features.empty:
                for col in ["player_name", "player_display_name", "position", "recent_team"]:
                    if col in player_features.columns:
                        prop_preds[col] = player_features.loc[prop_preds.index, col].values
            results[prop_name] = prop_preds
            print(f"  ✓ {prop_name}: {len(prop_preds)} predictions")
        except Exception as e:
            print(f"  ✗ {prop_name} failed: {e}")

    # --- Step 5: Find edges ---
    print("\n[5/6] Finding edges vs FanDuel lines...")

    best_bets = []

    if "spreads" in results and not current_odds.empty:
        try:
            spread_lines = current_odds[current_odds["market"] == "spreads"]
            edges = spread.find_edges(results["spreads"], spread_lines)
            edges_with_value = edges[edges["confidence"] != "no_bet"]
            results["spread_edges"] = edges_with_value
            best_bets.append(edges_with_value.assign(bet_type="spread"))
            print(f"  ✓ Spread edges: {len(edges_with_value)} actionable")
        except Exception as e:
            print(f"  ✗ Spread edge detection: {e}")

    if "totals" in results and not current_odds.empty:
        try:
            total_lines = current_odds[current_odds["market"] == "totals"]
            edges = totals.find_edges(results["totals"], total_lines)
            edges_with_value = edges[edges["confidence"] != "no_bet"]
            results["totals_edges"] = edges_with_value
            best_bets.append(edges_with_value.assign(bet_type="total"))
            print(f"  ✓ Totals edges: {len(edges_with_value)} actionable")
        except Exception as e:
            print(f"  ✗ Totals edge detection: {e}")

    # --- Step 6: Apply human notes ---
    print("\n[6/6] Applying human notes...")
    notes = load_weekly_notes()
    if notes:
        for key, preds in results.items():
            if isinstance(preds, pd.DataFrame):
                results[key] = apply_notes_to_predictions(preds, notes)
        print(f"  ✓ Applied notes from {len(notes.get('player_notes', []))} player entries")
    else:
        print("  No human notes found for this week")

    # --- Save outputs ---
    print("\n--- SAVING OUTPUTS ---")

    # Combine best bets
    if best_bets:
        all_bets = pd.concat(best_bets, ignore_index=True)
        all_bets_sorted = all_bets.sort_values("edge_abs", ascending=False)
        top_bets = all_bets_sorted.head(settings["output"]["top_bets_count"])
        top_bets.to_parquet(output_dir / "best_bets_latest.parquet", index=False)
        print(f"  ✓ Best bets: {len(top_bets)} saved")

    # Save all predictions
    for key, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_parquet(output_dir / f"{key}_latest.parquet", index=False)

    print("\n" + "=" * 70)
    print("PREDICTION RUN COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_all_models()
    else:
        run_weekly_predictions()
