"""
Moneyline Prediction Model.
Predicts win probability, then converts to implied odds for edge detection.
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class MoneylineModel(BaseModel):
    """Predicts home team win probability."""

    def __init__(self):
        super().__init__(model_name="moneyline_model", model_type="classifier")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Feature columns for win probability."""
        feature_patterns = [
            # Performance differentials
            "pass_yards_diff_roll3", "pass_yards_diff_roll5",
            "rush_yards_diff_roll3", "rush_yards_diff_roll5",
            # Home team stats
            "home_total_pass_yards_roll5", "home_total_rush_yards_roll5",
            "home_total_pass_tds_roll5", "home_total_rush_tds_roll5",
            "home_total_interceptions_roll5", "home_total_fumbles_lost_roll5",
            # Away team stats
            "away_total_pass_yards_roll5", "away_total_rush_yards_roll5",
            "away_total_pass_tds_roll5", "away_total_rush_tds_roll5",
            "away_total_interceptions_roll5", "away_total_fumbles_lost_roll5",
            # Situational
            "rest_differential", "tz_travel", "is_division_game",
            "game_in_dome", "game_on_grass", "high_altitude",
            "surface_change", "short_rest_home", "short_rest_away",
            # Weather
            "pass_suppression", "total_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: Did home team win? (1 = yes, 0 = no)."""
        return (df["home_margin"] > 0).astype(int)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed games (exclude ties)."""
        data = df.copy()
        if "game_type" in data.columns:
            data = data[data["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        if "home_margin" in data.columns:
            data = data[data["home_margin"].notna() & (data["home_margin"] != 0)]
        return data.reset_index(drop=True)

    def predict_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get win probability predictions (not just binary).
        Uses predict_proba for calibrated probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        data = self.prepare_data(df)
        X = data[self.feature_columns].fillna(0)

        probas = self.model.predict_proba(X)
        return pd.DataFrame({
            "home_win_prob": probas[:, 1],
            "away_win_prob": probas[:, 0],
        }, index=data.index)

    def find_edges(self, df: pd.DataFrame, current_lines: pd.DataFrame) -> pd.DataFrame:
        """
        Compare model win probability to implied probability from moneyline odds.
        
        American odds conversion:
            Favorite (negative): implied_prob = -odds / (-odds + 100)
            Underdog (positive): implied_prob = 100 / (odds + 100)
        """
        # Get model probabilities
        probas = self.predict_probability(df)
        data = self.prepare_data(df)
        result = data[["game_id", "home_team", "away_team"]].copy()
        result["model_home_prob"] = probas["home_win_prob"].values
        result["model_away_prob"] = probas["away_win_prob"].values

        # Merge with current lines
        merged = result.merge(current_lines, on="game_id", how="inner")

        # Convert American odds to implied probability
        def american_to_implied(odds):
            if odds < 0:
                return -odds / (-odds + 100)
            else:
                return 100 / (odds + 100)

        if "home_moneyline" in merged.columns:
            merged["market_home_prob"] = merged["home_moneyline"].apply(american_to_implied)
            merged["market_away_prob"] = merged["away_moneyline"].apply(american_to_implied)

            # Edge = model probability - market implied probability
            merged["home_edge"] = merged["model_home_prob"] - merged["market_home_prob"]
            merged["away_edge"] = merged["model_away_prob"] - merged["market_away_prob"]

            # Best bet side
            merged["bet_side"] = np.where(
                merged["home_edge"] > merged["away_edge"], "home", "away"
            )
            merged["edge"] = np.maximum(merged["home_edge"], merged["away_edge"])

            # Kelly criterion for unit sizing
            # Kelly fraction = (edge * odds) / odds = edge / (1 - implied_prob)
            # We use quarter-Kelly for safety
            merged["kelly_fraction"] = merged["edge"] / (1 - merged["market_home_prob"])
            merged["units"] = np.clip(merged["kelly_fraction"] * 4, 0, 3).round(1)

            merged["confidence"] = pd.cut(
                merged["edge"],
                bins=[-1, 0.02, 0.05, 0.10, 1],
                labels=["no_bet", "low", "medium", "high"],
            )

        return merged.sort_values("edge", ascending=False)
