"""
Spread Prediction Model.
Predicts the home team margin of victory, then compares to FanDuel spread.
Edge = (predicted margin) - (market spread) → positive edge means bet home.
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class SpreadModel(BaseModel):
    """Predicts home team margin (home_score - away_score)."""

    def __init__(self):
        super().__init__(model_name="spread_model", model_type="regressor")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Feature columns for spread prediction."""
        feature_patterns = [
            # Rolling offensive performance differentials
            "pass_yards_diff_roll3", "pass_yards_diff_roll5",
            "rush_yards_diff_roll3", "rush_yards_diff_roll5",
            # Home team rolling stats
            "home_total_pass_yards_roll5", "home_total_rush_yards_roll5",
            "home_total_pass_tds_roll5", "home_total_rush_tds_roll5",
            "home_total_interceptions_roll5", "home_total_fumbles_lost_roll5",
            # Away team rolling stats
            "away_total_pass_yards_roll5", "away_total_rush_yards_roll5",
            "away_total_pass_tds_roll5", "away_total_rush_tds_roll5",
            "away_total_interceptions_roll5", "away_total_fumbles_lost_roll5",
            # Situational
            "rest_differential", "tz_travel", "is_division_game",
            "game_in_dome", "game_on_grass", "high_altitude",
            "surface_change", "short_rest_home", "short_rest_away",
            # Weather impact
            "pass_suppression", "total_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: home margin (home_score - away_score)."""
        return df["home_margin"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed games with valid features."""
        data = df.copy()
        # Only use regular season and playoff games with final scores
        if "game_type" in data.columns:
            data = data[data["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        if "home_score" in data.columns:
            data = data[data["home_score"].notna()]
        return data.reset_index(drop=True)

    def find_edges(self, predictions: pd.DataFrame, current_lines: pd.DataFrame) -> pd.DataFrame:
        """
        Compare model predictions to FanDuel spread lines.
        
        Args:
            predictions: DataFrame with 'prediction' (predicted home margin)
            current_lines: DataFrame with game_id, spread_line (negative = home favored)
            
        Returns:
            DataFrame with edge calculations and bet recommendations.
        """
        merged = predictions.merge(current_lines, on="game_id", how="inner")

        # Edge calculation
        # spread_line is from home team perspective (e.g., -3 means home favored by 3)
        # Our prediction is home_margin (positive = home wins by that much)
        # Edge: predicted_margin - (-spread_line) = predicted_margin + spread_line
        merged["predicted_margin"] = merged["prediction"]
        merged["edge"] = merged["predicted_margin"] + merged["spread_line"]

        # Positive edge = bet home team to cover
        # Negative edge = bet away team to cover
        merged["bet_side"] = np.where(merged["edge"] > 0, "home", "away")
        merged["edge_abs"] = merged["edge"].abs()

        # Confidence tier
        merged["confidence"] = pd.cut(
            merged["edge_abs"],
            bins=[0, 2, 3, 5, 100],
            labels=["no_bet", "low", "medium", "high"],
        )

        # Unit sizing (Kelly-inspired)
        merged["units"] = np.clip(merged["edge_abs"] / 3, 0, 3).round(1)

        return merged.sort_values("edge_abs", ascending=False)


class SpreadCoverModel(BaseModel):
    """
    Binary classifier: Will the home team cover the spread?
    Useful as a secondary signal alongside the regression model.
    """

    def __init__(self):
        super().__init__(model_name="spread_cover_model", model_type="classifier")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Same features as spread model plus the spread itself."""
        feature_patterns = [
            "spread_line",
            "pass_yards_diff_roll3", "pass_yards_diff_roll5",
            "rush_yards_diff_roll3", "rush_yards_diff_roll5",
            "home_total_pass_yards_roll5", "home_total_rush_yards_roll5",
            "away_total_pass_yards_roll5", "away_total_rush_yards_roll5",
            "home_total_interceptions_roll5", "away_total_interceptions_roll5",
            "rest_differential", "tz_travel", "is_division_game",
            "game_in_dome", "high_altitude", "surface_change",
            "pass_suppression", "total_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: Did home team cover? (1 = yes, 0 = no)."""
        return df["home_cover"].astype(int)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed games with known cover results."""
        data = df.copy()
        if "game_type" in data.columns:
            data = data[data["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        if "home_cover" in data.columns:
            data = data[data["home_cover"].notna()]
        return data.reset_index(drop=True)
