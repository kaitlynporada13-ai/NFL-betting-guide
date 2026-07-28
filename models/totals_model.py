"""
Totals (Over/Under) Prediction Model.
Predicts total combined points, then compares to FanDuel total line.
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class TotalsModel(BaseModel):
    """Predicts total combined points (home + away)."""

    def __init__(self):
        super().__init__(model_name="totals_model", model_type="regressor")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Feature columns for totals prediction."""
        feature_patterns = [
            # Combined offensive firepower
            "home_total_pass_yards_roll5", "home_total_rush_yards_roll5",
            "home_total_pass_tds_roll5", "home_total_rush_tds_roll5",
            "away_total_pass_yards_roll5", "away_total_rush_yards_roll5",
            "away_total_pass_tds_roll5", "away_total_rush_tds_roll5",
            # Turnovers (create scoring opportunities)
            "home_total_interceptions_roll5", "home_total_fumbles_lost_roll5",
            "away_total_interceptions_roll5", "away_total_fumbles_lost_roll5",
            # Pace (more plays = more points)
            "home_total_attempts_roll5", "home_total_carries_roll5",
            "away_total_attempts_roll5", "away_total_carries_roll5",
            # Situational
            "game_in_dome", "game_on_grass", "high_altitude",
            "is_division_game",
            # Weather (huge impact on totals)
            "pass_suppression", "total_suppression",
            "wind_impact", "cold_game", "rain_likely",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: total points scored."""
        return df["total_points"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed games."""
        data = df.copy()
        if "game_type" in data.columns:
            data = data[data["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        if "total_points" in data.columns:
            data = data[data["total_points"].notna()]
        return data.reset_index(drop=True)

    def find_edges(self, predictions: pd.DataFrame, current_lines: pd.DataFrame) -> pd.DataFrame:
        """
        Compare predicted total to FanDuel total line.
        
        Positive edge = bet OVER
        Negative edge = bet UNDER
        """
        merged = predictions.merge(current_lines, on="game_id", how="inner")

        merged["predicted_total"] = merged["prediction"]
        merged["edge"] = merged["predicted_total"] - merged["total_line"]

        merged["bet_side"] = np.where(merged["edge"] > 0, "over", "under")
        merged["edge_abs"] = merged["edge"].abs()

        merged["confidence"] = pd.cut(
            merged["edge_abs"],
            bins=[0, 2, 3.5, 5, 100],
            labels=["no_bet", "low", "medium", "high"],
        )

        merged["units"] = np.clip(merged["edge_abs"] / 3, 0, 3).round(1)

        return merged.sort_values("edge_abs", ascending=False)


class OverUnderClassifier(BaseModel):
    """Binary classifier: Will the game go over the posted total?"""

    def __init__(self):
        super().__init__(model_name="over_under_classifier", model_type="classifier")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Features including the total line itself."""
        feature_patterns = [
            "total_line",
            "home_total_pass_yards_roll5", "home_total_rush_yards_roll5",
            "home_total_pass_tds_roll5", "away_total_pass_yards_roll5",
            "away_total_rush_yards_roll5", "away_total_pass_tds_roll5",
            "home_total_interceptions_roll5", "away_total_interceptions_roll5",
            "game_in_dome", "high_altitude", "is_division_game",
            "pass_suppression", "total_suppression",
            "wind_impact", "cold_game", "rain_likely",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: Did the game go over? (1 = yes, 0 = no)."""
        return df["over_hit"].astype(int)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed games."""
        data = df.copy()
        if "game_type" in data.columns:
            data = data[data["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        if "over_hit" in data.columns:
            data = data[data["over_hit"].notna()]
        return data.reset_index(drop=True)
