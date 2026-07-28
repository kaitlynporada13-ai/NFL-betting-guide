"""
Passing Yards Prediction Model.
Predicts individual QB passing yards per game.
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class PassingYardsModel(BaseModel):
    """Predicts passing yards for a QB in a given game."""

    def __init__(self):
        super().__init__(model_name="prop_passing_yards", model_type="regressor")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Features for passing yards prediction."""
        feature_patterns = [
            # Player rolling stats
            "passing_yards_roll3", "passing_yards_roll5", "passing_yards_roll10",
            "attempts_roll3", "attempts_roll5",
            "completions_roll3", "completions_roll5",
            "passing_tds_roll5",
            "interceptions_roll5",
            # Consistency
            "passing_yards_std5", "passing_yards_std10",
            "passing_yards_max5", "passing_yards_min5",
            # Situational (from merged game features)
            "game_in_dome", "game_on_grass", "high_altitude",
            "rest_differential", "tz_travel",
            # Weather impact on passing
            "pass_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: actual passing yards."""
        return df["passing_yards"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to QB games with meaningful passing volume."""
        data = df.copy()
        if "position" in data.columns:
            data = data[data["position"] == "QB"]
        if "attempts" in data.columns:
            data = data[data["attempts"] >= 10]  # meaningful starts only
        return data.reset_index(drop=True)

    def find_edges(self, player_preds: pd.DataFrame, prop_lines: pd.DataFrame) -> pd.DataFrame:
        """
        Compare predicted passing yards to FanDuel prop lines.
        
        Args:
            player_preds: DataFrame with player_id/name and prediction
            prop_lines: DataFrame with player_name, line (yards), over_odds, under_odds
        """
        merged = player_preds.merge(
            prop_lines,
            on="player_name",
            how="inner",
            suffixes=("_pred", "_line"),
        )

        merged["edge"] = merged["prediction"] - merged["line"]
        merged["bet_side"] = np.where(merged["edge"] > 0, "over", "under")
        merged["edge_abs"] = merged["edge"].abs()

        # Confidence based on edge relative to typical variance
        merged["confidence"] = pd.cut(
            merged["edge_abs"],
            bins=[0, 15, 25, 40, 500],
            labels=["no_bet", "low", "medium", "high"],
        )

        merged["units"] = np.clip(merged["edge_abs"] / 20, 0, 3).round(1)

        return merged.sort_values("edge_abs", ascending=False)
