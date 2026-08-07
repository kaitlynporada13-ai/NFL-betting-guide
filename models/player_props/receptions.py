"""
Receptions Prediction Model.
Predicts individual receptions per game.
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class ReceptionsModel(BaseModel):
    """Predicts receptions for a player in a given game."""

    def __init__(self):
        super().__init__(model_name="prop_receptions", model_type="regressor")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Features for receptions prediction."""
        feature_patterns = [
            # Player rolling stats
            "receptions_roll3", "receptions_roll5", "receptions_roll10",
            "targets_roll3", "targets_roll5", "targets_roll10",
            "receiving_yards_roll5",
            # Target share
            "target_share_roll3", "target_share_roll5",
            # Consistency
            "receptions_std5", "receptions_std10",
            "receptions_max5", "receptions_min5",
            # Air yards / route type (new)
            "avg_air_yards", "avg_yac", "deep_rate",
            # Snap count stability (new)
            "snap_pct_roll3", "snap_pct_roll5",
            # Situational
            "game_in_dome", "rest_differential", "is_division_game",
            # Weather
            "pass_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: actual receptions."""
        return df["receptions"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to pass catchers."""
        data = df.copy()
        if "position" in data.columns:
            data = data[data["position"].isin(["WR", "TE", "RB"])]
        if "targets" in data.columns:
            data = data[data["targets"] >= 1]
        return data.reset_index(drop=True)

    def find_edges(self, player_preds: pd.DataFrame, prop_lines: pd.DataFrame) -> pd.DataFrame:
        """Compare predicted receptions to FanDuel prop lines."""
        merged = player_preds.merge(
            prop_lines,
            on="player_name",
            how="inner",
            suffixes=("_pred", "_line"),
        )

        merged["edge"] = merged["prediction"] - merged["line"]
        merged["bet_side"] = np.where(merged["edge"] > 0, "over", "under")
        merged["edge_abs"] = merged["edge"].abs()

        # Receptions are lower variance so smaller edges matter
        merged["confidence"] = pd.cut(
            merged["edge_abs"],
            bins=[0, 0.8, 1.5, 2.5, 100],
            labels=["no_bet", "low", "medium", "high"],
        )

        merged["units"] = np.clip(merged["edge_abs"] / 1.5, 0, 3).round(1)

        return merged.sort_values("edge_abs", ascending=False)
