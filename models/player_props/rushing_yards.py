"""
Rushing Yards Prediction Model.
Predicts individual rushing yards per game (RBs and mobile QBs).
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class RushingYardsModel(BaseModel):
    """Predicts rushing yards for a player in a given game."""

    def __init__(self):
        super().__init__(model_name="prop_rushing_yards", model_type="regressor")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Features for rushing yards prediction."""
        feature_patterns = [
            # Player rolling stats
            "rushing_yards_roll3", "rushing_yards_roll5", "rushing_yards_roll10",
            "carries_roll3", "carries_roll5", "carries_roll10",
            "rushing_tds_roll5",
            # Volume indicators
            "targets_roll5",  # receiving involvement affects rush volume
            # Consistency
            "rushing_yards_std5", "rushing_yards_std10",
            "rushing_yards_max5", "rushing_yards_min5",
            # Game script indicators (teams ahead rush more)
            "passing_yards_roll5",  # team passing volume context
            # Goal-line carries (new)
            "gl_carries", "gl_tds",
            # Snap count stability (new)
            "snap_pct_roll3", "snap_pct_roll5",
            # Situational
            "game_in_dome", "game_on_grass", "high_altitude",
            "rest_differential", "is_division_game",
            # Weather (bad weather = more rushing)
            "rush_boost", "pass_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: actual rushing yards."""
        return df["rushing_yards"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to RB/QB games with rushing attempts."""
        data = df.copy()
        if "position" in data.columns:
            data = data[data["position"].isin(["RB", "QB"])]
        if "carries" in data.columns:
            data = data[data["carries"] >= 3]  # at least some carries
        return data.reset_index(drop=True)

    def find_edges(self, player_preds: pd.DataFrame, prop_lines: pd.DataFrame) -> pd.DataFrame:
        """Compare predicted rushing yards to FanDuel prop lines."""
        merged = player_preds.merge(
            prop_lines,
            on="player_name",
            how="inner",
            suffixes=("_pred", "_line"),
        )

        merged["edge"] = merged["prediction"] - merged["line"]
        merged["bet_side"] = np.where(merged["edge"] > 0, "over", "under")
        merged["edge_abs"] = merged["edge"].abs()

        merged["confidence"] = pd.cut(
            merged["edge_abs"],
            bins=[0, 8, 15, 25, 500],
            labels=["no_bet", "low", "medium", "high"],
        )

        merged["units"] = np.clip(merged["edge_abs"] / 12, 0, 3).round(1)

        return merged.sort_values("edge_abs", ascending=False)
