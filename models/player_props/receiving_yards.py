"""
Receiving Yards Prediction Model.
Predicts individual receiving yards per game (WR/TE/RB).
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class ReceivingYardsModel(BaseModel):
    """Predicts receiving yards for a player in a given game."""

    def __init__(self):
        super().__init__(model_name="prop_receiving_yards", model_type="regressor")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Features for receiving yards prediction."""
        feature_patterns = [
            # Player rolling stats
            "receiving_yards_roll3", "receiving_yards_roll5", "receiving_yards_roll10",
            "targets_roll3", "targets_roll5", "targets_roll10",
            "receptions_roll3", "receptions_roll5",
            "receiving_tds_roll5",
            # Target share (% of team targets)
            "target_share_roll3", "target_share_roll5",
            # Consistency
            "receiving_yards_std5", "receiving_yards_std10",
            "receiving_yards_max5", "receiving_yards_min5",
            # Air yards profile (new)
            "avg_air_yards", "avg_yac", "deep_rate",
            # Red zone involvement (new)
            "rz_targets", "rz_target_share",
            # Snap count stability (new)
            "snap_pct_roll3", "snap_pct_roll5",
            # Situational
            "game_in_dome", "game_on_grass", "is_division_game",
            "rest_differential", "tz_travel",
            # Weather (wind/rain suppresses deep passing)
            "pass_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: actual receiving yards."""
        return df["receiving_yards"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to pass catchers with meaningful targets."""
        data = df.copy()
        if "position" in data.columns:
            data = data[data["position"].isin(["WR", "TE", "RB"])]
        if "targets" in data.columns:
            data = data[data["targets"] >= 1]  # at least targeted
        return data.reset_index(drop=True)

    def find_edges(self, player_preds: pd.DataFrame, prop_lines: pd.DataFrame) -> pd.DataFrame:
        """Compare predicted receiving yards to FanDuel prop lines."""
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
