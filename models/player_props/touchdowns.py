"""
Anytime Touchdown Scorer Model.
Predicts probability a player scores at least one TD.
Binary classification — output is probability, compare to FanDuel implied odds.
"""

import pandas as pd
import numpy as np

from models.base_model import BaseModel


class AnytimeTDModel(BaseModel):
    """Predicts probability of scoring at least one TD."""

    def __init__(self):
        super().__init__(model_name="prop_anytime_td", model_type="classifier")

    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Features for TD probability."""
        feature_patterns = [
            # Recent TD scoring rate
            "rushing_tds_roll5", "rushing_tds_roll10",
            "receiving_tds_roll5", "receiving_tds_roll10",
            # Volume (more touches = more TD opportunities)
            "carries_roll5", "targets_roll5", "receptions_roll5",
            "rushing_yards_roll5", "receiving_yards_roll5",
            # Red zone proxy (TDs correlate with yards)
            "rushing_yards_roll3", "receiving_yards_roll3",
            # Target share
            "target_share_roll5",
            # Situational
            "game_in_dome", "high_altitude",
            # Weather (dome games = more passing TDs)
            "pass_suppression",
        ]
        return [f for f in feature_patterns if f in df.columns]

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Target: Did player score at least 1 TD? (1 = yes, 0 = no)."""
        total_tds = df.get("rushing_tds", 0).fillna(0) + df.get("receiving_tds", 0).fillna(0)
        return (total_tds > 0).astype(int)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to skill position players with real playing time."""
        data = df.copy()
        if "position" in data.columns:
            data = data[data["position"].isin(["RB", "WR", "TE", "QB"])]
        # Only players with meaningful volume
        if "carries" in data.columns and "targets" in data.columns:
            data = data[(data["carries"].fillna(0) + data["targets"].fillna(0)) >= 3]
        return data.reset_index(drop=True)

    def predict_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get TD probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        data = self.prepare_data(df)
        X = data[self.feature_columns].fillna(0)
        probas = self.model.predict_proba(X)

        return pd.DataFrame({
            "td_probability": probas[:, 1],
            "no_td_probability": probas[:, 0],
        }, index=data.index)

    def find_edges(self, player_preds: pd.DataFrame, prop_lines: pd.DataFrame) -> pd.DataFrame:
        """
        Compare model TD probability to FanDuel anytime TD odds.
        
        FanDuel anytime TD odds are typically in American format (e.g., +120, -150).
        Convert to implied probability and compare.
        """
        merged = player_preds.merge(
            prop_lines,
            on="player_name",
            how="inner",
            suffixes=("_pred", "_line"),
        )

        # Convert American odds to implied probability
        def american_to_implied(odds):
            if pd.isna(odds):
                return np.nan
            if odds < 0:
                return -odds / (-odds + 100)
            else:
                return 100 / (odds + 100)

        if "outcome_price" in merged.columns:
            merged["market_td_prob"] = merged["outcome_price"].apply(american_to_implied)
            merged["edge"] = merged["td_probability"] - merged["market_td_prob"]

            # Only bet when our model says higher probability than market
            merged["bet_side"] = np.where(merged["edge"] > 0, "yes_td", "no_td")
            merged["edge_abs"] = merged["edge"].abs()

            merged["confidence"] = pd.cut(
                merged["edge_abs"],
                bins=[0, 0.03, 0.07, 0.12, 1],
                labels=["no_bet", "low", "medium", "high"],
            )

            merged["units"] = np.clip(merged["edge_abs"] * 15, 0, 3).round(1)

        return merged.sort_values("edge_abs", ascending=False)
