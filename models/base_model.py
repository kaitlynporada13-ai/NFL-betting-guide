"""
Base model class for all betting models.
Provides shared training, evaluation, and persistence logic.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, XGBClassifier

from pipeline.config_loader import get_project_root


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, model_name: str, model_type: str = "regressor"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.metrics = {}
        self.trained_at = None
        self.model_dir = get_project_root() / "models" / "trained"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_features(self, df: pd.DataFrame) -> list[str]:
        """Return the list of feature column names for this model."""
        pass

    @abstractmethod
    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Return the target variable for training."""
        pass

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and filter data for this specific model."""
        pass

    def build_model(self, **kwargs) -> XGBRegressor | XGBClassifier:
        """Create the XGBoost model with default hyperparameters."""
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(kwargs)

        if self.model_type == "classifier":
            return XGBClassifier(**default_params, eval_metric="logloss")
        return XGBRegressor(**default_params, eval_metric="rmse")

    def train(self, df: pd.DataFrame, **model_kwargs) -> dict:
        """
        Train the model with time-series cross-validation.
        
        Returns dict of evaluation metrics.
        """
        print(f"\n{'='*60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'='*60}")

        # Prepare data
        data = self.prepare_data(df)
        if data.empty:
            print("  ERROR: No data after preparation")
            return {}

        self.feature_columns = self.get_features(data)
        target = self.get_target(data)

        # Filter to rows where we have the target (features can have NaN — filled below)
        valid_mask = target.notna()
        X = data.loc[valid_mask, self.feature_columns].copy()
        y = target[valid_mask]

        # Coerce object columns to numeric (booleans from merges can become objects)
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            elif X[col].dtype == "bool":
                X[col] = X[col].astype(int)
        X = X.fillna(0)

        print(f"  Data: {len(X)} samples, {len(self.feature_columns)} features")

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.build_model(**model_kwargs)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            preds = model.predict(X_val)

            if self.model_type == "regressor":
                mae = mean_absolute_error(y_val, preds)
                cv_scores.append(mae)
            else:
                accuracy = (preds.round() == y_val).mean()
                cv_scores.append(accuracy)

        # Train final model on all data
        self.model = self.build_model(**model_kwargs)
        self.model.fit(X, y, verbose=False)
        self.trained_at = datetime.utcnow()

        # Evaluation metrics
        if self.model_type == "regressor":
            self.metrics = {
                "cv_mae_mean": np.mean(cv_scores),
                "cv_mae_std": np.std(cv_scores),
                "r2_score": r2_score(y, self.model.predict(X)),
                "n_samples": len(X),
                "n_features": len(self.feature_columns),
            }
            print(f"  CV MAE: {self.metrics['cv_mae_mean']:.3f} (+/- {self.metrics['cv_mae_std']:.3f})")
            print(f"  R² (in-sample): {self.metrics['r2_score']:.3f}")
        else:
            self.metrics = {
                "cv_accuracy_mean": np.mean(cv_scores),
                "cv_accuracy_std": np.std(cv_scores),
                "n_samples": len(X),
                "n_features": len(self.feature_columns),
            }
            print(f"  CV Accuracy: {self.metrics['cv_accuracy_mean']:.3f} (+/- {self.metrics['cv_accuracy_std']:.3f})")

        # Feature importance
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(ascending=False)
        print(f"\n  Top 10 features:")
        for feat, imp in importances.head(10).items():
            print(f"    {feat}: {imp:.4f}")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data."""
        if self.model is None:
            raise ValueError(f"Model '{self.model_name}' has not been trained. Call train() first.")

        data = self.prepare_data(df)
        X = data[self.feature_columns].copy()

        # Handle missing features gracefully
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            print(f"  WARNING: Missing features filled with 0: {missing}")
            for col in missing:
                X[col] = 0

        # Coerce types
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            elif X[col].dtype == "bool":
                X[col] = X[col].astype(int)
        X = X.fillna(0)

        predictions = self.model.predict(X)
        return pd.Series(predictions, index=data.index)

    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals.
        Uses the model's training error to estimate uncertainty.
        """
        predictions = self.predict(df)
        data = self.prepare_data(df)

        result = pd.DataFrame({
            "prediction": predictions,
        }, index=data.index)

        if self.model_type == "regressor" and "cv_mae_mean" in self.metrics:
            mae = self.metrics["cv_mae_mean"]
            result["pred_low"] = predictions - (1.5 * mae)
            result["pred_high"] = predictions + (1.5 * mae)
            result["confidence_range"] = 2 * 1.5 * mae

        return result

    def save(self):
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        model_path = self.model_dir / f"{self.model_name}.joblib"
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "trained_at": str(self.trained_at),
        }

        joblib.dump({"model": self.model, "metadata": metadata}, model_path)
        print(f"  Saved model to {model_path}")

    def load(self):
        """Load a previously trained model."""
        model_path = self.model_dir / f"{self.model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")

        data = joblib.load(model_path)
        self.model = data["model"]
        metadata = data["metadata"]
        self.feature_columns = metadata["feature_columns"]
        self.metrics = metadata["metrics"]
        self.trained_at = metadata["trained_at"]
        print(f"  Loaded model '{self.model_name}' (trained: {self.trained_at})")
