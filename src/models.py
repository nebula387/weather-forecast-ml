"""Shared model classes — must be importable wherever models are loaded from pickle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

RAIN_THRESHOLD = 0.5  # mm — boundary for classifier target


class TwoStagePrecipModel:
    """Sklearn-compatible two-stage model for sparse precipitation prediction.

    Stage 1: XGBClassifier — will it rain? (handles 87% zero imbalance via scale_pos_weight)
    Stage 2: XGBRegressor  — how much?   (trained only on rainy hours)
    Predict: amount if P(rain) > clf_threshold else 0
    """

    def __init__(self, clf_threshold: float = 0.30):
        self.clf_threshold = clf_threshold
        self.clf: XGBClassifier | None = None
        self.reg: XGBRegressor | None  = None
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "TwoStagePrecipModel":
        import logging
        log = logging.getLogger(__name__)

        y_train_bin = (y_train > RAIN_THRESHOLD).astype(int)
        y_val_bin   = (y_val   > RAIN_THRESHOLD).astype(int)

        neg = int((y_train_bin == 0).sum())
        pos = int((y_train_bin == 1).sum())
        spw = neg / max(pos, 1)
        log.info("Precip clf  neg=%d  pos=%d  scale_pos_weight=%.1f", neg, pos, spw)

        self.clf = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            early_stopping_rounds=50,
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
        )
        self.clf.fit(X_train, y_train_bin, eval_set=[(X_val, y_val_bin)], verbose=False)

        rain_tr = y_train > RAIN_THRESHOLD
        rain_va = y_val   > RAIN_THRESHOLD
        log.info("Precip reg  train_rain=%d  val_rain=%d", rain_tr.sum(), rain_va.sum())

        self.reg = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            early_stopping_rounds=30,
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        )
        self.reg.fit(
            X_train[rain_tr], y_train[rain_tr],
            eval_set=[(X_val[rain_va], y_val[rain_va])],
            verbose=False,
        )

        self.feature_importances_ = (
            0.6 * self.clf.feature_importances_
            + 0.4 * self.reg.feature_importances_
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rain_prob = self.clf.predict_proba(X)[:, 1]
        amount    = self.reg.predict(X).clip(0)
        return np.where(rain_prob > self.clf_threshold, amount, 0.0)
