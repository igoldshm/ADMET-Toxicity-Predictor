"""
Baseline Models
===============
Classical ML baselines trained on ECFP/MACCS fingerprints.
Used to contextualise GNN performance — a GNN that doesn't substantially
outperform Random Forest on ECFP is not adding value.

Baselines included:
    • Random Forest        — strong non-linear baseline, interpretable feature importance
    • XGBoost              — gradient-boosted trees, often best fingerprint performer
    • Logistic Regression  — linear baseline; low AUC reveals non-linear structure
    • Naive Bayes          — fast probabilistic baseline; common in cheminformatics
    • Majority Classifier  — sanity-check floor (always predicts dominant class)

Multi-task strategy: OneVsRest wrapper per task, fit individually.
Missing labels (w=0 in Tox21) are masked before fitting each task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


@dataclass
class BaselineResults:
    model_name: str
    task_aucs: Dict[str, float] = field(default_factory=dict)
    task_f1s:  Dict[str, float] = field(default_factory=dict)

    @property
    def mean_auc(self) -> float:
        return float(np.mean(list(self.task_aucs.values()))) if self.task_aucs else 0.0

    @property
    def mean_f1(self) -> float:
        return float(np.mean(list(self.task_f1s.values()))) if self.task_f1s else 0.0

    def to_series(self) -> pd.Series:
        return pd.Series({
            "model":    self.model_name,
            "mean_AUC": round(self.mean_auc, 4),
            "mean_F1":  round(self.mean_f1, 4),
        })


class BaselineEvaluator:
    """
    Trains and evaluates a suite of classical ML baselines on Tox21.

    Parameters
    ----------
    tasks : list of str
        Tox21 task names.  Defaults to all 12.

    Usage
    -----
    >>> evaluator = BaselineEvaluator()
    >>> results = evaluator.run(X_train, y_train, w_train, X_test, y_test, w_test)
    >>> print(evaluator.leaderboard(results))
    """

    def __init__(self, tasks: Optional[List[str]] = None, seed: int = 42):
        self.tasks = tasks or TASK_NAMES
        self.seed  = seed
        self._models = self._build_model_zoo()

    def _build_model_zoo(self) -> Dict[str, object]:
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.seed,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=10,   # compensate for imbalance
                eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed,
                verbosity=0,
            ),
            "LogisticRegression": LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=self.seed,
            ),
            "NaiveBayes": BernoulliNB(alpha=1.0),
        }

    def run(
        self,
        X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray, w_test:  np.ndarray,
    ) -> List[BaselineResults]:
        """
        Train all baselines and evaluate on test set.

        Missing labels (w == 0) are excluded from both training and evaluation
        on a per-task basis, matching DeepChem's convention.
        """
        # Scale features for logistic regression (tree models are invariant)
        scaler  = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        all_results: List[BaselineResults] = []

        for model_name, estimator in self._models.items():
            logger.info("Training %s ...", model_name)
            result = BaselineResults(model_name=model_name)

            use_scaled = model_name == "LogisticRegression"
            Xtr = X_train_sc if use_scaled else X_train
            Xte = X_test_sc  if use_scaled else X_test

            for task_idx, task in enumerate(self.tasks):
                # Mask missing labels
                train_mask = w_train[:, task_idx] != 0
                test_mask  = w_test[:,  task_idx] != 0

                if train_mask.sum() < 10 or test_mask.sum() < 5:
                    logger.warning("Insufficient labelled data for task %s — skipping", task)
                    continue

                y_tr = y_train[train_mask, task_idx].astype(int)
                y_te = y_test[test_mask,   task_idx].astype(int)

                # Skip tasks with only one class in training set
                if len(np.unique(y_tr)) < 2:
                    logger.warning("Task %s has only one class in training — skipping", task)
                    continue

                try:
                    clone = self._clone_estimator(estimator)
                    clone.fit(Xtr[train_mask], y_tr)
                    y_prob = clone.predict_proba(Xte[test_mask])[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)

                    result.task_aucs[task] = roc_auc_score(y_te, y_prob)
                    result.task_f1s[task]  = f1_score(y_te, y_pred, zero_division=0)

                except Exception as exc:
                    logger.error("Error on %s / task %s: %s", model_name, task, exc)

            all_results.append(result)
            logger.info("%s — mean AUC: %.4f", model_name, result.mean_auc)

        return all_results

    @staticmethod
    def leaderboard(results: List[BaselineResults]) -> pd.DataFrame:
        """Return a sorted comparison DataFrame."""
        rows = [r.to_series() for r in results]
        df   = pd.DataFrame(rows).sort_values("mean_AUC", ascending=False)
        df   = df.reset_index(drop=True)
        df.index += 1   # rank from 1
        return df

    @staticmethod
    def _clone_estimator(estimator):
        """Deep-copy an estimator to avoid cross-task contamination."""
        from sklearn.base import clone as sklearn_clone
        try:
            return sklearn_clone(estimator)
        except Exception:
            import copy
            return copy.deepcopy(estimator)
