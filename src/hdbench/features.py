"""Preprocessing and feature selection, built so that every step is fitted on training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SELECTOR_METHODS = ("anova", "mi", "rfe", "l1", "rf_importance")


# --------------------------------------------------------------------------- preprocessing
def build_preprocessor(cfg: dict, feature_cols: list[str], scale: bool) -> ColumnTransformer:
    """Impute (+ missing indicator) -> one-hot / scale. Scaling only for LR and SVM (scale=True)."""
    p = cfg["preprocessing"]
    num = [c for c in p["numeric"] if c in feature_cols]
    binary = [c for c in p["binary"] if c in feature_cols]
    cat = [c for c in p["categorical"] if c in feature_cols]

    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=p["missing_indicator"],
                                          keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    bin_steps = [("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                 ("onehot", OneHotEncoder(categories=[p["categories"][c] for c in cat],
                                          handle_unknown="ignore", sparse_output=False))]
    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps), num))
    if binary:
        transformers.append(("bin", Pipeline(bin_steps), binary))
    if cat:
        transformers.append(("cat", Pipeline(cat_steps), cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform="default")


def base_variable(name: str, categorical: list[str]) -> str:
    """Collapse an encoded column name back to the clinical variable it came from."""
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):]
    for c in categorical:
        if name.startswith(c + "_"):
            return c
    return name


# --------------------------------------------------------------------------- selection
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Rank the (encoded) columns with one scoring method and keep the top k.

    method:  anova | mi              -> filter
             rfe                     -> wrapper (RFE around L2 logistic regression)
             l1 | rf_importance      -> embedded (|coef| of L1 logistic regression; RF importance)
    k_frac:  share of columns kept (tuned in the inner loop). ``k`` overrides it with an absolute count.
    Scoring uses an internal standardised copy so the downstream classifier can stay unscaled.
    """

    def __init__(self, method: str = "anova", k_frac: float = 1.0, k: int | None = None, random_state: int = 0):
        self.method = method
        self.k_frac = k_frac
        self.k = k
        self.random_state = random_state

    def _scores(self, Xs: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        m = self.method
        if m == "anova":
            s, _ = f_classif(Xs, y)
            return np.nan_to_num(s, nan=0.0, posinf=0.0)
        if m == "mi":
            return mutual_info_classif(Xs, y, discrete_features=False, n_neighbors=3, random_state=self.random_state)
        if m == "rfe":
            base = LogisticRegression(C=1.0, max_iter=2000)
            rfe = RFE(base, n_features_to_select=k, step=1).fit(Xs, y)
            return -rfe.ranking_.astype(float)
        if m == "l1":
            lr = LogisticRegression(l1_ratio=1.0, C=0.3, solver="saga", max_iter=5000, random_state=self.random_state)
            lr.fit(Xs, y)
            # |coef| first, ANOVA F as tie-breaker so zero coefficients are ranked deterministically
            f, _ = f_classif(Xs, y)
            return np.abs(lr.coef_.ravel()) + 1e-9 * np.nan_to_num(f) / (1e-12 + np.nanmax(np.nan_to_num(f)))
        if m == "rf_importance":
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=self.random_state, n_jobs=1)
            return rf.fit(Xs, y).feature_importances_
        raise ValueError(f"unknown selector {m!r}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        k = int(self.k) if self.k is not None else int(np.ceil(self.k_frac * d))
        k = int(np.clip(k, 1, d))
        Xs = StandardScaler().fit_transform(X)
        Xs = np.nan_to_num(Xs)
        scores = self._scores(Xs, np.asarray(y), k)
        order = np.argsort(-scores, kind="stable")
        mask = np.zeros(d, dtype=bool)
        mask[order[:k]] = True
        self.scores_, self.support_, self.n_features_in_, self.k_ = scores, mask, d, k
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self):
        return self.support_
