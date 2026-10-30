"""The four classifiers, a dummy baseline, their search spaces and the full pipelines."""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .features import FeatureSelector, build_preprocessor

SCALED_MODELS = {"logreg", "svm"}
MODEL_LABELS = {"dummy": "Dummy", "logreg": "LogReg", "rf": "RandomForest", "svm": "SVM", "xgb": "XGBoost"}


def make_classifier(name: str, seed: int):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":  # l1_ratio 0 = L2, 1 = L1, in between = elastic net (sklearn >= 1.8 API)
        return LogisticRegression(solver="saga", max_iter=5000, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "svm":
        return SVC(probability=True, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", n_jobs=1, eval_metric="logloss", random_state=seed, verbosity=0)
    raise ValueError(name)


def classifier_space(name: str, prefix: str = "clf__") -> dict:
    spaces = {
        "dummy": {},
        "logreg": {"C": loguniform(1e-3, 1e2), "l1_ratio": [0.0, 0.5, 1.0]},
        "rf": {"n_estimators": randint(100, 401), "max_depth": [2, 3, 4, 6, 8, 12, None],
               "min_samples_leaf": randint(1, 11), "max_features": ["sqrt", "log2", 0.5]},
        "svm": {"kernel": ["linear", "rbf"], "C": loguniform(1e-2, 1e2), "gamma": loguniform(1e-4, 1e0)},
        "xgb": {"n_estimators": randint(50, 401), "learning_rate": loguniform(1e-2, 3e-1),
                "max_depth": randint(2, 7), "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-2, 1e1)},
    }
    return {prefix + k: v for k, v in spaces[name].items()}


def build_pipeline(model: str, selector: str, cfg: dict, feature_cols: list[str], seed: int) -> Pipeline:
    steps = [("pre", build_preprocessor(cfg, feature_cols, scale=model in SCALED_MODELS))]
    if selector != "none":
        steps.append(("select", FeatureSelector(method=selector, k_frac=1.0, random_state=seed)))
    steps.append(("clf", make_classifier(model, seed)))
    return Pipeline(steps)


def search_space(model: str, selector: str, cfg: dict) -> dict:
    space = classifier_space(model)
    if selector != "none" and model != "dummy":
        space["select__k_frac"] = list(cfg["k_frac_grid"])  # k is tuned like any other hyper-parameter
    return space
