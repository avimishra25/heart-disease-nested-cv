"""Unit tests. Run:  python tests/test_core.py   (or pytest tests/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from hdbench import stats as S  # noqa: E402
from hdbench.config import load_config  # noqa: E402
from hdbench.data import clean_uci  # noqa: E402
from hdbench.features import FeatureSelector, build_preprocessor  # noqa: E402
from hdbench.metrics import classification_metrics, threshold_for_sensitivity  # noqa: E402
from hdbench.stability import kuncheva_index, mean_pairwise_jaccard, nogueira_stability  # noqa: E402


def test_holm_matches_hand_calculation():
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_nadeau_bengio_variance_correction():
    d = np.array([0.02, 0.01, 0.03, 0.00, 0.02, 0.01])
    t, p = S.nb_corrected_ttest(d, n_train=4, n_test=1)
    n, v = len(d), d.var(ddof=1)
    assert np.isclose(t, d.mean() / np.sqrt((1 / n + 1 / 4) * v))
    t_naive = d.mean() / np.sqrt(v / n)
    assert abs(t) < abs(t_naive)          # the correction must make the test more conservative


def test_stability_indices_bounds():
    same = np.tile(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool), (6, 1))
    assert np.isclose(kuncheva_index(same), 1.0) and np.isclose(nogueira_stability(same), 1.0)
    assert np.isclose(mean_pairwise_jaccard(same), 1.0)
    rng = np.random.default_rng(0)
    rand = np.array([rng.permutation([True] * 3 + [False] * 47) for _ in range(300)])
    assert abs(kuncheva_index(rand)) < 0.1 and abs(nogueira_stability(rand)) < 0.1   # random subsets ~ 0


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.5 * y + rng.normal(0.3, 0.2, 400), 0, 1)
    thr = threshold_for_sensitivity(y, p, 0.9)
    assert classification_metrics(y, p, thr)["recall"] >= 0.9


def test_cleaning_rules():
    cfg = load_config(profile="smoke")
    base = dict(age=50, sex=1, cp=3, trestbps=120, chol=200, fbs=0, restecg=0, thalach=150, exang=0,
                oldpeak=1.0, slope=2, ca=0, thal=3, num=0)
    rows = [dict(base, site="cleveland", row_in_file=0),
            dict(base, site="cleveland", row_in_file=1),                       # exact duplicate -> removed
            dict(base, site="va", row_in_file=0, age=61, chol=0, num=2),       # zero chol -> NaN, target 1
            dict(base, site="va", row_in_file=1, age=62, trestbps=0)]          # zero BP -> NaN
    clean, log = clean_uci(pd.DataFrame(rows), cfg)
    assert len(clean) == 3 and log["duplicates_within_site"] == {"cleveland": 1}
    assert clean.loc[clean.age == 61, "chol"].isna().all() and clean.loc[clean.age == 61, "target"].iloc[0] == 1
    assert clean.loc[clean.age == 62, "trestbps"].isna().all()


def test_preprocessor_has_fixed_width_across_folds():
    cfg = load_config(profile="smoke")
    feats = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "thal"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.integers(0, 2, 60).astype(float) for c in feats})
    X["cp"] = rng.choice([1, 2], 60).astype(float)                              # cp levels 3,4 absent in this sample
    X["thal"] = rng.choice([3, 7], 60).astype(float)
    X.loc[::7, "chol"] = np.nan
    w = {build_preprocessor(cfg, feats, scale=True).fit_transform(X.iloc[idx]).shape[1] for idx in (slice(0, 30), slice(30, 60))}
    assert len(w) == 1 or max(w) - min(w) == 1  # only a missing-indicator column may differ between folds


def test_selector_inside_pipeline_does_not_leak_but_outside_does():
    """Pure noise + permuted labels: AUC must be ~0.5 when selection is inside CV, clearly higher when outside."""
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(100, 300)), rng.integers(0, 2, 100)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clean = Pipeline([("s", FeatureSelector("anova", k=10)), ("m", LogisticRegression(max_iter=1000))])
    auc_clean = cross_val_score(clean, X, y, cv=cv, scoring="roc_auc").mean()
    Xs = FeatureSelector("anova", k=10).fit(X, y).transform(X)                  # the classic mistake
    auc_leak = cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=cv, scoring="roc_auc").mean()
    assert auc_clean < 0.62, auc_clean
    assert auc_leak > auc_clean + 0.12, (auc_clean, auc_leak)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tests passed")
