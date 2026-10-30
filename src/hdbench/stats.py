"""Statistical comparison of classifiers on resampled folds (Nadeau-Bengio, Wilcoxon, Holm, Friedman-Nemenyi, bootstrap)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score, roc_auc_score


def nb_corrected_ttest(d, n_train: int, n_test: int) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau & Bengio 2003) on paired fold differences ``d``."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    var = d.var(ddof=1)
    if var == 0:
        return (0.0, 1.0) if d.mean() == 0 else (np.inf * np.sign(d.mean()), 0.0)
    t = d.mean() / np.sqrt((1.0 / n + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def wilcoxon_p(d) -> float:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0 if len(d) else np.nan
    return float(stats.wilcoxon(d).pvalue)


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def paired_fold_diffs(folds: pd.DataFrame, cfg_a: tuple, cfg_b: tuple, metric: str) -> np.ndarray:
    """a - b on identical outer folds; cfg = (model, selector)."""
    def sub(c):
        m = folds[(folds.model == c[0]) & (folds.selector == c[1])]
        return m.set_index(["repeat", "fold"])[metric]
    a, b = sub(cfg_a), sub(cfg_b)
    idx = a.index.intersection(b.index)
    return (a.loc[idx] - b.loc[idx]).to_numpy()


def friedman_nemenyi(M: np.ndarray, names: list[str], alpha: float = 0.05) -> dict:
    """M: blocks x treatments (higher is better). Returns Friedman test, mean ranks and the Nemenyi CD."""
    n, k = M.shape
    ranks = np.vstack([stats.rankdata(-row) for row in M])  # rank 1 = best
    mean_ranks = ranks.mean(axis=0)
    out = {"n_blocks": n, "k": k, "mean_ranks": dict(zip(names, mean_ranks.round(3)))}
    if n >= 2 and k >= 3:
        stat, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
        out.update(chi2=float(stat), p=float(p), nemenyi_cd=float(q * np.sqrt(k * (k + 1) / (6.0 * n))))
    return out


def _metric(y, p, which: str) -> float:
    if which == "roc_auc":
        return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return recall_score(y, (p >= 0.5).astype(int), zero_division=0)


def bootstrap_ci(y: np.ndarray, p_by_repeat: list[np.ndarray], which: str = "roc_auc",
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Patient-level percentile bootstrap; the metric is averaged over CV repeats inside each resample."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(np.nanmean([_metric(y, p, which) for p in p_by_repeat]))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(np.nanmean([_metric(y[idx], p[idx], which) for p in p_by_repeat]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_mean(d, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
