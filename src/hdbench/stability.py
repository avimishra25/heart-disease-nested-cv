"""Stability of selected feature sets (Jaccard, Kuncheva 2007, Nogueira et al. 2018) and clinical plausibility."""
from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit

from .data import stratification_key
from .features import FeatureSelector, base_variable, build_preprocessor


def mean_pairwise_jaccard(Z: np.ndarray) -> float:
    vals = []
    for a, b in combinations(range(len(Z)), 2):
        union = (Z[a] | Z[b]).sum()
        vals.append((Z[a] & Z[b]).sum() / union if union else 1.0)
    return float(np.mean(vals))


def kuncheva_index(Z: np.ndarray) -> float:
    """Mean pairwise Kuncheva consistency index; requires all subsets to have the same size k (0 < k < d)."""
    M, d = Z.shape
    ks = Z.sum(axis=1)
    if len(set(ks)) != 1:
        raise ValueError("Kuncheva index needs equal-size subsets; use nogueira_stability for variable k")
    k = int(ks[0])
    if k in (0, d):
        return np.nan
    vals = [((Z[a] & Z[b]).sum() * d - k * k) / (k * (d - k)) for a, b in combinations(range(M), 2)]
    return float(np.mean(vals))


def nogueira_stability(Z: np.ndarray) -> float:
    """Nogueira, Sechidis & Brown (2018); valid for subsets of different sizes. 1 = identical, ~0 = random."""
    M, d = Z.shape
    pf = Z.mean(axis=0)
    kbar = Z.sum(axis=1).mean()
    if kbar in (0, d) or M < 2:
        return np.nan
    s2 = M / (M - 1) * pf * (1 - pf)
    return float(1 - s2.mean() / ((kbar / d) * (1 - kbar / d)))


def stability_from_benchmark(folds: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    """Stability of the sets the nested CV actually chose (variable k), at clinical-variable level."""
    rows = []
    for (fs, model, sel), g in folds[folds.selector != "none"].groupby(["feature_set", "model", "selector"]):
        Z = np.array([[v in set(str(s).split("|")) for v in universe] for s in g["selected_base"]], dtype=bool)
        rows.append(dict(feature_set=fs, model=model, selector=sel, n_subsets=len(Z),
                         mean_n_variables=Z.sum(axis=1).mean(), jaccard=mean_pairwise_jaccard(Z),
                         nogueira=nogueira_stability(Z)))
    return pd.DataFrame(rows)


def run_stability_experiment(cfg: dict, df: pd.DataFrame, feats: list[str], out_dir: Path, log=print) -> dict:
    """Controlled fixed-k experiment: random 80% subsamples, each selector, each k."""
    st, seed = cfg["stability"], cfg["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_ni = copy.deepcopy(cfg)
    cfg_ni["preprocessing"]["missing_indicator"] = False      # fixed column universe for Kuncheva
    cat = cfg["preprocessing"]["categorical"]
    y = df["target"].to_numpy()
    n_res = cfg["run"].get("stability_resamples", st["n_resamples"])
    sss = StratifiedShuffleSplit(n_splits=n_res, train_size=0.8, random_state=seed)
    methods = [s for s in cfg["run"]["selectors"] if s != "none"]
    store = {(m, k): [] for m in methods for k in st["k_values"]}
    names = None
    for tr, _ in sss.split(np.zeros(len(y)), stratification_key(df, 2)):
        pre = build_preprocessor(cfg_ni, feats, scale=True)
        Xt = pre.fit_transform(df[feats].iloc[tr])
        names = list(pre.get_feature_names_out())
        for m in methods:
            for k in st["k_values"]:
                store[(m, k)].append(FeatureSelector(m, k=k, random_state=seed).fit(Xt, y[tr]).support_)
    base = [base_variable(n, cat) for n in names]
    variables = [v for v in feats]
    strong, weak = st["clinical_prior_strong"], st["clinical_prior_weak"]
    summ, freq_rows = [], []
    for (m, k), subsets in store.items():
        Z = np.array(subsets)
        freq_base = {v: float(np.mean([any(Zi[j] for j, b in enumerate(base) if b == v) for Zi in Z])) for v in variables}
        sv = [freq_base[v] for v in strong if v in freq_base]
        wk = [freq_base[v] for v in weak if v in freq_base]
        summ.append(dict(selector=m, k=k, d=Z.shape[1], n_resamples=len(Z), kuncheva=kuncheva_index(Z),
                         jaccard=mean_pairwise_jaccard(Z), nogueira=nogueira_stability(Z),
                         prior_gap=float(np.mean(sv) - np.mean(wk)) if sv and wk else np.nan))
        freq_rows += [dict(selector=m, k=k, variable=v, frequency=f) for v, f in freq_base.items()]
    summary, freq = pd.DataFrame(summ), pd.DataFrame(freq_rows)
    # cross-selector agreement of selection frequencies (Spearman), per k
    agree = []
    for k in st["k_values"]:
        piv = freq[freq.k == k].pivot(index="variable", columns="selector", values="frequency")
        for a, b in combinations(piv.columns, 2):
            rho = spearmanr(piv[a], piv[b]).statistic if piv[a].nunique() > 1 and piv[b].nunique() > 1 else np.nan
            agree.append(dict(k=k, selector_a=a, selector_b=b, spearman=rho))
    agree = pd.DataFrame(agree)
    summary.to_csv(out_dir / "stability_fixed_k.csv", index=False)
    freq.to_csv(out_dir / "stability_selection_frequency.csv", index=False)
    agree.to_csv(out_dir / "stability_selector_agreement.csv", index=False)
    log(f"[stability] {len(store)} selector/k cells x {n_res} resamples")
    return dict(summary=summary, freq=freq, agreement=agree)
