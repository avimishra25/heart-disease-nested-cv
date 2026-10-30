"""Leakage: how much does fitting preprocessing / feature selection on the whole dataset inflate performance?

Three protocols share identical outer splits, the same fixed k and the same inner tuning budget:
  clean              imputation, scaling, selection and classifier are all fitted inside each training fold
  leak_prep          imputer / encoder / scaler fitted on ALL rows first (unsupervised leakage)
  leak_prep_select   as above AND the feature selector fitted on ALL rows with ALL labels (supervised leakage)
A label-permutation "null" experiment (no real signal, many noise columns) shows the same effect in its purest form.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from .features import FeatureSelector, build_preprocessor
from .models import SCALED_MODELS, build_pipeline, classifier_space, make_classifier
from .nested_cv import _task, make_outer_splits

PROTOCOLS = ["clean", "leak_prep", "leak_prep_select"]


def _jobs_for(cfg, df, feats, model, selector, k_frac, protocol, splits, n_iter, seed):
    run = cfg["run"]
    y = df["target"].to_numpy()
    cat = cfg["preprocessing"]["categorical"]
    space = classifier_space(model)
    kw0 = dict(inner_folds=run["inner_folds"], n_iter=n_iter, extras=False, op_cfg=cfg["operating_point"], categorical=cat)
    if protocol == "clean":
        X = df[feats]
        mk = lambda s: build_pipeline(model, selector, cfg, feats, s).set_params(select__k_frac=k_frac)
    else:
        pre = build_preprocessor(cfg, feats, scale=model in SCALED_MODELS)
        X = pre.fit_transform(df[feats])                      # <-- fitted on every row: the leak
        if protocol == "leak_prep":
            mk = lambda s: Pipeline([("select", FeatureSelector(selector, k_frac, random_state=s)),
                                     ("clf", make_classifier(model, s))])
        else:
            X = FeatureSelector(selector, k_frac, random_state=seed).fit(X, y).transform(X)   # <-- labels of test folds used
            mk = lambda s: Pipeline([("clf", make_classifier(model, s))])
    jobs = []
    for r, f, tr, te in splits:
        s = seed + 1000 * r + f
        meta = dict(model=model, selector=selector, k_frac=k_frac, protocol=protocol, repeat=r, fold=f)
        jobs.append(delayed(_task)(mk(s), space, X, y, tr, te, meta, dict(kw0, seed=s)))
    return jobs


def run_leakage(cfg: dict, df: pd.DataFrame, feats: list[str], out_dir: Path, log=print) -> pd.DataFrame:
    run, lk, seed = cfg["run"], cfg["leakage"], cfg["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "leakage_folds.csv"
    if path.exists():
        log("[skip] leakage_folds.csv exists")
        return pd.read_csv(path)
    models = [m for m in run.get("leakage_models", run["models"]) if m != "dummy"]
    selectors = [s for s in lk["selectors"] if s in run["selectors"]]
    k_fracs = run.get("leakage_k_fracs", lk["k_fracs"])
    n_iter = min(run["n_iter"], lk["n_iter"])
    splits = make_outer_splits(df, df["target"].to_numpy(), run["outer_repeats"], run["outer_folds"], seed)
    rows = []
    for model in models:
        for selector in selectors:
            for k in k_fracs:
                t0 = time.time()
                jobs = []
                for protocol in PROTOCOLS:
                    jobs += _jobs_for(cfg, df, feats, model, selector, k, protocol, splits, n_iter, seed)
                res = Parallel(n_jobs=run["n_jobs"])(jobs)
                part = pd.DataFrame([r for r, _ in res])
                rows.append(part)
                g = part.groupby("protocol").roc_auc.mean()
                log(f"[leak] {model}/{selector}/k={k}: " + "  ".join(f"{p}={g[p]:.3f}" for p in PROTOCOLS) + f" ({time.time()-t0:.0f}s)")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def run_null_experiment(cfg: dict, df: pd.DataFrame, feats: list[str], out_dir: Path, log=print) -> pd.DataFrame:
    """Permuted labels + noise columns: any AUC above 0.5 is pure selection bias."""
    run, nl, seed = cfg["run"], cfg["leakage"]["null_experiment"], cfg["seed"]
    path = out_dir / "leakage_null.csv"
    if not nl["enabled"]:
        return pd.DataFrame()
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(seed)
    pre = build_preprocessor(cfg, feats, scale=True)
    Xp_all = pre.fit_transform(df[feats])
    cat = cfg["preprocessing"]["categorical"]
    rows = []
    for size in nl.get("sample_sizes", [150, None]):
        for perm in range(nl["n_permutations"]):
            idx = rng.permutation(len(df))[:size] if size else np.arange(len(df))
            n = len(idx)
            y = rng.permutation(df["target"].to_numpy()[idx])           # labels carry no information
            X = np.hstack([Xp_all[idx], rng.normal(size=(n, nl["n_noise_features"]))])
            k = 10
            sub = df.iloc[idx].reset_index(drop=True).assign(target=y)
            splits = make_outer_splits(sub, y, run["outer_repeats"], run["outer_folds"], seed + perm)
            kw0 = dict(inner_folds=run["inner_folds"], n_iter=min(run["n_iter"], 10), extras=False,
                       op_cfg=cfg["operating_point"], categorical=cat)
            for selector in ("anova", "l1"):
                for protocol in ("clean", "leak_select"):
                    Xp = X if protocol == "clean" else FeatureSelector(selector, k=k, random_state=seed).fit(X, y).transform(X)
                    mk = (lambda s: Pipeline([("select", FeatureSelector(selector, k=k, random_state=s)), ("clf", make_classifier("logreg", s))])
                          ) if protocol == "clean" else (lambda s: Pipeline([("clf", make_classifier("logreg", s))]))
                    jobs = []
                    for r, f, tr, te in splits:
                        s = seed + 1000 * r + f
                        meta = dict(sample_size=size or len(df), permutation=perm, selector=selector, protocol=protocol, repeat=r, fold=f)
                        jobs.append(delayed(_task)(mk(s), classifier_space("logreg"), Xp, y, tr, te, meta, dict(kw0, seed=s)))
                    rows += [r for r, _ in Parallel(n_jobs=run["n_jobs"])(jobs)]
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    g = out.groupby(["sample_size", "selector", "protocol"]).roc_auc.mean().round(3)
    log("[null] mean AUC on permuted labels:\n" + g.to_string())
    return out
