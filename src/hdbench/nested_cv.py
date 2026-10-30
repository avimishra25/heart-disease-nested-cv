"""Leakage-free nested cross-validation engine (shared by the benchmark, leakage and LOSO runs).

Outer loop  : repeated stratified K-fold  -> unbiased performance estimate
Inner loop  : stratified K-fold inside each outer-training set -> hyper-parameters AND k
Pipeline    : impute -> encode -> (scale) -> select -> classifier, refitted inside every fold
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from .data import stratification_key
from .features import base_variable
from .metrics import (classification_metrics, expected_calibration_error, operating_point_metrics,
                      threshold_for_sensitivity)
from .models import build_pipeline, search_space


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def selection_info(best, categorical: list[str]) -> dict:
    pre, sel = best.named_steps.get("pre"), best.named_steps.get("select")
    if pre is None:
        return {}
    names = list(pre.get_feature_names_out())
    mask = sel.support_ if sel is not None else np.ones(len(names), dtype=bool)
    enc = [n for n, m in zip(names, mask) if m]
    base = sorted({base_variable(n, categorical) for n in enc})
    return {"n_encoded_total": len(names), "n_selected": int(mask.sum()),
            "selected_encoded": "|".join(enc), "selected_base": "|".join(base)}


def fit_and_score(est, space: dict, X, y, tr, te, *, inner_folds: int, n_iter: int, seed: int,
                  extras: bool, op_cfg: dict, categorical: list[str]) -> tuple[dict, pd.DataFrame]:
    """Tune on X[tr] only, score once on X[te]. Returns (metric row, out-of-fold predictions)."""
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, yte = _take(X, tr), y[tr], _take(X, te), y[te]
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    t0 = time.time()
    if space:
        search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="roc_auc", cv=inner,
                                    random_state=seed, n_jobs=1, refit=True, error_score="raise")
        search.fit(Xtr, ytr)
        best, params, inner_auc = search.best_estimator_, search.best_params_, float(search.best_score_)
    else:
        best, params, inner_auc = clone(est).fit(Xtr, ytr), {}, np.nan
    p = best.predict_proba(Xte)[:, 1]
    row = classification_metrics(yte, p)
    row["ece"] = expected_calibration_error(yte, p, op_cfg["ece_bins"])
    row["inner_cv_auc"] = inner_auc
    row["best_params"] = json.dumps({k: _py(v) for k, v in params.items()}, default=str)
    row["best_k_frac"] = _py(params.get("select__k_frac", np.nan))
    row.update(selection_info(best, categorical))
    oof = pd.DataFrame({"pos": te, "y": yte, "p": p})

    if extras and space:
        # Operating point: threshold chosen from INNER out-of-fold scores of the training part only
        oof_inner = cross_val_predict(clone(best), Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
        thr = threshold_for_sensitivity(ytr, oof_inner, op_cfg["target_sensitivity"])
        row.update(operating_point_metrics(yte, p, thr))
        for name, method in (("platt", "sigmoid"), ("isotonic", "isotonic")):
            if name in op_cfg["calibration_methods"]:
                cal = CalibratedClassifierCV(clone(best), method=method, cv=inner).fit(Xtr, ytr)
                pc = cal.predict_proba(Xte)[:, 1]
                row[f"brier_{name}"] = classification_metrics(yte, pc)["brier"]
                row[f"ece_{name}"] = expected_calibration_error(yte, pc, op_cfg["ece_bins"])
                oof[f"p_{name}"] = pc
    row["fit_seconds"] = round(time.time() - t0, 2)
    return row, oof


def _task(est, space, X, y, tr, te, meta, kw):
    row, oof = fit_and_score(est, space, X, y, tr, te, **kw)
    row.update(meta)
    for k, v in meta.items():
        oof[k] = v
    return row, oof


def make_outer_splits(df: pd.DataFrame, y: np.ndarray, repeats: int, folds: int, seed: int):
    """Identical for every model/selector so that results can be compared pair-wise."""
    key = stratification_key(df, folds)
    out = []
    for r in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), key)):
            out.append((r, f, tr, te))
    return out


def run_pooled_benchmark(cfg: dict, df: pd.DataFrame, feats: list[str], out_dir: Path,
                         feature_set: str = "full13", log=print) -> pd.DataFrame:
    run, seed = cfg["run"], cfg["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = df[feats], df["target"].to_numpy()
    splits = make_outer_splits(df, y, run["outer_repeats"], run["outer_folds"], seed)
    cat = cfg["preprocessing"]["categorical"]
    kw0 = dict(inner_folds=run["inner_folds"], n_iter=run["n_iter"], extras=run["extras"],
               op_cfg=cfg["operating_point"], categorical=cat)
    site = df["site"].to_numpy()
    all_rows = []
    for model in run["models"]:
        for selector in run["selectors"]:
            if model == "dummy" and selector != "none":
                continue
            tag = f"{feature_set}__{model}__{selector}"
            f_path, o_path = out_dir / f"folds__{tag}.csv", out_dir / f"oof__{tag}.csv"
            if f_path.exists() and o_path.exists():
                log(f"[skip] {tag} (already done)")
                all_rows.append(pd.read_csv(f_path))
                continue
            t0 = time.time()
            space = search_space(model, selector, cfg)
            meta_base = dict(model=model, selector=selector, feature_set=feature_set,
                             family=cfg["selector_families"][selector])
            jobs = []
            for r, f, tr, te in splits:
                s = seed + 1000 * r + f
                est = build_pipeline(model, selector, cfg, feats, s)
                meta = dict(meta_base, repeat=r, fold=f)
                jobs.append(delayed(_task)(est, space, X, y, tr, te, meta, dict(kw0, seed=s)))
            res = Parallel(n_jobs=run["n_jobs"])(jobs)
            rows = pd.DataFrame([r for r, _ in res])
            oof = pd.concat([o for _, o in res], ignore_index=True)
            oof["site"] = site[oof["pos"].to_numpy()]
            rows.to_csv(f_path, index=False)
            oof.to_csv(o_path, index=False)
            all_rows.append(rows)
            log(f"[done] {tag}: AUC {rows.roc_auc.mean():.3f} recall {rows.recall.mean():.3f} ({time.time() - t0:.0f}s)")
    return pd.concat(all_rows, ignore_index=True)
