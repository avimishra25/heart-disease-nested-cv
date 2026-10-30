"""Leave-one-site-out generalisation (+ optional external validation on Z-Alizadeh Sani)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import load_zalizadeh_harmonised
from .models import build_pipeline, search_space
from .nested_cv import _task


def run_loso(cfg: dict, df: pd.DataFrame, feats: list[str], out_dir: Path, feature_set: str = "full13", log=print) -> pd.DataFrame:
    run, lo, seed = cfg["run"], cfg["loso"], cfg["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, site = df[feats], df["target"].to_numpy(), df["site"].to_numpy()
    n_iter = min(run["n_iter"], lo["n_iter"])
    n_seeds = run.get("loso_seeds", lo["seeds"])
    kw0 = dict(inner_folds=lo["inner_folds"], n_iter=n_iter, extras=False, op_cfg=cfg["operating_point"],
               categorical=cfg["preprocessing"]["categorical"])
    all_rows = []
    for model in run["models"]:
        for selector in run["selectors"]:
            if model == "dummy" and selector != "none":
                continue
            tag = f"{feature_set}__{model}__{selector}"
            f_path, o_path = out_dir / f"loso__{tag}.csv", out_dir / f"loso_oof__{tag}.csv"
            if f_path.exists():
                all_rows.append(pd.read_csv(f_path))
                continue
            t0, space, jobs = time.time(), search_space(model, selector, cfg), []
            for held in sorted(set(site)):
                tr, te = np.where(site != held)[0], np.where(site == held)[0]
                for i in range(n_seeds):
                    s = seed + 7919 * i + 13
                    meta = dict(model=model, selector=selector, feature_set=feature_set, held_out=held, seed_i=i,
                                n_test=len(te), n_test_pos=int(y[te].sum()), n_test_neg=int(len(te) - y[te].sum()))
                    jobs.append(delayed(_task)(build_pipeline(model, selector, cfg, feats, s), space, X, y, tr, te,
                                               meta, dict(kw0, seed=s)))
            res = Parallel(n_jobs=run["n_jobs"])(jobs)
            rows = pd.DataFrame([r for r, _ in res])
            small = np.minimum(rows.n_test_pos, rows.n_test_neg) < lo["min_class_count"]
            rows["auc_defined"] = ~small
            rows.loc[small, "roc_auc"] = np.nan                      # too few events/non-events for a stable AUC
            oof = pd.concat([o for _, o in res], ignore_index=True)
            rows.to_csv(f_path, index=False)
            oof.to_csv(o_path, index=False)
            all_rows.append(rows)
            g = rows.groupby("held_out").roc_auc.mean().round(3).to_dict()
            log(f"[loso] {tag}: {g} ({time.time()-t0:.0f}s)")
    return pd.concat(all_rows, ignore_index=True)


def run_external(cfg: dict, df: pd.DataFrame, feats: list[str], out_dir: Path, log=print) -> pd.DataFrame | None:
    """Train on all UCI rows, test on Z-Alizadeh Sani using only the variables both sources share."""
    from .config import resolve
    path = resolve(cfg["data"]["zalizadeh_path"])
    if not path.exists():
        log(f"[external] {path.name} not found -> skipped (optional)")
        return None
    z = load_zalizadeh_harmonised(path)
    shared = [c for c in ("age", "sex", "trestbps", "fbs") if c in feats and c in z.columns]
    both = pd.concat([df[shared + ["target"]], z[shared + ["target"]]], ignore_index=True)
    tr, te = np.arange(len(df)), np.arange(len(df), len(both))
    run, seed = cfg["run"], cfg["seed"]
    pcfg = {**cfg, "preprocessing": {**cfg["preprocessing"]}}
    kw0 = dict(inner_folds=run["inner_folds"], n_iter=min(run["n_iter"], cfg["loso"]["n_iter"]), extras=False,
               op_cfg=cfg["operating_point"], categorical=cfg["preprocessing"]["categorical"])
    rows = []
    for model in [m for m in run["models"]]:
        est = build_pipeline(model, "none", pcfg, shared, seed)
        row, _ = _task(est, search_space(model, "none", cfg), both[shared], both["target"].to_numpy(), tr, te,
                       dict(model=model, selector="none", shared_features="|".join(shared)), dict(kw0, seed=seed))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "external_zalizadeh.csv", index=False)
    log(f"[external] trained on {len(tr)} UCI rows, tested on {len(te)} Z-Alizadeh rows with {shared}")
    return out
