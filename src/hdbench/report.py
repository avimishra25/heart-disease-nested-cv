"""Aggregate all result files into tables, figures and a summary (no modelling happens here)."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import stats as S
from .stability import stability_from_benchmark
from .util import md_table

METRICS = ["roc_auc", "recall", "specificity", "precision", "accuracy", "f1", "mcc", "brier"]
SEL_ORDER = ["none", "anova", "mi", "rfe", "l1", "rf_importance"]
MODEL_ORDER = ["dummy", "logreg", "rf", "svm", "xgb"]
COLORS = {"none": "#6b7280", "anova": "#2563eb", "mi": "#0891b2", "rfe": "#d97706", "l1": "#16a34a", "rf_importance": "#9333ea"}


def _load(run_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(run_dir.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _save(df: pd.DataFrame, tdir: Path, name: str, index: bool = False, md: bool = True) -> None:
    df.to_csv(tdir / f"{name}.csv", index=index)
    if md:
        (tdir / f"{name}.md").write_text(md_table(df, index=index), encoding="utf-8")


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_m"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df["_s"] = df["selector"].map({s: i for i, s in enumerate(SEL_ORDER)})
    return df.sort_values(["_m", "_s"]).drop(columns=["_m", "_s"]).reset_index(drop=True)


# --------------------------------------------------------------------------- main results
def table_main(folds: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRICS if c in folds]
    per_rep = folds.groupby(["model", "selector", "repeat"])[cols + ["n_selected"]].mean().reset_index()
    g = per_rep.groupby(["model", "selector"])
    mean, sd = g[cols].mean(), g[cols].std()
    out = mean.add_suffix("_mean").join(sd.add_suffix("_sd")).join(g["n_selected"].mean().rename("n_selected_mean")).reset_index()
    return _order(out)


def table_selection_effect(folds: pd.DataFrame, K: int) -> pd.DataFrame:
    rows = []
    for model in [m for m in folds.model.unique() if m != "dummy"]:
        for sel in [s for s in folds[folds.model == model].selector.unique() if s != "none"]:
            for metric in ("roc_auc", "recall"):
                d = S.paired_fold_diffs(folds, (model, sel), (model, "none"), metric)
                tmp = folds[folds.model.isin([model]) & folds.selector.isin([sel, "none"])]
                rep = (tmp.pivot_table(index="repeat", columns="selector", values=metric)).pipe(lambda t: t[sel] - t["none"])
                t, p = S.nb_corrected_ttest(d, K - 1, 1)
                mean, lo, hi = S.paired_bootstrap_mean(d)
                rows.append(dict(model=model, selector=sel, metric=metric, delta_mean=mean, ci_lo=lo, ci_hi=hi,
                                 nb_t=t, p_nb=p, p_wilcoxon_repeat_means=S.wilcoxon_p(rep.to_numpy())))
    out = pd.DataFrame(rows)
    for metric in out.metric.unique():
        m = out.metric == metric
        out.loc[m, "p_nb_holm"] = S.holm(out.loc[m, "p_nb"].to_numpy())
    return out


def table_model_comparison(folds: pd.DataFrame, K: int) -> pd.DataFrame:
    rows = []
    models = [m for m in MODEL_ORDER if m in set(folds.model) and m != "dummy"]
    for sel in folds.selector.unique():
        for metric in ("roc_auc", "recall"):
            part = []
            for a, b in combinations(models, 2):
                if not (((folds.model == a) & (folds.selector == sel)).any() and ((folds.model == b) & (folds.selector == sel)).any()):
                    continue
                d = S.paired_fold_diffs(folds, (a, sel), (b, sel), metric)
                t, p = S.nb_corrected_ttest(d, K - 1, 1)
                part.append(dict(selector=sel, metric=metric, model_a=a, model_b=b, delta_mean=np.nanmean(d), p_nb=p))
            if part:
                adj = S.holm([r["p_nb"] for r in part])
                for r, pa in zip(part, adj):
                    r["p_nb_holm"] = pa
                rows += part
    return pd.DataFrame(rows)


def table_bootstrap_ci(run_dir: Path, n_boot: int) -> pd.DataFrame:
    rows = []
    for f in sorted(run_dir.glob("oof__*.csv")):
        o = pd.read_csv(f)
        P = o.pivot_table(index="repeat", columns="pos", values="p")
        y = o.drop_duplicates("pos").set_index("pos")["y"].loc[P.columns].to_numpy()
        pr = [P.loc[r].to_numpy() for r in P.index]
        for which in ("roc_auc", "recall"):
            est, lo, hi = S.bootstrap_ci(y, pr, which, n_boot=n_boot)
            rows.append(dict(model=o.model.iloc[0], selector=o.selector.iloc[0], metric=which, estimate=est, ci_lo=lo, ci_hi=hi))
    return _order(pd.DataFrame(rows)) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- extensions
def table_leakage(lk: pd.DataFrame, K: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keys = ["model", "selector", "k_frac"]
    for key, g in lk.groupby(keys):
        clean = g[g.protocol == "clean"].set_index(["repeat", "fold"])
        for proto in ("leak_prep", "leak_prep_select"):
            leak = g[g.protocol == proto].set_index(["repeat", "fold"])
            for metric in ("roc_auc", "recall", "accuracy"):
                d = (leak[metric] - clean[metric]).dropna().to_numpy()
                mean, lo, hi = S.paired_bootstrap_mean(d)
                t, p = S.nb_corrected_ttest(d, K - 1, 1)
                rows.append(dict(zip(keys, key), protocol=proto, metric=metric, clean=clean[metric].mean(),
                                 leaky=leak[metric].mean(), inflation=mean, ci_lo=lo, ci_hi=hi, p_nb=p))
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["protocol", "metric"]).agg(inflation_mean=("inflation", "mean"), inflation_max=("inflation", "max"),
                                                         n_cells=("inflation", "size")).reset_index()
    return detail, summary


def table_loso(loso: pd.DataFrame, oof_pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loso_site = loso.groupby(["model", "selector", "held_out"]).agg(
        loso_auc=("roc_auc", "mean"), loso_recall=("recall", "mean"), loso_specificity=("specificity", "mean"),
        n_test=("n_test", "first"), n_pos=("n_test_pos", "first"), n_neg=("n_test_neg", "first")).reset_index()
    rows = []
    for (model, sel), g in oof_pooled.groupby(["model", "selector"]):
        for site, h in g.groupby("site"):
            aucs = [roc_auc_score(r.y, r.p) for _, r in h.groupby("repeat") if r.y.nunique() == 2]
            rows.append(dict(model=model, selector=sel, held_out=site, pooled_cv_auc=np.mean(aucs) if aucs else np.nan))
    gap = loso_site.merge(pd.DataFrame(rows), on=["model", "selector", "held_out"], how="left")
    gap["transfer_gap"] = gap["pooled_cv_auc"] - gap["loso_auc"]
    # selection benefit under site shift: selector - none, per site and model
    base = gap[gap.selector == "none"].set_index(["model", "held_out"])["loso_auc"]
    gap["loso_delta_vs_none"] = gap.apply(lambda r: r.loso_auc - base.get((r.model, r.held_out), np.nan), axis=1)
    fr = {}
    piv = gap[(gap.selector == "none") & (gap.model != "dummy")].pivot(index="held_out", columns="model", values="loso_auc").dropna()
    if len(piv) >= 2 and piv.shape[1] >= 3:
        fr = S.friedman_nemenyi(piv.to_numpy(), list(piv.columns))
    return gap, pd.DataFrame([fr]) if fr else pd.DataFrame()


# --------------------------------------------------------------------------- figures
def fig_main(main: pd.DataFrame, path: Path) -> None:
    models = [m for m in MODEL_ORDER if m in set(main.model) and m != "dummy"]
    sels = [s for s in SEL_ORDER if s in set(main.selector)]
    w = 0.8 / len(sels)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, s in enumerate(sels):
        sub = main[main.selector == s].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + j * w, sub.roc_auc_mean, w, yerr=sub.roc_auc_sd, color=COLORS[s], label=s, capsize=2, linewidth=0)
    lo = max(0.5, float(main[main.model != "dummy"].roc_auc_mean.min()) - 0.04)
    ax.set_ylim(lo, 1.0); ax.set_xticks(np.arange(len(models)) + 0.4 - w / 2, models)
    ax.set_ylabel("Nested-CV ROC-AUC\n(mean ± sd over repeats)"); ax.legend(ncol=3, fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_leakage(detail: pd.DataFrame, path: Path) -> None:
    d = detail[detail.metric == "roc_auc"].groupby(["model", "protocol"]).inflation.mean().unstack()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    d.plot(kind="bar", ax=ax, color=["#d97706", "#dc2626"], width=0.7)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("ROC-AUC inflation vs. clean pipeline"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_stability(fixed: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, col in zip(axes, ["nogueira", "kuncheva"]):
        for s, g in fixed.groupby("selector"):
            ax.plot(g.k, g[col], marker="o", label=s, color=COLORS.get(s))
        ax.set_title(col.capitalize()); ax.set_xlabel("k (selected columns)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("stability (1 = identical)"); axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_loso(gap: pd.DataFrame, path: Path) -> None:
    g = gap[gap.model != "dummy"].copy()
    g["cfg"] = g.model + "/" + g.selector
    piv = g.pivot(index="cfg", columns="held_out", values="loso_auc")
    fig, ax = plt.subplots(figsize=(5.5, max(3, 0.22 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, label="LOSO ROC-AUC"); ax.set_xlabel("held-out site")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_operating(op: pd.DataFrame, path: Path, target: float) -> None:
    d = op[op.model != "dummy"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for s, g in d.groupby("selector"):
        ax.scatter(g.op_recall_mean, g.op_specificity_mean, color=COLORS.get(s), label=s, s=26)
        for _, r in g.iterrows():
            ax.annotate(r.model, (r.op_recall_mean, r.op_specificity_mean), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axvline(target, ls="--", color="k", lw=0.8); ax.set_xlabel("achieved sensitivity on held-out folds")
    ax.set_ylabel("specificity"); ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_calibration(run_dir: Path, feature_set: str, best: tuple[str, str], path: Path) -> None:
    o = pd.read_csv(run_dir / f"oof__{feature_set}__{best[0]}__{best[1]}.csv")
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    for col, lab in (("p", "uncalibrated"), ("p_platt", "Platt"), ("p_isotonic", "isotonic")):
        if col in o:
            bins = pd.cut(o[col], np.linspace(0, 1, 11), include_lowest=True)
            g = o.groupby(bins, observed=True).agg(x=(col, "mean"), y=("y", "mean"))
            ax.plot(g.x, g.y, marker="o", ms=3, label=lab)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.set_title("/".join(best), fontsize=8)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# --------------------------------------------------------------------------- driver
def build_report(cfg: dict, run_dir: Path, feature_set: str = "full13", log=print) -> dict:
    tdir, fdir = run_dir / "tables", run_dir / "figures"
    tdir.mkdir(exist_ok=True); fdir.mkdir(exist_ok=True)
    bdir = run_dir / "benchmark"
    folds = _load(bdir, f"folds__{feature_set}__*.csv")
    if folds.empty:
        raise SystemExit("no benchmark results found - run `python run.py benchmark` first")
    K = cfg["run"]["outer_folds"]
    n_boot = cfg["run"].get("n_boot", cfg["stats"]["n_boot"])
    sections = ["# Results summary", f"Run: `{run_dir.name}` | profile `{cfg['profile']}` | feature set `{feature_set}`", ""]
    if (run_dir / "SYNTHETIC_DATA.txt").exists():
        sections += ["> **SYNTHETIC DATA.** These numbers come from generated stand-in data and exist only to show that the pipeline runs. They say nothing about heart disease.", ""]

    main = table_main(folds); _save(main, tdir, "T1_main_results")
    fig_main(main, fdir / "F1_auc_by_config.png")
    show = main[["model", "selector"] + [f"{m}_mean" for m in ("roc_auc", "recall", "specificity", "f1", "mcc", "brier")] + ["n_selected_mean"]]
    sections += ["## accuracy of the four classifiers (outer-fold means)", md_table(show, index=False), ""]

    eff = table_selection_effect(folds, K); _save(eff, tdir, "T2_selection_effect")
    sections += ["## value of feature selection (selector minus no-selection, paired folds)", md_table(eff, index=False), ""]
    cmp_ = table_model_comparison(folds, K); _save(cmp_, tdir, "T3_model_comparison")
    sections += ["## pairwise model comparison (Nadeau-Bengio, Holm)", md_table(cmp_, index=False), ""]
    ci = table_bootstrap_ci(bdir.parent / "benchmark", n_boot); _save(ci, tdir, "T4_bootstrap_ci")

    oof_all = _load(bdir, f"oof__{feature_set}__*.csv")
    lk_path = run_dir / "leakage" / "leakage_folds.csv"
    if lk_path.exists():
        detail, summ = table_leakage(pd.read_csv(lk_path), K)
        _save(detail, tdir, "T5_leakage_detail"); _save(summ, tdir, "T5_leakage_summary")
        fig_leakage(detail, fdir / "F2_leakage.png")
        sections += ["## leakage inflation (leaky minus clean, same folds, same k)", md_table(summ, index=False), ""]
        null_path = run_dir / "leakage" / "leakage_null.csv"
        if null_path.exists():
            nl = pd.read_csv(null_path).groupby(["sample_size", "selector", "protocol"]).roc_auc.agg(["mean", "std"]).reset_index()
            _save(nl, tdir, "T5_leakage_null")
            sections += ["### Permuted-label null experiment (true AUC = 0.5)", md_table(nl, index=False), ""]

    stab_dir = run_dir / "stability"
    if (stab_dir / "stability_fixed_k.csv").exists():
        fixed = pd.read_csv(stab_dir / "stability_fixed_k.csv")
        _save(fixed, tdir, "T6_stability_fixed_k")
        fig_stability(fixed, fdir / "F3_stability.png")
        sections += ["## stability of selected features (fixed k, 80% resamples)", md_table(fixed, index=False), ""]
        universe = [c for c in cfg["data"]["feature_sets"]["cheap"]] if feature_set == "cheap" else sorted(
            {v for s in folds.selected_base.dropna() for v in str(s).split("|")})
        sb = stability_from_benchmark(folds, universe); _save(sb, tdir, "T6_stability_nested_cv")
        sections += ["### Stability of the sets chosen inside the nested CV (variable k)", md_table(sb, index=False), ""]

    loso_dir = run_dir / "loso"
    loso = _load(loso_dir, f"loso__{feature_set}__*.csv") if loso_dir.exists() else pd.DataFrame()
    if not loso.empty:
        gap, fr = table_loso(loso, oof_all); _save(gap, tdir, "T7_loso")
        fig_loso(gap, fdir / "F4_loso_auc.png")
        sections += ["## leave-one-site-out", md_table(gap[gap.selector == "none"], index=False), ""]
        if not fr.empty:
            sections += ["### Friedman-Nemenyi across held-out sites (selector = none)", md_table(fr, index=False), ""]

    if "op_recall" in folds:
        op = folds[folds.model != "dummy"].groupby(["model", "selector"]).agg(
            op_recall_mean=("op_recall", "mean"), op_specificity_mean=("op_specificity", "mean"),
            op_threshold_mean=("op_threshold", "mean"), brier=("brier", "mean"), brier_platt=("brier_platt", "mean"),
            brier_isotonic=("brier_isotonic", "mean"), ece=("ece", "mean"), ece_platt=("ece_platt", "mean"),
            ece_isotonic=("ece_isotonic", "mean")).reset_index()
        op = _order(op); _save(op, tdir, "T8_operating_point_calibration")
        tgt = cfg["operating_point"]["target_sensitivity"]
        fig_operating(op, fdir / "F5_operating_point.png", tgt)
        best = main[main.model != "dummy"].sort_values("roc_auc_mean", ascending=False).iloc[0]
        fig_calibration(bdir, feature_set, (best.model, best.selector), fdir / "F6_calibration.png")
        sections += [f"## operating point (target sensitivity {tgt}) and calibration", md_table(op, index=False), ""]

    (run_dir / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    log(f"[report] wrote {run_dir/'summary.md'}")
    return {"main": main, "selection_effect": eff}
