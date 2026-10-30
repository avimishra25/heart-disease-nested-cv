"""Data audit: everything that must be known about the files BEFORE any model is fitted."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .util import md_table
from .data import (PREDICTORS, clean_uci, kaggle_uci_overlap, load_kaggle_harmonised, load_uci_raw,
                   sha256, UCI_FILES)

PLAUSIBLE = {"age": (18, 100), "trestbps": (60, 250), "chol": (80, 700), "thalach": (50, 250), "oldpeak": (-3, 8)}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def raw_audit_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables computed on the files exactly as published."""
    miss = raw.groupby("site")[PREDICTORS].apply(lambda g: g.isna().mean()).round(3)
    zeros = pd.DataFrame({c: raw.groupby("site")[c].apply(lambda s: int((s == 0).sum())) for c in ("chol", "trestbps")})
    oor = {}
    for c, (lo, hi) in PLAUSIBLE.items():
        bad = raw[c].notna() & ((raw[c] < lo) | (raw[c] > hi))
        oor[c] = raw[bad].groupby("site").size()
    oor = pd.DataFrame(oor).reindex(raw.site.unique()).fillna(0).astype(int)
    summary = raw.groupby("site").agg(n=("num", "size"), n_disease=("num", lambda s: int((s > 0).sum())))
    summary["prevalence"] = (summary.n_disease / summary.n).round(3)
    sev = pd.crosstab(raw["site"], raw["num"])
    desc = raw.groupby("site")[["age", "trestbps", "chol", "thalach", "oldpeak"]].describe().T.round(2)
    return dict(summary=summary, severity=sev, missing_frac=miss, zero_counts=zeros, out_of_range=oor, describe=desc)


def sensitivity_sample_sizes(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for dedupe in (True, False):
        for zeros in (cfg["data"]["zero_as_missing"], []):
            df, log = clean_uci(raw, cfg, zero_as_missing=zeros, drop_duplicates=dedupe)
            rows.append(dict(drop_duplicates=dedupe, zero_as_missing="|".join(zeros) or "none", n=len(df),
                             predictors_kept="|".join(log["predictors_kept"]), prevalence=round(df.target.mean(), 3)))
    return pd.DataFrame(rows)


def figures(raw: pd.DataFrame, tables: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    m = tables["missing_frac"]
    im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if m.values[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="share missing ('?')")
    ax.set_title("Raw missingness by site")
    fig.tight_layout(); fig.savefig(out_dir / "audit_missingness.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    s = tables["severity"]
    s.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis", width=0.8)
    axes[0].set_title("Severity (num) by site"); axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=30)
    tables["zero_counts"].plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].set_title("Exact zeros (raw)"); axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out_dir / "audit_balance_zeros.png", dpi=160); plt.close(fig)


def run_audit(cfg: dict, raw_dir: Path, out_dir: Path, kaggle_csv: Path | None = None, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_uci_raw(raw_dir, cfg["data"]["sites"])
    tables = raw_audit_tables(raw)
    clean, clog = clean_uci(raw, cfg)
    sens = sensitivity_sample_sizes(raw, cfg)
    for name, t in tables.items():
        t.to_csv(out_dir / f"audit_{name}.csv")
    sens.to_csv(out_dir / "audit_sensitivity_sample_sizes.csv", index=False)
    figures(raw, tables, out_dir)

    report = {"cleaning_log": clog,
              "checksums": {UCI_FILES[s]: sha256(raw_dir / UCI_FILES[s]) for s in cfg["data"]["sites"]},
              "synthetic_data": (raw_dir / "README_SYNTHETIC.txt").exists()}
    if kaggle_csv and Path(kaggle_csv).exists():
        report["kaggle_overlap"] = kaggle_uci_overlap(raw, load_kaggle_harmonised(Path(kaggle_csv)))
        report["kaggle_sha256"] = sha256(Path(kaggle_csv))
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2, default=_jsonable), encoding="utf-8")

    md = ["# Data audit report", ""]
    if report["synthetic_data"]:
        md += ["> **SYNTHETIC DATA** - produced by `hdbench.synthetic`, not the UCI files. Counts below test the code only.", ""]
    md += ["## Rows per site and class balance (raw)", md_table(tables["summary"]), "",
           "## Severity labels (num) by site", md_table(tables["severity"]), "",
           "## Raw missingness (share of '?')", md_table(tables["missing_frac"]), "",
           "## Exact zeros in the raw files", md_table(tables["zero_counts"]), "",
           "## Values outside plausible range (zeros excluded from this check only if inside range)",
           md_table(tables["out_of_range"]), "",
           "## Cleaning log", "```json", json.dumps(clog, indent=2, default=_jsonable), "```", "",
           "## Sample size sensitivity to cleaning rules", md_table(sens, index=False), ""]
    if "kaggle_overlap" in report:
        md += ["## Kaggle vs UCI overlap", json.dumps(report["kaggle_overlap"]), ""]
    md += ["## File checksums (SHA-256)"] + [f"- `{k}`: `{v}`" for k, v in report["checksums"].items()]
    (out_dir / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[audit] wrote {out_dir/'audit_report.md'}")
    return report
