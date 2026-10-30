"""Command line: python run.py <command> [--profile smoke|pilot|full] [--data-dir DIR] [--run-id NAME]."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .config import ROOT, load_config, resolve


def _env_info() -> dict:
    import numpy, pandas, scipy, sklearn
    try:
        import xgboost
        xgb = xgboost.__version__
    except ImportError:
        xgb = None
    return dict(python=sys.version.split()[0], platform=platform.platform(), numpy=numpy.__version__,
                pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, xgboost=xgb)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__)
    ap.add_argument("command", choices=["synth", "download", "audit", "benchmark", "leakage", "stability", "loso",
                                        "external", "report", "all"])
    ap.add_argument("--profile", default="smoke", choices=["smoke", "pilot", "full"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None, help="folder holding processed.*.data (default: data/raw)")
    ap.add_argument("--run-id", default=None, help="results sub-folder (default: profile name)")
    ap.add_argument("--feature-set", default=None, help="full13 or cheap (see configs/default.yaml)")
    ap.add_argument("--synthetic", action="store_true", help="mark the run as synthetic-data (adds a warning to reports)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.profile)
    raw_dir = resolve(a.data_dir) if a.data_dir else resolve(cfg["data"]["raw_dir"])
    fs = a.feature_set or cfg["data"]["default_feature_set"]
    run_id = a.run_id or a.profile
    run_dir = resolve(cfg["output_root"]) / run_id
    log = lambda m: print(m, flush=True)

    if a.command == "synth":
        from .synthetic import write_synthetic_uci
        write_synthetic_uci(raw_dir, cfg["seed"])
        log(f"[synth] wrote SYNTHETIC UCI-format files to {raw_dir}")
        return
    if a.command == "download":
        from .data import download_uci
        m = download_uci(raw_dir, cfg["data"]["sites"])
        log(json.dumps(m, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    if a.synthetic or (raw_dir / "README_SYNTHETIC.txt").exists():
        (run_dir / "SYNTHETIC_DATA.txt").write_text("Generated from synthetic data; not valid evidence.\n", encoding="utf-8")
    cmds = ["audit", "benchmark", "leakage", "stability", "loso", "external", "report"] if a.command == "all" else [a.command]

    from .data import prepare_data
    df, feats, dlog = prepare_data(cfg, raw_dir, fs)
    (run_dir / "run_info.json").write_text(json.dumps(dict(
        profile=a.profile, seed=cfg["seed"], run=cfg["run"], feature_set=fs, features=feats, n_rows=len(df),
        env=_env_info(), cleaning_log=dlog), indent=2, default=str), encoding="utf-8")
    df.to_csv(run_dir / "clean_data.csv", index=False)

    for c in cmds:
        if c == "audit":
            from .audit import run_audit
            run_audit(cfg, raw_dir, run_dir / "audit", resolve(cfg["data"]["kaggle_csv"]), log)
        elif c == "benchmark":
            from .nested_cv import run_pooled_benchmark
            run_pooled_benchmark(cfg, df, feats, run_dir / "benchmark", fs, log)
        elif c == "leakage":
            from .leakage import run_leakage, run_null_experiment
            run_leakage(cfg, df, feats, run_dir / "leakage", log)
            run_null_experiment(cfg, df, feats, run_dir / "leakage", log)
        elif c == "stability":
            from .stability import run_stability_experiment
            run_stability_experiment(cfg, df, feats, run_dir / "stability", log)
        elif c == "loso":
            from .loso import run_loso
            run_loso(cfg, df, feats, run_dir / "loso", fs, log)
        elif c == "external":
            from .loso import run_external
            run_external(cfg, df, feats, run_dir, log)
        elif c == "report":
            from .report import build_report
            build_report(cfg, run_dir, fs, log)
