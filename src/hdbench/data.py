"""Loading, harmonising and cleaning the UCI Heart Disease data (+ Kaggle / Z-Alizadeh Sani).

Every cleaning decision is logged so the data audit can quote exact counts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "num"]
PREDICTORS = UCI_COLUMNS[:-1]
UCI_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"


# --------------------------------------------------------------------------- raw I/O
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_uci(raw_dir: Path, sites: list[str]) -> dict:
    """Fetch the four processed.*.data files and write a checksum manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for site in sites:
        url = UCI_BASE_URL + UCI_FILES[site]
        dest = raw_dir / UCI_FILES[site]
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                dest.write_bytes(resp.read())
        except urllib.error.URLError:
            # Python's bundled CA store can be stale; curl uses the OS store (verification stays ON)
            subprocess.run(["curl", "-sS", "-f", "-m", "60", "-o", str(dest), url], check=True)
    return write_manifest(raw_dir, sites)


def write_manifest(raw_dir: Path, sites: list[str]) -> dict:
    manifest = {
        "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "source": UCI_BASE_URL,
        "files": {},
    }
    for site in sites:
        p = raw_dir / UCI_FILES[site]
        manifest["files"][UCI_FILES[site]] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (raw_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_uci_raw(raw_dir: Path, sites: list[str]) -> pd.DataFrame:
    """Read the four site files exactly as published ('?' -> NaN, nothing else touched)."""
    frames = []
    for site in sites:
        path = raw_dir / UCI_FILES[site]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python run.py download` or place the file there.")
        df = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values=["?"], skipinitialspace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.insert(0, "site", site)
        df.insert(1, "row_in_file", np.arange(len(df)))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- cleaning
def clean_uci(raw: pd.DataFrame, cfg: dict, zero_as_missing: list[str] | None = None,
              drop_duplicates: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply the documented cleaning rules and return (clean frame, log).

    Order matters and is part of the protocol:
      1. binarise target (num > 0)
      2. remove exact duplicate rows (all 14 raw columns identical; first occurrence kept)
      3. physiologically impossible zeros -> NaN
      4. drop predictors missing in more than ``max_missing_frac_global`` of all rows
    Nothing here is fitted on outcome-dependent statistics, so doing it once before
    cross-validation is not leakage. Imputation/scaling/encoding stay inside the pipeline.
    """
    dcfg = cfg["data"]
    zero_cols = dcfg["zero_as_missing"] if zero_as_missing is None else zero_as_missing
    dedupe = dcfg["drop_exact_duplicates"] if drop_duplicates is None else drop_duplicates
    log: dict = {"n_raw_by_site": raw.groupby("site").size().to_dict(), "n_raw": int(len(raw))}
    df = raw.copy()
    df["target"] = (df["num"] > 0).astype(int) if dcfg["binarise_target"] else df["num"]

    # duplicates: within-site and across-site are reported separately
    key = PREDICTORS + ["num"]
    dup_within = df.duplicated(subset=["site"] + key, keep="first")
    dup_any = df.duplicated(subset=key, keep="first")
    log["duplicates_within_site"] = df[dup_within].groupby("site").size().to_dict()
    log["duplicates_across_sites_only"] = int((dup_any & ~dup_within).sum())
    if dedupe:
        df = df[~dup_any].reset_index(drop=True)
    log["n_after_dedup"] = int(len(df))

    # impossible zeros -> NaN (counted per site)
    zero_counts = {}
    for col in zero_cols:
        mask = df[col] == 0
        zero_counts[col] = df[mask].groupby("site").size().to_dict()
        df.loc[mask, col] = np.nan
    log["zeros_set_to_nan"] = zero_counts
    log["negative_oldpeak_rows"] = int((df["oldpeak"] < 0).sum())  # flagged, kept (ST depression can be coded negative)

    # global missingness rule
    miss = df[PREDICTORS].isna().mean().round(4)
    dropped = miss[miss > dcfg["max_missing_frac_global"]].index.tolist()
    log["missing_frac_by_predictor"] = miss.to_dict()
    log["predictors_dropped_for_missingness"] = dropped
    log["predictors_kept"] = [c for c in PREDICTORS if c not in dropped]
    log["class_balance_by_site"] = (
        df.groupby("site")["target"].agg(n="size", n_pos="sum").assign(prevalence=lambda t: (t.n_pos / t.n).round(3))
        .to_dict(orient="index")
    )
    log["n_final"] = int(len(df))
    df = df.drop(columns=dropped)
    return df.reset_index(drop=True), log


def resolve_features(cfg: dict, clean: pd.DataFrame, feature_set: str | None = None) -> list[str]:
    name = feature_set or cfg["data"]["default_feature_set"]
    spec = cfg["data"]["feature_sets"][name]
    avail = [c for c in PREDICTORS if c in clean.columns]
    return avail if spec == "all" else [c for c in spec if c in avail]


def prepare_data(cfg: dict, raw_dir: Path | None = None, feature_set: str | None = None,
                 **clean_kwargs) -> tuple[pd.DataFrame, list[str], dict]:
    from .config import resolve
    raw_dir = raw_dir or resolve(cfg["data"]["raw_dir"])
    raw = load_uci_raw(raw_dir, cfg["data"]["sites"])
    clean, log = clean_uci(raw, cfg, **clean_kwargs)
    feats = resolve_features(cfg, clean, feature_set)
    log["feature_set"] = feature_set or cfg["data"]["default_feature_set"]
    log["features_used"] = feats
    return clean, feats, log


# --------------------------------------------------------------------------- splits
def stratification_key(df: pd.DataFrame, n_splits: int) -> np.ndarray:
    """site x class strata when every stratum can fill all folds, else class only."""
    combo = df["site"].astype(str) + "_" + df["target"].astype(str)
    if combo.value_counts().min() >= n_splits:
        return combo.to_numpy()
    return df["target"].to_numpy()


# --------------------------------------------------------------------------- secondary datasets
def load_kaggle_harmonised(path: Path) -> pd.DataFrame:
    """Kaggle 'Heart Failure Prediction' CSV mapped onto the UCI coding. Not independent of UCI."""
    k = pd.read_csv(path)
    out = pd.DataFrame({
        "age": k["Age"],
        "sex": (k["Sex"].astype(str).str.upper().str.startswith("M")).astype(int),
        "cp": k["ChestPainType"].map({"TA": 1, "ATA": 2, "NAP": 3, "ASY": 4}),
        "trestbps": k["RestingBP"],
        "chol": k["Cholesterol"],
        "fbs": k["FastingBS"],
        "restecg": k["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2}),
        "thalach": k["MaxHR"],
        "exang": (k["ExerciseAngina"].astype(str).str.upper() == "Y").astype(int),
        "oldpeak": k["Oldpeak"],
        "slope": k["ST_Slope"].map({"Up": 1, "Flat": 2, "Down": 3}),
        "target": k["HeartDisease"].astype(int),
    })
    out.insert(0, "site", "kaggle")
    return out


OVERLAP_KEY = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]


def kaggle_uci_overlap(uci_raw: pd.DataFrame, kaggle: pd.DataFrame) -> dict:
    """Share of Kaggle rows that also appear in UCI on the shared columns.

    NaN and 0 are treated as equal for chol/trestbps because the two sources code
    missing values differently. This is a fuzzy key, so the result is an estimate.
    """
    def key(df: pd.DataFrame) -> pd.Series:
        t = df[OVERLAP_KEY].copy()
        for c in ("chol", "trestbps"):
            t[c] = t[c].fillna(0)
        return t.round(3).astype(str).agg("|".join, axis=1)

    ku, kk = set(key(uci_raw)), key(kaggle)
    hit = kk.isin(ku)
    return {"n_kaggle": int(len(kaggle)), "n_found_in_uci": int(hit.sum()),
            "share_found_in_uci": round(float(hit.mean()), 4)}


def load_zalizadeh_harmonised(path: Path) -> pd.DataFrame:
    """Z-Alizadeh Sani CAD set reduced to the variables it shares with UCI.

    Only age, sex, resting BP and fasting glucose (>120 mg/dl -> fbs) are comparable, so any
    external validation must refit the models on this 4-variable subset. Column names follow the
    UCI file ('Age', 'Sex', 'BP', 'FBS', 'Cath'); verify them against the downloaded copy.
    """
    path = Path(path)
    z = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    out = pd.DataFrame({
        "age": z["Age"],
        "sex": z["Sex"].astype(str).str.upper().str.startswith("M").astype(int),
        "trestbps": z["BP"],
        "fbs": (z["FBS"] > 120).astype(int),
        "target": (z["Cath"].astype(str).str.strip().str.lower() == "cad").astype(int),
    })
    out.insert(0, "site", "zalizadeh")
    return out
