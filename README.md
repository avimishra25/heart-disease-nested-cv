# hdbench: leakage-free nested cross-validation toolkit

Code for benchmarking Logistic Regression, Random Forest, SVM and XGBoost on the UCI Heart Disease data,
with and without feature selection (filter, wrapper, embedded). Every data-dependent step is a stage of one
scikit-learn pipeline that is refitted inside each training fold.

## Layout

```
configs/default.yaml      all settings; profiles smoke | pilot | full
run.py                    command-line entry point
src/hdbench/              data, features, models, nested_cv, leakage, stability, loso, stats, audit, report
tests/test_core.py        unit tests
data/raw/                 the UCI files go here (downloaded by the script, not stored in the repo)
```

## Setup and use

```bash
pip install -r requirements.txt     # Python >= 3.10, scikit-learn >= 1.8
python tests/test_core.py
python run.py download             # fetches the four UCI files and writes checksums
python run.py all --profile pilot --run-id my_run
```

Commands: `synth, download, audit, benchmark, leakage, stability, loso, external, report, all`.
Results are written to `results/<run-id>/`.
