"""Entry point: python run.py <command> --profile smoke|pilot|full  (see README.md)."""
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")  # joblib workers must import hdbench too

if __name__ == "__main__":
    from hdbench.cli import main
    main()
