"""Small helpers shared by several modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    """GitHub-flavoured markdown table without the optional `tabulate` dependency."""
    d = df.reset_index() if index else df.copy()
    d.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in d.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join("---" for _ in d.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)
