"""Helpers for client-provided rows (no server-side parquet)."""

from __future__ import annotations

from typing import Any


def materialize_sample_from_row(row_materializer: Any, row_dict: dict) -> dict:
    """
    Turn one JSON/decoded row into a training sample using the dataset class helper
    (e.g. ``CustomRLHFDataset.sample_from_row_dict`` from ``from_config_only``).
    """
    if hasattr(row_materializer, "sample_from_row_dict"):
        return row_materializer.sample_from_row_dict(row_dict)
    raise TypeError(
        f"{type(row_materializer).__name__} has no sample_from_row_dict(); "
        "use CustomRLHFDataset.from_config_only(...) for streaming."
    )


def pandas_row_to_jsonable(row) -> dict:
    """Convert one pandas Series / row to a plain dict for ``sample_from_row_dict`` / JSON."""
    import math

    import numpy as np

    try:
        import pandas as pd
    except ImportError:
        pd = None

    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    out = {}
    for k, v in d.items():
        if pd is not None:
            if isinstance(v, pd.Timestamp):
                out[k] = v.isoformat()
                continue
            try:
                if pd.isna(v) and not isinstance(v, (list, dict)):
                    out[k] = None
                    continue
            except Exception:
                pass
        if v is None:
            out[k] = None
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out
