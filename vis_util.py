"""Plotting utilities for DRL training pipelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out.index):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def plot_account_values(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _ensure_datetime_index(df)

    plt.figure(figsize=(14, 5))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column.upper())
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
