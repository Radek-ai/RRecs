"""Plot benchmark results: best run per model family."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def best_per_family(results: pd.DataFrame, ndcg_col: str | None = None) -> pd.DataFrame:
    """Return one row per *model_class* with the highest NDCG."""
    if ndcg_col is None:
        for c in results.columns:
            if c.startswith("NDCG@"):
                ndcg_col = c
                break
        if ndcg_col is None:
            raise ValueError("No NDCG column found")

    idx = results.groupby("model_class")[ndcg_col].idxmax()
    return results.loc[idx].reset_index(drop=True)


def plot_best_per_family(
    results: pd.DataFrame,
    output_path: str,
    ndcg_col: str | None = None,
    title: str = "Best config per model (by NDCG)",
) -> str:
    """Horizontal grouped bar chart: Precision, Recall, NDCG for best row per model.

    Returns the path to the saved PNG.
    """
    if ndcg_col is None:
        for c in results.columns:
            if c.startswith("NDCG@"):
                ndcg_col = c
                break
        if ndcg_col is None:
            raise ValueError("No NDCG column found")

    prec_col = next(c for c in results.columns if c.startswith("Precision@"))
    rec_col = next(c for c in results.columns if c.startswith("Recall@"))

    best = best_per_family(results, ndcg_col=ndcg_col)
    best = best.sort_values(ndcg_col, ascending=True)

    plot_df = best.set_index("model_class")[[prec_col, rec_col, ndcg_col]]
    plot_df.columns = ["Precision", "Recall", "NDCG"]

    ax = plot_df.plot(kind="barh", figsize=(10, max(4, len(best) * 0.6)), width=0.85)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(plot_df.max().max() * 1.15, 0.01))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
