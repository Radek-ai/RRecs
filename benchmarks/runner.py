"""Benchmark runner -- fit models, evaluate, collect results."""

from __future__ import annotations

import inspect
import json
import os
import time
from datetime import datetime

import pandas as pd

from benchmarks.datasets import BenchmarkData
from benchmarks.metrics import evaluate
from recs.base import Recommender


def run_config(
    model: Recommender,
    data: BenchmarkData,
    k: int = 10,
) -> dict:
    """Fit *model*, generate recommendations, and return metrics + timing.

    Returns a flat dict with metric values, timing, model class name, and
    a JSON-serialised copy of the model's ``__init__`` parameters.
    """
    t0 = time.perf_counter()
    model.fit(data.customers, data.products, data.train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    recs_df = model.recommend(
        customer_ids=[str(c) for c in data.eval_customers], n=k,
    )
    recommend_time = time.perf_counter() - t0

    recs_dict = (
        recs_df.sort_values(["customer_id", "rank"])
        .groupby("customer_id")["product_id"]
        .apply(list)
        .to_dict()
    )

    metrics = evaluate(
        recs_dict, data.ground_truth, data.eval_customers, k=k,
    )

    params = _extract_params(model)

    return {
        **metrics,
        "fit_time_s": round(fit_time, 3),
        "recommend_time_s": round(recommend_time, 3),
        "model_class": type(model).__name__,
        "params_json": json.dumps(params, default=str),
    }


def run_sweep(
    configs: list[tuple[str, Recommender]],
    data: BenchmarkData,
    k: int = 10,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run a list of ``(name, model)`` configs and return a results DataFrame.

    Parameters
    ----------
    configs : list of (name, model) tuples
        Each *model* must implement :class:`recs.base.Recommender`.
    data : BenchmarkData
        Dataset to benchmark against.
    k : int
        Cut-off for evaluation metrics.
    output_path : str | None
        If given, save results to a timestamped CSV in this directory (or
        exact file path if it ends with ``.csv``).

    Returns
    -------
    DataFrame
        One row per config, sorted by NDCG@K descending.
    """
    rows: list[dict] = []
    total = len(configs)

    for i, (name, model) in enumerate(configs, 1):
        print(f"  [{i}/{total}] {name} …", end=" ", flush=True)
        result = run_config(model, data, k=k)
        result["name"] = name
        rows.append(result)
        ndcg_key = f"NDCG@{k}"
        print(
            f"NDCG={result[ndcg_key]:.4f}  "
            f"fit={result['fit_time_s']:.1f}s  "
            f"rec={result['recommend_time_s']:.1f}s"
        )

    df = pd.DataFrame(rows)
    col_order = [
        "name", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}",
        "fit_time_s", "recommend_time_s", "model_class", "params_json",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(f"NDCG@{k}", ascending=False).reset_index(drop=True)

    if output_path is not None:
        csv_path = _resolve_output_path(output_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    return df


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_params(model: Recommender) -> dict:
    """Pull constructor parameters from a model instance.

    Models that inherit the default ``object.__init__`` have no
    inspectable ``__code__``; we use :func:`inspect.signature` instead.
    """
    params: dict = {}
    try:
        sig = inspect.signature(type(model).__init__)
    except (TypeError, ValueError):
        return _extract_params_from_instance_dict(model)

    for name in sig.parameters:
        if name == "self":
            continue
        if hasattr(model, name):
            params[name] = getattr(model, name)

    if not params:
        return _extract_params_from_instance_dict(model)
    return params


def _extract_params_from_instance_dict(model: Recommender) -> dict:
    """Fallback: public non-callable attributes (typical hyperparameters)."""
    out = {}
    for k, v in getattr(model, "__dict__", {}).items():
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def _resolve_output_path(path: str) -> str:
    """If *path* ends with ``.csv``, insert a timestamp before the extension.
    Otherwise treat it as a directory and generate a filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.endswith(".csv"):
        base, ext = os.path.splitext(path)
        return f"{base}_{ts}{ext}"
    return os.path.join(path, f"sweep_{ts}.csv")
