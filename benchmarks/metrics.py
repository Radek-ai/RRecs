"""Evaluation metrics for recommendation benchmarks."""

from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    rec = recommended[:k]
    if not rec:
        return 0.0
    return len(set(rec) & relevant) / len(rec)


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    rec = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(rec) & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    rec = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, pid in enumerate(rec) if pid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    recommendations: dict[str, list],
    ground_truth: dict[str, set],
    eval_customers: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute mean Precision@K, Recall@K and NDCG@K across *eval_customers*.

    Parameters
    ----------
    recommendations : dict
        ``{customer_id: [product_id, …]}`` — ordered recommendation lists.
    ground_truth : dict
        ``{customer_id: {product_id, …}}`` — held-out test items.
    eval_customers : list
        Customer IDs to average over.
    k : int
        Cut-off for all metrics.
    """
    prec, rec, ndcg = [], [], []
    for cid in eval_customers:
        recs_list = recommendations.get(str(cid), [])
        rel = ground_truth.get(cid, set())
        prec.append(precision_at_k(recs_list, rel, k))
        rec.append(recall_at_k(recs_list, rel, k))
        ndcg.append(ndcg_at_k(recs_list, rel, k))
    return {
        f"Precision@{k}": float(np.mean(prec)),
        f"Recall@{k}": float(np.mean(rec)),
        f"NDCG@{k}": float(np.mean(ndcg)),
    }
