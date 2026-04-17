"""Benchmark script: refined hyperparameter grids, baselines, CSV + plot.

Grids are tuned from prior sweeps:
- WeightedSimilarity overlap: dense *top_k* in 10–50 (sweet spot); cosine kept sparse.
- BipartiteGraphRec: extend *top_k* past 150 for 3-hop count; lighter multi-hop probes.

Run from the project root::

    python -m benchmarks.run_benchmark

Requires matplotlib for the summary plot (``pip install matplotlib``).
"""

from __future__ import annotations

from benchmarks.baselines import (
    ItemCFBaseline,
    MostPopularBaseline,
    SVDBaseline,
    UserCFBaseline,
)
from benchmarks.datasets import load_online_retail
from benchmarks.plotting import plot_best_per_family
from benchmarks.runner import run_sweep
from recs import BipartiteGraphRec, WeightedSimilarity, BeliefPropagationRec

K = 10

# --- Grid constants (edit here to explore further) ---
WS_OVERLAP_K = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # fine steps where optimum likely lies
WS_OVERLAP_K_TAIL = [60, 75]  # check decay after 50
WS_COSINE_K = [30, 50, 100]  # sparse: cosine was uniformly weak vs overlap

GRAPH_3HOP_COUNT_K = [75, 100, 125, 150, 175, 200, 250]  # extend past 150; anchor low
GRAPH_3HOP_MULT_K = [30, 50, 100]  # multiply underperforms; few reference points


def build_configs() -> list[tuple[str, object]]:
    """Return (name, model) pairs for the benchmark sweep."""
    configs: list[tuple[str, object]] = []

    # ---- Classic baselines ----
    configs.append(("Popular", MostPopularBaseline()))
    for n_neigh in [30, 40, 50, 100]:
        configs.append((f"UserCF nn={n_neigh}", UserCFBaseline(top_neighbors=n_neigh)))
    configs.append(("ItemCF", ItemCFBaseline()))
    for n_comp in [20, 50, 80]:
        configs.append((f"SVD n={n_comp}", SVDBaseline(n_components=n_comp)))

    # ---- WeightedSimilarity: overlap, fine top_k in 10–50 ----
    for top_k in WS_OVERLAP_K:
        configs.append((
            f"WS overlap k={top_k}",
            WeightedSimilarity(
                w_product_metadata=0,
                w_product_interactions=1,
                w_customer_metadata=0,
                w_customer_interactions=1,
                metric="overlap",
                top_k=top_k,
            ),
        ))
    for top_k in WS_OVERLAP_K_TAIL:
        configs.append((
            f"WS overlap k={top_k}",
            WeightedSimilarity(
                w_product_metadata=0,
                w_product_interactions=1,
                w_customer_metadata=0,
                w_customer_interactions=1,
                metric="overlap",
                top_k=top_k,
            ),
        ))

    # ---- WeightedSimilarity: cosine (sparse reference only) ----
    for top_k in WS_COSINE_K:
        configs.append((
            f"WS cosine k={top_k}",
            WeightedSimilarity(
                w_product_metadata=0,
                w_product_interactions=1,
                w_customer_metadata=0,
                w_customer_interactions=1,
                metric="cosine",
                top_k=top_k,
            ),
        ))

    # ---- WeightedSimilarity: small metadata blend (overlap) ----
    for meta_w in [0.02, 0.05, 0.08]:
        configs.append((
            f"WS overlap meta={meta_w} k=50",
            WeightedSimilarity(
                w_product_metadata=meta_w,
                w_product_interactions=1 - meta_w,
                w_customer_metadata=meta_w,
                w_customer_interactions=1 - meta_w,
                metric="overlap",
                top_k=50,
            ),
        ))

    # ---- WeightedSimilarity: per-matrix top_k (overlap, interaction-only) ----
    for k_ic, k_ip in [(25, 100), (30, 100), (40, 120), (50, 100), (50, 150)]:
        configs.append((
            f"WS ovl k_ic={k_ic} k_ip={k_ip}",
            WeightedSimilarity(
                w_product_metadata=0,
                w_product_interactions=1,
                w_customer_metadata=0,
                w_customer_interactions=1,
                metric="overlap",
                top_k=50,
                top_k_icsm=k_ic,
                top_k_ipsm=k_ip,
            ),
        ))

    # ---- BeliefPropagationRec: item–item loopy BP (reference points) ---- commented due to run time
    # for beta in [0.8, 1.2]:
    #     configs.append((
    #         f"BP item beta={beta} k=100",
    #         BeliefPropagationRec(top_k=100, beta=beta, max_iter=15, damping=0.5, show_progress=True),
    #     ))

    # ---- BipartiteGraphRec: 3-hop count, extend top_k ----
    for top_k in GRAPH_3HOP_COUNT_K:
        configs.append((
            f"Graph 3hop count k={top_k}",
            BipartiteGraphRec(
                hop_weights={3: 1.0},
                aggregation="count",
                top_k=top_k,
            ),
        ))

    # ---- BipartiteGraphRec: 3-hop multiply (reference only) ----
    for top_k in GRAPH_3HOP_MULT_K:
        configs.append((
            f"Graph 3hop multiply k={top_k}",
            BipartiteGraphRec(
                hop_weights={3: 1.0},
                aggregation="multiply",
                top_k=top_k,
            ),
        ))

    # ---- BipartiteGraphRec: lighter multi-hop (5-hop down-weighted vs prior sweep) ----
    for w5 in [0.05, 0.1]:
        configs.append((
            f"Graph 3+5 count w5={w5} k=100",
            BipartiteGraphRec(
                hop_weights={3: 1.0, 5: w5},
                aggregation="count",
                top_k=100,
            ),
        ))
    configs.append((
        "Graph 3+5+7 count k=100",
        BipartiteGraphRec(
            hop_weights={3: 1.0, 5: 0.15, 7: 0.05},
            aggregation="count",
            top_k=100,
        ),
    ))

    # ---- BipartiteGraphRec: penalties & groups (aligned with strong top_k) ----
    for pen in [0.4, 0.5, 0.6]:
        configs.append((
            f"Graph 3hop cnt k=150 pen={pen}",
            BipartiteGraphRec(
                hop_weights={3: 1.0},
                aggregation="count",
                top_k=150,
                customer_penalties={"country": pen},
            ),
        ))

    configs.append((
        "Graph 3hop cnt k=150 grp",
        BipartiteGraphRec(
            hop_weights={3: 1.0},
            aggregation="count",
            top_k=150,
            customer_group_cols=["country"],
        ),
    ))

    return configs


def main() -> None:
    print("Loading dataset …")
    data = load_online_retail()

    configs = build_configs()
    print(f"\nRunning {len(configs)} configs (K={K}) …\n")

    results = run_sweep(
        configs,
        data,
        k=K,
        output_path="benchmarks/results/sweep.csv",
    )

    print(f"\n{'=' * 80}")
    print(results.to_string(index=False))

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("\nInstall matplotlib to generate the summary plot: pip install matplotlib")
        return

    plot_path = plot_best_per_family(
        results,
        output_path="benchmarks/results/best_per_family.png",
        title=f"Best run per model (by NDCG@{K})",
    )
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
