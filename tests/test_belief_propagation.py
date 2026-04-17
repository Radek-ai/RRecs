import numpy as np
import pandas as pd
import pytest

from recs.models.belief_propagation import BeliefPropagationRec


def test_bp_prefers_co_occurring_item():
    """Item adjacent to the basket in S should score above an isolated SKU."""
    customers = pd.DataFrame({
        "customer_id": ["c0", "c1", "c2", "c3"],
        "x": [0, 0, 0, 0],
    })
    products = pd.DataFrame({
        "product_id": ["p0", "p1", "p2", "p3"],
        "x": [0, 0, 0, 0],
    })
    interactions = pd.DataFrame({
        "customer_id": ["c0", "c1", "c2", "c2", "c3"],
        "product_id": ["p0", "p1", "p0", "p1", "p0"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    })

    model = BeliefPropagationRec(top_k=10, beta=1.5, max_iter=25, damping=0.4)
    model.fit(customers, products, interactions)

    s = model._scores_for_user(
        model._interactions.getrow(model._customer_idx.get_loc("c3")),
        model._S,
        len(model._product_idx),
    )
    i1 = model._product_idx.get_loc("p1")
    i2 = model._product_idx.get_loc("p2")
    assert s[i1] > s[i2]


def test_bp_chain_propagation():
    """p0–p1–p2 chain: buying p0 should lift p2 vs isolated p3."""
    customers = pd.DataFrame({"customer_id": ["c0", "c1", "c2"], "x": [0, 0, 0]})
    products = pd.DataFrame({
        "product_id": ["p0", "p1", "p2", "p3"],
        "x": [0, 0, 0, 0],
    })
    interactions = pd.DataFrame({
        "customer_id": ["c0", "c0", "c1", "c1", "c2", "c2"],
        "product_id": ["p0", "p1", "p1", "p2", "p2", "p3"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    model = BeliefPropagationRec(top_k=10, beta=1.2, max_iter=25, damping=0.45)
    model.fit(customers, products, interactions)

    s = model._scores_for_user(
        model._interactions.getrow(0),
        model._S,
        len(model._product_idx),
    )
    # c0 only bought p0; p2 is two hops via p1, p3 only touches c2
    i2 = model._product_idx.get_loc("p2")
    i3 = model._product_idx.get_loc("p3")
    assert s[i2] > s[i3]


def test_fit_recommend_smoke():
    rng = np.random.default_rng(0)
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(20)],
        "z": rng.integers(0, 3, 20),
    })
    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(15)],
        "z": rng.integers(0, 3, 15),
    })
    cids = rng.choice(customers["customer_id"].values, 80)
    pids = rng.choice(products["product_id"].values, 80)
    interactions = pd.DataFrame({
        "customer_id": cids,
        "product_id": pids,
        "weight": rng.uniform(0.5, 2.0, 80),
    }).drop_duplicates(subset=["customer_id", "product_id"])

    model = BeliefPropagationRec(top_k=8, max_iter=8)
    model.fit(customers, products, interactions)
    recs = model.recommend(n=5)
    assert set(recs.columns) == {"customer_id", "product_id", "score", "rank"}
    assert recs["rank"].max() <= 5


def test_not_fitted_raises():
    m = BeliefPropagationRec()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.recommend()
