import numpy as np
import pandas as pd
import pytest

from recs.models.bipartite_graph import BipartiteGraphRec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(n_customers=100, n_products=50, n_interactions=500, seed=42):
    rng = np.random.default_rng(seed)

    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n_customers)],
        "region": rng.choice(["north", "south", "east", "west"], n_customers),
        "segment": rng.choice(["budget", "mid", "premium"], n_customers),
    })

    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(n_products)],
        "category": rng.choice(["catA", "catB", "catC", "catD"], n_products),
    })

    cids = rng.choice(customers["customer_id"].values, n_interactions)
    pids = rng.choice(products["product_id"].values, n_interactions)
    interactions = pd.DataFrame({
        "customer_id": cids,
        "product_id": pids,
        "weight": rng.uniform(0.1, 5.0, n_interactions).round(2),
    }).drop_duplicates(subset=["customer_id", "product_id"])

    return customers, products, interactions


@pytest.fixture()
def data():
    return _make_data()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicFitRecommend:
    def test_3hop(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(hop_weights={3: 1.0}, top_k=20)
        model.fit(customers, products, interactions)

        recs = model.recommend(n=5)
        assert set(recs.columns) == {"customer_id", "product_id", "score", "rank"}
        assert recs["rank"].max() <= 5
        assert recs["score"].min() > 0

    def test_3hop_and_5hop(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(hop_weights={3: 1.0, 5: 0.3}, top_k=20)
        model.fit(customers, products, interactions)

        recs = model.recommend(n=5)
        assert not recs.empty
        assert recs["score"].min() > 0

    def test_excludes_already_seen(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(hop_weights={3: 1.0}, top_k=20)
        model.fit(customers, products, interactions)

        recs = model.recommend(n=50)
        seen = set(zip(interactions["customer_id"], interactions["product_id"]))
        recommended = set(zip(recs["customer_id"], recs["product_id"]))
        assert recommended.isdisjoint(seen)

    def test_include_seen(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(hop_weights={3: 1.0}, top_k=20)
        model.fit(customers, products, interactions)

        recs = model.recommend(n=50, exclude_seen=False)
        seen = set(zip(interactions["customer_id"], interactions["product_id"]))
        recommended = set(zip(recs["customer_id"], recs["product_id"]))
        assert len(recommended & seen) > 0

    def test_recommend_subset(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(hop_weights={3: 1.0}, top_k=20)
        model.fit(customers, products, interactions)

        subset = ["c0", "c1", "c2"]
        recs = model.recommend(customer_ids=subset, n=3)
        assert set(recs["customer_id"].unique()).issubset(set(subset))

    def test_not_fitted_raises(self):
        model = BipartiteGraphRec()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.recommend()


class TestAggregation:
    def test_count_vs_multiply_differ(self, data):
        customers, products, interactions = data

        model_mul = BipartiteGraphRec(
            hop_weights={3: 1.0}, aggregation="multiply", top_k=20
        )
        model_mul.fit(customers, products, interactions)

        model_cnt = BipartiteGraphRec(
            hop_weights={3: 1.0}, aggregation="count", top_k=20
        )
        model_cnt.fit(customers, products, interactions)

        recs_mul = model_mul.recommend(customer_ids=["c0"], n=5)
        recs_cnt = model_cnt.recommend(customer_ids=["c0"], n=5)

        if not recs_mul.empty and not recs_cnt.empty:
            scores_mul = recs_mul.set_index("product_id")["score"]
            scores_cnt = recs_cnt.set_index("product_id")["score"]
            common = scores_mul.index.intersection(scores_cnt.index)
            if len(common) > 0:
                assert not np.allclose(
                    scores_mul.loc[common].values,
                    scores_cnt.loc[common].values,
                )


class TestStrictGroups:
    def test_customer_group_filtering(self):
        """With strict customer groups, paths should not cross groups."""
        customers = pd.DataFrame({
            "customer_id": ["c0", "c1", "c2", "c3"],
            "region": ["A", "A", "B", "B"],
        })
        products = pd.DataFrame({
            "product_id": ["p0", "p1", "p2"],
            "category": ["x", "x", "x"],
        })
        # c0,c1 in group A interact with p0; c2,c3 in group B interact with p2
        # p1 is bought by c1 (group A) only
        interactions = pd.DataFrame({
            "customer_id": ["c0", "c1", "c1", "c2", "c3"],
            "product_id":  ["p0", "p0", "p1", "p2", "p2"],
            "weight":      [1.0,  1.0,  1.0,  1.0,  1.0],
        })

        model = BipartiteGraphRec(
            hop_weights={3: 1.0},
            customer_group_cols=["region"],
            top_k=10,
        )
        model.fit(customers, products, interactions)

        # c2 is in group B, should NOT get p1 recommended (only reachable
        # through c1 who is in group A)
        recs = model.recommend(customer_ids=["c2"], n=10)
        rec_products = set(recs["product_id"])
        assert "p1" not in rec_products

    def test_product_group_filtering(self):
        customers = pd.DataFrame({
            "customer_id": ["c0", "c1"],
            "region": ["A", "A"],
        })
        products = pd.DataFrame({
            "product_id": ["p0", "p1", "p2"],
            "category": ["X", "X", "Y"],
        })
        interactions = pd.DataFrame({
            "customer_id": ["c0", "c0", "c1", "c1"],
            "product_id":  ["p0", "p2", "p0", "p1"],
            "weight":      [1.0,  1.0,  1.0,  1.0],
        })

        # With product group filtering on category, p0-p2 link is cut
        # (different categories), but p0-p1 link preserved (same category X)
        model = BipartiteGraphRec(
            hop_weights={3: 1.0},
            product_group_cols=["category"],
            top_k=10,
        )
        model.fit(customers, products, interactions)
        # This mainly tests that it runs without error
        recs = model.recommend(n=10)
        assert isinstance(recs, pd.DataFrame)


class TestSoftPenalties:
    def test_penalties_reduce_scores(self, data):
        customers, products, interactions = data

        model_no_pen = BipartiteGraphRec(hop_weights={3: 1.0}, top_k=20)
        model_no_pen.fit(customers, products, interactions)

        model_pen = BipartiteGraphRec(
            hop_weights={3: 1.0}, top_k=20,
            customer_penalties={"region": 0.1, "segment": 0.1},
        )
        model_pen.fit(customers, products, interactions)

        recs_no = model_no_pen.recommend(customer_ids=["c0"], n=5)
        recs_pen = model_pen.recommend(customer_ids=["c0"], n=5)

        if not recs_no.empty and not recs_pen.empty:
            avg_no = recs_no["score"].mean()
            avg_pen = recs_pen["score"].mean()
            assert avg_pen < avg_no

    def test_single_float_penalty(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(
            hop_weights={3: 1.0}, top_k=20,
            customer_penalties=0.5,
        )
        model.fit(customers, products, interactions)
        recs = model.recommend(n=5)
        assert not recs.empty

    def test_per_feature_dict_penalty(self, data):
        customers, products, interactions = data
        model = BipartiteGraphRec(
            hop_weights={3: 1.0}, top_k=20,
            product_penalties={"category": 0.5},
        )
        model.fit(customers, products, interactions)
        recs = model.recommend(n=5)
        assert not recs.empty


class TestColdStart:
    def test_new_customer_gets_recs_and_excludes_synthetic_basket(self):
        customers = pd.DataFrame({
            "customer_id": ["seen", "cold"],
            "region": ["X", "X"],
        })
        products = pd.DataFrame({
            "product_id": ["p0", "p1", "p2", "p3"],
            "category": ["a", "a", "a", "a"],
        })
        interactions = pd.DataFrame({
            "customer_id": ["seen"] * 6,
            "product_id": ["p0", "p0", "p1", "p1", "p2", "p3"],
            "weight": [3.0, 2.0, 2.0, 1.0, 0.5, 0.5],
        })

        model = BipartiteGraphRec(
            hop_weights={3: 1.0},
            top_k=10,
            cold_start_n=2,
        )
        model.fit(customers, products, interactions)

        recs = model.recommend(customer_ids=["cold"], n=10, exclude_seen=True)
        out = set(recs["product_id"].tolist())
        assert "p0" not in out and "p1" not in out
        assert len(out) > 0

    def test_cold_start_falls_back_to_global_when_group_empty(self):
        customers = pd.DataFrame({
            "customer_id": ["a", "b"],
            "region": ["G1", "G2"],
        })
        products = pd.DataFrame({"product_id": ["p0", "p1"], "x": [0, 0]})
        interactions = pd.DataFrame({
            "customer_id": ["a", "a"],
            "product_id": ["p0", "p1"],
            "weight": [5.0, 1.0],
        })
        model = BipartiteGraphRec(
            hop_weights={3: 1.0},
            top_k=5,
            cold_start_n=1,
            customer_group_cols=["region"],
        )
        model.fit(customers, products, interactions)
        recs = model.recommend(customer_ids=["b"], n=2, exclude_seen=True)
        assert not recs.empty


class TestEdgeCases:
    def test_1hop(self):
        customers = pd.DataFrame({
            "customer_id": ["c0", "c1"],
            "region": ["A", "B"],
        })
        products = pd.DataFrame({
            "product_id": ["p0", "p1"],
            "category": ["x", "y"],
        })
        interactions = pd.DataFrame({
            "customer_id": ["c0", "c1"],
            "product_id": ["p0", "p1"],
            "weight": [1.0, 2.0],
        })
        model = BipartiteGraphRec(hop_weights={1: 1.0}, top_k=5)
        model.fit(customers, products, interactions)
        recs = model.recommend(n=5, exclude_seen=False)
        assert isinstance(recs, pd.DataFrame)

    def test_invalid_hop_raises(self):
        with pytest.raises(ValueError, match="positive odd"):
            BipartiteGraphRec(hop_weights={4: 1.0})

    def test_customer_no_interactions(self):
        customers = pd.DataFrame({
            "customer_id": ["c0", "c1"],
            "region": ["A", "B"],
        })
        products = pd.DataFrame({
            "product_id": ["p0", "p1"],
            "category": ["x", "y"],
        })
        interactions = pd.DataFrame({
            "customer_id": ["c0"],
            "product_id": ["p0"],
            "weight": [1.0],
        })
        model = BipartiteGraphRec(hop_weights={3: 1.0}, top_k=5)
        model.fit(customers, products, interactions)
        recs = model.recommend(n=5)
        assert isinstance(recs, pd.DataFrame)
