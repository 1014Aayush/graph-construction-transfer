import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipes import DEFAULT_RECIPES, V1_RECIPES, Recipe, build, jaccard, stats  # noqa: E402


def data(n=300, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, d)) + 2.0 * np.eye(3, d)[y]      # class-separated features
    return X, y


def undirected(ei):
    u, v = ei
    return set(zip(np.minimum(u, v).tolist(), np.maximum(u, v).tolist()))


def given_graph(y, seed=0, p_in=0.05, p_out=0.005):
    """Toy 'real' graph: mostly same-class links."""
    rng = np.random.default_rng(seed)
    n = len(y)
    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    A = np.triu(rng.random((n, n)) < P, 1)
    u, v = np.nonzero(A)
    return np.concatenate([np.stack([u, v]), np.stack([v, u])], 1)


@pytest.mark.parametrize("r", V1_RECIPES + DEFAULT_RECIPES, ids=lambda r: r.name)
def test_symmetric_no_self_loops(r):
    X, y = data()
    ei = build(X, r, given=given_graph(y))
    arcs = set(map(tuple, ei.T))
    assert all((b, a) in arcs for a, b in arcs)
    assert not (ei[0] == ei[1]).any()
    assert ei.shape[1] == 2 * len(undirected(ei))


def test_knn_min_degree_is_k():
    X, _ = data()
    ei = build(X, Recipe("knn", 10, "cosine"))
    deg = np.bincount(ei[0], minlength=len(X))
    assert deg.min() >= 10          # union of top-k lists: every node has at least k neighbours


def test_mutual_is_subset_of_knn():
    X, _ = data()
    knn = undirected(build(X, Recipe("knn", 10, "cosine")))
    mut = undirected(build(X, Recipe("mutual_knn", 10, "cosine")))
    assert mut <= knn and len(mut) < len(knn)


def test_eps_matches_knn_edge_count():
    X, _ = data()
    n_knn = len(undirected(build(X, Recipe("knn", 10, "cosine"))))
    n_eps = len(undirected(build(X, Recipe("eps", 10, "cosine"))))
    assert n_eps == n_knn


def test_more_neighbours_more_edges_and_features_give_homophily():
    X, y = data()
    s3 = stats(build(X, Recipe("knn", 3, "cosine")), y, len(X))
    s30 = stats(build(X, Recipe("knn", 30, "cosine")), y, len(X))
    assert s30["avg_degree"] > s3["avg_degree"]
    assert s3["homophily"] > 0.5      # class-separated features -> mostly same-class links


def test_descriptors_sensible():
    from recipes import descriptors
    X, _ = data()
    d3 = descriptors(build(X, Recipe("knn", 3, "cosine")), X, len(X))
    d30 = descriptors(build(X, Recipe("knn", 30, "cosine")), X, len(X))
    assert d30["avg_degree"] > d3["avg_degree"]
    assert d3["feat_sim"] > d30["feat_sim"]            # nearer neighbours are more similar
    assert 0.0 <= d3["spectral_gap"] <= 2.0 and d30["spectral_gap"] >= d3["spectral_gap"] - 0.05
    assert d3["isolated"] == 0.0
    # two disconnected cliques -> spectral gap ~ 0
    ei = np.array([[0, 1, 1, 0, 2, 3, 3, 2], [1, 0, 0, 1, 3, 2, 2, 3]])
    assert descriptors(ei, np.eye(4), 4)["spectral_gap"] < 1e-3


def test_jaccard_bounds():
    X, _ = data()
    a = build(X, Recipe("knn", 10, "cosine"))
    assert jaccard(a, a) == 1.0
    b = build(X, Recipe("knn", 10, "euclidean"))
    assert 0.0 <= jaccard(a, b) <= 1.0


def test_given_recipes():
    X, y = data()
    g = given_graph(y)
    G = undirected(build(X, Recipe("given", 0, "none"), given=g))
    assert G == undirected(g)
    GK = undirected(build(X, Recipe("given_knn", 5, "pca"), given=g))
    assert G <= GK and len(GK) > len(G)
    with pytest.raises(ValueError):
        build(X, Recipe("given", 0, "none"))


def test_ppr_degree_and_homophily():
    X, y = data()
    g = given_graph(y)
    ei = build(X, Recipe("ppr", 16, "none"), given=g)
    deg = np.bincount(ei[0], minlength=len(X))
    linked = np.bincount(g[0], minlength=len(X)) > 0
    assert deg[linked].min() >= 1 and np.median(deg) >= 16   # isolated nodes stay isolated (nothing to diffuse)
    assert (deg[~linked] == 0).all()
    assert stats(ei, y, len(X))["homophily"] > 0.6          # diffusion of a homophilous graph stays homophilous


def test_default_recipe_set():
    assert len(DEFAULT_RECIPES) == 11 and len({r.name for r in DEFAULT_RECIPES}) == 11


def test_embknn_uses_embeddings():
    X, y = data()
    emb = np.eye(3)[y] + 0.01 * np.random.default_rng(1).normal(size=(len(y), 3))   # near-perfect embeddings
    ei = build(X, Recipe("embknn", 10, "none"), emb=emb)
    assert stats(ei, y, len(X))["homophily"] > 0.95
    with pytest.raises(ValueError):
        build(X, Recipe("embknn", 10, "none"))


def _deg(ei, n):
    return np.bincount(ei[0], minlength=n)


@pytest.mark.parametrize("kind", ["rw_keep", "rw_mix"])
def test_rewire_preserves_degree_and_changes_requested_share(kind):
    X, y = data(n=400)
    g = build(X, Recipe("given", 0, "none"), given=given_graph(y))
    r = build(X, Recipe(kind, 50, "none"), given=given_graph(y), labels=y)
    assert np.array_equal(_deg(g, len(y)), _deg(r, len(y)))
    changed = 1 - len(undirected(g) & undirected(r)) / len(undirected(g))
    assert 0.48 <= changed <= 0.52
    assert np.all(r[0] != r[1]) and undirected(r) == {(min(a, b), max(a, b)) for a, b in zip(*r)}


def test_rewire_keep_preserves_neighbour_classes_mix_does_not():
    X, y = data(n=400)
    G = given_graph(y)
    g = build(X, Recipe("given", 0, "none"), given=G)

    def neigh_class_counts(ei):
        C = np.zeros((len(y), 3))
        np.add.at(C, (ei[0], y[ei[1]]), 1)
        return C
    keep = build(X, Recipe("rw_keep", 50, "none"), given=G, labels=y)
    mix = build(X, Recipe("rw_mix", 50, "none"), given=G, labels=y)
    assert np.array_equal(neigh_class_counts(g), neigh_class_counts(keep))
    assert stats(mix, y, len(y))["homophily"] < stats(g, y, len(y))["homophily"] - 0.1
    with pytest.raises(ValueError):
        build(X, Recipe("rw_mix", 50, "none"), given=G)


from recipes import V5_REALISTIC  # noqa: E402


def test_v5_realistic_recipes_valid_and_sensible():
    X, y = data(n=400)
    G = given_graph(y)
    g = build(X, Recipe("given", 0, "none"), given=G)
    m = len(undirected(g))
    for r in V5_REALISTIC:
        ei = build(X, r, given=G)
        assert np.all(ei[0] != ei[1]) and undirected(ei) == {(min(a, b), max(a, b)) for a, b in zip(*ei)}, r.name
        assert ei.shape[1] > 0, r.name
    assert abs(len(undirected(build(X, Recipe("drop", 50, "none"), given=G))) / m - 0.5) < 0.06
    pr = build(X, Recipe("prune", 30, "pca"), given=G)
    assert undirected(pr) <= undirected(g) and abs(len(undirected(pr)) / m - 0.7) < 0.02
    assert stats(pr, y, len(y))["homophily"] >= stats(g, y, len(y))["homophily"] - 0.01   # pruning dissimilar edges
    two = build(X, Recipe("twohop", 10, "none"), given=G)
    assert undirected(g) < undirected(two)
    heat = build(X, Recipe("heat", 16, "none"), given=G)
    assert stats(heat, y, len(y))["homophily"] > 0.6
