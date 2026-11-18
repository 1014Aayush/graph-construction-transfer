import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controlled import CONTROLLED, graph_profile, matched_graph, unit_features  # noqa: E402


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    n, k = 900, 3
    y = rng.integers(0, k, n)
    X = rng.normal(size=(n, 20)) + 1.5 * np.eye(k, 20)[y]
    P = np.where(y[:, None] == y[None, :], 0.03, 0.006)          # homophilous "given" graph, h ~ 0.7
    A = np.triu(rng.random((n, n)) < P, 1)
    u, v = np.nonzero(A)
    given = np.concatenate([np.stack([u, v]), np.stack([v, u])], 1)
    return X, y, given, unit_features(X, 10)


def prof(ei, w):
    X, y, given, Z = w
    return graph_profile(ei, y, Z, len(y))


def test_all_controlled_graphs_valid_and_keep_homophily(world):
    X, y, given, Z = world
    base = prof(given, world)
    for name, (same, cross, deg, counts) in CONTROLLED.items():
        ei = matched_graph(Z, y, given, same, cross, deg, counts, seed=1)
        assert np.all(ei[0] != ei[1])
        assert {(a, b) for a, b in zip(*ei)} == {(b, a) for a, b in zip(*ei)}
        p = prof(ei, world)
        assert abs(p["h1"] - base["h1"]) < 0.06, (name, p["h1"], base["h1"])


def test_each_factor_moves_its_own_descriptor(world):
    X, y, given, Z = world
    g = lambda name, seed=1: prof(matched_graph(Z, y, given, *CONTROLLED[name], seed=seed), world)  # noqa: E731
    m0 = g("M0")
    assert g("same_near")["sim_same"] > m0["sim_same"] + 0.1 > g("same_far")["sim_same"] + 0.2
    assert g("cross_near")["sim_cross"] > m0["sim_cross"] + 0.1
    assert 1.6 < g("deg_double")["avg_degree"] / m0["avg_degree"] < 2.4
    assert 0.35 < g("deg_half")["avg_degree"] / m0["avg_degree"] < 0.65
    assert g("h_polarized")["h_node_sd"] > g("h_uniform")["h_node_sd"] + 0.1
    a, b = matched_graph(Z, y, given, *CONTROLLED["M0"], seed=1), matched_graph(Z, y, given, *CONTROLLED["M0"], seed=2)
    ea, eb = {tuple(e) for e in a.T}, {tuple(e) for e in b.T}
    assert len(ea & eb) / len(ea | eb) < 0.2          # resampled M0 shares few edges: a pure edge-identity change


def test_profile_two_hop_homophily_bounds(world):
    X, y, given, Z = world
    p = prof(given, world)
    assert 0 < p["h2"] < 1 and 0 < p["h1"] < 1 and p["isolated"] < 0.05
