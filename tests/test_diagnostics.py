import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diagnostics import (adjacency, agg_shift, compat, compat_shift, local_homophily,  # noqa: E402
                         neighbour_mean, pseudo_homophily)


def sym(pairs):
    p = np.array(pairs).T
    return np.concatenate([p, p[::-1]], 1)


def test_local_homophily_and_isolated_skip():
    y = np.array([0, 0, 1, 1, 0])
    ei = sym([(0, 1), (0, 2), (2, 3)])            # node 4 isolated
    A = adjacency(ei, 5)
    # node0: neighbours 1(same),2(diff) -> 0.5 ; node2: 0(diff),3(same) -> 0.5 ; node4 skipped
    assert abs(local_homophily(A, y, np.array([0, 2, 4])) - 0.5) < 1e-9


def test_compat_and_shift():
    y = np.array([0, 0, 1, 1])
    same = sym([(0, 1), (2, 3)])
    cross = sym([(0, 2), (1, 3)])
    assert np.allclose(compat(same, y, 2), np.eye(2))
    assert np.allclose(compat(cross, y, 2), 1 - np.eye(2))
    assert compat_shift(compat(same, y, 2), compat(same, y, 2)) == 0.0
    assert abs(compat_shift(compat(same, y, 2), compat(cross, y, 2)) - 1.0) < 1e-9


def test_agg_shift_zero_for_same_graph_and_positive_otherwise():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(6, 4)).astype(np.float32)
    a = adjacency(sym([(0, 1), (1, 2), (3, 4), (4, 5)]), 6)
    b = adjacency(sym([(0, 3), (1, 4), (2, 5)]), 6)
    nodes = np.arange(6)
    assert abs(agg_shift(neighbour_mean(a, Z), neighbour_mean(a, Z), nodes)) < 1e-6
    assert agg_shift(neighbour_mean(a, Z), neighbour_mean(b, Z), nodes) > 0.1


def test_pseudo_homophily():
    pred = np.array([0, 0, 1, 1])
    assert pseudo_homophily(sym([(0, 1), (2, 3)]), pred) == 1.0
    assert pseudo_homophily(sym([(0, 2)]), pred) == 0.0


from diagnostics import atc_estimate, atc_threshold, doc_estimate, ego_distance, ego_spectra, soft_shift  # noqa: E402


def test_atc_threshold_reproduces_source_accuracy():
    rng = np.random.default_rng(0)
    conf = rng.random(1000)
    correct = rng.random(1000) < 0.7
    t = atc_threshold(conf, correct)
    assert abs(atc_estimate(conf, t) - correct.mean()) < 0.002
    assert atc_threshold(conf, np.ones(10, bool)) == -np.inf and atc_estimate(conf, -np.inf) == 1.0


def test_doc_and_soft_shift_identities():
    rng = np.random.default_rng(1)
    c = rng.random(100)
    assert abs(doc_estimate(0.8, c, c) - 0.8) < 1e-12
    assert abs(doc_estimate(0.8, c, c - 0.1) - 0.7) < 1e-9
    P = rng.dirichlet(np.ones(4), 50)
    nodes = np.arange(50)
    assert soft_shift(P, P, nodes) == 0.0
    Q = np.roll(np.eye(4), 1, 1)[np.argmax(P, 1)]          # all mass moved to another class
    assert soft_shift(np.eye(4)[np.argmax(P, 1)], Q, nodes) == 1.0


def test_ego_distance_zero_for_same_graph_positive_for_different():
    ring = adjacency(sym([(i, (i + 1) % 30) for i in range(30)]), 30)
    star = adjacency(sym([(0, i) for i in range(1, 30)]), 30)
    nodes = np.arange(0, 30, 3)
    sa, sb = ego_spectra(ring, nodes), ego_spectra(star, nodes)
    assert ego_distance(sa, sa) == 0.0
    assert ego_distance(sa, sb) > 0.3
    assert np.allclose(sa.sum(1), 1.0)


from diagnostics import (cot_estimate, cott_estimate, cott_threshold, energy_score, frechet_distance,  # noqa: E402
                         gcn_operator, mano_score, nuclear_score, op_shift, softmax_np, spectrum_distance, top_spectrum)


def test_prediction_matrix_scores_prefer_confident_spread_predictions():
    confident = np.eye(4)[np.arange(400) % 4] * 0.97 + 0.0075
    unsure = np.full((400, 4), 0.25)
    collapsed = np.tile([0.97, 0.01, 0.01, 0.01], (400, 1))
    assert nuclear_score(confident) > nuclear_score(collapsed) > nuclear_score(unsure) - 1e-9
    big = np.log(confident)
    assert mano_score(big) > mano_score(np.zeros((400, 4)))
    lg = np.eye(4)[np.arange(400) % 4]
    assert energy_score(5 * lg) > energy_score(lg)                 # larger confident logit -> higher score


def test_cot_is_exact_for_perfect_confident_predictions_and_lower_when_unsure():
    y = np.arange(400) % 4
    prior = np.bincount(y) / len(y)
    P = np.eye(4)[y] * 0.999 + 0.00025
    assert cot_estimate(P, prior) > 0.97
    assert cot_estimate(np.full((400, 4), 0.25), prior) < 0.3


def test_cott_threshold_reproduces_source_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    P = softmax_np(3 * np.eye(3)[y] + rng.normal(0, 1.5, (600, 3)))
    prior = np.bincount(y, minlength=3) / len(y)
    t = cott_threshold(P, y, prior)
    acc = float(np.mean(P.argmax(1) == y))
    assert abs(cott_estimate(P, prior, t) - acc) < 0.03


def test_frechet_operator_and_spectrum():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(300, 5))
    assert abs(frechet_distance(L, L)) < 1e-6 and frechet_distance(L, L + 2) > 19
    ring = sym([(i, (i + 1) % 40) for i in range(40)])
    star = sym([(0, i) for i in range(1, 40)])
    Z = rng.normal(size=(40, 6))
    Sa, Sb = gcn_operator(ring, 40), gcn_operator(star, 40)
    assert op_shift(Sa, Sa, Z, np.arange(40)) == 0.0 and op_shift(Sa, Sb, Z, np.arange(40)) > 0.3
    ea, eb = top_spectrum(Sa, 8), top_spectrum(Sb, 8)
    assert abs(ea[0] - 1) < 1e-6 and spectrum_distance(ea, ea) == 0 and spectrum_distance(ea, eb) > 0.05


from diagnostics import (corrected_pseudo_homophily, feature_mixture_homophily, labelled_edge_homophily,  # noqa: E402
                         pseudo_match_rates, pseudo_same_class_similarity)


def _homophilous_world(h_target, n=3000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    Z = rng.normal(size=(n, 16)) + 1.2 * np.eye(k, 16)[y]
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    m = 20000
    u = rng.integers(0, n, m)
    same = rng.random(m) < h_target
    v = np.empty(m, int)
    for i in range(m):
        pool = np.flatnonzero((y == y[u[i]]) if same[i] else (y != y[u[i]]))
        v[i] = rng.choice(pool)
    keep = u != v
    return y, Z, np.stack([u[keep], v[keep]]), rng


@pytest.mark.parametrize("h", [0.3, 0.6, 0.85])
def test_label_light_homophily_estimators_recover_truth(h):
    y, Z, ei, rng = _homophilous_world(h)
    true = float((y[ei[0]] == y[ei[1]]).mean())
    lab = np.zeros(len(y), bool)
    lab[rng.choice(len(y), 600, replace=False)] = True
    hl, n_e = labelled_edge_homophily(ei, y, lab)
    assert n_e > 100 and abs(hl - true) < 0.08
    acc = 0.75
    pseudo = np.where(rng.random(len(y)) < acc, y, (y + rng.integers(1, 4, len(y))) % 4)   # symmetric errors
    hc, obs = corrected_pseudo_homophily(ei, pseudo, acc, 4)
    assert abs(hc - true) < 0.04 and abs(obs - true) > abs(hc - true)                  # correction helps
    hf = feature_mixture_homophily(ei, Z, y, np.flatnonzero(lab))
    assert abs(hf - true) < 0.06


def test_pseudo_match_rates_and_similarity():
    ms, md = pseudo_match_rates(1.0, 5)
    assert ms == 1.0 and md == 0.0
    ms, md = pseudo_match_rates(0.2, 5)                       # chance-level classifier: no information
    assert abs(ms - md) < 1e-12
    y, Z, ei, rng = _homophilous_world(0.8, n=800)
    assert pseudo_same_class_similarity(ei, Z, y) > pseudo_same_class_similarity(ei, Z, rng.integers(0, 4, len(y)))


from diagnostics import em_homophily  # noqa: E402


def _correlated_error_world(h_target, seed=0, n=2400, k=4):
    """Linked nodes are chosen to look alike, so a nearest-class-mean classifier makes correlated mistakes on
    the two ends of an edge: the exact situation that broke the v6 correction."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    Z = rng.normal(size=(n, 8)) + 0.9 * np.eye(k, 8)[y]
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    means = np.stack([Z[y == c].mean(0) for c in range(k)])
    pseudo = (Z @ means.T).argmax(1)
    S = Z @ Z.T
    np.fill_diagonal(S, -9)
    rows, cols = [], []
    for i in range(n):
        same = rng.random() < h_target
        cand = np.flatnonzero((y == y[i]) if same else (y != y[i]))
        j = cand[np.argsort(-S[i, cand])[rng.integers(0, 15)]]          # one of the 15 most similar
        rows.append(i), cols.append(j)
    ei = np.array([rows + cols, cols + rows])
    return y, Z, pseudo, ei, rng


def _setup(y, Z, rng):
    from sklearn.linear_model import LogisticRegression
    perm = rng.permutation(len(y))
    tr, va = perm[:160], perm[160:400]
    lab = np.zeros(len(y), bool)
    lab[np.concatenate([tr, va])] = True
    Q = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z)
    return lab, Q, np.bincount(y[tr], minlength=4) / len(tr)


@pytest.mark.parametrize("h", [0.5, 0.8])
def test_em_homophily_recovers_truth_when_edges_are_not_feature_chosen(h):
    y, Z, pseudo, _, rng = _correlated_error_world(h)
    r, c = [], []
    for i in range(len(y)):
        same = rng.random() < h
        j = rng.choice(np.flatnonzero((y == y[i]) if same else (y != y[i])))
        r.append(i), c.append(j)
    ei = np.array([r + c, c + r])
    true = float((y[ei[0]] == y[ei[1]]).mean())
    lab, Q, prior = _setup(y, Z, rng)
    est, n_e = em_homophily(ei, y, lab, Q, prior)
    assert n_e > 300 and abs(est - true) < 0.05


def test_known_limit_feature_chosen_edges_bias_every_feature_based_estimator_upward():
    """On kNN-like graphs (neighbours picked because they look alike) features cannot separate same-class from
    look-alike different-class neighbours; the estimators over-estimate. Documented limitation, not a bug."""
    y, Z, pseudo, ei, rng = _correlated_error_world(0.5)
    true = float((y[ei[0]] == y[ei[1]]).mean())
    lab, Q, prior = _setup(y, Z, rng)
    assert em_homophily(ei, y, lab, Q, prior)[0] > true + 0.1
    acc = float((pseudo[lab] == y[lab]).mean())
    assert corrected_pseudo_homophily(ei, pseudo, acc, 4)[0] > true + 0.1
