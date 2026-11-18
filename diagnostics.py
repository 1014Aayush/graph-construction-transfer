"""Torch-free diagnostics for the two-step analysis (unit-tested locally)."""
import numpy as np
import scipy.sparse as sp


def adjacency(ei, n):
    return sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()


def neighbour_mean(A, Z):
    """Each node's mean neighbour feature vector, and whether it has any neighbour."""
    deg = np.asarray(A.sum(1)).ravel()
    return (A @ Z) / np.maximum(deg, 1)[:, None], deg > 0


def agg_shift(nm_a, nm_b, nodes):
    """Mean cosine distance between nodes' neighbour-averages under two graphs (label-free)."""
    (Ma, oka), (Mb, okb) = nm_a, nm_b
    m = nodes[oka[nodes] & okb[nodes]]
    if not len(m):
        return float("nan")
    num = (Ma[m] * Mb[m]).sum(1)
    den = np.linalg.norm(Ma[m], axis=1) * np.linalg.norm(Mb[m], axis=1)
    return float((1 - num / np.maximum(den, 1e-12)).mean())


def local_homophily(A, labels, nodes):
    """Mean share of same-label neighbours over the given nodes (nodes without neighbours skipped)."""
    sub = A[nodes].tocoo()
    same = (labels[nodes][sub.row] == labels[sub.col]).astype(np.float32)
    num = np.bincount(sub.row, weights=same, minlength=len(nodes))
    den = np.bincount(sub.row, minlength=len(nodes))
    ok = den > 0
    return float((num[ok] / den[ok]).mean()) if ok.any() else float("nan")


def compat(ei, labels, k):
    """Row-normalised class-to-class link matrix."""
    C = np.zeros((k, k))
    np.add.at(C, (labels[ei[0]], labels[ei[1]]), 1)
    return C / np.maximum(C.sum(1, keepdims=True), 1)


def compat_shift(ca, cb):
    """Mean total-variation distance between two class-to-class link patterns."""
    return float(0.5 * np.abs(ca - cb).sum(1).mean())


def pseudo_homophily(ei, pred):
    """Share of edges whose endpoints get the same predicted label (label-free homophily)."""
    return float((pred[ei[0]] == pred[ei[1]]).mean()) if ei.shape[1] else float("nan")


# ---- v4: comparing two graphs at three levels (structure / model behaviour / calibrated confidence) ----

def atc_threshold(conf_val, correct_val):
    """ATC (Garg et al., ICLR 2022): threshold t such that the share of source-validation nodes with
    confidence > t equals the source-validation accuracy."""
    s = np.sort(np.asarray(conf_val, dtype=float))
    k = int(round((1 - float(np.mean(correct_val))) * len(s)))
    if k <= 0:
        return -np.inf
    if k >= len(s):
        return np.inf
    return float(s[k - 1])


def atc_estimate(conf, t):
    """Estimated accuracy on the target: share of unlabelled target nodes with confidence above t."""
    return float(np.mean(np.asarray(conf) > t))


def doc_estimate(acc_src, conf_src, conf_tgt):
    """DoC (Guillory et al., 2021): source accuracy minus the drop in average confidence."""
    return float(acc_src - (np.mean(conf_src) - np.mean(conf_tgt)))


def soft_shift(P_a, P_b, nodes):
    """Mean total-variation distance between the model's class-probability vectors under graph A and B."""
    return float(0.5 * np.abs(P_a[nodes] - P_b[nodes]).sum(1).mean())


def ego_spectra(A, nodes, hops=2, max_size=100, bins=20, seed=0):
    """EGI-style structural signature (Zhu et al., NeurIPS 2021): for each node, the eigenvalue histogram
    of the normalised Laplacian of its `hops`-hop ego-graph (capped at `max_size` nodes by sampling).
    Histograms on [0, 2] make graphs of different sizes comparable. Structure only: no features, no model."""
    rng = np.random.default_rng(seed)
    A = sp.csr_matrix(A)
    out = np.zeros((len(nodes), bins))
    for i, v in enumerate(nodes):
        ego, frontier = {int(v)}, {int(v)}
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                nxt.update(A.indices[A.indptr[u]:A.indptr[u + 1]].tolist())
            frontier = nxt - ego
            ego |= nxt
        ego = list(ego)
        if len(ego) > max_size:
            others = [u for u in ego if u != v]
            ego = [int(v)] + rng.choice(others, max_size - 1, replace=False).tolist()
        S = A[ego][:, ego].toarray()
        d = S.sum(1)
        dinv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
        L = np.eye(len(ego)) - dinv[:, None] * S * dinv[None, :]
        ev = np.clip(np.linalg.eigvalsh(L), 0, 2)
        out[i] = np.histogram(ev, bins=bins, range=(0, 2))[0] / len(ev)
    return out


def ego_distance(spec_a, spec_b):
    """Mean per-node L1 distance between ego-graph spectra of the same nodes under two graphs."""
    return float(np.abs(spec_a - spec_b).sum(1).mean())


# ---- v4b: estimators from the label-free accuracy-estimation literature (ODP-Bench, ICCV 2025) ----

def softmax_np(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nuclear_score(P):
    """Normalised nuclear norm of the prediction matrix (Deng et al., ICML 2023): high when predictions
    are both confident and spread over classes."""
    n, c = P.shape
    ev = np.clip(np.linalg.eigvalsh(P.T @ P), 0, None)
    return float(np.sqrt(ev).sum() / np.sqrt(min(n, c) * n))


def mano_score(logits, p=4):
    """MaNo (Xie et al., NeurIPS 2024): normalised L_p norm of the softmax prediction matrix."""
    P = softmax_np(logits)
    return float(np.mean(np.abs(P) ** p) ** (1 / p))


def energy_score(logits):
    """Mean negative energy, log-sum-exp of the logits (energy-based evaluation, MDE, Peng et al., ICLR 2024)."""
    m = logits.max(1, keepdims=True)
    return float(np.mean(m.ravel() + np.log(np.exp(logits - m).sum(1))))


def _sinkhorn_plan(cost, a, b, eps=0.02, iters=150):
    """Entropic optimal transport plan (log domain)."""
    f, g = np.zeros(len(a)), np.zeros(len(b))
    la, lb = np.log(np.maximum(a, 1e-300)), np.log(np.maximum(b, 1e-300))
    K = -cost / eps
    for _ in range(iters):
        f = eps * (la - _lse(K + g[None, :] / eps, 1))
        g = eps * (lb - _lse(K + f[:, None] / eps, 0))
    return np.exp(K + f[:, None] / eps + g[None, :] / eps)


def _lse(x, axis):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def cot_plan(P, prior):
    """Transport the predicted probability vectors onto one-hot labels with the source class prior
    (COT, Lu et al., NeurIPS 2023). Cost of sending p to class j is 1 - p_j (= half the L1 distance)."""
    cost = 1 - P
    return _sinkhorn_plan(cost, np.full(len(P), 1 / len(P)), prior), cost


def cot_estimate(P, prior):
    """COT accuracy estimate: 1 - optimal transport cost."""
    plan, cost = cot_plan(P, prior)
    return float(1 - (plan * cost).sum())


def cott_threshold(P_val, y_val, prior):
    """COTT: threshold t on transport cost such that the transported mass with cost > t equals the
    source-validation error."""
    plan, cost = cot_plan(P_val, prior)
    err = 1 - float(np.mean(P_val.argmax(1) == y_val))
    order = np.argsort(-cost.ravel())
    mass = np.cumsum(plan.ravel()[order])
    k = int(np.searchsorted(mass, err))
    return float(cost.ravel()[order][min(k, len(order) - 1)])


def cott_estimate(P, prior, t):
    plan, cost = cot_plan(P, prior)
    return float(1 - plan[cost > t].sum())


def frechet_distance(La, Lb):
    """Frechet distance between two sets of output vectors (AutoEval, Deng & Zheng, CVPR 2021)."""
    from scipy.linalg import sqrtm
    ma, mb = La.mean(0), Lb.mean(0)
    Ca, Cb = np.cov(La, rowvar=False), np.cov(Lb, rowvar=False)
    s = sqrtm(Ca @ Cb)
    s = s.real if np.iscomplexobj(s) else s
    return float(((ma - mb) ** 2).sum() + np.trace(Ca + Cb - 2 * s))


def gcn_operator(ei, n):
    """Symmetric-normalised adjacency with self-loops, D^-1/2 (A + I) D^-1/2."""
    A = adjacency(ei, n) + sp.eye(n, format="csr")
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d))
    return (Dm @ A @ Dm).tocsr()


def op_shift(Sa, Sb, Z, nodes):
    """Relative change of one propagation step on the features, ||(S_B - S_A) Z|| / ||S_A Z|| on the given
    nodes: the graph-difference term in GNN transferability bounds (graphon / spectral theory)."""
    Ha, Hb = (Sa @ Z)[nodes], (Sb @ Z)[nodes]
    return float(np.linalg.norm(Hb - Ha) / max(np.linalg.norm(Ha), 1e-12))


def top_spectrum(S, k=32):
    """Largest k eigenvalues of a symmetric propagation operator, sorted."""
    from scipy.sparse.linalg import eigsh
    k = min(k, S.shape[0] - 2)
    try:
        ev = eigsh(S, k=k, which="LA", return_eigenvectors=False, tol=1e-4, maxiter=5000)
    except Exception:
        return np.full(k, np.nan)
    return np.sort(ev)[::-1]


def spectrum_distance(ea, eb):
    return float(np.mean(np.abs(ea - eb)))


def cot_both(P, prior, t):
    """COT and COTT accuracy estimates from one transport plan."""
    plan, cost = cot_plan(P, prior)
    return float(1 - (plan * cost).sum()), float(1 - plan[cost > t].sum())


# ---- v6: label-free (label-light) estimates of the two causes found in v5 -----------------------------
# Only the already-labelled training/validation nodes, node features and a feature-only MLP are used;
# nothing is labelled on the deployment graph.

def labelled_edge_homophily(ei, y, lab):
    """Homophily over edges whose both ends are labelled nodes (unbiased, few edges). Returns (h, n_edges)."""
    m = lab[ei[0]] & lab[ei[1]]
    return (float((y[ei[0]][m] == y[ei[1]][m]).mean()) if m.any() else float("nan")), int(m.sum())


def pseudo_match_rates(acc, k):
    """Symmetric-error model of a classifier with accuracy `acc` over k classes: probability that the two
    ends of an edge get the same predicted label, for a same-class edge and for a different-class edge."""
    e = (1 - acc) / (k - 1)
    m_same = acc ** 2 + (k - 1) * e ** 2
    m_diff = 2 * acc * e + (k - 2) * e ** 2
    return m_same, m_diff


def corrected_pseudo_homophily(ei, pseudo, acc, k):
    """Share of edges whose ends get the same pseudo-label, corrected for pseudo-label errors:
    h = (observed - m_diff) / (m_same - m_diff), clipped to [0, 1]."""
    obs = float((pseudo[ei[0]] == pseudo[ei[1]]).mean())
    ms, md = pseudo_match_rates(acc, k)
    return float(np.clip((obs - md) / max(ms - md, 1e-9), 0, 1)), obs


def feature_mixture_homophily(ei, Z, y, lab_idx, bins=40, iters=200, n_pairs=20000, seed=0):
    """Edge cosine similarities are a mixture of 'same-class pair' and 'different-class pair' similarity
    distributions; both distributions are learned from random pairs of labelled nodes, the mixing weight
    (= homophily) is fitted to the graph's edges by EM with the components held fixed."""
    rng = np.random.default_rng(seed)
    a, b = rng.choice(lab_idx, n_pairs), rng.choice(lab_idx, n_pairs)
    ok = a != b
    a, b = a[ok], b[ok]
    cos_pairs = (Z[a] * Z[b]).sum(1)
    same = y[a] == y[b]
    edges = np.linspace(-1, 1, bins + 1)
    f_s = np.histogram(cos_pairs[same], edges, density=True)[0] + 1e-3
    f_d = np.histogram(cos_pairs[~same], edges, density=True)[0] + 1e-3
    c = np.clip(np.digitize((Z[ei[0]] * Z[ei[1]]).sum(1), edges) - 1, 0, bins - 1)
    ps, pd = f_s[c], f_d[c]
    h = 0.5
    for _ in range(iters):
        h = float(np.mean(h * ps / (h * ps + (1 - h) * pd)))
    return h


def pseudo_same_class_similarity(ei, Z, pseudo):
    """Mean feature cosine over edges whose ends share a pseudo-label (label-free estimate of sim_same)."""
    m = pseudo[ei[0]] == pseudo[ei[1]]
    return float((Z[ei[0]][m] * Z[ei[1]][m]).sum(1).mean()) if m.any() else float("nan")


# ---- v7: latent-label (EM) homophily. v6's independent-error correction over-shot because linked nodes look
# alike; tried and dropped (synthetic test, 2026-10-05): half-labelled edges and similarity-conditioned rates,
# which still missed by ~0.17. The EM estimator below works unless edges are chosen by feature similarity.

def em_homophily(ei, y, lab, Q, prior, iters=200):
    """Latent-label homophily over edges with a labelled source u: the neighbour's unknown class y_v gets a
    feature posterior Q[v] (calibrated feature-only classifier) and an edge prior P(y_v = y_u) = h; h is fitted
    by EM. Labelled neighbours enter exactly. Identifiable when edges are not chosen by feature similarity;
    biased upward on feature-built (kNN-like) graphs, where a different-class neighbour is picked BECAUSE it
    looks like u's class (shown in tests/test_diagnostics.py). Returns (h, n_edges)."""
    m = lab[ei[0]]
    u, v = ei[0][m], ei[1][m]
    if not len(u):
        return float("nan"), 0
    known = lab[v]
    q = np.clip(Q[v, y[u]], 1e-6, 1 - 1e-6)
    pi = np.clip(prior[y[u]], 1e-6, 1 - 1e-6)
    exact = (y[u] == y[v]).astype(float)
    h = 0.5
    for _ in range(iters):
        a = h * q / pi
        b = (1 - h) * (1 - q) / (1 - pi)
        h = float(np.where(known, exact, a / (a + b)).mean())
    return h, int(len(u))
