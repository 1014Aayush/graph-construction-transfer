"""v5: controlled homophilous graphs. Each graph keeps (approximately) the same 1-hop homophily as the
dataset's given graph and changes ONE factor, so a transfer between two of them isolates that factor.

Generator: every node i keeps a budget of same-class links s_i and different-class links c_i (taken from the
given graph, optionally redistributed or rescaled), and picks WHICH nodes fill them by a policy:
    same  = "near" | "random" | "far"   feature similarity of the same-class neighbours
    cross = "random" | "near"          different-class neighbours: random, or look-alikes (feature-similar)
    deg   = 0.5 | 1 | 2                 scale both budgets (ratio unchanged)
    counts= "orig" | "uniform" | "polarized"
        orig: budgets from the given graph; uniform: every node gets the global ratio;
        polarized: half the nodes are pure (all same-class), the other half get ratio 2h-1 (same global mean).
Each node picks its full budget; after symmetrising, same-class and different-class edges are trimmed at
random to exact targets, so global homophily equals the given graph's by construction.
These graphs use all labels: a laboratory tool to isolate causes, never a deployable construction.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from recipes import _symmetric

MAX_PICK = 64          # per node and side, keeps hub nodes tractable

CONTROLLED = {         # name -> (same, cross, deg, counts); "M0" is the reference every factor is compared with
    "M0": ("random", "random", 1.0, "orig"),
    "same_near": ("near", "random", 1.0, "orig"),
    "same_far": ("far", "random", 1.0, "orig"),
    "cross_near": ("random", "near", 1.0, "orig"),
    "deg_half": ("random", "random", 0.5, "orig"),
    "deg_double": ("random", "random", 2.0, "orig"),
    "h_uniform": ("random", "random", 1.0, "uniform"),
    "h_polarized": ("random", "random", 1.0, "polarized"),
}
# one-factor contrasts: (factor, graph 1, graph 2); both graphs differ only in that factor
CONTRASTS = [
    ("0 edge identity (control)", "M0", "M0_resample"),
    ("1 same-class neighbour similarity", "same_near", "same_far"),
    ("2 wrong-class look-alikes", "M0", "cross_near"),
    ("3 degree", "deg_half", "deg_double"),
    ("4 homophily spread over nodes", "h_uniform", "h_polarized"),
]


def unit_features(X, dim=64, seed=0):
    Z = PCA(n_components=min(dim, X.shape[1] - 1, X.shape[0] - 1), random_state=seed).fit_transform(X)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def budgets(given, y, n):
    u, v = given
    same = np.bincount(u, weights=(y[u] == y[v]).astype(float), minlength=n)
    deg = np.bincount(u, minlength=n).astype(float)
    return same, deg - same


def _pick(Z, members, queries, k, policy, rng):
    """For each query node pick k[q] nodes from `members` (never itself) by policy."""
    out_r, out_c = [], []
    kmax = int(min(max(k.max(), 0) + 1, len(members)))
    if kmax <= 1:
        return np.array([], int), np.array([], int)
    if policy == "random":
        for q, kk in zip(queries, k):
            if kk <= 0:
                continue
            cand = rng.choice(members, min(int(kk) + 1, len(members)), replace=False)
            cand = cand[cand != q][: int(kk)]
            out_r += [q] * len(cand)
            out_c += cand.tolist()
        return np.array(out_r, int), np.array(out_c, int)
    nn = NearestNeighbors(n_neighbors=kmax, metric="euclidean").fit(Z[members])   # unit vectors: euclid ~ cosine
    Q = Z[queries] if policy == "near" else -Z[queries]                             # "far" = nearest to the opposite
    idx = nn.kneighbors(Q, return_distance=False)
    for q, kk, row in zip(queries, k, idx):
        cand = members[row]
        cand = cand[cand != q][: int(kk)]
        out_r += [q] * len(cand)
        out_c += cand.tolist()
    return np.array(out_r, int), np.array(out_c, int)


def matched_graph(Z, y, given, same="random", cross="random", deg=1.0, counts="orig", seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    g = _symmetric(given[0], given[1], n)
    s, c = budgets(g, y, n)
    d = s + c
    h = s.sum() / max(d.sum(), 1)
    cross_pool = np.ones(n, bool)
    if counts == "uniform":
        s = d * h
    elif counts == "polarized":
        pure = rng.random(n) < 0.5
        s = np.where(pure, d, d * max(2 * h - 1, 0))
        cross_pool = ~pure                     # pure nodes never receive different-class links either
    c = d - s
    s_pick = np.minimum(np.ceil(deg * s), MAX_PICK).astype(int)
    c_pick = np.minimum(np.ceil(deg * c), MAX_PICK).astype(int) * cross_pool
    rows, cols = [], []
    for cl in np.unique(y):
        inside = np.flatnonzero(y == cl)
        outside = np.flatnonzero((y != cl) & cross_pool)
        r, k = _pick(Z, inside, inside, s_pick[inside], same, rng)
        rows.append(r), cols.append(k)
        q = inside[cross_pool[inside]]
        if len(outside) and len(q):
            r, k = _pick(Z, outside, q, c_pick[q], cross, rng)
            rows.append(r), cols.append(k)
    ei = _symmetric(np.concatenate(rows), np.concatenate(cols), n)
    # trim to exact undirected targets so that global homophily matches the given graph by construction
    u, v = ei[:, ei[0] < ei[1]]
    is_same = y[u] == y[v]
    keep = np.zeros(len(u), bool)
    for mask, target in ((is_same, deg * s.sum() / 2), (~is_same, deg * c.sum() / 2)):
        idx = np.flatnonzero(mask)
        t = int(round(min(target, len(idx))))
        keep[rng.choice(idx, t, replace=False)] = True
    return _symmetric(u[keep], v[keep], n)


def graph_profile(ei, y, Z, n):
    """Label-based and label-free descriptors of one graph (for explaining, not for prediction)."""
    u, v = ei
    deg = np.bincount(u, minlength=n).astype(float)
    same = (y[u] == y[v])
    loc_num = np.bincount(u, weights=same.astype(float), minlength=n)
    has = deg > 0
    loc = loc_num[has] / deg[has]
    cos = (Z[u] * Z[v]).sum(1)
    # 2-hop homophily: share of length-2 paths i-j-k (k != i) whose ends share the label
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()
    Y = sp.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, int(y.max()) + 1))
    AY = A @ Y                                         # per node: neighbour class counts
    paths_same = (A @ AY).multiply(Y).sum() - deg.sum()          # remove i-j-i returns (same label by definition)
    paths_all = (A @ deg).sum() - deg.sum()
    return dict(avg_degree=float(deg.mean()), degree_cv=float(deg.std() / max(deg.mean(), 1e-12)),
                h1=float(same.mean()) if len(u) else np.nan, h2=float(paths_same / max(paths_all, 1)),
                h_node_sd=float(loc.std()) if len(loc) else np.nan,
                sim_same=float(cos[same].mean()) if same.any() else np.nan,
                sim_cross=float(cos[~same].mean()) if (~same).any() else np.nan,
                feat_sim=float(cos.mean()) if len(u) else np.nan, isolated=float(1 - has.mean()))
