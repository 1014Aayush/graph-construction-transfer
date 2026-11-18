"""Graph-construction recipes: build an undirected graph from node features.

Every recipe returns a symmetric (2, 2E) edge array (both directions, no self-loops) over the
rows of X. Recipes differ in exactly the choices practitioners make when building graphs:
    knn          each node links to its k most similar nodes (union -> symmetric)
    mutual_knn   keep a link only if each node is in the other's top-k
    eps          link all pairs with similarity above a threshold, the threshold chosen so the
                 average degree matches a kNN graph with the same k (degree-matched comparison)
    given        the dataset's own graph (citations, co-purchases): the reference graph
    given_knn    given graph plus a kNN graph from features (common augmentation)
    ppr          personalised-PageRank diffusion of the given graph, top-k per node (GDC-style,
                 Gasteiger et al. 2019): links each node to the nodes random walks reach most
Similarity: "cosine" (on L2-normalised features), "euclidean", or "pca" (cosine on the top-128
principal components: denoised features).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Recipe:
    kind: str      # "knn" | "mutual_knn" | "eps" | "given" | "given_knn" | "ppr" | "embknn" | "rw_keep" | "rw_mix"
    k: int         # neighbours (for eps: the kNN degree to match)
    metric: str    # "cosine" | "euclidean"

    @property
    def name(self):
        if self.kind == "given":
            return "given"
        if self.kind in ("ppr", "embknn"):
            return f"{self.kind}-k{self.k}"
        if self.kind in ("rw_keep", "rw_mix", "drop", "prune"):
            return f"{self.kind}-{self.k}"
        if self.kind in ("heat", "twohop"):
            return f"{self.kind}-k{self.k}"
        return f"{self.kind}-k{self.k}-{self.metric[:3]}"


def _prep(X, metric):
    X = np.asarray(X, dtype=np.float32)
    if metric == "pca":
        from sklearn.decomposition import PCA
        X = PCA(n_components=min(128, X.shape[1] - 1, len(X) - 1), random_state=0).fit_transform(X).astype(np.float32)
        metric = "cosine"
    if metric == "cosine":
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def _topk(X, k):
    """Indices of each row's k nearest other rows (Euclidean on prepared X)."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]   # drop self


def _symmetric(rows, cols, n):
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    keep = a != b
    pairs = np.unique(np.stack([a[keep], b[keep]], 1), axis=0)
    return np.concatenate([pairs.T, pairs.T[::-1]], axis=1)


def _ppr_topk(given, n, k, alpha=0.15, iters=10, keep=None):
    """Approximate personalised-PageRank matrix of the given graph (row-pruned power iteration),
    then each node's top-k targets (excluding itself), symmetrised."""
    import scipy.sparse as sp
    keep = keep or 4 * k
    A = sp.coo_matrix((np.ones(given.shape[1]), (given[0], given[1])), shape=(n, n)).tocsr()
    A = ((A + sp.eye(n)) > 0).astype(np.float64)
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    T = (sp.diags(d) @ A @ sp.diags(d)).tocsr()

    def prune(M, m):
        M = M.tocoo()
        order = np.lexsort((-M.data, M.row))
        r, c, v = M.row[order], M.col[order], M.data[order]
        start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
        rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
        sel = rank < m
        return sp.csr_matrix((v[sel], (r[sel], c[sel])), shape=M.shape)

    I = sp.eye(n, format="csr")
    X = alpha * I
    for _ in range(iters):
        X = prune((1 - alpha) * (T @ X) + alpha * I, keep)
    X = X.tocoo()
    m = X.row != X.col
    X = prune(sp.csr_matrix((X.data[m], (X.row[m], X.col[m])), shape=(n, n)), k).tocoo()
    return _symmetric(X.row, X.col, n)


def build(X, recipe: Recipe, given=None, emb=None, labels=None):
    n = len(X)
    if recipe.kind in ("rw_keep", "rw_mix"):
        if given is None or labels is None:
            raise ValueError(f"{recipe.name} needs the given graph and labels (laboratory intervention)")
        return rewire(given, n, labels, recipe.k / 100, mix=recipe.kind == "rw_mix")
    if recipe.kind == "embknn":
        # learned ("dynamic") graph: kNN on representations learned by a model (ViG / DGCNN / GSL idea)
        if emb is None:
            raise ValueError("embknn needs learned embeddings (emb=...)")
        return build(emb, Recipe("knn", recipe.k, "cosine"))
    if recipe.kind in ("given", "given_knn", "ppr", "heat", "twohop", "drop", "prune"):
        if given is None:
            raise ValueError(f"{recipe.name} needs the dataset's given edge_index")
        g = _symmetric(given[0], given[1], n)
        if recipe.kind == "given":
            return g
        if recipe.kind == "heat":
            return _heat_topk(g, n, recipe.k)
        if recipe.kind == "twohop":
            return _twohop(g, n, recipe.k)
        if recipe.kind == "drop":
            return _drop(g, n, recipe.k / 100)
        if recipe.kind == "prune":
            return _prune_dissimilar(X, g, n, recipe.k / 100)
        if recipe.kind == "ppr":
            return _ppr_topk(g, n, recipe.k)
        kn = build(X, Recipe("knn", recipe.k, recipe.metric))
        both = np.concatenate([g, kn], axis=1)
        return _symmetric(both[0], both[1], n)
    Xp = _prep(X, recipe.metric)
    nbr = _topk(Xp, recipe.k)
    rows = np.repeat(np.arange(n), nbr.shape[1])
    cols = nbr.ravel()
    if recipe.kind == "knn":
        return _symmetric(rows, cols, n)
    if recipe.kind == "mutual_knn":
        directed = set(zip(rows.tolist(), cols.tolist()))
        m = np.array([(r, c) in directed and (c, r) in directed for r, c in zip(rows, cols)], dtype=bool)
        return _symmetric(rows[m], cols[m], n)
    if recipe.kind == "eps":
        target_pairs = _symmetric(rows, cols, n).shape[1] // 2      # match the kNN graph's edge count
        k2 = min(n - 1, max(4 * recipe.k, 50))                       # candidate pool per node
        nn = NearestNeighbors(n_neighbors=k2 + 1).fit(Xp)
        dist, idx = nn.kneighbors(Xp)
        dist, idx = dist[:, 1:], idx[:, 1:]
        r = np.repeat(np.arange(n), k2)
        c, d = idx.ravel(), dist.ravel()
        a, b = np.minimum(r, c), np.maximum(r, c)
        pairs, first = np.unique(np.stack([a, b], 1), axis=0, return_index=True)
        order = np.argsort(d[first], kind="stable")[:target_pairs]  # globally closest pairs = threshold
        kept = pairs[order]
        return np.concatenate([kept.T, kept.T[::-1]], axis=1)
    raise ValueError(recipe.kind)


def stats(ei, y, n):
    """Average degree and edge homophily of a symmetric edge array."""
    E = ei.shape[1] // 2
    u, v = ei[:, :E]
    return dict(avg_degree=2 * E / n, homophily=float((y[u] == y[v]).mean()) if E else float("nan"))


def descriptors(ei, X, n):
    """Label-free measurements of a graph, computable before deployment:
       avg_degree, degree_cv (spread of degrees), feat_sim (mean cosine similarity of linked nodes'
       features), spectral_gap (1 - second-largest eigenvalue of the normalised adjacency; 0 if
       the graph is disconnected), isolated (share of nodes with no edge)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    E = ei.shape[1] // 2
    deg = np.bincount(ei[0], minlength=n).astype(float)
    Xn = _prep(X, "cosine")
    u, v = ei[:, :E]
    feat_sim = float((Xn[u] * Xn[v]).sum(1).mean()) if E else float("nan")
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    d = np.where(deg > 0, deg, 1.0) ** -0.5
    An = sp.diags(d) @ A @ sp.diags(d)
    try:
        vals = eigsh(An, k=2, which="LA", return_eigenvectors=False, tol=1e-4, maxiter=5000)
        gap = float(1.0 - sorted(vals)[0])
    except Exception:
        gap = float("nan")
    return dict(avg_degree=float(deg.mean()), degree_cv=float(deg.std() / max(deg.mean(), 1e-12)),
                feat_sim=feat_sim, spectral_gap=max(gap, 0.0), isolated=float((deg == 0).mean()))


def jaccard(ei_a, ei_b):
    """Overlap of two graphs' undirected edge sets."""
    def s(ei):
        u, v = ei
        return set(zip(np.minimum(u, v).tolist(), np.maximum(u, v).tolist()))
    A, B = s(ei_a), s(ei_b)
    return len(A & B) / max(1, len(A | B))


# v1 recipe set (feature-only graphs; kept for reference)
V1_RECIPES = [Recipe("knn", k, m) for k in (3, 5, 10, 20, 30) for m in ("cosine", "euclidean")] + \
             [Recipe("mutual_knn", 10, "cosine"), Recipe("mutual_knn", 10, "euclidean"),
              Recipe("eps", 10, "cosine"), Recipe("eps", 30, "cosine")]

# v2: popular construction strategies that tend to help, plus feature-kNN references (11 recipes)
DEFAULT_RECIPES = [Recipe("given", 0, "none"), Recipe("given_knn", 5, "pca"),
                   Recipe("ppr", 16, "none"), Recipe("ppr", 32, "none")] + \
                  [Recipe("knn", k, "pca") for k in (5, 10, 20)] + \
                  [Recipe("knn", k, "cosine") for k in (5, 10, 20)] + [Recipe("eps", 10, "pca")]


# v3: v2 recipes plus one learned ("dynamic") recipe, built per seed from a model trained on that seed's labels
LEARNED_RECIPES = [Recipe("embknn", 10, "none")]


# ---- v3 intervention: change ONLY class mixing, keep everything else fixed ----------------------------
# Degree-preserving double-edge swaps on the given graph: (u,v),(x,w) -> (u,w),(x,v).
#   "rw_keep": only swaps with label(v) == label(w) and label(u) == label(x): every node keeps its degree AND the class make-up of
#              its neighbourhood (local homophily and class-to-class links unchanged) -> edges change, cause not.
#   "rw_mix":  only swaps with label(v) != label(w) and label(u) != label(x): degrees kept, the same share
#              of edges replaced, but every touched node's neighbour classes change -> the cause changes.
# Uses all labels, so these graphs are a laboratory intervention only (excluded from the predictor).
INTERVENTION_RECIPES = [Recipe(kd, p, "none") for p in (20, 50) for kd in ("rw_keep", "rw_mix")]


def rewire(given, n, labels, frac, mix, seed=0, max_tries_factor=200):
    """Swap edges until `frac` of the ORIGINAL undirected edges are gone (see above).
    keep: label(u)==label(x) and label(v)==label(w)  -> all four nodes keep their neighbour-class counts
    mix:  label(u)!=label(x) and label(v)!=label(w)  -> all four nodes' neighbour classes change"""
    rng = np.random.default_rng(seed)
    g = _symmetric(given[0], given[1], n)
    keep = g[0] < g[1]
    E = [tuple(e) for e in g[:, keep].T.tolist()]
    orig, present = set(E), set(E)
    target, lost, tries = int(round(frac * len(E))), 0, 0
    while lost < target and tries < max_tries_factor * max(target, 1):
        tries += 1
        i, j = rng.integers(len(E), size=2)
        (u, v), (x, w) = E[i], E[j]
        if rng.random() < 0.5:
            x, w = w, x
        if len({u, v, x, w}) < 4:
            continue
        same = (labels[u] == labels[x], labels[v] == labels[w])
        if (mix and (same[0] or same[1])) or (not mix and not (same[0] and same[1])):
            continue
        a, b = (min(u, w), max(u, w)), (min(x, v), max(x, v))
        if a in present or b in present:
            continue
        for e in (E[i], E[j]):
            present.discard(e)
            lost += e in orig
        for e in (a, b):
            present.add(e)
            lost -= e in orig
        E[i], E[j] = a, b
    e = np.array(E, dtype=np.int64).T
    return _symmetric(e[0], e[1], n)


# ---- v5: further popular real-world constructions derived from the given graph ------------------------
def _operator(g, n):
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(g.shape[1]), (g[0], g[1])), shape=(n, n)).tocsr()
    A = ((A + sp.eye(n)) > 0).astype(np.float64)
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    return (sp.diags(d) @ A @ sp.diags(d)).tocsr()


def _row_topk(M, m):
    import scipy.sparse as sp
    M = M.tocoo()
    order = np.lexsort((-M.data, M.row))
    r, c, v = M.row[order], M.col[order], M.data[order]
    start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    sel = rank < m
    return sp.csr_matrix((v[sel], (r[sel], c[sel])), shape=M.shape)


def _no_diag_topk(M, k, n):
    import scipy.sparse as sp
    M = M.tocoo()
    m = M.row != M.col
    M = _row_topk(sp.csr_matrix((M.data[m], (M.row[m], M.col[m])), shape=(n, n)), k).tocoo()
    return _symmetric(M.row, M.col, n)


def _heat_topk(g, n, k, t=3.0, terms=10):
    """Heat-kernel diffusion exp(-t(I - T)) = sum_j e^-t t^j / j! T^j (GDC's second kernel), row-pruned,
    then each node's top-k targets (excluding itself), symmetrised."""
    import math
    import scipy.sparse as sp
    T = _operator(g, n)
    term = sp.eye(n, format="csr")
    X = math.exp(-t) * term
    for j in range(1, terms + 1):
        term = _row_topk(T @ term, 4 * k)
        X = _row_topk(X + math.exp(-t) * t ** j / math.factorial(j) * term, 4 * k)
    return _no_diag_topk(X, k, n)


def _twohop(g, n, k):
    """Multi-hop graph (MixHop / SIGN style): the given edges plus each node's k strongest 2-hop neighbours
    (by number of connecting paths)."""
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(g.shape[1]), (g[0], g[1])), shape=(n, n)).tocsr()
    A2 = (A @ A).tolil()
    A2 = A2.tocsr() - A2.tocsr().multiply(A > 0)             # only genuinely 2-hop pairs
    two = _no_diag_topk(A2, k, n)
    both = np.concatenate([g, two], 1)
    return _symmetric(both[0], both[1], n)


def _drop(g, n, frac, seed=0):
    """Random edge dropping (missing links / edge sampling): remove a share of undirected edges."""
    rng = np.random.default_rng(seed)
    u, v = g[:, g[0] < g[1]]
    keep = rng.random(len(u)) >= frac
    return _symmetric(u[keep], v[keep], n)


def _prune_dissimilar(X, g, n, frac):
    """Similarity-based edge pruning (Jaccard / GNNGuard-style denoising): remove the least feature-similar
    share of the given edges (cosine on PCA features)."""
    Xp = _prep(X, "pca")
    u, v = g[:, g[0] < g[1]]
    sim = (Xp[u] * Xp[v]).sum(1)
    keep = sim >= np.quantile(sim, frac)
    return _symmetric(u[keep], v[keep], n)


V5_REALISTIC = [Recipe("given", 0, "none"), Recipe("drop", 20, "none"), Recipe("drop", 50, "none"),
                Recipe("prune", 30, "pca"), Recipe("ppr", 16, "none"), Recipe("ppr", 32, "none"),
                Recipe("heat", 16, "none"), Recipe("twohop", 10, "none"), Recipe("given_knn", 5, "pca"),
                Recipe("knn", 5, "cosine"), Recipe("knn", 10, "cosine"), Recipe("knn", 20, "cosine"),
                Recipe("knn", 10, "pca"), Recipe("eps", 10, "pca")]
