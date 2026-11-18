"""v6 (label-free step): estimate the causes found in v5 (homophily; same-class neighbour similarity) WITHOUT
labelling anything on the graph. Rebuilds exactly the graphs of v4 and v5 (all deterministic) and, per seed,
uses only that seed's labelled training/validation nodes and a feature-only classifier.

Per (dataset, seed, recipe):
  true (labels, for validation only): h_true, sim_same_true
  label-light estimates:
    h_lab        homophily over edges between labelled nodes (+ n_lab_edges)
    h_pseudo     share of edges with equal pseudo-labels, corrected for the classifier's validation accuracy
    h_pseudo_raw the same without correction
    h_feat       mixture fit of edge feature-similarities (components learned from labelled pairs)
    sim_same_pseudo  mean feature cosine over edges with equal pseudo-labels

    python labelfree.py --datasets cora citeseer --out results/labelfree_0.csv
"""
import argparse
import csv
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from controlled import CONTROLLED, matched_graph, unit_features
from diagnostics import (corrected_pseudo_homophily, em_homophily, feature_mixture_homophily, labelled_edge_homophily,
                         pseudo_same_class_similarity)
from recipes import DEFAULT_RECIPES, INTERVENTION_RECIPES, V5_REALISTIC, build as build_graph
from run import load, split

FIELDS = ["dataset", "seed", "recipe", "h_true", "sim_same_true", "h_lab", "n_lab_edges", "h_pseudo", "h_pseudo_raw",
          "h_feat", "sim_same_pseudo", "pseudo_acc", "h_em", "n_em_edges", "feature_built", "h_rule"]
# graphs whose edges are chosen or filtered by feature similarity: features alone cannot identify their homophily
FEATURE_BUILT = ("knn", "mutual_knn", "eps", "given_knn", "embknn", "prune", "same_near", "same_far", "cross_near")


def all_graphs(X, y, given, Z):
    g = {}
    for r in list(DEFAULT_RECIPES) + list(INTERVENTION_RECIPES) + list(V5_REALISTIC):
        if r.name not in g:
            g[r.name] = build_graph(X, r, given=given, labels=y)
    for name, prm in CONTROLLED.items():
        g[name] = matched_graph(Z, y, given, *prm, seed=1)
    g["M0_resample"] = matched_graph(Z, y, given, *CONTROLLED["M0"], seed=2)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "photo", "computers", "cs", "wikics",
                                                       "roman", "ratings"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/labelfree.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for ds in args.datasets:
            t0 = time.time()
            X, y, given = load(ds, args.root)
            n, k = len(y), int(y.max()) + 1
            Z = unit_features(X)
            graphs = all_graphs(X, y, given, Z)
            print(f"[{ds}] {len(graphs)} graphs in {time.time() - t0:.0f}s", flush=True)
            for seed in args.seeds:
                tr, va, te = split(y, seed)
                lab = np.zeros(n, bool)
                lab[np.concatenate([tr, va])] = True
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y[tr])      # feature-only, trained on tr
                acc = float((clf.predict(Z[va]) == y[va]).mean())                     # its accuracy, from val
                pseudo = clf.predict(Z)
                Q = clf.predict_proba(Z)
                prior = np.bincount(y[tr], minlength=k) / len(tr)
                for name, ei in graphs.items():
                    hl, ne = labelled_edge_homophily(ei, y, lab)
                    hp, raw = corrected_pseudo_homophily(ei, pseudo, acc, k)
                    he, n_em = em_homophily(ei, y, lab, Q, prior)
                    fb = name.split("-")[0] in FEATURE_BUILT
                    same = y[ei[0]] == y[ei[1]]
                    w.writerow(dict(dataset=ds, seed=seed, recipe=name, h_true=round(float(same.mean()), 4),
                                    sim_same_true=round(float((Z[ei[0]][same] * Z[ei[1]][same]).sum(1).mean()), 4),
                                    h_lab=round(hl, 4), n_lab_edges=ne, h_pseudo=round(hp, 4), h_pseudo_raw=round(raw, 4),
                                    h_feat=round(feature_mixture_homophily(ei, Z, y, np.flatnonzero(lab), seed=seed), 4),
                                    sim_same_pseudo=round(pseudo_same_class_similarity(ei, Z, pseudo), 4),
                                    pseudo_acc=round(acc, 4), h_em=round(he, 4), n_em_edges=n_em, feature_built=int(fb),
                                    h_rule=round(hl if fb else he, 4)))
                fh.flush()
            print(f"[{ds}] done in {time.time() - t0:.0f}s", flush=True)
    print("RUN DONE", flush=True)


if __name__ == "__main__":
    main()
