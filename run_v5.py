"""v5: what must two HOMOPHILOUS graphs share for a trained GNN to transfer between them?

Homophilous datasets only. Graphs per dataset (built once, shared by all seeds):
  realistic   14 popular constructions: given; random edge dropping 20/50%; similarity pruning 30%; PPR top-16/32;
              heat-kernel top-16; two-hop (given + 10 strongest 2-hop); given + kNN; kNN cosine 5/10/20; kNN PCA 10; eps
  controlled  homophily matched to the given graph, one factor changed (controlled.py), plus
              M0_resample (same recipe as M0, new random draw) and rw_keep-50 (degree-preserving swaps)
For every seed, model and training graph A: train two models (different initialisations), predict on every
graph B. Per (A, B): accuracy, and label-free monitors (two-model agreement, ATC, DoC, confidence).

    python run_v5.py --datasets cora citeseer --out results/v5_gpu0.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from controlled import CONTROLLED, graph_profile, matched_graph, unit_features
from diagnostics import atc_estimate, atc_threshold, doc_estimate
from recipes import INTERVENTION_RECIPES, V5_REALISTIC as REALISTIC, build as build_graph, jaccard
from run import load, split, train

FIELDS = ["dataset", "seed", "model", "recipe_train", "recipe_test", "acc_test", "acc_val", "disagree",
          "conf_B", "atc_est", "doc_est", "gde_est", "agree_est", "seconds"]


def build_all(X, y, given):
    Z = unit_features(X)
    graphs = {r.name: build_graph(X, r, given=given) for r in REALISTIC}
    keep = [r for r in INTERVENTION_RECIPES if r.name == "rw_keep-50"][0]
    graphs[keep.name] = build_graph(X, keep, given=given, labels=y)
    for name, prm in CONTROLLED.items():
        graphs[name] = matched_graph(Z, y, given, *prm, seed=1)
    graphs["M0_resample"] = matched_graph(Z, y, given, *CONTROLLED["M0"], seed=2)
    return graphs, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "photo", "computers", "cs", "wikics"])
    ap.add_argument("--models", nargs="+", default=["gcn", "sage", "gat", "gatsep", "gcnres", "mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/v5.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["dataset"], int(r["seed"]), r["model"], r["recipe_train"]) for r in csv.DictReader(fh)}
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, FIELDS)
    if new:
        w.writeheader()
    gpath, ppath = args.out.replace(".csv", "_graphs.csv"), args.out.replace(".csv", "_pairs.csv")

    for ds in args.datasets:
        X, y, given = load(ds, args.root)
        n, n_cls = len(y), int(y.max()) + 1
        t0 = time.time()
        graphs, Z = build_all(X, y, given)
        names = list(graphs)
        prof = {k: graph_profile(v, y, Z, n) for k, v in graphs.items()}
        pd_rows = [dict(dataset=ds, recipe=k, **prof[k]) for k in names]
        pr_rows = [dict(dataset=ds, recipe_train=a, recipe_test=b, jaccard=jaccard(graphs[a], graphs[b])) for a in names for b in names]
        for path, rows in ((gpath, pd_rows), (ppath, pr_rows)):
            first = not os.path.exists(path)
            with open(path, "a", newline="") as gf:
                gw = csv.DictWriter(gf, list(rows[0]))
                if first:
                    gw.writeheader()
                gw.writerows(rows)
        print(f"[{ds}] built {len(names)} graphs in {time.time() - t0:.0f}s: "
              + ", ".join(f"{k} h={prof[k]['h1']:.2f} d={prof[k]['avg_degree']:.1f}" for k in names), flush=True)
        x_t, y_t = torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(y, device=dev)
        ei = {k: torch.as_tensor(v, dtype=torch.long, device=dev) for k, v in graphs.items()}
        for seed in args.seeds:
            tr, va, te = split(y, seed)
            tr_t, va_t = (torch.as_tensor(a, device=dev) for a in (tr, va))
            for m in args.models:
                if all((ds, seed, m, a) in done for a in names):
                    continue
                if os.environ.get("DEADLINE") and time.time() > float(os.environ["DEADLINE"]):
                    print(f"[{ds}] seed {seed} {m}: skipped (time budget reached)", flush=True)
                    continue
                for ra in names:
                    t1 = time.time()
                    m1 = train(m, x_t, ei[ra], y_t, tr_t, va_t, n_cls, seed, dev)
                    m2 = train(m, x_t, ei[ra], y_t, tr_t, va_t, n_cls, seed + 1000, dev)
                    with torch.no_grad():
                        P = {rb: F.softmax(m1(x_t, ei[rb]), 1).cpu().numpy() for rb in names}
                        P2 = {rb: m2(x_t, ei[rb]).argmax(1).cpu().numpy() for rb in names}
                    own = P[ra].argmax(1)
                    acc_val_A = float((own[va] == y[va]).mean())
                    conf_val = P[ra][va].max(1)
                    t_atc = atc_threshold(conf_val, own[va] == y[va])
                    agree_val_A = float((own[va] == P2[ra][va]).mean())
                    for rb in names:
                        pr = P[rb].argmax(1)
                        conf_te = P[rb][te].max(1)
                        agree = float((pr[te] == P2[rb][te]).mean())
                        w.writerow(dict(dataset=ds, seed=seed, model=m, recipe_train=ra, recipe_test=rb,
                                        acc_test=round(float((pr[te] == y[te]).mean()), 4),
                                        acc_val=round(float((pr[va] == y[va]).mean()), 4),
                                        disagree=round(float((pr[te] != own[te]).mean()), 4),
                                        conf_B=round(float(conf_te.mean()), 4), atc_est=round(atc_estimate(conf_te, t_atc), 4),
                                        doc_est=round(doc_estimate(acc_val_A, conf_val, conf_te), 4),
                                        gde_est=round(agree, 4), agree_est=round(agree - (agree_val_A - acc_val_A), 4),
                                        seconds=round(time.time() - t1, 2)))
                    fh.flush()
                print(f"[{ds}] seed {seed} {m}: done", flush=True)
    fh.close()
    print("RUN DONE", flush=True)


if __name__ == "__main__":
    main()
