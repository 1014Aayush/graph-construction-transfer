"""Construction-transfer experiment, v4 = v3 + comparing graphs A and B at three levels, label-free on B:
  structure    ego_dist      EGI-style ego-graph spectral distance (no features, no model)
  behaviour    soft_shift    change of the model's class probabilities; pseudo_compat_shift: class-link shift
                             measured with the model's own predictions (Pair-Align/StruRW-style)
  confidence   atc_est / doc_est: accuracy on B estimated from the model's confidence, calibrated on
                             graph A's labelled validation nodes (ATC, Garg et al. 2022; DoC, Guillory et al. 2021)

v3: v2 + a learned ("dynamic") recipe + per-node predictions
+ diagnostics for the two-step analysis (find the true cause with labels, then a label-free stand-in).

Per dataset and seed: 20 labelled nodes per class (30/class validation, rest test); all graphs span all
nodes. For each training recipe A and model: train on A, predict on every deployment recipe B.

Per (A, B) row, besides accuracy:
  label-based (step 1, diagnosis only)
    h_test_local_A/B  mean share of same-label neighbours around TEST nodes
    compat_shift      how much the class-to-class link pattern changes (total variation, row-normalised)
  label-free (step 2, usable at deployment)
    disagree          share of test-node predictions that flip when graph A is swapped for graph B
    pseudo_h_A/B      homophily of graphs A / B measured with the model's own predictions (on A) as labels
    agg_shift         mean cosine distance between each test node's neighbour-average features under A vs B
All predictions are saved to results/preds/ for later analysis.

    python run.py --datasets cora citeseer photo --out results/part0.csv
"""
import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, Planetoid, WikiCS

from models import build as build_model
from diagnostics import (adjacency, agg_shift, atc_estimate, atc_threshold, compat, compat_shift, cot_both,
                         cott_threshold, doc_estimate, ego_distance, ego_spectra, energy_score, frechet_distance,
                         gcn_operator, local_homophily, mano_score, neighbour_mean, nuclear_score, op_shift,
                         pseudo_homophily, soft_shift, spectrum_distance, top_spectrum)
from recipes import DEFAULT_RECIPES, INTERVENTION_RECIPES, LEARNED_RECIPES, build as build_graph, descriptors, jaccard, stats

EPOCHS, LR, WD = 200, 0.01, 5e-4
N_TRAIN, N_VAL = 20, 30
FIELDS = ["dataset", "seed", "model", "recipe_train", "recipe_test", "acc_test", "acc_val",
          "train_deg", "train_h", "test_deg", "test_h", "jaccard_full",
          "h_test_local_A", "h_test_local_B", "compat_shift",
          "disagree", "pseudo_h_A", "pseudo_h_B", "agg_shift",
          "conf_B", "atc_est", "doc_est", "soft_shift", "pseudo_compat_shift", "ego_dist",
          "nuclear_B", "mano_B", "energy_B", "cot_est", "cott_est", "frechet", "op_shift", "spec_dist",
          "gde_est", "agree_est", "ni_B", "ni_est", "projnorm", "seconds"]
NI_DROP, NI_DRAWS, FT_EPOCHS, COT_NODES = 0.2, 2, 10, 1000
GF = ["dataset", "seed", "recipe", "nodes", "avg_degree", "degree_cv", "feat_sim", "spectral_gap", "isolated",
      "homophily", "h_test_local"]


def load(name, root):
    if name in ("cora", "citeseer", "pubmed"):
        d = Planetoid(root, {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name])[0]
    elif name in ("photo", "computers"):
        d = Amazon(root, name.capitalize())[0]
    elif name == "cs":
        d = Coauthor(root, "CS")[0]
    elif name == "wikics":
        d = WikiCS(f"{root}/WikiCS")[0]
    elif name in ("roman", "ratings"):   # heterophilous: linked nodes usually have different labels
        d = HeterophilousGraphDataset(root, {"roman": "Roman-empire", "ratings": "Amazon-ratings"}[name])[0]
    else:
        raise ValueError(name)
    return d.x.numpy(), d.y.numpy(), d.edge_index.numpy()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def split(y, seed):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        tr += idx[:N_TRAIN].tolist()
        va += idx[N_TRAIN:N_TRAIN + N_VAL].tolist()
    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    return tr, va, np.setdiff1d(np.arange(len(y)), np.concatenate([tr, va]))


def train(model_name, x, ei, y, tr, va, n_cls, seed, dev):
    """Up to EPOCHS epochs; keep the weights with the best validation accuracy on the training graph
    (early stopping by model selection: with 20 labels per class, the final epoch is noisy)."""
    seed_all(seed)
    model = build_model(model_name, x.shape[1], n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[tr], y[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = (model(x, ei)[va].argmax(1) == y[va]).float().mean().item()
        if v > best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "photo", "computers", "cs", "wikics", "roman", "ratings"])
    ap.add_argument("--models", nargs="+", default=["gcn", "sage", "gat", "gatsep", "gcnres", "mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    pred_dir = os.path.join(os.path.dirname(args.out) or ".", "preds")
    os.makedirs(pred_dir, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["dataset"], int(r["seed"]), r["model"], r["recipe_train"]) for r in csv.DictReader(fh)}
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, FIELDS)
    if new:
        w.writeheader()
    gpath = args.out.replace(".csv", "_graphs.csv")
    gnew = not os.path.exists(gpath)
    gfh = open(gpath, "a", newline="")
    gw = csv.DictWriter(gfh, GF)
    if gnew:
        gw.writeheader()

    for ds in args.datasets:
        X, y, given = load(ds, args.root)
        n, n_cls = len(y), int(y.max()) + 1
        t0 = time.time()
        base = {r.name: build_graph(X, r, given=given) for r in DEFAULT_RECIPES}
        # laboratory intervention: rewire the given graph changing only class mixing (rw_mix) or not (rw_keep)
        base.update({r.name: build_graph(X, r, given=given, labels=y) for r in INTERVENTION_RECIPES})
        ops = {k: gcn_operator(v, n) for k, v in base.items()}
        tops = {k: top_spectrum(S) for k, S in ops.items()}
        Z = PCA(n_components=min(64, X.shape[1] - 1), random_state=0).fit_transform(X).astype(np.float32)
        x_t, y_t = torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(y, device=dev)
        print(f"[{ds}] built {len(base)} base graphs in {time.time() - t0:.0f}s", flush=True)
        for seed in args.seeds:
            tr, va, te = split(y, seed)
            tr_t, va_t, te_t = (torch.as_tensor(a, device=dev) for a in (tr, va, te))
            # learned ("dynamic") graph: kNN on hidden features of an MLP trained on this seed's labels
            mlp = train("mlp", x_t, None, y_t, tr_t, va_t, n_cls, seed, dev)
            with torch.no_grad():
                emb = F.relu(mlp.l1(x_t)).cpu().numpy()
            graphs = dict(base)
            for r in LEARNED_RECIPES:
                graphs[r.name] = build_graph(X, r, emb=emb)
            names = list(graphs)
            st = {k: stats(v, y, n) for k, v in graphs.items()}
            desc = {k: descriptors(v, X, n) for k, v in graphs.items()}
            adj = {k: adjacency(v, n) for k, v in graphs.items()}
            nm = {k: neighbour_mean(a, Z) for k, a in adj.items()}
            h_loc = {k: local_homophily(a, y, te) for k, a in adj.items()}
            cm = {k: compat(v, y, n_cls) for k, v in graphs.items()}
            jac = {(a, b): jaccard(graphs[a], graphs[b]) for a in names for b in names}
            agg = {(a, b): agg_shift(nm[a], nm[b], te) for a in names for b in names}
            smp = np.random.default_rng(seed).choice(te, min(200, len(te)), replace=False)
            spec = {k: ego_spectra(a, smp, seed=seed) for k, a in adj.items()}
            ego = {(a, b): ego_distance(spec[a], spec[b]) for a in names for b in names}
            for k in names:
                if k not in ops or k in {r.name for r in LEARNED_RECIPES}:
                    ops[k] = gcn_operator(graphs[k], n)
                    tops[k] = top_spectrum(ops[k])
            opsh = {(a, b): op_shift(ops[a], ops[b], Z, te) for a in names for b in names}
            specd = {(a, b): spectrum_distance(tops[a], tops[b]) for a in names for b in names}
            cot_nodes = np.random.default_rng(seed + 7).choice(te, min(COT_NODES, len(te)), replace=False)
            lab = np.concatenate([tr, va])
            prior = np.bincount(y[lab], minlength=n_cls) / len(lab)
            for k in names:
                gw.writerow(dict(dataset=ds, seed=seed, recipe=k, nodes=n, homophily=round(st[k]["homophily"], 4),
                                 h_test_local=round(h_loc[k], 4), **{a: round(b, 5) for a, b in desc[k].items()}))
            gfh.flush()
            ei = {k: torch.as_tensor(v, dtype=torch.long, device=dev) for k, v in graphs.items()}
            print(f"[{ds}] seed {seed}: " + ", ".join(f"{k} h={st[k]['homophily']:.2f}" for k in names), flush=True)
            for m in args.models:
                if all((ds, seed, m, a) in done for a in names):
                    continue
                if os.environ.get("DEADLINE") and time.time() > float(os.environ["DEADLINE"]):
                    print(f"[{ds}] seed {seed} {m}: skipped (time budget reached)", flush=True)
                    continue
                P = np.zeros((len(names), len(names), n), dtype=np.int8)
                for ia, ra in enumerate(names):
                    t1 = time.time()
                    model = train(m, x_t, ei[ra], y_t, tr_t, va_t, n_cls, seed, dev)
                    model2 = train(m, x_t, ei[ra], y_t, tr_t, va_t, n_cls, seed + 1000, dev)   # for agreement (GDE)
                    gen = torch.Generator(device=dev).manual_seed(seed)
                    with torch.no_grad():
                        logits = [model(x_t, ei[rb]).cpu().numpy() for rb in names]
                        pred2 = [model2(x_t, ei[rb]).argmax(1).cpu().numpy() for rb in names]
                        ni = []                       # neighbourhood invariance: predictions under random edge dropping
                        for ib, rb in enumerate(names):
                            base_pred = torch.as_tensor(logits[ib].argmax(1), device=dev)
                            same = torch.zeros(n, device=dev)
                            for _ in range(NI_DRAWS):
                                keep = torch.rand(ei[rb].shape[1], device=dev, generator=gen) >= NI_DROP
                                same += (model(x_t, ei[rb][:, keep]).argmax(1) == base_pred).float()
                            ni.append((same / NI_DRAWS).cpu().numpy())
                    probs = [np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True) for l in logits]
                    proj = []                         # ProjNorm-style: fine-tune on B's pseudo-labels, measure weight change
                    theta = torch.cat([q.detach().flatten() for q in model.parameters()])
                    for ib, rb in enumerate(names):
                        ft = copy.deepcopy(model).train()
                        opt_ft = torch.optim.Adam(ft.parameters(), lr=LR)
                        pseudo = torch.as_tensor(probs[ib].argmax(1), device=dev)
                        torch.manual_seed(seed)
                        for _ in range(FT_EPOCHS):
                            opt_ft.zero_grad()
                            F.cross_entropy(ft(x_t, ei[rb])[te_t], pseudo[te_t]).backward()
                            opt_ft.step()
                        th2 = torch.cat([q.detach().flatten() for q in ft.parameters()])
                        proj.append(float((th2 - theta).norm() / theta.norm()))
                        del ft, opt_ft
                    for ib in range(len(names)):
                        P[ia, ib] = probs[ib].argmax(1).astype(np.int8)
                    own = P[ia, ia].astype(np.int64)
                    PA = probs[ia]
                    conf_val = PA[va].max(1)
                    acc_val_A = float((own[va] == y[va]).mean())
                    t_atc = atc_threshold(conf_val, own[va] == y[va])
                    cm_own_A = compat(graphs[ra], own, n_cls)
                    t_cott = cott_threshold(PA[va], y[va], prior)
                    agree_val_A = float((own[va] == pred2[ia][va]).mean())
                    ni_val_A = float(ni[ia][va].mean())
                    for ib, rb in enumerate(names):
                        pr = P[ia, ib].astype(np.int64)
                        conf_te = probs[ib][te].max(1)
                        cot_e, cott_e = cot_both(probs[ib][cot_nodes], prior, t_cott)
                        agree_B = float((pr[te] == pred2[ib][te]).mean())
                        ga, gb = graphs[ra], graphs[rb]
                        w.writerow(dict(
                            dataset=ds, seed=seed, model=m, recipe_train=ra, recipe_test=rb,
                            acc_test=round(float((pr[te] == y[te]).mean()), 4), acc_val=round(float((pr[va] == y[va]).mean()), 4),
                            train_deg=round(st[ra]["avg_degree"], 2), train_h=round(st[ra]["homophily"], 4),
                            test_deg=round(st[rb]["avg_degree"], 2), test_h=round(st[rb]["homophily"], 4),
                            jaccard_full=round(jac[(ra, rb)], 4),
                            h_test_local_A=round(h_loc[ra], 4), h_test_local_B=round(h_loc[rb], 4),
                            compat_shift=round(compat_shift(cm[ra], cm[rb]), 4),
                            disagree=round(float((pr[te] != own[te]).mean()), 4),
                            pseudo_h_A=round(pseudo_homophily(ga, own), 4), pseudo_h_B=round(pseudo_homophily(gb, own), 4),
                            agg_shift=round(agg[(ra, rb)], 5),
                            conf_B=round(float(conf_te.mean()), 4), atc_est=round(atc_estimate(conf_te, t_atc), 4),
                            doc_est=round(doc_estimate(acc_val_A, conf_val, conf_te), 4),
                            soft_shift=round(soft_shift(PA, probs[ib], te), 4),
                            pseudo_compat_shift=round(compat_shift(cm_own_A, compat(gb, own, n_cls)), 4),
                            ego_dist=round(ego[(ra, rb)], 4),
                            nuclear_B=round(nuclear_score(probs[ib][te]), 4), mano_B=round(mano_score(logits[ib][te]), 4),
                            energy_B=round(energy_score(logits[ib][te]), 4), cot_est=round(cot_e, 4), cott_est=round(cott_e, 4),
                            frechet=round(frechet_distance(logits[ia][te], logits[ib][te]), 4),
                            op_shift=round(opsh[(ra, rb)], 4), spec_dist=round(specd[(ra, rb)], 5),
                            gde_est=round(agree_B, 4), agree_est=round(agree_B - (agree_val_A - acc_val_A), 4),
                            ni_B=round(float(ni[ib][te].mean()), 4), ni_est=round(float(ni[ib][te].mean()) - (ni_val_A - acc_val_A), 4),
                            projnorm=round(proj[ib], 5), seconds=round(time.time() - t1, 2)))
                    fh.flush()
                np.savez_compressed(os.path.join(pred_dir, f"{ds}_s{seed}_{m}.npz"), preds=P, recipes=np.array(names),
                                    test=te, y=y)
                print(f"[{ds}] seed {seed} {m}: done", flush=True)
    fh.close()
    gfh.close()
    print("RUN DONE", flush=True)


if __name__ == "__main__":
    main()
