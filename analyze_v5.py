"""v5 analysis: with homophily held fixed, which other similarity must two graphs share for transfer?

mismatch(g1, g2) = mean of gap(g1 -> g2) and gap(g2 -> g1), where gap(A -> B) = acc(A->B) - acc(B->B):
the accuracy lost purely because the model was trained on a different graph (B's own quality cancels).

    python analyze_v5.py "results/v5_gpu*.csv" --outdir results/analysis_v5
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from controlled import CONTRASTS

NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "gatsep": "GAT-sep", "gcnres": "GCN-res"}
SELF_PATH, NO_SELF_PATH = ["GraphSAGE", "GAT-sep", "GCN-res"], ["GCN", "GAT"]
DESCRIPTORS = {"degree (|log ratio|)": "d_logdeg", "2-hop homophily": "d_h2", "homophily spread over nodes": "d_hsd",
               "same-class neighbour similarity": "d_simsame", "wrong-class neighbour similarity": "d_simcross",
               "degree spread": "d_degcv", "edge change (1 - Jaccard)": "edge_change", "1-hop homophily": "d_h1"}
REALISTIC_PAIRS = [("given", "drop-20"), ("given", "drop-50"), ("given", "prune-30"), ("given", "ppr-k16"),
                   ("given", "heat-k16"), ("given", "twohop-k10"), ("given", "given_knn-k5-pca"), ("ppr-k16", "heat-k16"),
                   ("knn-k5-cos", "knn-k20-cos"), ("knn-k10-cos", "knn-k10-pca"), ("given", "knn-k10-cos")]


def load(pattern):
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(("_graphs.csv", "_pairs.csv")))
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = pd.concat([pd.read_csv(f.replace(".csv", "_graphs.csv")) for f in files], ignore_index=True).drop_duplicates(["dataset", "recipe"])
    pr = pd.concat([pd.read_csv(f.replace(".csv", "_pairs.csv")) for f in files], ignore_index=True).drop_duplicates(["dataset", "recipe_train", "recipe_test"])
    for c in ("acc_test", "acc_val", "conf_B", "atc_est", "doc_est", "gde_est", "agree_est"):
        r[c] = 100 * r[c]
    key = ["dataset", "seed", "model"]
    own = r[r.recipe_train == r.recipe_test][key + ["recipe_test", "acc_test", "acc_val"]]
    r = r.merge(own.rename(columns={"acc_test": "acc_BB", "acc_val": "val_BB"}).drop(columns="acc_val", errors="ignore"),
                on=key + ["recipe_test"], how="left")
    r = r.merge(own.rename(columns={"recipe_test": "recipe_train", "acc_test": "acc_AA", "acc_val": "val_AA"}),
                on=key + ["recipe_train"], how="left")
    r["gap"] = r.acc_test - r.acc_BB
    r["drop"] = r.acc_test - r.acc_AA
    for name, c in (("agreement", "agree_est"), ("DoC", "doc_est"), ("ATC", "atc_est")):
        r["err " + name] = (r[c] - r.acc_test).abs()
    r["err nothing changed"] = (r.val_AA - r.acc_test).abs()
    gi = g.set_index(["dataset", "recipe"])
    A = gi.loc[list(zip(r.dataset, r.recipe_train))].reset_index(drop=True)
    B = gi.loc[list(zip(r.dataset, r.recipe_test))].reset_index(drop=True)
    r = r.reset_index(drop=True)
    r["d_h1"] = (B.h1 - A.h1).abs()
    r["d_logdeg"] = np.log(B.avg_degree / A.avg_degree).abs()
    r["d_h2"] = (B.h2 - A.h2).abs()
    r["d_hsd"] = (B.h_node_sd - A.h_node_sd).abs()
    r["d_simsame"] = (B.sim_same - A.sim_same).abs()
    r["d_simcross"] = (B.sim_cross - A.sim_cross).abs()
    r["d_degcv"] = (B.degree_cv - A.degree_cv).abs()
    r = r.merge(pr, on=["dataset", "recipe_train", "recipe_test"], how="left")
    r["edge_change"] = 1 - r.jaccard
    return r, g


def cluster_ci(values, clusters, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = clusters.unique()
    by = {k: values[clusters == k].to_numpy() for k in keys}
    boots = [np.concatenate([by[k] for k in rng.choice(keys, len(keys))]).mean() for _ in range(n_boot)]
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pair_table(r, g1, g2):
    """Per (dataset, seed, model): mismatch averaged over both directions, plus drops and monitor errors."""
    f = r[((r.recipe_train == g1) & (r.recipe_test == g2)) | ((r.recipe_train == g2) & (r.recipe_test == g1))]
    return f.groupby(["dataset", "seed", "model"]).agg(mismatch=("gap", "mean"), drop=("drop", "mean"),
                                                       err_agree=("err agreement", "mean"),
                                                       err_none=("err nothing changed", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--outdir", default="results/analysis_v5")
    args = ap.parse_args()
    r, g = load(args.pattern)
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 250)

    print("0. GRAPHS: 1-hop homophily (controlled graphs should match 'given')")
    print(g.pivot(index="recipe", columns="dataset", values="h1").round(2).to_string())
    mlp = r[(r.model == "mlp") & (r.recipe_train == "given") & (r.recipe_test == "given")].groupby("dataset").acc_test.mean()
    gnn = r[(r.model != "mlp") & (r.recipe_train == "given") & (r.recipe_test == "given")].groupby("dataset").acc_test.mean()
    print("\n   GNN (mean of 5) vs MLP on the given graph:", {d: f"{gnn[d]:.1f} vs {mlp[d]:.1f}" for d in gnn.index}, "\n")

    q = r[r.model != "mlp"].copy()
    q["model"] = q.model.map(NAMES)
    rows, results = [], {}
    for factor, g1, g2 in CONTRASTS:
        t = pair_table(q, g1, g2)
        t["cl"] = t.dataset + "|" + t.seed.astype(str)
        for m, tm in t.groupby("model"):
            mean, lo, hi = cluster_ci(tm.mismatch, tm.cl)
            rows.append(dict(factor=factor, model=m, mismatch=mean, ci_lo=lo, ci_hi=hi, drop=tm["drop"].mean(),
                             err_agreement=tm.err_agree.mean(), err_nothing_changed=tm.err_none.mean()))
        results[factor] = t
    c = pd.DataFrame(rows)
    c.to_csv(f"{args.outdir}/contrasts.csv", index=False)
    print("1. ONE-FACTOR CONTRASTS (homophily held fixed). mismatch = points lost because the model was trained on")
    print("   the other graph (both directions averaged); 95% CI over (dataset, seed) clusters")
    print(c.round(2).to_string(index=False), "\n")
    per_ds = pd.concat({f: results[f].groupby("dataset").mismatch.mean() for f, _, _ in CONTRASTS}, axis=1)
    print("   mismatch per dataset (mean over models):")
    print(per_ds.round(1).to_string(), "\n")

    rp = []
    for a, b in REALISTIC_PAIRS:
        t = pair_table(q, a, b)
        if len(t):
            rp.append(dict(pair=f"{a} <-> {b}", mismatch=t.mismatch.mean(), worst_model=t.groupby("model").mismatch.mean().min(),
                           err_agreement=t.err_agree.mean(), err_nothing_changed=t.err_none.mean()))
    print("2. REALISTIC CONSTRUCTION CHANGES (mean over datasets, seeds, models)")
    print(pd.DataFrame(rp).round(2).to_string(index=False), "\n")

    off = q[q.recipe_train != q.recipe_test]
    rows = {}
    for sub, s in (("all homophily-matched pairs (|dh| < 0.03)", off[off.d_h1 < 0.03]),
                   ("realistic graphs only, homophily-matched", off[(off.d_h1 < 0.03) & ~off.recipe_train.isin(list(CONTROLLED_NAMES)) & ~off.recipe_test.isin(list(CONTROLLED_NAMES))])):
        rows[sub] = {name: np.nanmean([spearmanr(-d[col], d.gap)[0] for _, d in s.groupby(["dataset", "model"]) if d[col].nunique() > 2])
                     for name, col in DESCRIPTORS.items()}
        rows[sub]["(transfers)"] = len(s)
    ex = pd.DataFrame(rows)
    ex.to_csv(f"{args.outdir}/explain.csv")
    print("3. WHICH DIFFERENCE EXPLAINS THE MISMATCH? Spearman(-|difference|, gap) inside each dataset and model, averaged")
    print(ex.round(2).to_string(), "\n")

    n = c.model.nunique()
    get = lambda f: c[c.factor.str.startswith(f)].set_index("model")  # noqa: E731
    c0, c1, c2, c3, c4 = (get(str(i)) for i in range(5))
    k0 = int((c0.mismatch.abs() < 2).sum())
    k1 = int(((c1.mismatch <= -2) & (c1.ci_hi < 0)).sum())
    k2 = int(((c2.mismatch.abs() >= 2) & ((c2.ci_hi < 0) | (c2.ci_lo > 0))).sum())
    k3a = bool((c3.mismatch.reindex(NO_SELF_PATH) <= -2).all())
    k3b = bool(c3.mismatch.reindex(SELF_PATH).mean() > c3.mismatch.reindex(NO_SELF_PATH).mean())
    k4 = int(((c4.mismatch <= -2) & (c4.ci_hi < 0)).sum())
    mon = sum(int(((cf.err_agreement < cf.err_nothing_changed).sum()) >= round(0.6 * n)) for cf in (c1, c2, c3, c4))
    print("4. PRE-REGISTERED CHECKS")
    print(f"   C0 control: replacing edges alone costs < 2 points                       {k0}/{n} -> {'PASS' if k0 >= round(0.6 * n) else 'FAIL'}")
    print(f"   C1 same-class neighbour similarity matters (<= -2, CI < 0)              {k1}/{n} -> {'PASS' if k1 >= round(0.6 * n) else 'FAIL'}")
    print(f"   C2 wrong-class look-alikes matter (|.| >= 2, CI excludes 0)             {k2}/{n} -> {'PASS' if k2 >= round(0.6 * n) else 'FAIL'}")
    print(f"   C3 degree: GCN and GAT lose >= 2, self-path models lose less             {'PASS' if k3a and k3b else 'FAIL'} (GCN/GAT <= -2: {k3a}; self-path less: {k3b})")
    print(f"   C4 homophily spread over nodes matters (<= -2, CI < 0)                  {k4}/{n} -> {'PASS' if k4 >= round(0.6 * n) else 'FAIL'}")
    print(f"   M1 agreement beats 'nothing changed' (most models) for factors 1-4      {mon}/4 -> {'PASS' if mon >= 3 else 'FAIL'}")
    print(f"\nsaved {args.outdir}/contrasts.csv, explain.csv")


from controlled import CONTROLLED  # noqa: E402
CONTROLLED_NAMES = set(CONTROLLED) | {"M0_resample", "rw_keep-50"}

if __name__ == "__main__":
    main()
