"""v7: label-light homophily by construction type (EM where edges are not feature-chosen, labelled edges where
they are), and the transfer formula, fitted on v4 and CONFIRMED on v5 outcomes it has never seen.

    python analyze_v7.py --labelfree "results_v7/ct/results/labelfree_*.csv" --v4 "results_v4/ct/results/p?_gpu?.csv" \
        --v5 "results_v5/ct/results/v5_gpu?.csv" --outdir results_v7/analysis
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analyze_v4
import analyze_v5
from controlled import CONTROLLED

LAB_GRAPHS = set(CONTROLLED) | {"M0_resample"}


def within_rho(df, a, b, by):
    return float(np.nanmean([spearmanr(g[a], g[b])[0] for _, g in df.groupby(by) if g[a].nunique() > 2]))


def attach(df, lf, cols, ds="dataset", seed="seed", a="recipe_train", b="recipe_test"):
    key = lf.set_index(["dataset", "seed", "recipe"])
    for c in cols:
        A = key[c].reindex(list(zip(df[ds], df[seed], df[a]))).to_numpy()
        B = key[c].reindex(list(zip(df[ds], df[seed], df[b]))).to_numpy()
        df["d_" + c] = B - A
    return df


def mlp_val(files, read):
    raw = pd.concat([read(f) for f in files])
    return raw[(raw.model == "mlp") & (raw.recipe_train == raw.recipe_test)].groupby(["dataset", "seed"]).acc_val.mean()


def fit_c(x, y):
    ok = x.notna() & y.notna()
    return float((x[ok] * y[ok]).sum() / max((x[ok] ** 2).sum(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelfree", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--v5", required=True)
    ap.add_argument("--outdir", default="results_v7/analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 250)
    lf = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.labelfree))], ignore_index=True)
    lf = lf[~lf.recipe.isin(LAB_GRAPHS)]

    print("E. HOMOPHILY ESTIMATES BY CONSTRUCTION TYPE (mean abs error; rank corr with truth across graphs)")
    rows = []
    for (ds, fb), g in lf.groupby(["dataset", "feature_built"]):
        for est in ("h_em", "h_lab", "h_rule"):
            rows.append(dict(dataset=ds, graphs="feature-built" if fb else "not feature-built", estimator=est,
                             mae=float((g[est] - g.h_true).abs().mean()), bias=float((g[est] - g.h_true).mean()),
                             rho=within_rho(g, est, "h_true", "seed")))
    te = pd.DataFrame(rows)
    te.to_csv(f"{args.outdir}/estimates.csv", index=False)
    print(te.pivot_table(index="dataset", columns=["graphs", "estimator"], values="mae").round(3).to_string())
    print("   bias (estimate - truth):")
    print(te.pivot_table(index="dataset", columns=["graphs", "estimator"], values="bias").round(3).to_string())
    print("   rank correlation with truth, all graphs:", {ds: round(within_rho(g, "h_rule", "h_true", "seed"), 2) for ds, g in lf.groupby("dataset")}, "\n")

    v4files = sorted(f for f in glob.glob(args.v4) if not f.endswith("_graphs.csv"))
    p4, _ = analyze_v4.load(v4files)
    p4 = p4[~p4.recipe_train.str.startswith("embknn") & ~p4.recipe_test.str.startswith("embknn")].copy()
    p4 = attach(p4, lf, ["h_true", "h_rule", "h_em", "h_lab"])
    m4 = mlp_val(v4files, lambda f: pd.read_csv(f, usecols=["dataset", "seed", "model", "recipe_train", "recipe_test", "acc_val"])) * 100
    p4["benefit"] = (p4.acc_val_AA - m4.reindex(list(zip(p4.dataset, p4.seed))).to_numpy()).clip(lower=0)

    print("R. DOES THE ESTIMATED HOMOPHILY CHANGE TRACK THE DROP? (v4, Spearman inside datasets)")
    t2 = pd.DataFrame({m: {c: within_rho(g, "d_" + c, "drop", "dataset") for c in ("h_true", "h_rule", "h_em", "h_lab")} |
                       {"agreement": within_rho(g, "agree_est_drop", "drop", "dataset")} for m, g in p4.groupby("model")}).T
    print(t2.round(2).to_string(), "\n")

    rows, consts = [], {}
    for version, col in (("label-light (h_rule)", "d_h_rule"), ("oracle (true h)", "d_h_true")):
        g4 = p4.assign(x=-p4.benefit * (-p4[col]).clip(lower=0))
        for m, g in g4.groupby("model"):
            g = g.copy()
            g["pred"] = np.nan
            for ds in g.dataset.unique():
                trn, tst = g.dataset != ds, g.dataset == ds
                g.loc[tst, "pred"] = fit_c(g.x[trn], g["drop"][trn]) * g.x[tst]
            ok = g.pred.notna()
            consts[(version, m)] = fit_c(g.x, g["drop"])
            rows.append(dict(version=version, model=m, c=consts[(version, m)], within_rho=within_rho(g[ok], "pred", "drop", "dataset"),
                             mae=float((g.pred - g["drop"])[ok].abs().mean()), nothing_mae=float(g["drop"][ok].abs().mean()),
                             agreement_within=within_rho(g[ok], "agree_est_drop", "drop", "dataset"),
                             agreement_mae=float((g.agree_est_drop - g["drop"])[ok].abs().mean())))
    tf = pd.DataFrame(rows)
    tf.to_csv(f"{args.outdir}/formula_v4.csv", index=False)
    print("F. FORMULA ON v4 (c fitted leave-one-dataset-out):  drop = -c x (acc_A - acc_MLP) x max(0, h_A - h_B)")
    print(tf.round(3).to_string(index=False), "\n")

    v5 = analyze_v5.load(args.v5)[0]
    v5 = v5[(v5.model != "mlp") & (v5.recipe_train != v5.recipe_test) & ~v5.recipe_train.isin(LAB_GRAPHS | {"rw_keep-50"})
            & ~v5.recipe_test.isin(LAB_GRAPHS | {"rw_keep-50"})].copy()
    v5["model"] = v5.model.map(analyze_v5.NAMES)
    own = analyze_v5.load(args.v5)[0]
    own = own[own.recipe_train == own.recipe_test][["dataset", "seed", "model", "recipe_train", "agree_est"]]
    own["model"] = own.model.map(analyze_v5.NAMES)
    v5 = v5.merge(own.rename(columns={"agree_est": "agree_AA"}), on=["dataset", "seed", "model", "recipe_train"], how="left")
    v5["agree_est_drop"] = v5.agree_est - v5.agree_AA
    v5files = sorted(f for f in glob.glob(args.v5) if not f.endswith(("_graphs.csv", "_pairs.csv")))
    m5 = mlp_val(v5files, lambda f: pd.read_csv(f, usecols=["dataset", "seed", "model", "recipe_train", "recipe_test", "acc_val"])) * 100
    v5["benefit"] = (v5.val_AA - m5.reindex(list(zip(v5.dataset, v5.seed))).to_numpy()).clip(lower=0)
    v5 = attach(v5, lf, ["h_rule", "h_true"])
    rows = []
    for version, col in (("label-light (h_rule)", "d_h_rule"), ("oracle (true h)", "d_h_true")):
        for m, g in v5.groupby("model"):
            g = g.assign(pred=consts[(version, m)] * -g.benefit * (-g[col]).clip(lower=0))
            ok = g.pred.notna()
            rows.append(dict(version=version, model=m, within_rho=within_rho(g[ok], "pred", "drop", "dataset"),
                             mae=float((g.pred - g["drop"])[ok].abs().mean()), nothing_mae=float(g["drop"][ok].abs().mean()),
                             agreement_within=within_rho(g[ok], "agree_est_drop", "drop", "dataset"),
                             agreement_mae=float((g.agree_est_drop - g["drop"])[ok].abs().mean()), n=int(ok.sum())))
    tc = pd.DataFrame(rows)
    tc.to_csv(f"{args.outdir}/formula_confirm_v5.csv", index=False)
    print("C. CONFIRMATION ON v5 OUTCOMES (new run, 5 new recipes; c taken from v4, nothing refitted)")
    print(tc.round(3).to_string(index=False), "\n")

    n_ds = lf.dataset.nunique()
    nf = te[(te.graphs == "not feature-built") & (te.estimator == "h_em")].set_index("dataset")
    fbt = te[(te.graphs == "feature-built") & (te.estimator == "h_em")].set_index("dataset")
    e1 = int((nf.mae < 0.05).sum())   # rank corr uninformative: these graphs have near-identical homophily
    e2 = int((fbt.bias > 0.05).sum())
    e3 = int((t2.h_rule >= 0.8 * t2.h_true).sum())
    fl = tf[tf.version.str.startswith("label")].set_index("model")
    f1 = int((fl.mae < fl.nothing_mae).sum())
    f2 = [m for m in ("GCN", "GAT") if fl.loc[m, "within_rho"] >= 0.8 * fl.loc[m, "agreement_within"]]
    cl = tc[tc.version.str.startswith("label")].set_index("model")
    c1 = int((cl.mae < cl.nothing_mae).sum())
    c2 = [m for m in ("GCN", "GAT") if cl.loc[m, "within_rho"] >= 0.8 * cl.loc[m, "agreement_within"]]
    n = len(fl)
    print("PRE-REGISTERED CHECKS (v7)")
    print(f"   E1 EM accurate on graphs not built from feature similarity (error < 0.05)         {e1}/{n_ds} -> {'PASS' if e1 >= 7 else 'FAIL'}")
    print(f"   E2 EM over-estimates on feature-built graphs (bias > +0.05): the known limit       {e2}/{n_ds} -> {'PASS' if e2 >= 7 else 'FAIL'}")
    print(f"   E3 rule-based estimate tracks the drop >= 80% as well as true homophily         {e3}/{n} -> {'PASS' if e3 >= round(0.6 * n) else 'FAIL'}")
    print(f"   F1 formula (rule) beats 'nothing changed' on v4 (leave-one-dataset-out)          {f1}/{n} -> {'PASS' if f1 >= round(0.6 * n) else 'FAIL'}")
    print(f"   F2 formula >= 80% of agreement's rho for GCN and GAT on v4                       {len(f2)}/2 -> {'PASS' if len(f2) == 2 else 'FAIL'}")
    print(f"   C1 CONFIRMATION: formula beats 'nothing changed' on unseen v5 outcomes           {c1}/{n} -> {'PASS' if c1 >= round(0.6 * n) else 'FAIL'}")
    print(f"   C2 CONFIRMATION: formula >= 80% of agreement's rho for GCN and GAT on v5          {len(c2)}/2 -> {'PASS' if len(c2) == 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
