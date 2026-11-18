"""v4 analysis: how should two graphs be compared to predict how a trained GNN transfers?

Seven families of label-free comparisons (structure -> model behaviour -> calibrated estimators), drawn from
graph transferability theory and the label-free accuracy-estimation literature (ODP-Bench, ICCV 2025).
Target: the reused model's accuracy acc(A->B) and its drop = acc(A->B) - acc(A->A).
Nothing is fitted across datasets (estimators are calibrated on graph A's own validation labels), so every
number is valid on a dataset never seen before.

    python analyze_v4.py "results/p*_gpu*.csv" --outdir results/analysis_v4
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "gatsep": "GAT-sep", "gcnres": "GCN-res"}
SELF_PATH, NO_SELF_PATH = ["GraphSAGE", "GAT-sep", "GCN-res"], ["GCN", "GAT"]
STRUCTURE_FAMILIES = ["1 structure", "1b structure+features"]
# measure -> (family, sign); sign +1: higher means less degradation. "*_drop" = value(A->B) - value(A->A).
MEASURES = {
    "ego_dist": ("1 structure", -1), "jaccard_full": ("1 structure", +1), "spec_dist": ("1 structure", -1),
    "agg_shift": ("1b structure+features", -1), "op_shift": ("1b structure+features", -1),
    "disagree": ("2 output change", -1), "soft_shift": ("2 output change", -1),
    "pseudo_compat_shift": ("2 output change", -1), "frechet": ("2 output change", -1),
    "conf_B_drop": ("3a confidence", +1), "atc_est_drop": ("3a confidence", +1), "doc_est_drop": ("3a confidence", +1),
    "cot_est_drop": ("3a confidence", +1), "cott_est_drop": ("3a confidence", +1),
    "nuclear_B_drop": ("3b prediction matrix", +1), "mano_B_drop": ("3b prediction matrix", +1),
    "energy_B_drop": ("3b prediction matrix", +1),
    "gde_est_drop": ("3c agreement/invariance", +1), "agree_est_drop": ("3c agreement/invariance", +1),
    "ni_est_drop": ("3c agreement/invariance", +1),
    "projnorm_drop": ("3d retraining behaviour", -1),
}
# accuracy estimators (in [0,1] in the raw file): name -> column
ESTIMATORS = {"AC": "conf_B", "ATC": "atc_est", "DoC": "doc_est", "COT": "cot_est", "COTT": "cott_est",
              "GDE": "gde_est", "Agreement (calibrated)": "agree_est", "NI (calibrated)": "ni_est"}
ODP_TOP4 = {"ATC", "Agreement (calibrated)", "NI (calibrated)", "COTT"}
PCT = ["acc_test", "acc_val"] + list(ESTIMATORS.values())
DROP_OF = ["conf_B", "atc_est", "doc_est", "cot_est", "cott_est", "nuclear_B", "mano_B", "energy_B",
           "gde_est", "agree_est", "ni_est", "projnorm"]


def load(files):
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    r = r[r.model != "mlp"].copy()
    r["model"] = r.model.map(NAMES)
    for c in PCT:
        r[c] = 100 * r[c]
    key = ["dataset", "seed", "model", "recipe_train"]
    own = r[r.recipe_train == r.recipe_test][key + ["acc_test", "acc_val"] + DROP_OF]
    own = own.rename(columns={c: c + "_AA" for c in ["acc_test", "acc_val"] + DROP_OF})
    p = r[r.recipe_train != r.recipe_test].merge(own, on=key)
    p["drop"] = p.acc_test - p.acc_test_AA
    for c in DROP_OF:
        p[c + "_drop"] = p[c] - p[c + "_AA"]
    for name, c in ESTIMATORS.items():
        p["err " + name] = (p[c] - p.acc_test).abs()
    p["err nothing changed"] = (p.acc_val_AA - p.acc_test).abs()
    lab = p.recipe_train.str.startswith("rw_") | p.recipe_test.str.startswith("rw_")
    return p[~lab].copy(), p[(p.recipe_train == "given") & p.recipe_test.str.startswith("rw_")].copy()


def within_rho(p, col, sign):
    """Spearman(sign * measure, drop) inside each dataset, averaged over datasets (per model)."""
    out = {}
    for m, g in p.groupby("model"):
        vals = [spearmanr(sign * d[col], d["drop"], nan_policy="omit")[0]
                for _, d in g.groupby("dataset") if d[col].nunique() > 1]
        out[m] = float(np.nanmean(vals)) if vals else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outdir", default="results/analysis_v4")
    args = ap.parse_args()
    files = sorted({f for pat in args.files for f in glob.glob(pat) if not f.endswith("_graphs.csv")})
    p, stress = load(files)
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 250)
    print(f"transfers (A != B, no laboratory graphs): {len(p)} over {p.dataset.nunique()} datasets\n")

    t = pd.DataFrame({c: within_rho(p, c, s) for c, (fam, s) in MEASURES.items() if c in p})
    t.to_csv(f"{args.outdir}/measures.csv")
    print("1. HOW WELL EACH COMPARISON TRACKS THE REUSED MODEL'S DROP (Spearman inside each dataset, averaged;")
    print("   sign-corrected so higher = better)")
    print(t.T.round(2).to_string(), "\n")
    fams = list(dict.fromkeys(f for f, _ in MEASURES.values()))
    fam = pd.DataFrame({f: t[[c for c, (f2, _) in MEASURES.items() if f2 == f and c in t]].max(1) for f in fams})
    fam.to_csv(f"{args.outdir}/families.csv")
    print("   best measure per family:")
    print(fam.round(2).to_string(), "\n")

    err_cols = ["err " + k for k in ESTIMATORS] + ["err nothing changed"]
    mae = p.groupby("model")[err_cols].mean()
    mae.columns = [c[4:] for c in mae.columns]
    mae.to_csv(f"{args.outdir}/estimation_error.csv")
    print("2. ESTIMATING THE ACCURACY ON B: mean absolute error (points); lower = better")
    print(mae.round(2).to_string())
    pooled = pd.DataFrame({k: {m: spearmanr(g[c], g.acc_test)[0] for m, g in p.groupby("model")} for k, c in ESTIMATORS.items()})
    print("   rank correlation with true accuracy, pooled over ALL datasets (no per-dataset fitting):")
    print(pooled.round(2).to_string())
    print("   ATC error per dataset:")
    print(p.groupby(["dataset", "model"])["err ATC"].mean().unstack().round(1).to_string(), "\n")

    s = stress.groupby(["model", "recipe_test"])[["drop"] + ["err " + k for k in ESTIMATORS]].mean()
    s.to_csv(f"{args.outdir}/stress.csv")
    print("3. STRESS TEST: given graph -> rewired copy (keep = same class mixing, mix = changed class mixing)")
    print(s.round(1).to_string(), "\n")

    n = len(t)
    best_model_based = fam[[f for f in fams if f not in STRUCTURE_FAMILIES]].max(1)
    best_structure = fam[STRUCTURE_FAMILIES].max(1)
    h1 = int((best_model_based - best_structure >= 0.1).sum())   # margin: model-based families hold more measures
    est_only = mae.drop(columns="nothing changed")
    h2 = int(((est_only.min(1) < 5) & (est_only.min(1) < mae["nothing changed"])).sum())
    winner = est_only.idxmin(1)
    h2b = int(winner.isin(ODP_TOP4).sum())
    e = stress.groupby(["model", "recipe_test"])["err ATC"].mean().unstack()
    h3 = int((e["rw_mix-50"] - e["rw_keep-50"] >= 2).sum()) if {"rw_mix-50", "rw_keep-50"} <= set(e.columns) else 0
    sp_err, nsp_err = mae["ATC"].reindex(SELF_PATH).mean(), mae["ATC"].reindex(NO_SELF_PATH).mean()
    print("4. PRE-REGISTERED CHECKS")
    print(f"   H1  a model-based family beats every structure family by >= 0.1             {h1}/{n} -> {'PASS' if h1 >= round(0.6 * n) else 'FAIL'}")
    print(f"   H2  some estimator: error < 5 points and better than 'nothing changed'      {h2}/{n} -> {'PASS' if h2 >= round(0.6 * n) else 'FAIL'}")
    print(f"   H2b best estimator is one of ODP-Bench's top 4 (ATC/Agreement/NI/COTT)     {h2b}/{n} -> {'PASS' if h2b >= round(0.6 * n) else 'FAIL'}"
          f"   (winners: {winner.to_dict()})")
    print(f"   H3  ATC error >= 2 points larger under class mixing (50% rewired)           {h3}/{n} -> {'PASS' if h3 >= round(0.6 * n) else 'FAIL'}")
    print(f"   H4  ATC error >= 1 point lower for self-path models: {sp_err:.2f} vs {nsp_err:.2f}       -> {'PASS' if nsp_err - sp_err >= 1 else 'FAIL'}")
    print(f"\nsaved {args.outdir}/measures.csv, families.csv, estimation_error.csv, stress.csv")


if __name__ == "__main__":
    main()
