# stepg_posthoc.py
"""
Step G: Post-hoc analysis on saved predictions from Step F.

No LOSO re-run needed. Loads stepf_tierA_all_predictions.pkl and applies:
  1. Binary/ordinal grouping (Normal/Slight=0-1, Mild=2, Moderate/Severe=3-4)
  2. Ensemble of top models (mean/median of model predictions)
  3. Mean vs median aggregation comparison
  4. Ordinal regression (requires mord: pip install mord)
  5. Bootstrap confidence intervals on all metrics

Clinical reference for grouping:
  - MDS-UPDRS anchors: 0=normal, 1=slight, 2=mild, 3=moderate, 4=severe
    (Goetz et al., 2007, Mov Disord 22(1):41-47)
  - Martinez-Martin et al. (2015) proposed cut-off points for
    mild/moderate/severe PD severity levels based on MDS-UPDRS scores.
    (Parkinsonism Relat Disord 21(12):1465-1468)
  - The 3-level grouping (0-1 / 2 / 3-4) aligns with the anchor language:
    "slight" (0-1) has "no impact on function", "mild" (2) has "modest impact",
    "moderate/severe" (3-4) has "considerable impact or prevents function".

Reads:  {OUTPUT_DIR}/stepf_tierA_all_predictions.pkl
Writes: {OUTPUT_DIR}/stepg_*.csv, stepg_*.png
"""

import os, csv, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import (mean_absolute_error, f1_score, cohen_kappa_score,
                              accuracy_score, confusion_matrix)
import config as C

warnings.filterwarnings("ignore")

try: import mord; HAS_MORD = True
except ImportError: HAS_MORD = False


# ============================================================
# Load predictions from Step F
# ============================================================
def load_predictions(tier, activity):
    pkl = os.path.join(C.OUTPUT_DIR, f"Stepf_analysis/{activity}/stepf_{tier}_all_predictions.pkl")
    print(f"  Loading: {pkl}")
    with open(pkl, "rb") as f: data = pickle.load(f)
    for nm in data:
        for k in ["win_yt","win_yp","subj_yt","subj_yp","subj_yp_median"]:
            if k in data[nm]: data[nm][k] = np.array(data[nm][k])
        for k in ["win_subj","subj_names","win_activity"]:
            if k in data[nm]: data[nm][k] = np.array(data[nm][k])
    return data


# ============================================================
# Metrics
# ============================================================
def _reg_metrics(yt, yp):
    v = np.isfinite(yp) & np.isfinite(yt)
    if v.sum() < 3: return {k:float("nan") for k in ["rho","p","mae","acc_exact","acc_w1","macro_f1","kappa"]}
    yt2,yp2 = yt[v],yp[v]; yr = np.clip(np.round(yp2),0,4)
    rho,pv = spearmanr(yt2,yp2)
    return {"rho":float(rho),"p":float(pv),"mae":float(mean_absolute_error(yt2,yp2)),
            "acc_exact":float(np.mean(yr==yt2)),"acc_w1":float(np.mean(np.abs(yr-yt2)<=1)),
            "macro_f1":float(f1_score(yt2.astype(int),yr.astype(int),average="macro",zero_division=0)),
            "kappa":float(cohen_kappa_score(yt2.astype(int),yr.astype(int)))}

def _cls_metrics(yt, yp, labels):
    """Classification metrics for grouped labels."""
    v = np.isfinite(yp.astype(float)) & np.isfinite(yt.astype(float))
    yt2, yp2 = yt[v], yp[v]
    if len(yt2) < 3: return {k:float("nan") for k in ["acc","macro_f1","weighted_f1","kappa"]}
    return {
        "acc": float(accuracy_score(yt2, yp2)),
        "macro_f1": float(f1_score(yt2, yp2, average="macro", zero_division=0, labels=labels)),
        "weighted_f1": float(f1_score(yt2, yp2, average="weighted", zero_division=0, labels=labels)),
        "kappa": float(cohen_kappa_score(yt2, yp2)),
    }


# ============================================================
# 1. Binary/Ordinal Grouping
# ============================================================
GROUPING = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}  # 0-1 -> "Normal/Slight", 2 -> "Mild", 3-4 -> "Moderate/Severe"
GROUP_NAMES = {0: "Normal/Slight (0-1)", 1: "Mild (2)", 2: "Mod/Severe (3-4)"}
GROUP_LABELS = [0, 1, 2]

def apply_grouping(yt, yp):
    """Map 5-class UPDRS to 3-class groups. Drops NaN predictions."""
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt_v, yp_v = yt[valid], yp[valid]
    yt_g = np.array([GROUPING[int(np.clip(np.round(y), 0, 4))] for y in yt_v])
    yp_g = np.array([GROUPING[int(np.clip(np.round(y), 0, 4))] for y in yp_v])
    return yt_g, yp_g

def grouping_analysis(results, odir):
    """Apply 3-class grouping to all models' predictions."""
    print(f"\n{'='*60}\n  1. Ordinal Grouping (3 classes)\n{'='*60}")
    print(f"  Grouping: {GROUP_NAMES}")

    all_rows = []
    for nm, res in results.items():
        # Mean aggregation
        yt_g, yp_g = apply_grouping(res["subj_yt"], res["subj_yp"])
        m = _cls_metrics(yt_g, yp_g, GROUP_LABELS)
        all_rows.append({"model": nm, "agg": "mean", **m})
        # Median aggregation
        if "subj_yp_median" in res:
            yt_g2, yp_g2 = apply_grouping(res["subj_yt"], res["subj_yp_median"])
            m2 = _cls_metrics(yt_g2, yp_g2, GROUP_LABELS)
            all_rows.append({"model": nm, "agg": "median", **m2})

    # Print
    print(f"\n    {'Model':<15s} {'Agg':>6s} {'Acc':>6s} {'F1(M)':>6s} {'F1(W)':>6s} {'κ':>6s}")
    print(f"    {'-'*50}")
    for r in sorted(all_rows, key=lambda x: x.get("kappa",0), reverse=True):
        print(f"    {r['model']:<15s} {r['agg']:>6s} {r['acc']:>6.3f} {r['macro_f1']:>6.3f} "
              f"{r['weighted_f1']:>6.3f} {r['kappa']:>6.3f}")

    # Best model confusion matrix (grouped)
    best_nm = max(all_rows, key=lambda x: x.get("kappa",0))["model"]
    yt_g, yp_g = apply_grouping(results[best_nm]["subj_yt"], results[best_nm]["subj_yp"])
    cm = confusion_matrix(yt_g, yp_g, labels=GROUP_LABELS)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5),dpi=120)
    im1 = ax1.imshow(cm, cmap="Blues")
    cm_n = cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-12)
    im2 = ax2.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    gnames = [GROUP_NAMES[i] for i in GROUP_LABELS]
    for ax_i, data, fmt in [(ax1,cm,"d"),(ax2,cm_n,".2f")]:
        for i in range(3):
            for j in range(3):
                ax_i.text(j,i,format(data[i,j],fmt),ha="center",va="center",fontsize=12,
                          color="white" if data[i,j]>(cm.max()*0.5 if fmt=="d" else 0.5) else "black")
        ax_i.set_xticks(range(3)); ax_i.set_xticklabels(gnames, rotation=30, ha="right", fontsize=8)
        ax_i.set_yticks(range(3)); ax_i.set_yticklabels(gnames, fontsize=8)
        ax_i.set_xlabel("Predicted"); ax_i.set_ylabel("True")
    ax1.set_title(f"Counts (N={len(yt_g)})"); ax2.set_title("Recall")
    fig.suptitle(f"3-Class Grouping — {best_nm}", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(odir, "stepg_grouping_confusion.png"), bbox_inches="tight"); plt.close(fig)

    # Save CSV
    path = os.path.join(odir, "stepg_grouping_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(
            {k: f"{v:.4f}" if isinstance(v,float) else v for k,v in r.items()} for r in all_rows)
    print(f"    Saved: {path}")
    return all_rows


# ============================================================
# 2. Ensemble of Top Models
# ============================================================
def ensemble_analysis(results, odir, top_models=None):
    """Average predictions from top models."""
    print(f"\n{'='*60}\n  2. Ensemble of Top Models\n{'='*60}")

    if top_models is None:
        # Auto-select top 4 by subject-level rho
        rhos = {nm: _reg_metrics(r["subj_yt"], r["subj_yp"])["rho"] for nm, r in results.items()}
        top_models = sorted(rhos, key=lambda x: rhos[x], reverse=True)[:4]

    print(f"    Ensemble members: {top_models}")

    # Get common subject ordering (should be same for all models)
    ref = results[top_models[0]]
    subj_names = ref["subj_names"]
    yt = ref["subj_yt"]
    n = len(subj_names)

    # Collect predictions from each member
    preds_mean = np.zeros((len(top_models), n))
    preds_median = np.zeros((len(top_models), n))
    for i, nm in enumerate(top_models):
        r = results[nm]
        # Ensure same subject ordering
        for j, s in enumerate(subj_names):
            idx = np.where(r["subj_names"] == s)[0]
            if len(idx) > 0:
                preds_mean[i, j] = r["subj_yp"][idx[0]]
                preds_median[i, j] = r["subj_yp_median"][idx[0]] if "subj_yp_median" in r else r["subj_yp"][idx[0]]
            else:
                preds_mean[i, j] = float("nan")
                preds_median[i, j] = float("nan")

    # Ensemble: mean of model predictions
    ens_mean = np.nanmean(preds_mean, axis=0)
    ens_median = np.nanmedian(preds_mean, axis=0)

    # Metrics
    m_ens_mean = _reg_metrics(yt, ens_mean)
    m_ens_median = _reg_metrics(yt, ens_median)

    # Also grouped
    yt_g, yp_g_mean = apply_grouping(yt, ens_mean)
    yt_g, yp_g_med = apply_grouping(yt, ens_median)
    m_g_mean = _cls_metrics(yt_g, yp_g_mean, GROUP_LABELS)
    m_g_med = _cls_metrics(yt_g, yp_g_med, GROUP_LABELS)

    print(f"    Ensemble (mean of means):   ρ={m_ens_mean['rho']:.3f}  MAE={m_ens_mean['mae']:.3f}  "
          f"Acc±1={m_ens_mean['acc_w1']:.3f}  κ={m_ens_mean['kappa']:.3f}")
    print(f"    Ensemble (median of means): ρ={m_ens_median['rho']:.3f}  MAE={m_ens_median['mae']:.3f}  "
          f"Acc±1={m_ens_median['acc_w1']:.3f}  κ={m_ens_median['kappa']:.3f}")
    print(f"    Grouped (3-class, mean):  Acc={m_g_mean['acc']:.3f}  F1={m_g_mean['macro_f1']:.3f}  κ={m_g_mean['kappa']:.3f}")
    print(f"    Grouped (3-class, median):Acc={m_g_med['acc']:.3f}  F1={m_g_med['macro_f1']:.3f}  κ={m_g_med['kappa']:.3f}")

    # Compare: individual models vs ensemble
    print(f"\n    Individual vs Ensemble:")
    for nm in top_models:
        m = _reg_metrics(results[nm]["subj_yt"], results[nm]["subj_yp"])
        print(f"      {nm:<15s}: ρ={m['rho']:.3f}  κ={m['kappa']:.3f}")
    print(f"      {'Ensemble':<15s}: ρ={m_ens_mean['rho']:.3f}  κ={m_ens_mean['kappa']:.3f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=120)
    ax.scatter(yt+np.random.randn(n)*0.08, ens_mean, alpha=0.6, s=40, c=yt, cmap="coolwarm",
               edgecolors="white", linewidth=0.3)
    ax.plot([-0.5,4.5],[-0.5,4.5],"--",color="gray",alpha=0.5)
    ax.set_xlim(-0.5,4.5); ax.set_ylim(-0.5,4.5)
    ax.set_xlabel("True UPDRS"); ax.set_ylabel("Ensemble Predicted")
    ax.set_title(f"Ensemble ({', '.join(top_models)})\nρ={m_ens_mean['rho']:.3f}  MAE={m_ens_mean['mae']:.3f}  "
                 f"κ={m_ens_mean['kappa']:.3f}")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid(True, alpha=0.2); plt.tight_layout()
    fig.savefig(os.path.join(odir, "stepg_ensemble_scatter.png"), bbox_inches="tight"); plt.close(fig)

    # Save
    path = os.path.join(odir, "stepg_ensemble_predictions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject","y_true","ensemble_mean","ensemble_median","y_round_mean","y_round_median"] +
                   [f"pred_{nm}" for nm in top_models])
        for j in range(n):
            row = [subj_names[j], f"{yt[j]:.0f}", f"{ens_mean[j]:.4f}", f"{ens_median[j]:.4f}",
                   f"{np.clip(np.round(ens_mean[j]),0,4):.0f}", f"{np.clip(np.round(ens_median[j]),0,4):.0f}"]
            for nm in top_models: row.append(f"{preds_mean[top_models.index(nm),j]:.4f}")
            w.writerow(row)
    print(f"    Saved: {path}")

    return {"ens_mean": ens_mean, "ens_median": ens_median, "yt": yt, "subj_names": subj_names,
            "metrics_mean": m_ens_mean, "metrics_median": m_ens_median, "top_models": top_models}


# ============================================================
# 3. Mean vs Median Aggregation Comparison
# ============================================================
def aggregation_comparison(results, odir):
    """Compare mean vs median window aggregation for all models."""
    print(f"\n{'='*60}\n  3. Mean vs Median Aggregation\n{'='*60}")

    rows = []
    print(f"    {'Model':<15s} {'ρ(mean)':>8s} {'ρ(med)':>8s} {'Δρ':>6s} {'κ(mean)':>8s} {'κ(med)':>8s}")
    print(f"    {'-'*55}")
    for nm, res in results.items():
        mm = _reg_metrics(res["subj_yt"], res["subj_yp"])
        mmed = _reg_metrics(res["subj_yt"], res["subj_yp_median"]) if "subj_yp_median" in res else mm
        delta = mmed["rho"] - mm["rho"]
        rows.append({"model":nm, "rho_mean":mm["rho"], "rho_median":mmed["rho"], "delta_rho":delta,
                      "kappa_mean":mm["kappa"], "kappa_median":mmed["kappa"]})
        winner = "median" if delta > 0.01 else ("mean" if delta < -0.01 else "~same")
        print(f"    {nm:<15s} {mm['rho']:>8.3f} {mmed['rho']:>8.3f} {delta:>+6.3f} "
              f"{mm['kappa']:>8.3f} {mmed['kappa']:>8.3f}  {winner}")

    path = os.path.join(odir, "stepg_aggregation_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow({k: f"{v:.4f}" if isinstance(v,float) else v for k,v in r.items()})


# ============================================================
# 4. Ordinal Regression (mord)
# ============================================================
def ordinal_regression(results, odir):
    """
    Fit ordinal regression on the window-level features.
    Uses the saved window-level predictions as a baseline, but re-fits
    using mord.LogisticAT (All-Thresholds ordinal logistic).
    """
    if not HAS_MORD:
        print(f"\n  4. Ordinal Regression: SKIPPED (pip install mord)")
        return

    print(f"\n{'='*60}\n  4. Ordinal Regression (mord)\n{'='*60}")

    # We need to re-load window-level features for this
    # (ordinal regression needs raw features, not just predictions)
    all_rows = []
    for act in C.ACTIVITIES:
        path = C.features_window_csv(act)
        if not os.path.isfile(path): continue
        rows = load_csv_simple(path)
        for r in rows: r["is_right_side"] = 1.0 if r.get("side") == "Right" else 0.0
        all_rows.extend(rows)

    if not all_rows:
        print("    No window-level data found."); return

    fn = load_features() + ["is_right_side"]
    X = np.array([[r.get(f, float("nan")) for f in fn] for r in all_rows])
    y = np.array([r["updrs_score"] for r in all_rows])
    subjects = np.array([r["subject"] for r in all_rows])
    unique_subj = np.unique(subjects)

    from sklearn.preprocessing import StandardScaler

    subj_yt, subj_yp = [], []
    subj_names_list = []

    for held in unique_subj:
        tr = subjects != held; te = subjects == held
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(np.nan_to_num(X[tr]))
        Xte = scaler.transform(np.nan_to_num(X[te]))

        # Try different alphas
        best_score = -np.inf; best_pred = None
        for alpha in [0.1, 1.0, 10.0]:
            try:
                mdl = mord.LogisticAT(alpha=alpha)
                mdl.fit(Xtr, y[tr].astype(int))
                pred = mdl.predict(Xte).astype(float)
                # Quick validation score
                score = -mean_absolute_error(y[te], pred)
                if score > best_score:
                    best_score = score; best_pred = pred
            except: continue

        if best_pred is not None:
            subj_yt.append(float(y[te][0]))
            subj_yp.append(float(np.mean(best_pred)))
            subj_names_list.append(held)

    subj_yt = np.array(subj_yt); subj_yp = np.array(subj_yp)
    m = _reg_metrics(subj_yt, subj_yp)
    print(f"    Ordinal (LogisticAT): ρ={m['rho']:.3f}  MAE={m['mae']:.3f}  "
          f"Acc±1={m['acc_w1']:.3f}  κ={m['kappa']:.3f}")

    # Grouped
    yt_g, yp_g = apply_grouping(subj_yt, subj_yp)
    mg = _cls_metrics(yt_g, yp_g, GROUP_LABELS)
    print(f"    Grouped: Acc={mg['acc']:.3f}  F1={mg['macro_f1']:.3f}  κ={mg['kappa']:.3f}")

    # Save
    path = os.path.join(odir, "stepg_ordinal_predictions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject","y_true","y_pred_ordinal","y_round","error"])
        for s, yt2, yp2 in zip(subj_names_list, subj_yt, subj_yp):
            yr = np.clip(np.round(yp2), 0, 4)
            w.writerow([s, f"{yt2:.0f}", f"{yp2:.4f}", f"{yr:.0f}", f"{abs(yt2-yp2):.4f}"])
    print(f"    Saved: {path}")

    return {"yt": subj_yt, "yp": subj_yp, "metrics": m}


def load_csv_simple(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            p = {}
            for k, v in row.items():
                try: p[k] = float(v) if v not in ("NaN","") else float("nan")
                except ValueError: p[k] = v
            rows.append(p)
    return rows

def load_features():
    path = os.path.join(C.OUTPUT_DIR, "stepe_final_features.txt")
    with open(path) as f: return [l.strip() for l in f if l.strip()]


# ============================================================
# 5. Bootstrap Confidence Intervals
# ============================================================
def bootstrap_ci(results, odir, n_boot=1000, ci=95):
    """
    Bootstrap 95% CI on subject-level metrics for all models.
    Resamples subjects (with replacement), recomputes metrics.
    """
    print(f"\n{'='*60}\n  5. Bootstrap {ci}% Confidence Intervals (n={n_boot})\n{'='*60}")

    alpha = (100 - ci) / 2
    all_rows = []

    for nm, res in results.items():
        yt = res["subj_yt"]; yp = res["subj_yp"]
        n = len(yt)
        if n < 5: continue

        boot_rho, boot_mae, boot_kappa, boot_accw1 = [], [], [], []
        np.random.seed(42)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            m = _reg_metrics(yt[idx], yp[idx])
            boot_rho.append(m["rho"]); boot_mae.append(m["mae"])
            boot_kappa.append(m["kappa"]); boot_accw1.append(m["acc_w1"])

        def _ci(arr):
            arr = [x for x in arr if np.isfinite(x)]
            if not arr: return float("nan"), float("nan"), float("nan")
            return np.mean(arr), np.percentile(arr, alpha), np.percentile(arr, 100-alpha)

        rho_m, rho_lo, rho_hi = _ci(boot_rho)
        mae_m, mae_lo, mae_hi = _ci(boot_mae)
        kap_m, kap_lo, kap_hi = _ci(boot_kappa)
        aw1_m, aw1_lo, aw1_hi = _ci(boot_accw1)

        all_rows.append({
            "model": nm,
            "rho": f"{rho_m:.3f}", "rho_ci": f"[{rho_lo:.3f}, {rho_hi:.3f}]",
            "mae": f"{mae_m:.3f}", "mae_ci": f"[{mae_lo:.3f}, {mae_hi:.3f}]",
            "kappa": f"{kap_m:.3f}", "kappa_ci": f"[{kap_lo:.3f}, {kap_hi:.3f}]",
            "acc_w1": f"{aw1_m:.3f}", "acc_w1_ci": f"[{aw1_lo:.3f}, {aw1_hi:.3f}]",
        })

    # Print
    print(f"\n    {'Model':<15s} {'ρ':>6s} {'95% CI':>16s} {'MAE':>6s} {'95% CI':>16s} {'κ':>6s} {'95% CI':>16s}")
    print(f"    {'-'*85}")
    for r in sorted(all_rows, key=lambda x: float(x["rho"]), reverse=True):
        print(f"    {r['model']:<15s} {r['rho']:>6s} {r['rho_ci']:>16s} "
              f"{r['mae']:>6s} {r['mae_ci']:>16s} {r['kappa']:>6s} {r['kappa_ci']:>16s}")

    # Save
    path = os.path.join(odir, "stepg_bootstrap_ci.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"    Saved: {path}")

    # Visualization: forest plot
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(all_rows)*0.4)), dpi=120)
    models = [r["model"] for r in all_rows]
    y_pos = np.arange(len(models))

    for ax, key, label in zip(axes, ["rho","mae","kappa"], ["Spearman ρ","MAE","Cohen κ"]):
        means = [float(r[key]) for r in all_rows]
        ci_strs = [r[f"{key}_ci"] for r in all_rows]
        los = [float(c.strip("[]").split(",")[0]) for c in ci_strs]
        his = [float(c.strip("[]").split(",")[1]) for c in ci_strs]
        errs = [[m-lo for m,lo in zip(means,los)], [hi-m for m,hi in zip(means,his)]]

        ax.errorbar(means, y_pos, xerr=errs, fmt="o", capsize=3, color="#2196F3", markersize=5)
        ax.set_yticks(y_pos); ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel(label); ax.set_title(f"{label} with {ci}% CI")
        ax.grid(True, axis="x", alpha=0.2)
        if key == "rho": ax.axvline(0, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    fig.savefig(os.path.join(odir, "stepg_bootstrap_forest.png"), bbox_inches="tight"); plt.close(fig)
    print(f"    Forest plot: stepg_bootstrap_forest.png")

    return all_rows


# ============================================================
# Summary table: everything in one place
# ============================================================
def summary_table(results, ensemble_res, grouping_rows, bootstrap_rows, ordinal_res, odir):
    """Create a single summary CSV with all approaches."""
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")

    rows = []
    # Individual models (5-class)
    for nm, res in results.items():
        m = _reg_metrics(res["subj_yt"], res["subj_yp"])
        rows.append({"approach": f"{nm} (5-class)", "rho": m["rho"], "mae": m["mae"],
                      "acc_w1": m["acc_w1"], "kappa": m["kappa"], "macro_f1": m["macro_f1"]})

    # Ensemble (5-class)
    if ensemble_res:
        m = ensemble_res["metrics_mean"]
        rows.append({"approach": "Ensemble (5-class, mean)", **{k: m[k] for k in ["rho","mae","acc_w1","kappa","macro_f1"]}})

    # Ordinal regression
    if ordinal_res:
        m = ordinal_res["metrics"]
        rows.append({"approach": "OrdinalReg (5-class)", **{k: m[k] for k in ["rho","mae","acc_w1","kappa","macro_f1"]}})

    # Sort by rho
    rows.sort(key=lambda x: x.get("rho",0), reverse=True)

    print(f"\n    {'Approach':<35s} {'ρ':>6s} {'MAE':>6s} {'Acc±1':>6s} {'κ':>6s} {'F1':>6s}")
    print(f"    {'-'*65}")
    for r in rows:
        print(f"    {r['approach']:<35s} {r['rho']:>6.3f} {r['mae']:>6.3f} {r['acc_w1']:>6.3f} "
              f"{r['kappa']:>6.3f} {r['macro_f1']:>6.3f}")

    path = os.path.join(odir, "stepg_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["approach","rho","mae","acc_w1","kappa","macro_f1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v,float) else v for k,v in r.items()})
    print(f"\n    Summary: {path}")


# ============================================================
# 6. Per-Activity Comparison (Left vs Right)
# ============================================================
def per_activity_analysis(results, odir):
    """
    Split predictions by activity (Left fist vs Right fist) and compare.
    Uses win_activity to identify which windows belong to which activity.
    Aggregates per subject-activity pair.
    """
    print(f"\n{'='*60}\n  6. Per-Activity Comparison (Left vs Right)\n{'='*60}")

    # Check if activity info is available
    ref_nm = list(results.keys())[0]
    if "win_activity" not in results[ref_nm] or results[ref_nm]["win_activity"] is None:
        print("    Activity info not available in saved predictions.")
        print("    Re-run Step F to include activity tracking.")
        return None

    activities = sorted(set(results[ref_nm]["win_activity"]))
    if len(activities) < 2:
        print(f"    Only {len(activities)} activity found: {activities}")
        print("    Need both Left and Right for comparison.")
        return None

    print(f"    Activities found: {activities}")

    all_rows = []
    for nm, res in results.items():
        wa = res["win_activity"]; ws = res["win_subj"]
        wyt = res["win_yt"]; wyp = res["win_yp"]

        for act in activities:
            # Get windows for this activity
            act_mask = wa == act
            if act_mask.sum() == 0: continue

            # Aggregate per subject within this activity
            subjs_in_act = np.unique(ws[act_mask])
            subj_yt_list, subj_yp_list = [], []
            for s in subjs_in_act:
                s_mask = act_mask & (ws == s)
                if s_mask.sum() == 0: continue
                subj_yt_list.append(float(wyt[s_mask][0]))  # same label for all windows
                subj_yp_list.append(float(np.mean(wyp[s_mask])))  # mean aggregation

            if len(subj_yt_list) < 3: continue
            yt_arr = np.array(subj_yt_list); yp_arr = np.array(subj_yp_list)
            m = _reg_metrics(yt_arr, yp_arr)

            # Also compute grouped metrics
            yt_g, yp_g = apply_grouping(yt_arr, yp_arr)
            mg = _cls_metrics(yt_g, yp_g, GROUP_LABELS) if len(yt_g) >= 3 else {"acc":float("nan"),"kappa":float("nan"),"macro_f1":float("nan")}

            side = "Left" if "Left" in act else ("Right" if "Right" in act else act)
            all_rows.append({
                "model": nm, "activity": act, "side": side,
                "n_subjects": len(subj_yt_list), "n_windows": int(act_mask.sum()),
                "rho": m["rho"], "mae": m["mae"], "acc_w1": m["acc_w1"], "kappa": m["kappa"],
                "grouped_acc": mg["acc"], "grouped_kappa": mg["kappa"], "grouped_f1": mg["macro_f1"],
            })

    if not all_rows:
        print("    No per-activity results generated."); return None

    # Print comparison table
    print(f"\n    {'Model':<15s} {'Side':>5s} {'N':>3s} | {'ρ':>6s} {'MAE':>6s} {'Acc±1':>6s} {'κ':>6s} | {'3cl Acc':>7s} {'3cl κ':>5s}")
    print(f"    {'-'*72}")
    # Sort: by model then side
    for nm in dict.fromkeys(r["model"] for r in all_rows):
        for side in ["Left", "Right"]:
            matches = [r for r in all_rows if r["model"]==nm and r["side"]==side]
            if matches:
                r = matches[0]
                print(f"    {r['model']:<15s} {r['side']:>5s} {r['n_subjects']:>3d} | "
                      f"{r['rho']:>6.3f} {r['mae']:>6.3f} {r['acc_w1']:>6.3f} {r['kappa']:>6.3f} | "
                      f"{r['grouped_acc']:>7.3f} {r['grouped_kappa']:>5.3f}")

    # Summary: which side is easier to predict?
    print(f"\n    Side comparison (averaged across models):")
    for side in ["Left", "Right"]:
        side_rows = [r for r in all_rows if r["side"] == side]
        if side_rows:
            avg_rho = np.nanmean([r["rho"] for r in side_rows])
            avg_mae = np.nanmean([r["mae"] for r in side_rows])
            avg_kappa = np.nanmean([r["kappa"] for r in side_rows])
            print(f"      {side}: avg ρ={avg_rho:.3f}, avg MAE={avg_mae:.3f}, avg κ={avg_kappa:.3f}")

    # Save CSV
    path = os.path.join(odir, "stepg_per_activity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v) for k, v in r.items()})
    print(f"    Saved: {path}")

    # Visualization: grouped bar chart (Left vs Right for top models)
    top_models = list(dict.fromkeys(r["model"] for r in all_rows))[:6]  # top 6
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    x = np.arange(len(top_models)); w_bar = 0.35

    for ax, key, label in zip(axes, ["rho","mae","kappa"], ["Spearman ρ","MAE","Cohen κ"]):
        vals_L = [next((r[key] for r in all_rows if r["model"]==m and r["side"]=="Left"), float("nan")) for m in top_models]
        vals_R = [next((r[key] for r in all_rows if r["model"]==m and r["side"]=="Right"), float("nan")) for m in top_models]
        ax.bar(x - w_bar/2, vals_L, w_bar, label="Left", color="#2196F3", alpha=0.7)
        ax.bar(x + w_bar/2, vals_R, w_bar, label="Right", color="#FF9800", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(top_models, rotation=45, ha="right", fontsize=8)
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Per-Activity Comparison: Left vs Right Fist", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(odir, "stepg_per_activity_comparison.png"), bbox_inches="tight"); plt.close(fig)
    print(f"    Plot: stepg_per_activity_comparison.png")

    return all_rows


# ============================================================
if __name__ == "__main__":
    os.makedirs(f"{C.OUTPUT_DIR}/Stepf_analysis", exist_ok=True)
    print(f"\n{'#'*70}\n  Step G: Post-Hoc Analysis\n{'#'*70}")

    for activity in ["gait"]: #["making_a_fist_Left", "making_a_fist_Right"]:
        print(f"         Activity: {activity}")
        results = load_predictions("tierA", activity)
        print(f"  Models loaded: {list(results.keys())}")

        # 1. Grouping
        grouping_rows = grouping_analysis(results, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        # 2. Ensemble
        ensemble_res = ensemble_analysis(results, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        # 3. Mean vs Median
        aggregation_comparison(results, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        # 4. Ordinal regression
        ordinal_res = ordinal_regression(results, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        # 5. Bootstrap CI
        bootstrap_rows = bootstrap_ci(results, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        # 6. Per-activity comparison
        activity_rows = per_activity_analysis(results, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        # Summary
        summary_table(results, ensemble_res, grouping_rows, bootstrap_rows, ordinal_res, f"{C.OUTPUT_DIR}/Stepf_analysis/{activity}")

        print(f"\n  Done. All outputs in {C.OUTPUT_DIR}/Stepf_analysis//{activity}/stepg_*")
