# task3_analysis.py
"""
Task 3: Treatment Response Analysis for DBS ON/OFF gait data.

Layer 1: Feature-level treatment sensitivity (Cohen's d, Wilcoxon, paired plots)
Layer 2: Body-part-specific clinical annotation analysis (heatmap)
Layer 3: Responder classification benchmark (LOSO, OFF features -> good/bad)
Layer 4: Feature-to-annotation correlation (kinematic features vs clinician body-part scores)

Reads:  {INPUT_DIR}/task3_dbs_paired.csv
Writes: {INPUT_DIR}/figures/ and {INPUT_DIR}/tables/
"""

import os, csv, copy, warnings

import config
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon, spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

warnings.filterwarnings("ignore")

# ============================================================
INPUT_DIR = config.DBS_WORK_DIR
PAIRED_CSV = os.path.join(INPUT_DIR, "task3_dbs_paired.csv")
FIG_DIR = os.path.join(INPUT_DIR, "figures")
TAB_DIR = os.path.join(INPUT_DIR, "tables")

FEATURE_NAMES = [
    "stride_length", "stride_time", "step_length_asy", "step_time_asy",
    "velocity", "cadance", "hip_ROM_asy", "knee_ROM_asy",
    "elbow_ROM_asy", "shoulder_ROM_asy", "arm2arm_ROM", "leg2leg_ROM",
    "step_length", "step_time", "hip_ROM", "knee_ROM",
    "elbow_ROM", "shoulder_ROM"
]

FEATURE_DISPLAY = {
    "stride_length": "Stride Length", "stride_time": "Stride Time",
    "step_length_asy": "Step Length Asym.", "step_time_asy": "Step Time Asym.",
    "velocity": "Velocity", "cadance": "Cadence",
    "hip_ROM_asy": "Hip ROM Asym.", "knee_ROM_asy": "Knee ROM Asym.",
    "elbow_ROM_asy": "Elbow ROM Asym.", "shoulder_ROM_asy": "Shoulder ROM Asym.",
    "arm2arm_ROM": "Arm-Arm ROM", "leg2leg_ROM": "Leg-Leg ROM",
    "step_length": "Step Length", "step_time": "Step Time",
    "hip_ROM": "Hip ROM", "knee_ROM": "Knee ROM",
    "elbow_ROM": "Elbow ROM", "shoulder_ROM": "Shoulder ROM",
}

BP_DISPLAY = {
    "bp_armswing_max": "Arm Swing (max)", "bp_armswing_left": "Arm Swing (L)",
    "bp_armswing_right": "Arm Swing (R)", "bp_armswing_asy": "Arm Swing Asym.",
    "bp_stride_max": "Stride Length", "bp_stride_left": "Stride (L)", "bp_stride_right": "Stride (R)",
    "bp_elbow_max": "Elbow Flex. (max)", "bp_elbow_left": "Elbow Flex. (L)",
    "bp_elbow_right": "Elbow Flex. (R)", "bp_elbow_asy": "Elbow Flex. Asym.",
}

# Mapping: which kinematic features should correlate with which body-part annotation
FEATURE_TO_BP = {
    # arm swing annotations should correlate with shoulder/elbow/arm features
    "bp_armswing_left":  ["shoulder_ROM", "elbow_ROM", "arm2arm_ROM", "shoulder_ROM_asy", "elbow_ROM_asy"],
    "bp_armswing_right": ["shoulder_ROM", "elbow_ROM", "arm2arm_ROM", "shoulder_ROM_asy", "elbow_ROM_asy"],
    "bp_armswing_max":   ["shoulder_ROM", "elbow_ROM", "arm2arm_ROM"],
    # stride annotations should correlate with stride/step/velocity features
    "bp_stride_max":     ["stride_length", "step_length", "velocity", "hip_ROM", "knee_ROM", "leg2leg_ROM"],
    "bp_stride_left":    ["stride_length", "step_length", "velocity", "step_length_asy"],
    "bp_stride_right":   ["stride_length", "step_length", "velocity", "step_length_asy"],
    # elbow annotations should correlate with elbow features
    "bp_elbow_max":      ["elbow_ROM", "elbow_ROM_asy"],
    "bp_elbow_left":     ["elbow_ROM", "elbow_ROM_asy"],
    "bp_elbow_right":    ["elbow_ROM", "elbow_ROM_asy"],
}

N_BOOTSTRAP = 1000
SEED = 42


def load_paired():
    rows = []
    with open(PAIRED_CSV) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try: parsed[k] = float(v) if v not in ("", "nan", "NaN") else float("nan")
                except ValueError: parsed[k] = v
            rows.append(parsed)
    print(f"  Loaded {len(rows)} paired subjects")
    return rows


# ============================================================
# LAYER 1: Feature Sensitivity (unchanged from previous version)
# ============================================================
def layer1_feature_sensitivity(rows):
    print(f"\n{'='*70}\n  Layer 1: Feature-Level Treatment Sensitivity\n{'='*70}")
    rng = np.random.RandomState(SEED)
    results = []
    for fn in FEATURE_NAMES:
        off = np.array([r.get(f"OFF_{fn}", np.nan) for r in rows])
        on = np.array([r.get(f"ON_{fn}", np.nan) for r in rows])
        valid = np.isfinite(off) & np.isfinite(on)
        ov, nv = off[valid], on[valid]; n = len(ov)
        if n < 4:
            results.append({"feature": fn, "n": n, "cohens_d": np.nan, "d_ci_lo": np.nan,
                            "d_ci_hi": np.nan, "wilcoxon_p": np.nan, "mean_off": np.nan,
                            "mean_on": np.nan, "mean_delta": np.nan})
            continue
        diff = ov - nv; d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        boots = [np.mean(b := diff[rng.choice(n, n, replace=True)]) / (np.std(b, ddof=1) + 1e-12)
                 for _ in range(N_BOOTSTRAP)]
        ci = np.percentile(boots, [2.5, 97.5])
        try: _, wp = wilcoxon(diff)
        except: wp = np.nan
        results.append({"feature": fn, "n": n, "mean_off": float(np.mean(ov)),
                         "mean_on": float(np.mean(nv)), "mean_delta": float(np.mean(diff)),
                         "cohens_d": float(d), "d_ci_lo": float(ci[0]), "d_ci_hi": float(ci[1]),
                         "wilcoxon_p": float(wp)})
    results.sort(key=lambda x: abs(x.get("cohens_d", 0) if np.isfinite(x.get("cohens_d", 0)) else 0), reverse=True)
    print(f"\n  {'Feature':<25s} {'d':>7s} {'95% CI':>16s} {'p':>8s}")
    print(f"  {'-'*60}")
    for r in results:
        star = "***" if r["wilcoxon_p"]<.001 else "**" if r["wilcoxon_p"]<.01 else "*" if r["wilcoxon_p"]<.05 else ""
        print(f"  {r['feature']:<25s} {r['cohens_d']:>7.3f} [{r['d_ci_lo']:>6.2f}, {r['d_ci_hi']:>6.2f}] {r['wilcoxon_p']:>8.4f} {star}")
    _save_csv(results, os.path.join(TAB_DIR, "layer1_feature_sensitivity.csv"))
    _plot_paired(rows, results)
    _plot_forest(results)
    return results


def _plot_paired(rows, top):
    n = len(top)
    # fig, axes = plt.subplots(1, n, figsize=(3.2*n, 4.5), dpi=150)
    # if n == 1: axes = [axes]
    ncols, nrows = 6, n//6
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 4.5*nrows), dpi=150)
    axes = axes.flatten()
    resps = ["good" if r.get("responder","") in (1,1.0) else ("bad" if r.get("responder","") in (0,0.0) else "unk") for r in rows]
    cmap = {"good": "#2196F3", "bad": "#F44336", "unk": "#999"}
    for idx, res in enumerate(top):
        ax = axes[idx]; fn = res["feature"]
        offv = [r[f"OFF_{fn}"] for r in rows]; onv = [r[f"ON_{fn}"] for r in rows]
        bp = ax.boxplot([offv, onv], positions=[0,1], widths=0.35, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor("#FFCDD2"); bp["boxes"][1].set_facecolor("#C8E6C9")
        for i in range(len(rows)):
            if np.isfinite(offv[i]) and np.isfinite(onv[i]):
                ax.plot([0,1],[offv[i],onv[i]],"o-",color=cmap[resps[i]],alpha=0.6,markersize=5,linewidth=1.2,markeredgecolor="white",markeredgewidth=0.5)
        ax.set_xticks([0,1]); ax.set_xticklabels(["OFF","ON"],fontsize=10)
        ax.set_title(f"{FEATURE_DISPLAY.get(fn,fn)}\n(d={res['cohens_d']:.2f})",fontsize=9,fontweight="bold")
        ax.grid(True,axis="y",alpha=0.2)
    handles = [mpatches.Patch(fc="#2196F3",alpha=0.6,label="Good resp."),mpatches.Patch(fc="#F44336",alpha=0.6,label="Poor resp.")]
    axes[n-1].legend(handles=handles,loc="upper right",fontsize=7)
    fig.suptitle("Gait Feature Changes: DBS OFF vs ON",fontsize=12,fontweight="bold",y=1.02)
    plt.tight_layout()
    for ext in [".pdf",".png"]: fig.savefig(os.path.join(FIG_DIR,f"figure5a_paired_features{ext}"),bbox_inches="tight")
    plt.close(fig); print(f"  Saved: figure5a_paired_features.pdf")


def _plot_forest(results):
    fig, ax = plt.subplots(figsize=(6,7),dpi=150); n=len(results); y=np.arange(n)
    ds=[r["cohens_d"] for r in results]; lo=[r["d_ci_lo"] for r in results]; hi=[r["d_ci_hi"] for r in results]
    names=[FEATURE_DISPLAY.get(r["feature"],r["feature"]) for r in results]; ps=[r["wilcoxon_p"] for r in results]
    colors=["#2196F3" if abs(d)>=0.8 else "#64B5F6" if abs(d)>=0.5 else "#BBDEFB" for d in ds]
    for i in range(n):
        ax.barh(y[i],ds[i],color=colors[i],edgecolor="white",height=0.6,alpha=0.8)
        ax.plot([lo[i],hi[i]],[y[i],y[i]],"k-",linewidth=1.5); ax.plot([lo[i],hi[i]],[y[i],y[i]],"k|",markersize=6)
        if ps[i]<0.05: ax.text(max(hi[i],ds[i])+0.05,y[i],"*",va="center",fontsize=12,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names,fontsize=8)
    for v in [0.5,-0.5]: ax.axvline(v,color="gray",lw=0.5,ls="--",alpha=0.5)
    for v in [0.8,-0.8]: ax.axvline(v,color="gray",lw=0.5,ls=":",alpha=0.5)
    ax.axvline(0,color="black",lw=0.8); ax.set_xlabel("Cohen's d (OFF - ON)")
    ax.set_title("Treatment Effect Sizes",fontsize=11,fontweight="bold"); ax.invert_yaxis(); ax.grid(True,axis="x",alpha=0.2)
    plt.tight_layout()
    for ext in [".pdf",".png"]: fig.savefig(os.path.join(FIG_DIR,f"forest_plot_cohens_d{ext}"),bbox_inches="tight")
    plt.close(fig); print(f"  Saved: forest_plot_cohens_d.pdf")


# ============================================================
# LAYER 2: Body-Part Annotation Analysis (unchanged)
# ============================================================
def layer2_bodypart_analysis(rows):
    print(f"\n{'='*70}\n  Layer 2: Body-Part Annotation Analysis\n{'='*70}")
    bp_cols = [bp for bp in BP_DISPLAY if any(f"delta_{bp}" in r and r.get(f"delta_{bp}","") != "" for r in rows)]
    if not bp_cols: print("  No body-part annotations. Skipping."); return
    subjects = [r.get("subject",f"S{i+1}") for i,r in enumerate(rows)]
    n_s, n_bp = len(rows), len(bp_cols)
    matrix = np.full((n_bp,n_s),np.nan)
    for j,r in enumerate(rows):
        for i,bp in enumerate(bp_cols):
            v = r.get(f"delta_{bp}","")
            if v != "" and str(v) != "nan":
                try: matrix[i,j] = float(v)
                except: pass
    resp_labels = ["Good" if r.get("responder","") in (1,1.0) else ("Poor" if r.get("responder","") in (0,0.0) else "?") for r in rows]
    fig, ax = plt.subplots(figsize=(max(8,n_s*0.9+2),n_bp*0.55+2),dpi=150)
    vmax = max(abs(np.nanmin(matrix)),abs(np.nanmax(matrix)),1)
    norm = TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    im = ax.imshow(matrix,cmap="RdBu",norm=norm,aspect="auto")
    for i in range(n_bp):
        for j in range(n_s):
            v=matrix[i,j]
            if np.isfinite(v): ax.text(j,i,f"{'+' if v>0 else ''}{int(v)}",ha="center",va="center",fontsize=9,color="white" if abs(v)>=vmax*0.6 else "black")
    ax.set_yticks(range(n_bp)); ax.set_yticklabels([BP_DISPLAY.get(bp,bp) for bp in bp_cols],fontsize=9)
    ax.set_xticks(range(n_s)); ax.set_xticklabels([f"S{i+1}\n({rl})" for i,rl in enumerate(resp_labels)],fontsize=8)
    ax.set_title("Body-Part Severity Change (OFF - ON)\npositive = improvement",fontsize=11,fontweight="bold")
    fig.colorbar(im,ax=ax,shrink=0.6,label="Severity change"); plt.tight_layout()
    for ext in [".pdf",".png"]: fig.savefig(os.path.join(FIG_DIR,f"figure5b_bodypart_heatmap{ext}"),bbox_inches="tight")
    plt.close(fig); print(f"  Saved: figure5b_bodypart_heatmap.pdf")
    tab = []
    for i,bp in enumerate(bp_cols):
        vals = matrix[i,:][np.isfinite(matrix[i,:])]
        row = {"bodypart": BP_DISPLAY.get(bp,bp)}
        for j in range(n_s): row[f"S{j+1}_{resp_labels[j]}"] = int(matrix[i,j]) if np.isfinite(matrix[i,j]) else ""
        row["mean_delta"] = f"{np.mean(vals):.2f}" if len(vals)>0 else ""
        row["n_improved"] = int(np.sum(vals>0)) if len(vals)>0 else ""
        tab.append(row)
    _save_csv(tab, os.path.join(TAB_DIR,"layer2_bodypart_changes.csv"))
    print(f"\n  UPDRS-gait unchanged: {sum(1 for r in rows if r.get('delta_gait_updrs',0)==0)}/{len(rows)}")
    for i,bp in enumerate(bp_cols):
        vals = matrix[i,:][np.isfinite(matrix[i,:])]
        if len(vals)>0: print(f"    {BP_DISPLAY.get(bp,bp):25s}  improved: {int(np.sum(vals>0))}/{len(vals)}  mean delta={np.mean(vals):+.2f}")


# ============================================================
# LAYER 3: Responder Classification (unchanged)
# ============================================================
def layer3_responder_classification(rows):
    print(f"\n{'='*70}\n  Layer 3: Responder Classification (LOSO)\n{'='*70}")
    labeled = [r for r in rows if r.get("responder","") in (0,0.0,1,1.0)]
    if len(labeled)<4: print(f"  Only {len(labeled)} labeled. Skipping."); return
    y = np.array([int(r["responder"]) for r in labeled])
    X = np.array([[r.get(f"OFF_{fn}",np.nan) for fn in FEATURE_NAMES] for r in labeled])
    subjs = [r.get("subject",f"S{i}") for i,r in enumerate(labeled)]
    print(f"  Subjects: {len(labeled)} ({sum(y==1)} good, {sum(y==0)} bad)")
    models = {"LogReg": LogisticRegression(C=1.0,max_iter=1000,random_state=SEED),
              "SVM-Lin": SVC(C=1.0,kernel="linear",probability=True,random_state=SEED),
              "SVM-RBF": SVC(C=1.0,kernel="rbf",probability=True,random_state=SEED),
              "RF": RandomForestClassifier(n_estimators=100,max_depth=2,random_state=SEED,n_jobs=1)}
    all_res = {}
    for mn, mt in models.items():
        preds = np.zeros(len(y),dtype=int); probs = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j!=i]
            sc = StandardScaler(); Xtr = sc.fit_transform(np.nan_to_num(X[tr])); Xte = sc.transform(np.nan_to_num(X[[i]]))
            try: m = copy.deepcopy(mt); m.fit(Xtr,y[tr]); preds[i]=m.predict(Xte)[0]; probs[i]=m.predict_proba(Xte)[0,1]
            except: preds[i]=0; probs[i]=0.5
        acc=accuracy_score(y,preds); f1=f1_score(y,preds,average="macro",zero_division=0)
        try: kappa=cohen_kappa_score(y,preds)
        except: kappa=0.0
        all_res[mn]={"acc":acc,"f1":f1,"kappa":kappa,"preds":preds,"probs":probs}
        print(f"  {mn:<12s}  Acc={acc:.3f}  F1={f1:.3f}  Kappa={kappa:.3f}")
    pred_tab = []
    for i,r in enumerate(labeled):
        row = {"subject":subjs[i],"true":int(y[i]),"label":"Good" if y[i]==1 else "Poor",
               "delta_gait":r.get("delta_gait_updrs",""),"delta_combined":r.get("delta_combined_updrs","")}
        for mn in models: row[f"{mn}_pred"]=int(all_res[mn]["preds"][i]); row[f"{mn}_prob"]=f"{all_res[mn]['probs'][i]:.3f}"
        pred_tab.append(row)
    _save_csv(pred_tab, os.path.join(TAB_DIR,"layer3_responder_predictions.csv"))
    sc=StandardScaler(); Xa=sc.fit_transform(np.nan_to_num(X))
    rf=RandomForestClassifier(n_estimators=200,max_depth=2,random_state=SEED,n_jobs=1); rf.fit(Xa,y); imp=rf.feature_importances_
    order=np.argsort(imp)[::-1]
    print(f"\n  RF Feature Importance (top 10):")
    for rank,idx in enumerate(order[:10]): print(f"    {rank+1}. {FEATURE_NAMES[idx]:<25s}  {imp[idx]:.4f}")
    _save_csv([{"rank":rank+1,"feature":FEATURE_NAMES[idx],"importance":f"{imp[idx]:.6f}"} for rank,idx in enumerate(order)],
              os.path.join(TAB_DIR,"layer3_feature_importance.csv"))
    fig,ax=plt.subplots(figsize=(6,5),dpi=150); top_n=min(10,len(FEATURE_NAMES)); ti=order[:top_n]
    ax.barh(range(top_n),imp[ti],color="#64B5F6",edgecolor="white"); ax.set_yticks(range(top_n))
    ax.set_yticklabels([FEATURE_DISPLAY.get(FEATURE_NAMES[i],FEATURE_NAMES[i]) for i in ti],fontsize=9)
    ax.set_xlabel("Importance"); ax.set_title("RF Feature Importance: Predicting DBS Response",fontsize=10,fontweight="bold")
    ax.invert_yaxis(); ax.grid(True,axis="x",alpha=0.2); plt.tight_layout()
    for ext in [".pdf",".png"]: fig.savefig(os.path.join(FIG_DIR,f"layer3_feature_importance{ext}"),bbox_inches="tight")
    plt.close(fig); print(f"  Saved: layer3_feature_importance.pdf")
    return all_res


# ============================================================
# LAYER 4: Feature-to-Annotation Correlation (NEW)
# ============================================================
def layer4_feature_annotation_correlation(rows):
    """
    Correlate kinematic features with clinician body-part severity scores
    across both OFF and ON states (pooled: 10 subjects x 2 states = 20 observations).
    Also compute delta-to-delta correlations (does kinematic change match annotation change?).
    """
    print(f"\n{'='*70}\n  Layer 4: Feature-to-Annotation Correlation\n{'='*70}")

    results_pooled = []
    results_delta = []

    for bp_col, feat_list in FEATURE_TO_BP.items():
        bp_name = BP_DISPLAY.get(bp_col, bp_col)

        for fn in feat_list:
            fn_display = FEATURE_DISPLAY.get(fn, fn)

            # --- Pooled correlation (OFF + ON as separate observations) ---
            bp_vals, feat_vals = [], []
            for r in rows:
                for state in ["OFF", "ON"]:
                    bp_v = r.get(f"{state}_{bp_col}", "")
                    feat_v = r.get(f"{state}_{fn}", np.nan)
                    if bp_v != "" and str(bp_v) != "nan" and np.isfinite(float(bp_v)) and np.isfinite(feat_v):
                        bp_vals.append(float(bp_v))
                        feat_vals.append(feat_v)

            if len(bp_vals) >= 6:
                rho, p = spearmanr(bp_vals, feat_vals)
                results_pooled.append({
                    "bodypart": bp_name, "bodypart_col": bp_col,
                    "feature": fn_display, "feature_col": fn,
                    "rho": float(rho), "p": float(p), "n": len(bp_vals),
                    "analysis": "pooled_OFF_ON"
                })

            # --- Delta correlation (does kinematic change match annotation change?) ---
            delta_bp, delta_feat = [], []
            for r in rows:
                dbp = r.get(f"delta_{bp_col}", "")
                dfn = r.get(f"delta_{fn}", np.nan)
                if dbp != "" and str(dbp) != "nan" and np.isfinite(float(dbp)) and np.isfinite(dfn):
                    delta_bp.append(float(dbp))
                    delta_feat.append(dfn)

            if len(delta_bp) >= 5:
                rho, p = spearmanr(delta_bp, delta_feat)
                results_delta.append({
                    "bodypart": bp_name, "bodypart_col": bp_col,
                    "feature": fn_display, "feature_col": fn,
                    "rho": float(rho), "p": float(p), "n": len(delta_bp),
                    "analysis": "delta_OFF_minus_ON"
                })

    # Print pooled results
    print(f"\n  === Pooled Correlation (OFF+ON, ~20 obs per pair) ===")
    print(f"  {'Body Part':<22s} {'Feature':<22s} {'rho':>6s} {'p':>8s} {'n':>4s}")
    print(f"  {'-'*65}")
    for r in sorted(results_pooled, key=lambda x: abs(x["rho"]), reverse=True):
        star = "*" if r["p"]<0.05 else ""
        print(f"  {r['bodypart']:<22s} {r['feature']:<22s} {r['rho']:>6.3f} {r['p']:>8.4f} {r['n']:>4d} {star}")

    # Print delta results
    print(f"\n  === Delta Correlation (feature change vs annotation change, ~10 obs) ===")
    print(f"  {'Body Part':<22s} {'Feature':<22s} {'rho':>6s} {'p':>8s} {'n':>4s}")
    print(f"  {'-'*65}")
    for r in sorted(results_delta, key=lambda x: abs(x["rho"]), reverse=True):
        star = "*" if r["p"]<0.05 else ""
        print(f"  {r['bodypart']:<22s} {r['feature']:<22s} {r['rho']:>6.3f} {r['p']:>8.4f} {r['n']:>4d} {star}")

    # Save
    _save_csv(results_pooled, os.path.join(TAB_DIR, "layer4_pooled_correlation.csv"))
    _save_csv(results_delta, os.path.join(TAB_DIR, "layer4_delta_correlation.csv"))

    # Plot: heatmap of pooled correlations
    _plot_correlation_heatmap(results_pooled, "pooled")
    _plot_correlation_heatmap(results_delta, "delta")

    return results_pooled, results_delta


def _plot_correlation_heatmap(results, tag):
    """Heatmap: rows=body parts, cols=features, cells=Spearman rho."""
    if not results:
        return

    bp_names = sorted(set(r["bodypart"] for r in results))
    feat_names = sorted(set(r["feature"] for r in results))
    matrix = np.full((len(bp_names), len(feat_names)), np.nan)
    pmat = np.full((len(bp_names), len(feat_names)), np.nan)

    for r in results:
        i = bp_names.index(r["bodypart"])
        j = feat_names.index(r["feature"])
        matrix[i, j] = r["rho"]
        pmat[i, j] = r["p"]

    fig, ax = plt.subplots(figsize=(max(6, len(feat_names)*0.9+2), len(bp_names)*0.6+2), dpi=150)
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.5)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(bp_names)):
        for j in range(len(feat_names)):
            v = matrix[i, j]; p = pmat[i, j]
            if np.isfinite(v):
                star = "*" if p < 0.05 else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > vmax*0.6 else "black")

    ax.set_yticks(range(len(bp_names))); ax.set_yticklabels(bp_names, fontsize=9)
    ax.set_xticks(range(len(feat_names))); ax.set_xticklabels(feat_names, fontsize=7, rotation=45, ha="right")
    title = "Pooled (OFF+ON) Feature-Annotation Correlation" if tag == "pooled" else "Delta Feature vs Delta Annotation Correlation"
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ"); plt.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(FIG_DIR, f"layer4_{tag}_correlation_heatmap{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: layer4_{tag}_correlation_heatmap.pdf")


# ============================================================
# UPDRS insensitivity
# ============================================================
def updrs_insensitivity(rows, l1):
    print(f"\n{'='*70}\n  UPDRS Insensitivity Analysis\n{'='*70}")
    n=len(rows); g0=sum(1 for r in rows if r.get("delta_gait_updrs",0)==0)
    c0=sum(1 for r in rows if r.get("delta_combined_updrs",0)==0)
    sig=[r for r in l1 if np.isfinite(r.get("cohens_d",0)) and abs(r["cohens_d"])>=0.5]
    print(f"  UPDRS-gait unchanged: {g0}/{n}"); print(f"  Combined unchanged: {c0}/{n}")
    print(f"  Features with |d| >= 0.5: {len(sig)}/{len(l1)}")
    for r in sig: print(f"    {r['feature']:<25s}  d={r['cohens_d']:>+.3f}")


def _save_csv(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows: w.writerow({k:(f"{v:.6f}" if isinstance(v,float) and np.isfinite(v) else str(v)) for k,v in row.items()})
    print(f"  Saved: {path}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR,exist_ok=True); os.makedirs(TAB_DIR,exist_ok=True)
    print(f"\n{'#'*70}\n  Task 3: Treatment Response Analysis\n{'#'*70}\n")
    rows = load_paired()
    l1 = layer1_feature_sensitivity(rows)
    layer2_bodypart_analysis(rows)
    layer3_responder_classification(rows)
    layer4_feature_annotation_correlation(rows)
    updrs_insensitivity(rows, l1)
    print(f"\n{'#'*70}\n  Task 3 Complete\n{'#'*70}")
