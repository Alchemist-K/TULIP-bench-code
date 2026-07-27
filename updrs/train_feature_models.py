# stepf_analysis.py
"""
Step F : Slim grids for baselines, full grids for top models.

Saves EVERYTHING needed for post-hoc analysis (Step G) without re-running:
  - stepf_tierA_all_predictions.pkl: all model predictions (window + subject level)
  - stepf_tierA_predictions_{model}.csv: per-model subject-level CSV
  - stepf_tierA_metrics.csv: metrics at both levels
  - stepf_tierA_fold_details.pkl: per-fold params, scaler states, etc.

Grid philosophy:
  - Top models (GBR, XGBoost, LightGBM, RF): full grid
  - Baselines (Ridge, Lasso, ElasticNet, SVR, MLP): minimal grid (1-3 combos)
  - Classifiers: moderate grid
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv, warnings, pickle, numpy as np, matplotlib, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product as iprod
from scipy.stats import spearmanr, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, f1_score, cohen_kappa_score
from sklearn.inspection import permutation_importance
import config as C

warnings.filterwarnings("ignore")
try: from xgboost import XGBRegressor, XGBClassifier; HAS_XGB = True
except ImportError: HAS_XGB = False
try: from lightgbm import LGBMRegressor; HAS_LGBM = True
except ImportError: HAS_LGBM = False
try: from imblearn.over_sampling import SMOTE; HAS_SMOTE = True
except ImportError: HAS_SMOTE = False
try: from joblib import Parallel, delayed; HAS_JOBLIB = True
except ImportError: HAS_JOBLIB = False

# ---- Config ----
USE_SMOTE = True; N_WORKERS = 4; INNER_FOLDS = 5; SMOTE_K = 3

# ============================================================
def load_final_features():
    path = os.path.join(C.OUTPUT_DIR, "stepe_final_features.txt")
    with open(path) as f: return [l.strip() for l in f if l.strip()]

def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            p = {}
            for k, v in row.items():
                try: p[k] = float(v) if v not in ("NaN","") else float("nan")
                except ValueError: p[k] = v
            rows.append(p)
    return rows

# ============================================================
# Models: slim baselines + full top models
# ============================================================
def get_models():
    m = {}
    # ---- Baselines (minimal grids, fast) ----
    m["Ridge"]      = (Ridge,      {"alpha":[1,10]}, "reg")
    m["Lasso"]      = (Lasso,      {"alpha":[0.1,1]}, "reg")
    m["ElasticNet"] = (ElasticNet, {"alpha":[0.1,1],"l1_ratio":[0.5]}, "reg")
    m["SVR"]        = (SVR,        {"C":[1,10],"gamma":["scale"]}, "reg")
    m["MLP"]        = (MLPRegressor, {"hidden_layer_sizes":[(64,),(32,16)],"alpha":[0.01],"max_iter":[500]}, "reg")

    # ---- Top models (full grids) ----
    m["RF"]  = (RandomForestRegressor, {
        "n_estimators":[200,300,500],"max_depth":[2,3,5],
        "min_samples_leaf":[2,3,5],"n_jobs":[1]}, "reg")
    m["GBR"] = (GradientBoostingRegressor, {
        "n_estimators":[200,300,500],"max_depth":[2,3,5],
        "learning_rate":[0.01,0.05,0.1]}, "reg")
    if HAS_XGB:
        m["XGBoost"] = (XGBRegressor, {
            "n_estimators":[200,300,500],"max_depth":[2,3,5],
            "learning_rate":[0.01,0.05,0.1],
            "subsample":[0.8],"colsample_bytree":[0.8],
            "verbosity":[0],"n_jobs":[1]}, "reg")
    if HAS_LGBM:
        m["LightGBM"] = (LGBMRegressor, {
            "n_estimators":[200,300,500],"max_depth":[2,3,5],
            "learning_rate":[0.01,0.05,0.1],
            "num_leaves":[7,15,31],
            "verbose":[-1],"n_jobs":[1]}, "reg")

    # ---- Classifiers (moderate grids) ----
    m["RF-Cls"] = (RandomForestClassifier, {
        "n_estimators":[200,500],"max_depth":[3,5],
        "class_weight":["balanced"],"n_jobs":[1]}, "cls")
    m["GBR-Cls"] = (GradientBoostingClassifier, {
        "n_estimators":[200,300],"max_depth":[2,3],"learning_rate":[0.05,0.1]}, "cls")
    if HAS_XGB:
        m["XGB-Cls"] = (XGBClassifier, {
            "n_estimators":[200,500],"max_depth":[2,3],
            "learning_rate":[0.05,0.1],"verbosity":[0],"n_jobs":[1]}, "cls")
    return m

# def get_models():
#     m = {}
#     # ---- Baselines (minimal grids, fast) ----
#     m["Ridge"]      = (Ridge,      {"alpha":[1,10]}, "reg")
#     return m

# ============================================================
def _tune(cls, grid, X, y, groups, mtype, ni=INNER_FOLDS):
    keys = list(grid.keys()); vals = list(grid.values())
    best_s = -np.inf; best_p = {k: v[0] for k,v in grid.items()}
    na = min(ni, len(np.unique(groups)))
    if na < 2: return best_p
    gkf = GroupKFold(n_splits=na)
    for combo in iprod(*vals):
        params = dict(zip(keys, combo))
        if "num_leaves" in params and "max_depth" in params:
            if params["max_depth"] > 0 and params["num_leaves"] > 2**params["max_depth"]:
                continue
        try:
            sc = []
            for tr, va in gkf.split(X, y, groups):
                mdl = cls(**params)
                mdl.fit(X[tr], y[tr].astype(int) if mtype=="cls" else y[tr])
                sc.append(-mean_absolute_error(y[va], mdl.predict(X[va])))
            ms = np.mean(sc)
            if ms > best_s: best_s = ms; best_p = params
        except: continue
    return best_p

def _metrics(yt, yp):
    v = np.isfinite(yp) & np.isfinite(yt)
    if v.sum() < 3: return {k:float("nan") for k in ["rho","p","mae","acc_exact","acc_w1","macro_f1","kappa"]}
    yt2,yp2 = yt[v],yp[v]; yr = np.clip(np.round(yp2),0,4)
    rho,pv = spearmanr(yt2,yp2)
    return {"rho":float(rho),"p":float(pv),"mae":float(mean_absolute_error(yt2,yp2)),
            "acc_exact":float(np.mean(yr==yt2)),"acc_w1":float(np.mean(np.abs(yr-yt2)<=1)),
            "macro_f1":float(f1_score(yt2.astype(int),yr.astype(int),average="macro",zero_division=0)),
            "kappa":float(cohen_kappa_score(yt2.astype(int),yr.astype(int)))}

# ============================================================
def _run_one_fold(held_subj, X, y, subjects, activities, zoo, use_smote):
    train_mask = subjects != held_subj; test_mask = subjects == held_subj
    assert not np.any(subjects[train_mask] == held_subj)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_activities = activities[test_mask]  # track which activity each test window belongs to
    groups_train = subjects[train_mask]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.nan_to_num(X_train))
    X_te = scaler.transform(np.nan_to_num(X_test))
    # SMOTE
    X_sm, y_sm, g_sm = X_tr, y_train, groups_train
    if use_smote and HAS_SMOTE:
        try:
            cc = Counter(y_train.astype(int)); k = min(SMOTE_K, min(cc.values())-1)
            if k >= 1 and len(cc) > 1:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_sm, y_sm = sm.fit_resample(X_tr, y_train.astype(int))
                y_sm = y_sm.astype(float)
                g_sm = np.concatenate([groups_train, np.array([f"SMOTE_{i}" for i in range(len(X_sm)-len(X_tr))])])
        except: pass

    fold_results = {}
    for nm, (cls, grid, mtype) in zoo.items():
        bp = _tune(cls, grid, X_sm, y_sm, g_sm, mtype)
        try:
            mdl = cls(**bp)
            mdl.fit(X_sm, y_sm.astype(int) if mtype=="cls" else y_sm)
            wp = mdl.predict(X_te).astype(float)
            fold_results[nm] = {"win_preds":wp.tolist(),"win_true":y_test.tolist(),
                                "win_activities":test_activities.tolist(),
                                "subj_pred":float(np.mean(wp)),"subj_true":float(y_test[0]),
                                "params":bp,"n_windows":len(wp),
                                "subj_pred_median":float(np.median(wp))}
        except:
            fold_results[nm] = {"win_preds":[float("nan")]*len(y_test),"win_true":y_test.tolist(),
                                "win_activities":test_activities.tolist(),
                                "subj_pred":float("nan"),"subj_true":float(y_test[0]),
                                "params":{},"n_windows":len(y_test),"subj_pred_median":float("nan")}
    return held_subj, fold_results

# ============================================================
def loso_window_aggregate(rows, feat_names, use_smote=USE_SMOTE, n_workers=N_WORKERS):
    X = np.array([[r.get(fn, float("nan")) for fn in feat_names] for r in rows])
    y = np.array([r["updrs_score"] for r in rows])
    subjects = np.array([r["subject"] for r in rows])
    activities = np.array([r.get("activity", "unknown") for r in rows])  # track activity
    unique_subj = np.unique(subjects); zoo = get_models()

    # Count combos
    tc = 0
    for nm,(cls,grid,mt) in zoo.items():
        n = 1
        for v in grid.values(): n *= len(v)
        tc += n
    print(f"    LOSO: {len(unique_subj)} subj, {X.shape[0]} win, {X.shape[1]} feat, {len(zoo)} models, {tc} combos")
    print(f"    SMOTE: {use_smote and HAS_SMOTE}, Workers: {n_workers}")

    t0 = time.time()
    if n_workers > 1 and HAS_JOBLIB:
        folds = Parallel(n_jobs=n_workers, verbose=5)(
            delayed(_run_one_fold)(h, X, y, subjects, activities, zoo, use_smote) for h in unique_subj)
    else:
        folds = []
        for i, h in enumerate(unique_subj):
            folds.append(_run_one_fold(h, X, y, subjects, activities, zoo, use_smote))
            if (i+1)%10==0: print(f"      Fold {i+1}/{len(unique_subj)}")
    print(f"    Done in {time.time()-t0:.1f}s")

    results = {}
    for nm in zoo:
        results[nm] = {"win_yt":[],"win_yp":[],"win_subj":[],"win_activity":[],
                       "subj_yt":[],"subj_yp":[],"subj_yp_median":[],"subj_names":[],
                       "params":[],"type":zoo[nm][2]}
    for held, fr in folds:
        for nm in zoo:
            if nm not in fr: continue
            f = fr[nm]
            for wp, wt, wa in zip(f["win_preds"], f["win_true"], f.get("win_activities", ["unknown"]*len(f["win_preds"]))):
                results[nm]["win_yt"].append(float(wt)); results[nm]["win_yp"].append(float(wp))
                results[nm]["win_subj"].append(held); results[nm]["win_activity"].append(wa)
            results[nm]["subj_yt"].append(f["subj_true"])
            results[nm]["subj_yp"].append(f["subj_pred"])
            results[nm]["subj_yp_median"].append(f["subj_pred_median"])
            results[nm]["subj_names"].append(held)
            results[nm]["params"].append(f["params"])
    for nm in results:
        for k in ["win_yt","win_yp","subj_yt","subj_yp","subj_yp_median"]:
            results[nm][k] = np.array(results[nm][k])
        results[nm]["win_subj"] = np.array(results[nm]["win_subj"])
        results[nm]["win_activity"] = np.array(results[nm]["win_activity"])
        results[nm]["subj_names"] = np.array(results[nm]["subj_names"])
    return results

# ============================================================
# Save EVERYTHING
# ============================================================
def save_all(results, tier, odir, activity):
    """Save complete results: pkl (full), per-model CSVs, metrics CSV."""
    os.makedirs(f"{odir}/{activity}", exist_ok=True)
    # 1. Full pickle
    pkl_path = os.path.join(odir, activity, f"stepf_{tier}_all_predictions.pkl")
    save_data = {}
    for nm, res in results.items():
        save_data[nm] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in res.items()}
    with open(pkl_path, "wb") as f: pickle.dump(save_data, f, protocol=4)
    print(f"    Full pkl: {pkl_path}")

    # 2. Per-model CSVs
    for nm, res in results.items():
        csv_path = os.path.join(odir, activity, f"stepf_{tier}_predictions_{nm}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject","y_true","y_pred_mean","y_pred_median","y_round_mean","y_round_median","error_mean","error_median","n_windows"])
            for i, s in enumerate(res["subj_names"]):
                yt = res["subj_yt"][i]; ypm = res["subj_yp"][i]; ypmed = res["subj_yp_median"][i]
                yrm = np.clip(np.round(ypm),0,4) if np.isfinite(ypm) else float("nan")
                yrmed = np.clip(np.round(ypmed),0,4) if np.isfinite(ypmed) else float("nan")
                nw = int(np.sum(res["win_subj"]==s))
                w.writerow([s, f"{yt:.0f}", f"{ypm:.4f}", f"{ypmed:.4f}",
                            f"{yrm:.0f}", f"{yrmed:.0f}",
                            f"{abs(yt-ypm):.4f}", f"{abs(yt-ypmed):.4f}", nw])

    # 3. Metrics CSV (both levels, mean+median aggregation)
    met_path = os.path.join(odir, activity, f"stepf_{tier}_metrics.csv")
    with open(met_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","type","level","agg","rho","p","mae","acc_exact","acc_w1","macro_f1","kappa"])
        for nm, res in results.items():
            for level, yt, yp, agg in [
                ("window", res["win_yt"], res["win_yp"], "n/a"),
                ("subject", res["subj_yt"], res["subj_yp"], "mean"),
                ("subject", res["subj_yt"], res["subj_yp_median"], "median"),
            ]:
                m = _metrics(yt, yp)
                w.writerow([nm, res["type"], level, agg] +
                           [f"{m.get(k,0):.4f}" for k in ["rho","p","mae","acc_exact","acc_w1","macro_f1","kappa"]])
    print(f"    Metrics: {met_path}")
    print(f"    Per-model CSVs: {len(results)} files")

# ============================================================
# Minimal viz (full viz in Step G)
# ============================================================
def quick_summary(results, activity, tier):
    print(f"\n-------------------- Activity: {activity}")
    print(f"    {'Model':<15s} {'Type':>4s} | {'Win ρ':>6s} | {'Sub ρ(mean)':>11s} {'Sub ρ(med)':>10s} {'MAE':>6s} {'Acc±1':>6s} {'κ':>6s}")
    print(f"    {'-'*75}")
    for nm, res in results.items():
        wm = _metrics(res["win_yt"], res["win_yp"])
        sm = _metrics(res["subj_yt"], res["subj_yp"])
        smed = _metrics(res["subj_yt"], res["subj_yp_median"])
        print(f"    {nm:<15s} {res['type']:>4s} | {wm['rho']:>6.3f} | "
              f"{sm['rho']:>11.3f} {smed['rho']:>10.3f} {sm['mae']:>6.3f} {sm['acc_w1']:>6.3f} {sm['kappa']:>6.3f}")

# ============================================================
# Tier A
# ============================================================
def run_tier_a(feat_names, odir):
    print(f"\n{'='*70}\n  Tier A: Window-level train -> Subject-level aggregate\n{'='*70}")
    for act in C.ACTIVITIES:
        all_rows = []
        path = C.features_window_csv(act)
        if not os.path.isfile(path): continue
        rows = load_csv(path)
        #for r in rows: r["is_right_side"] = 1.0 if r.get("side")=="Right" else 0.0
        all_rows.extend(rows)
        if not all_rows: return
        fn = feat_names # feat_names + ["is_right_side"]
        vals, cnts = np.unique([r["updrs_score"] for r in all_rows], return_counts=True)
        print(f"    Windows: {len(all_rows)}, UPDRS: {', '.join(f'{int(v)}:{c}' for v,c in zip(vals,cnts))}")
        results = loso_window_aggregate(all_rows, fn)
        quick_summary(results, act, "tierA")
        save_all(results, "tierA", odir, act)
    return results

# ============================================================
# Tier B
# ============================================================
def run_tier_b(feat_names, odir):
    print(f"\n{'='*70}\n  Tier B: Asymmetry\n{'='*70}")
    sides = {}
    for act in C.ACTIVITIES:
        path = C.features_subject_csv(act)
        if not os.path.isfile(path): continue
        sides[C.ACTIVITY_SIDES[act]] = {r["subject"]:r for r in load_csv(path)}
    if "Left" not in sides or "Right" not in sides: return
    common = sorted(set(sides["Left"])&set(sides["Right"]))
    if len(common) < 10: return
    afn = [f"abs_diff_{f}" for f in feat_names]; rows = []
    for s in common:
        L=sides["Left"][s]; R=sides["Right"][s]
        row = {"subject":s, "updrs_score":max(L["updrs_score"],R["updrs_score"])}
        for f in feat_names:
            vL=L.get(f,float("nan")); vR=R.get(f,float("nan"))
            row[f"abs_diff_{f}"] = abs(vL-vR) if np.isfinite(vL) and np.isfinite(vR) else float("nan")
        rows.append(row)
    # Simple LOSO (no windows for Tier B)
    X = np.array([[r.get(f,float("nan")) for f in afn] for r in rows])
    y = np.array([r["updrs_score"] for r in rows]); subj = np.array([r["subject"] for r in rows])
    zoo = get_models(); unique = np.unique(subj)
    results = {nm:{"win_yt":[],"win_yp":[],"win_subj":[],"subj_yt":[],"subj_yp":[],"subj_yp_median":[],
                    "subj_names":[],"params":[],"type":zoo[nm][2]} for nm in zoo}
    for held in unique:
        tr=subj!=held; te=subj==held
        sc=StandardScaler(); Xtr=sc.fit_transform(np.nan_to_num(X[tr])); Xte=sc.transform(np.nan_to_num(X[te]))
        for nm,(cls,grid,mt) in zoo.items():
            bp=_tune(cls,grid,Xtr,y[tr],subj[tr],mt)
            try:
                mdl=cls(**bp); mdl.fit(Xtr,y[tr].astype(int) if mt=="cls" else y[tr])
                pred=float(mdl.predict(Xte)[0])
                for k,v in [("subj_yt",float(y[te][0])),("subj_yp",pred),("subj_yp_median",pred),("subj_names",held)]:
                    results[nm][k].append(v)
                results[nm]["win_yt"].append(float(y[te][0])); results[nm]["win_yp"].append(pred)
                results[nm]["win_subj"].append(held); results[nm]["params"].append(bp)
            except:
                results[nm]["subj_yt"].append(float(y[te][0])); results[nm]["subj_yp"].append(float("nan"))
                results[nm]["subj_yp_median"].append(float("nan")); results[nm]["subj_names"].append(held)
    for nm in results:
        for k in ["win_yt","win_yp","subj_yt","subj_yp","subj_yp_median"]: results[nm][k]=np.array(results[nm][k])
        results[nm]["win_subj"]=np.array(results[nm].get("win_subj",[])); results[nm]["subj_names"]=np.array(results[nm]["subj_names"])
    quick_summary(results, "SYMMETRY", "tierB")
    save_all(results, "tierB", odir, "symmetry")

# ============================================================
# Tier C
# ============================================================
def run_tier_c(feat_names, odir):
    print(f"\n{'='*70}\n  Tier C: Paired Wilcoxon\n{'='*70}")
    sides = {}
    for act in C.ACTIVITIES:
        path = C.features_subject_csv(act)
        if not os.path.isfile(path): continue
        sides[C.ACTIVITY_SIDES[act]] = {r["subject"]:r for r in load_csv(path)}
    if "Left" not in sides or "Right" not in sides: return
    common = sorted(set(sides["Left"])&set(sides["Right"]))
    if len(common) < 10: return
    results = []
    for fn in feat_names:
        mv,lv = [],[]
        for s in common:
            L=sides["Left"][s]; R=sides["Right"][s]
            vL=L.get(fn,float("nan")); vR=R.get(fn,float("nan"))
            if not(np.isfinite(vL) and np.isfinite(vR)): continue
            if L["updrs_score"]>=R["updrs_score"]: mv.append(vL); lv.append(vR)
            else: mv.append(vR); lv.append(vL)
        if len(mv)<5: continue
        diff=np.array(mv)-np.array(lv)
        try: _,pv=wilcoxon(diff)
        except: pv=float("nan")
        results.append({"feature":fn,"p":float(pv),"effect":float(np.mean(diff)/(np.std(diff)+1e-12)),"n":len(mv)})
    results.sort(key=lambda x:x["p"])
    sig=[r for r in results if r["p"]<0.05]
    print(f"    Significant: {len(sig)}/{len(results)}")
    for r in results:
        star="***" if r["p"]<0.001 else "**" if r["p"]<0.01 else "*" if r["p"]<0.05 else ""
        print(f"    {r['feature']:<40s} p={r['p']:.4f} eff={r['effect']:.3f} {star}")
    path=os.path.join(odir,"stepf_tierC_paired.csv")
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["feature","p","effect","n"]); w.writeheader()
        for r in results: w.writerow({k:f"{v:.6f}" if isinstance(v,float) else v for k,v in r.items()})

# ============================================================
def load_predictions(tier, odir):
    """Load saved predictions pkl for post-hoc analysis (Step G)."""
    pkl_path = os.path.join(odir, f"stepf_{tier}_all_predictions.pkl")
    with open(pkl_path, "rb") as f: data = pickle.load(f)
    for nm in data:
        for k in ["win_yt","win_yp","subj_yt","subj_yp","subj_yp_median"]:
            if k in data[nm]: data[nm][k] = np.array(data[nm][k])
        for k in ["win_subj","subj_names","win_activity"]:
            if k in data[nm]: data[nm][k] = np.array(data[nm][k])
    return data

# ============================================================
if __name__ == "__main__":
    os.makedirs(f"{C.OUTPUT_DIR}/Stepf_analysis", exist_ok=True)
    smote_str = f"ON (k={SMOTE_K})" if USE_SMOTE and HAS_SMOTE else "OFF"
    print(f"\n{'#'*70}\n  Step F: LOSO (slim baselines + full top models)\n  SMOTE: {smote_str} | Workers: {N_WORKERS}\n{'#'*70}")
    fn = load_final_features(); print(f"  Features: {len(fn)}")
    run_tier_a(fn, f"{C.OUTPUT_DIR}/Stepf_analysis")
    run_tier_b(fn, f"{C.OUTPUT_DIR}/Stepf_analysis")
    run_tier_c(fn, f"{C.OUTPUT_DIR}/Stepf_analysis")
    print(f"\n  Done. Run stepg_posthoc.py for ensemble/grouping/ordinal/bootstrap analysis.")
