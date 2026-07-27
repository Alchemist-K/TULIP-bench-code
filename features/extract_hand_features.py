# stepd_feature_extraction.py
"""
Step D: Extract 43 features per 10-rep window + UPDRS-stratified visualizations.

Reads:  {OUTPUT_DIR}/stepb_reps_{activity}.pkl
Writes: {OUTPUT_DIR}/stepd_features_window_{activity}.csv
        {OUTPUT_DIR}/stepd_features_subject_{activity}.csv
        {OUTPUT_DIR}/stepd_feature_matrix_{activity}.png
        {OUTPUT_DIR}/stepd_feature_boxplots_{activity}.png
"""

import os
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import config as C

# MediaPipe indices
WRIST = 0; MIDDLE_TIP = 12; INDEX_MCP = 5; PINKY_MCP = 17
BODY_L_SHOULDER = 11; BODY_R_SHOULDER = 12
BODY_L_ELBOW = 13; BODY_R_ELBOW = 14
BODY_L_WRIST = 15; BODY_R_WRIST = 16
BODY_L_HIP = 23; BODY_R_HIP = 24; BODY_NOSE = 0

# Import Rep from stepb
from stepb_rep_detection import Rep


# ============================================================
# Data containers
# ============================================================
@dataclass
class Rep:
    idx: int; start: int; peak: int; end: int; amplitude: float
    duration_frames: int = field(init=False)
    def __post_init__(self): self.duration_frames = self.end - self.start + 1

@dataclass
class Window10:
    window_idx: int; reps: List[Rep]
    start_frame: int; end_frame: int; rep_idx_range: Tuple[int, int]

@dataclass
class ClosureResult:
    subject: str
    activity: str
    distance_score: np.ndarray      # (T,) in [0,1]
    angle_score: np.ndarray         # (T,) in [0,1]
    fingertip_wrist_norm: np.ndarray  # (T,)
    palm_span_mm: np.ndarray        # (T,)
    mcp_angles_rad: np.ndarray      # (T,4)
    pip_angles_rad: np.ndarray      # (T,4)
    open_ref: Dict[str, float] = field(default_factory=dict)
    movement_quality: Dict[str, float] = field(default_factory=dict)

# ============================================================
# Utilities
# ============================================================
def _interp_nans(x):
    x = np.array(x, dtype=np.float64); bad = ~np.isfinite(x)
    if not bad.any(): return x
    good = ~bad
    if good.sum() < 2: x[bad] = 0.0; return x
    idx = np.arange(len(x)); x[bad] = np.interp(idx[bad], idx[good], x[good]); return x

def _cv(x):
    m = np.nanmean(x); return float(np.nanstd(x) / (abs(m) + 1e-12))

def _slope(x):
    if len(x) < 2: return float("nan")
    v = np.isfinite(x)
    if v.sum() < 2: return float("nan")
    return float(np.polyfit(np.arange(len(x))[v], x[v], 1)[0])

def _sample_entropy(x, m=2, r_frac=0.2):
    x = _interp_nans(x); N = len(x)
    if N < m+2: return float("nan")
    r = r_frac * np.std(x)
    if r < 1e-12: return float("nan")
    def _ct(L):
        c = 0; tot = 0
        for i in range(N-L):
            for j in range(i+1, N-L):
                if np.max(np.abs(x[i:i+L] - x[j:j+L])) < r: c += 1
                tot += 1
        return c / (tot + 1e-12)
    A = _ct(m+1); B = _ct(m)
    if B < 1e-12 or A < 1e-12: return float("nan")
    return float(-np.log(A / B))

def _sparc(speed, fps, fc=10.0, amp_th=0.05):
    speed = _interp_nans(speed); N = len(speed)
    if N < 8: return float("nan")
    nfft = int(2**np.ceil(np.log2(N)+4))
    freq = np.arange(0, fps, fps/nfft); Mf = np.abs(np.fft.fft(speed, nfft)); Mf = Mf/(np.max(Mf)+1e-12)
    fc_idx = np.searchsorted(freq, fc); above = np.where(Mf[:fc_idx] >= amp_th)[0]
    wc = max(2, min(above[-1]+1 if len(above) else 1, fc_idx))
    dw = freq[1] - freq[0]
    return float(-np.sum(np.sqrt(dw**2 + np.diff(Mf[:wc])**2)))


# ============================================================
# Feature categories
# ============================================================
def _amplitude(closure, ftw, mcp, palm_sp, reps, hand, fps):
    ra = np.array([r.amplitude for r in reps])
    ro = []; rc = []; rm = []
    for r in reps:
        seg = ftw[r.start:r.end+1]
        ro.append(float(np.nanmax(seg)) if len(seg) else float("nan"))
        rc.append(float(np.nanmin(seg)) if len(seg) else float("nan"))
        seg_m = mcp[r.start:r.end+1]
        if len(seg_m): mm = seg_m.mean(axis=1); rm.append(float(np.nanmax(mm)-np.nanmin(mm)))
        else: rm.append(float("nan"))
    return {"amp_mean_closure": float(np.nanmean(ra)), "amp_std_closure": float(np.nanstd(ra)),
            "amp_cv_closure": _cv(ra), "amp_max_open_ftw": float(np.nanmean(ro)),
            "amp_min_closed_ftw": float(np.nanmean(rc)), "amp_decrement_slope": _slope(ra),
            "amp_mcp_range_rad_mean": float(np.nanmean(rm))}

def _speed(closure, hand, reps, fps):
    peaks = np.array([r.peak for r in reps])
    periods = np.diff(peaks)/fps if len(peaks) > 1 else np.array([])
    freq = float(1/np.mean(periods)) if len(periods) and np.mean(periods) > 0 else float("nan")
    dc = np.gradient(closure) * fps
    cv_c = []; cv_o = []
    for r in reps:
        seg = dc[r.start:r.end+1]
        cv_c.append(float(np.nanmean(seg[seg>0])) if (seg>0).any() else 0)
        cv_o.append(float(np.nanmean(np.abs(seg[seg<0]))) if (seg<0).any() else 0)
    mc = float(np.nanmean(cv_c)); mo = float(np.nanmean(cv_o))
    tv = np.linalg.norm(np.gradient(hand[:, MIDDLE_TIP], axis=0)*fps, axis=1)
    rps = []; rpp = []
    for r in reps:
        seg = tv[r.start:r.end+1]
        rps.append(float(np.nanmean(seg)) if len(seg) else float("nan"))
        rpp.append(float(np.nanmax(seg)) if len(seg) else float("nan"))
    return {"speed_rep_freq_hz": freq, "speed_closure_vel_close": mc, "speed_closure_vel_open": mo,
            "speed_vel_ratio_open_close": float(mo/(mc+1e-12)),
            "speed_fingertip_mean": float(np.nanmean(rps)), "speed_fingertip_peak": float(np.nanmean(rpp)),
            "speed_decrement_slope": _slope(np.array(rps))}

def _rhythm(closure, reps, fps):
    peaks = np.array([r.peak for r in reps])
    if len(peaks) < 2:
        return {k: float("nan") if k != "rhythm_hesitation_count" and k != "rhythm_arrest_count" else 0
                for k in ["rhythm_period_mean_s","rhythm_period_cv","rhythm_hesitation_count",
                           "rhythm_arrest_count","rhythm_entropy","rhythm_trend_slope"]}
    periods = np.diff(peaks)/fps; med = float(np.nanmedian(periods))
    hes = int(np.sum(periods > 2*med))
    dc = np.gradient(closure)*fps; arr = 0; maf = max(3, int(0.05*fps))
    for r in reps:
        seg = np.abs(dc[r.start:r.end+1])
        if len(seg) < maf: continue
        inner = seg[2:-2] if len(seg) > 4 else seg
        th = 0.05 * np.nanmax(seg); run = 0
        for s in (inner < th):
            if s: run += 1
            else:
                if run >= maf: arr += 1
                run = 0
    return {"rhythm_period_mean_s": float(np.nanmean(periods)), "rhythm_period_cv": _cv(periods),
            "rhythm_hesitation_count": hes, "rhythm_arrest_count": arr,
            "rhythm_entropy": _sample_entropy(periods), "rhythm_trend_slope": _slope(periods)}

def _fatigue(closure, hand, reps, fps):
    n = len(reps)
    keys = ["fatigue_amp_first5_last5","fatigue_speed_first5_last5","fatigue_period_first5_last5",
            "fatigue_decrement_onset_rep","fatigue_recovery_count",
            "fatigue_amp_slope_windows","fatigue_speed_slope_windows","fatigue_index"]
    if n < 4: return {k: float("nan") for k in keys}
    ra = np.array([r.amplitude for r in reps]); peaks = np.array([r.peak for r in reps])
    periods = np.diff(peaks)/fps if len(peaks) > 1 else np.array([float("nan")])
    tv = np.linalg.norm(np.gradient(hand[:, MIDDLE_TIP], axis=0)*fps, axis=1)
    rs = np.array([float(np.nanmean(tv[r.start:r.end+1])) for r in reps])
    mid = n // 2
    af = np.nanmean(ra[:mid]); al = np.nanmean(ra[mid:])
    sf = np.nanmean(rs[:mid]); sl = np.nanmean(rs[mid:])
    pf = np.nanmean(periods[:max(1,mid-1)]); pl = np.nanmean(periods[max(1,mid-1):])
    onset = float("nan")
    if ra[0] > 1e-6:
        below = np.where(ra < 0.8*ra[0])[0]
        if len(below): onset = float(below[0])
    rec = sum(1 for i in range(2, len(ra)) if ra[i] > ra[i-1] and ra[i-1] < ra[i-2])
    return {"fatigue_amp_first5_last5": float(af/(al+1e-12)), "fatigue_speed_first5_last5": float(sf/(sl+1e-12)),
            "fatigue_period_first5_last5": float(pf/(pl+1e-12)), "fatigue_decrement_onset_rep": onset,
            "fatigue_recovery_count": float(rec), "fatigue_amp_slope_windows": _slope(ra),
            "fatigue_speed_slope_windows": _slope(rs), "fatigue_index": float(al/(af+1e-12))}

def _smoothness(closure, hand, reps, fps):
    tv = np.linalg.norm(np.gradient(hand[:, MIDDLE_TIP], axis=0)*fps, axis=1)
    sparc = _sparc(tv, fps)
    tip = hand[:, MIDDLE_TIP]
    if tip.shape[0] > 3:
        jerk = np.diff(tip, n=3, axis=0)*(fps**3); jm = np.linalg.norm(jerk, axis=1)
        mj = float(np.nanmean(jm)); dur = tip.shape[0]/fps
        pl = np.nansum(np.linalg.norm(np.diff(tip, axis=0), axis=1))
        ldlj = float(-np.log(dur**5/(pl**2+1e-12)*np.nanmean(jm**2)+1e-12)) if pl > 1e-6 else float("nan")
    else: mj = ldlj = float("nan")
    dc = np.gradient(closure)*fps
    duty = [float(np.mean(dc[r.start:r.end+1]>0)) for r in reps if r.end-r.start > 0]
    zc = [float(np.sum(np.diff(np.sign(dc[r.start:r.end+1]))!=0)) for r in reps if r.end-r.start > 2]
    return {"smooth_sparc": sparc, "smooth_log_dim_jerk": ldlj, "smooth_mean_jerk": mj,
            "smooth_duty_cycle": float(np.nanmean(duty)) if duty else float("nan"),
            "smooth_vel_zc_per_rep": float(np.nanmean(zc)) if zc else float("nan")}

def _spectral(closure, fps):
    sig = _interp_nans(closure - np.nanmean(closure)); N = len(sig)
    keys = ["spec_dominant_freq_hz","spec_centroid_hz","spec_entropy","spec_power_ratio_low_high","spec_edge95_hz"]
    if N < 16: return {k: float("nan") for k in keys}
    nfft = int(2**np.ceil(np.log2(N))); freqs = np.fft.rfftfreq(nfft, d=1/fps)
    psd = np.abs(np.fft.rfft(sig, nfft))**2
    m = freqs <= 10; fm = freqs[m]; pm = psd[m]; pt = pm.sum()+1e-12
    dom = float(fm[np.argmax(pm[1:])+1])
    cent = float(np.sum(fm*pm)/pt)
    pn = pm/pt; pn = pn[pn>1e-20]; ent = float(-np.sum(pn*np.log(pn)))
    pl = pm[fm<=2].sum(); ph = pm[(fm>2)&(fm<=10)].sum()+1e-12
    cum = np.cumsum(pm)/pt; e95 = float(fm[min(np.searchsorted(cum, 0.95), len(fm)-1)])
    return {"spec_dominant_freq_hz": dom, "spec_centroid_hz": cent, "spec_entropy": ent,
            "spec_power_ratio_low_high": float(pl/ph), "spec_edge95_hz": e95}

def _body(body, side, fps):
    keys = ["body_shoulder_elev_range","body_trunk_sway_rms","body_elbow_angle_range",
            "body_wrist_drift","body_head_movement"]
    T = body.shape[0]
    if T < 2: return {k: float("nan") for k in keys}
    si, ei, wi = (BODY_L_SHOULDER, BODY_L_ELBOW, BODY_L_WRIST) if side=="Left" else (BODY_R_SHOULDER, BODY_R_ELBOW, BODY_R_WRIST)
    sho_r = float(np.nanmax(body[:,si,2]) - np.nanmin(body[:,si,2]))
    ms = 0.5*(body[:,BODY_L_SHOULDER]+body[:,BODY_R_SHOULDER]); mh = 0.5*(body[:,BODY_L_HIP]+body[:,BODY_R_HIP])
    trunk = 0.5*(ms+mh); tc = trunk - trunk[0:1]; tsway = float(np.sqrt(np.nanmean(np.sum(tc**2, axis=1))))
    u = body[:,si]-body[:,ei]; l = body[:,wi]-body[:,ei]
    dot = np.einsum("ij,ij->i",u,l); ea = np.arccos(np.clip(dot/(np.linalg.norm(u,axis=1)*np.linalg.norm(l,axis=1)+1e-12),-1,1))
    er = float(np.nanmax(ea)-np.nanmin(ea))
    wd = float(np.nansum(np.linalg.norm(np.diff(body[:,wi], axis=0), axis=1)))
    head = body[:,BODY_NOSE]; hd = np.linalg.norm(head-head[0:1], axis=1)
    hr = float(np.nanmax(hd)-np.nanmin(hd))
    return {"body_shoulder_elev_range": sho_r, "body_trunk_sway_rms": tsway,
            "body_elbow_angle_range": er, "body_wrist_drift": wd, "body_head_movement": hr}


# ============================================================
# Feature names (43 total)
# ============================================================
FEATURE_NAMES = [
    "amp_mean_closure","amp_std_closure","amp_cv_closure","amp_max_open_ftw",
    "amp_min_closed_ftw","amp_decrement_slope","amp_mcp_range_rad_mean",
    "speed_rep_freq_hz","speed_closure_vel_close","speed_closure_vel_open",
    "speed_vel_ratio_open_close","speed_fingertip_mean","speed_fingertip_peak","speed_decrement_slope",
    "rhythm_period_mean_s","rhythm_period_cv","rhythm_hesitation_count",
    "rhythm_arrest_count","rhythm_entropy","rhythm_trend_slope",
    "fatigue_amp_first5_last5","fatigue_speed_first5_last5","fatigue_period_first5_last5",
    "fatigue_decrement_onset_rep","fatigue_recovery_count",
    "fatigue_amp_slope_windows","fatigue_speed_slope_windows","fatigue_index",
    "smooth_sparc","smooth_log_dim_jerk","smooth_mean_jerk","smooth_duty_cycle","smooth_vel_zc_per_rep",
    "spec_dominant_freq_hz","spec_centroid_hz","spec_entropy","spec_power_ratio_low_high","spec_edge95_hz",
    "body_shoulder_elev_range","body_trunk_sway_rms","body_elbow_angle_range","body_wrist_drift","body_head_movement",
]
assert len(FEATURE_NAMES) == 43


# ============================================================
# Window-level extraction
# ============================================================
def extract_window(closure, ftw, mcp, palm_sp, hand, body, reps, side, fps):
    f = {}
    f.update(_amplitude(closure, ftw, mcp, palm_sp, reps, hand, fps))
    f.update(_speed(closure, hand, reps, fps))
    f.update(_rhythm(closure, reps, fps))
    f.update(_fatigue(closure, hand, reps, fps))
    f.update(_smoothness(closure, hand, reps, fps))
    f.update(_spectral(closure, fps))
    f.update(_body(body, side, fps))
    return f


def extract_all(results, fps, level="window"):
    rows = []
    for r in results:
        cr = r["closure"]; s, e = r["clip_frames"]
        windows = r.get("windows_10", [])
        if not windows: continue
        dc = cr.distance_score[s:e]; ftw = cr.fingertip_wrist_norm[s:e]
        mcp = cr.mcp_angles_rad[s:e]; ps = cr.palm_span_mm[s:e]
        hc = r["hand"]; bc = r["body"]

        win_rows = []
        for win in windows:
            ws, we = win.start_frame, min(win.end_frame+1, len(dc))
            if we - ws < 10: continue
            # Offset reps to window start
            wreps = [Rep(rr.idx, rr.start-ws, rr.peak-ws, rr.end-ws, rr.amplitude) for rr in win.reps]
            feats = extract_window(dc[ws:we], ftw[ws:we], mcp[ws:we], ps[ws:we],
                                   hc[ws:we], bc[ws:we], wreps, r["side"], fps)
            row = {"subject": r["subject"], "activity": r["activity"], "side": r["side"],
                   "updrs_score": r["updrs_score"], "window_idx": win.window_idx}
            row.update(feats); win_rows.append(row)

        if level == "window":
            rows.extend(win_rows)
        elif level == "subject" and win_rows:
            avg = {"subject": r["subject"], "activity": r["activity"], "side": r["side"],
                   "updrs_score": r["updrs_score"], "n_windows": len(win_rows)}
            for fn in FEATURE_NAMES:
                vals = [w[fn] for w in win_rows if np.isfinite(w.get(fn, float("nan")))]
                avg[fn] = float(np.mean(vals)) if vals else float("nan")
            rows.append(avg)
    print(f"  Extracted: {len(rows)} rows ({level}-level)")
    return rows


def save_csv(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else str(v)) for k,v in row.items()})
    print(f"  Saved: {path}")


# ============================================================
# Visualizations
# ============================================================
def plot_feature_matrix(rows, activity, output_dir):
    """
    Heatmap: rows = subjects (sorted by UPDRS), columns = features.
    Z-scored per feature for visual comparability.
    """
    if not rows: return
    # Sort by UPDRS then subject
    rows_sorted = sorted(rows, key=lambda r: (r["updrs_score"], r["subject"]))
    subjects = [r["subject"] for r in rows_sorted]
    updrs = [r["updrs_score"] for r in rows_sorted]

    mat = np.array([[r.get(fn, float("nan")) for fn in FEATURE_NAMES] for r in rows_sorted])

    # Z-score per feature (column)
    means = np.nanmean(mat, axis=0); stds = np.nanstd(mat, axis=0) + 1e-12
    z = (mat - means) / stds
    z = np.clip(z, -3, 3)  # clip outliers for viz

    fig, ax = plt.subplots(figsize=(20, max(6, len(subjects)*0.35)), dpi=120)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")

    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"{s} (U={u})" for s, u in zip(subjects, updrs)], fontsize=7)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=6)
    ax.set_title(f"Feature Matrix (z-scored) — {activity}", fontsize=11)

    # Color bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")

    # UPDRS group separators
    prev_u = updrs[0]
    for i, u in enumerate(updrs):
        if u != prev_u:
            ax.axhline(i - 0.5, color="black", lw=1.5)
            prev_u = u

    plt.tight_layout()
    path = os.path.join(output_dir, f"stepd_feature_matrix_{activity}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Matrix plot: {path}")


def plot_feature_boxplots(rows, activity, output_dir, top_n=15):
    """
    Box plots of top features grouped by UPDRS score.
    Shows the top_n features with highest Spearman correlation to UPDRS.
    """
    if not rows or len(rows) < 5: return

    from scipy.stats import spearmanr

    updrs = np.array([r["updrs_score"] for r in rows])
    correlations = {}
    for fn in FEATURE_NAMES:
        vals = np.array([r.get(fn, float("nan")) for r in rows])
        valid = np.isfinite(vals)
        if valid.sum() < 5: continue
        rho, pval = spearmanr(updrs[valid], vals[valid])
        correlations[fn] = (abs(rho), rho, pval)

    # Top features by absolute correlation
    top = sorted(correlations.items(), key=lambda x: x[1][0], reverse=True)[:top_n]

    n_cols = 3; n_rows_grid = (len(top) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(15, n_rows_grid*3.5), dpi=120)
    axes = axes.flatten()

    updrs_groups = sorted(set(updrs))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(updrs_groups)))

    for idx, (fn, (abs_rho, rho, pval)) in enumerate(top):
        ax = axes[idx]
        data_by_group = []
        for u in updrs_groups:
            vals = [r[fn] for r in rows if r["updrs_score"] == u and np.isfinite(r.get(fn, float("nan")))]
            data_by_group.append(vals)

        bp = ax.boxplot(data_by_group, labels=[str(int(u)) for u in updrs_groups],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)

        ax.set_title(f"{fn}\nρ={rho:.3f}, p={pval:.4f}", fontsize=8)
        ax.set_xlabel("UPDRS", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide empty axes
    for idx in range(len(top), len(axes)): axes[idx].set_visible(False)

    fig.suptitle(f"Top {len(top)} Features by |Spearman ρ| — {activity}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"stepd_feature_boxplots_{activity}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Boxplots: {path}")


# ============================================================
if __name__ == "__main__":
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    for activity in C.ACTIVITIES:
        pkl_path = C.reps_pkl(activity)
        print(f"\n{'#'*60}\n  Step D: {activity}\n{'#'*60}")
        print(f"  Loading: {pkl_path}")
        with open(pkl_path, "rb") as f: results = pickle.load(f)

        # Window-level
        win_rows = extract_all(results, C.FPS, level="window")
        save_csv(win_rows, C.features_window_csv(activity))

        # Subject-level
        subj_rows = extract_all(results, C.FPS, level="subject")
        save_csv(subj_rows, C.features_subject_csv(activity))

        # Visualizations (on subject-level for cleaner plots)
        plot_feature_matrix(subj_rows, activity, C.OUTPUT_DIR)
        plot_feature_boxplots(subj_rows, activity, C.OUTPUT_DIR)
