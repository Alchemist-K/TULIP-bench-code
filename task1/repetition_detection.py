# stepb_rep_detection.py
"""
Step B: Detect repetitions from closure scores.

Reads:  {OUTPUT_DIR}/stepa_closure_{activity}.pkl
Writes: {OUTPUT_DIR}/stepb_reps_{activity}.pkl
        {OUTPUT_DIR}/stepb_report_{activity}.csv
"""

import os
import csv
import pickle
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np

import config as C

logger = logging.getLogger(__name__)


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
# Sensitivity tiers
# ============================================================
TIERS = [
    {"name":"default",   "hi_pct":60, "lo_pct":35, "min_prom":0.10, "min_peak_ms":150, "min_dur_ms":100, "min_gap_ms":120},
    {"name":"sensitive",  "hi_pct":55, "lo_pct":30, "min_prom":0.05, "min_peak_ms":120, "min_dur_ms":90,  "min_gap_ms":110},
    {"name":"ultra",      "hi_pct":50, "lo_pct":25, "min_prom":0.035,"min_peak_ms":100, "min_dur_ms":80,  "min_gap_ms":100},
    {"name":"maximum",    "hi_pct":45, "lo_pct":20, "min_prom":0.02, "min_peak_ms":80,  "min_dur_ms":60,  "min_gap_ms":80},
]


# ============================================================
# Core detection
# ============================================================
def _detect_single_tier(score, fps, hi_pct, lo_pct, min_prom, min_peak_ms, min_dur_ms, min_gap_ms, **_):
    T = len(score)
    if T < 10: return []
    th_hi = np.percentile(score, hi_pct)
    th_lo = np.percentile(score, lo_pct)
    min_dist = max(1, int(min_peak_ms/1000*fps))
    min_dur = max(1, int(min_dur_ms/1000*fps))
    min_gap = max(1, int(min_gap_ms/1000*fps))

    # Find local maxima
    cands = []
    i = 1
    while i < T-1:
        if score[i-1] < score[i] and score[i] >= score[i+1] and score[i] >= th_lo:
            cands.append(i); i += min_dist; continue
        i += 1

    reps = []; last_end = -10**9; ri = 0
    for pk in cands:
        L = pk
        while L > 0 and score[L-1] > th_lo: L -= 1
        R = pk
        while R < T-1 and score[R+1] > th_lo: R += 1
        if np.max(score[L:R+1]) < th_hi: continue
        if R - L + 1 < min_dur: continue
        if L - last_end < min_gap and reps:
            prev = reps[-1]; mL = prev.start; mR = max(R, prev.end)
            mpk = prev.peak if score[prev.peak] >= score[pk] else pk
            base = 0.5*(score[mL]+score[mR]); amp = score[mpk]-base
            if amp >= min_prom: reps[-1] = Rep(prev.idx, mL, mpk, mR, amp); last_end = mR
            continue
        base = 0.5*(score[L]+score[R]); amp = score[pk]-base
        if amp < min_prom: continue
        reps.append(Rep(ri, L, pk, R, amp)); last_end = R; ri += 1
    return reps


def detect_reps(score, fps):
    """Multi-tier detection with velocity fallback."""
    for tier in TIERS:
        reps = _detect_single_tier(score, fps, **tier)
        if len(reps) >= C.MIN_REPS_TARGET: return reps, tier["name"]
        vel = np.abs(np.gradient(score) * fps)
        reps_v = _detect_single_tier(vel, fps, **tier)
        if len(reps_v) >= C.MIN_REPS_TARGET: return reps_v, f"{tier['name']}+velocity"
    # Fallback: best from last tier
    best = _detect_single_tier(score, fps, **TIERS[-1])
    best_v = _detect_single_tier(np.abs(np.gradient(score)*fps), fps, **TIERS[-1])
    if len(best_v) > len(best): return best_v, f"{TIERS[-1]['name']}+velocity (sub-target)"
    return best, f"{TIERS[-1]['name']} (sub-target)"


def build_windows_10(reps, stride=1):
    if len(reps) < 10: return []
    reps_s = sorted(reps, key=lambda r: r.idx)
    windows = []
    for i in range(0, len(reps_s)-9, stride):
        g = reps_s[i:i+10]
        windows.append(Window10(len(windows), g, g[0].start, g[-1].end, (g[0].idx, g[-1].idx)))
    return windows


def rep_summary(reps, fps):
    if not reps: return {"n_reps": 0}
    amps = np.array([r.amplitude for r in reps])
    peaks = np.array([r.peak for r in reps])
    periods = np.diff(peaks)/fps if len(peaks) > 1 else np.array([])
    out = {"n_reps": len(reps), "amp_mean": float(np.mean(amps)), "amp_cv": float(np.std(amps)/(np.mean(amps)+1e-12))}
    if len(periods) > 0:
        out["period_mean_s"] = float(np.mean(periods))
        out["period_cv"] = float(np.std(periods)/(np.mean(periods)+1e-12))
        out["freq_hz"] = float(1/np.mean(periods))
    else:
        out["period_mean_s"] = out["period_cv"] = out["freq_hz"] = float("nan")
    if len(amps) >= 3: out["amp_slope"] = float(np.polyfit(np.arange(len(amps)), amps, 1)[0])
    else: out["amp_slope"] = float("nan")
    return out


# ============================================================
# Batch
# ============================================================
def process_activity(activity):
    pkl_path = C.closure_pkl(activity)
    print(f"\n  Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    # with open(f"Step0_Slicing_sequences/closure_outputs/closure_{activity}.pkl", "rb") as f:
    #     results = pickle.load(f)

    for r in results:
        cr = r["closure"]; s, e = r["clip_frames"]
        score_clip = cr.distance_score[s:e]
        reps, tier = detect_reps(score_clip, C.FPS)
        windows = build_windows_10(reps, stride=C.WINDOW_STRIDE)
        r["reps"] = reps
        r["windows_10"] = windows
        r["detection_tier"] = tier
        r["rep_summary"] = rep_summary(reps, C.FPS)
    return results


def generate_report(results, activity):
    rows = []
    for r in results:
        summ = r["rep_summary"]; mq = r["closure"].movement_quality
        issues = []
        if summ["n_reps"] < 10: issues.append(f"Only {summ['n_reps']} reps")
        pcv = summ.get("period_cv", float("nan"))
        if np.isfinite(pcv) and pcv > 0.5: issues.append(f"High period CV={pcv:.2f}")
        if "sub-target" in r["detection_tier"]: issues.append("Sub-target")
        elif r["detection_tier"] != "default": issues.append(f"Tier: {r['detection_tier']}")

        rows.append({
            "subject": r["subject"], "updrs": r["updrs_score"],
            "n_reps": summ["n_reps"], "n_windows": len(r["windows_10"]),
            "tier": r["detection_tier"], "movement_class": mq.get("movement_class","?"),
            "amp_mean": f"{summ.get('amp_mean',0):.3f}",
            "amp_cv": f"{summ.get('amp_cv',0):.3f}",
            "period_s": f"{summ.get('period_mean_s',0):.3f}",
            "period_cv": f"{summ.get('period_cv',0):.3f}",
            "freq_hz": f"{summ.get('freq_hz',0):.2f}",
            "issues": " | ".join(issues),
        })

    # Console
    print(f"\n{'='*100}\n  Step B: Rep Detection — {activity}\n{'='*100}")
    print(f"  {'Subject':<14s} {'UPDRS':>5s} {'Reps':>5s} {'Win':>4s} {'Tier':<18s} {'Class':>8s} {'AmpM':>6s} {'PerS':>6s} {'Hz':>5s}")
    print(f"  {'-'*96}")
    for row in rows:
        f = "*" if row["issues"] else " "
        print(f"{f} {row['subject']:<14s} {row['updrs']:>5d} {row['n_reps']:>5d} {row['n_windows']:>4d} "
              f"{row['tier']:<18s} {row['movement_class']:>8s} {row['amp_mean']:>6s} {row['period_s']:>6s} {row['freq_hz']:>5s}")

    # Tier usage
    print(f"\n  Tier usage: {dict(Counter(r['tier'] for r in rows))}")
    # UPDRS x reps
    for u in sorted(set(r["updrs"] for r in rows)):
        sub = [r for r in rows if r["updrs"] == u]
        rc = [r["n_reps"] for r in sub]
        print(f"  UPDRS {u}: n={len(sub)}, reps mean={np.mean(rc):.1f} [{min(rc)}-{max(rc)}]")
    print(f"{'='*100}\n")

    # CSV
    path = os.path.join(C.OUTPUT_DIR, f"stepb_report_{activity}.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  Report: {path}")


# ============================================================
if __name__ == "__main__":
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    for activity in C.ACTIVITIES:
        results = process_activity(activity)
        generate_report(results, activity)

        out = C.reps_pkl(activity)
        with open(out, "wb") as f: pickle.dump(results, f, protocol=4)
        print(f"  Saved {len(results)} -> {out}")
