# stepa_closure_scores.py
"""
Step A: Compute closure scores for all subjects.

Reads:  canonical 3D poses from POSE3D_ROOT
Writes: {OUTPUT_DIR}/stepa_closure_{activity}.pkl
        {OUTPUT_DIR}/stepa_report_{activity}.csv

Each pkl contains a list of dicts:
    subject, activity, side, updrs_score,
    closure (ClosureResult dataclass),
    hand (clipped np.ndarray), body (clipped), subhand (clipped or None),
    clip_frames (start, end), full_n_frames
"""

import os
import csv
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

import config as C

logger = logging.getLogger(__name__)


# ============================================================
# MediaPipe indices
# ============================================================
WRIST = 0; THUMB_TIP = 4; INDEX_TIP = 8; MIDDLE_TIP = 12; RING_TIP = 16; PINKY_TIP = 20
INDEX_MCP = 5; MIDDLE_MCP = 9; RING_MCP = 13; PINKY_MCP = 17
INDEX_PIP = 6; MIDDLE_PIP = 10; RING_PIP = 14; PINKY_PIP = 18
ALL_FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

MCP_JOINT_TRIPLETS = [
    (WRIST, INDEX_MCP, INDEX_PIP), (WRIST, MIDDLE_MCP, MIDDLE_PIP),
    (WRIST, RING_MCP, RING_PIP), (WRIST, PINKY_MCP, PINKY_PIP),
]
PIP_JOINT_TRIPLETS = [
    (INDEX_MCP, INDEX_PIP, INDEX_TIP), (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP), (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]


# ============================================================
# Data containers
# ============================================================
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
# Numeric utilities
# ============================================================
def _interp_nans(x):
    x = np.array(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if not bad.any(): return x
    good = ~bad
    if good.sum() < 2: x[bad] = 0.0; return x
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[good], x[good])
    return x

def _smooth(x, win):
    x = _interp_nans(x)
    win = max(5, win | 1)
    if len(x) < win: return x
    return np.convolve(x, np.ones(win)/win, mode="same")

def _vec_angle(u, v):
    dot = np.einsum("ij,ij->i", u, v)
    return np.arccos(np.clip(dot / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12), -1, 1))

def _normalize_01(x, lo_pct=5.0, hi_pct=95.0):
    lo, hi = np.nanpercentile(x, [lo_pct, hi_pct])
    if hi - lo < 1e-8: return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


# ============================================================
# Hand geometry
# ============================================================
def palm_span(hand): return np.linalg.norm(hand[:, INDEX_MCP] - hand[:, PINKY_MCP], axis=-1) + 1e-8

def fingertip_wrist_distances_normalized(hand):
    wrist = hand[:, WRIST]; span = palm_span(hand)
    dists = np.stack([np.linalg.norm(hand[:, t] - wrist, axis=-1) for t in ALL_FINGERTIPS], axis=1)
    return dists.mean(axis=1) / span

def joint_angles(hand, triplets):
    out = np.empty((hand.shape[0], len(triplets)))
    for i, (prox, jnt, dist) in enumerate(triplets):
        out[:, i] = _vec_angle(hand[:, prox] - hand[:, jnt], hand[:, dist] - hand[:, jnt])
    return out


# ============================================================
# Closure scores
# ============================================================
def closure_score_distance(hand, fps=80.0, smooth_ms=100):
    raw = fingertip_wrist_distances_normalized(hand)
    z = 1.0 - _normalize_01(raw)
    return _smooth(z, win=max(5, int(smooth_ms / 1000.0 * fps)))

def closure_score_angle(hand, fps=80.0, smooth_ms=100):
    mcp = joint_angles(hand, MCP_JOINT_TRIPLETS)
    pip = joint_angles(hand, PIP_JOINT_TRIPLETS)
    z = 1.0 - _normalize_01(np.concatenate([mcp, pip], axis=1).mean(axis=1))
    return _smooth(z, win=max(5, int(smooth_ms / 1000.0 * fps)))

def open_hand_reference(hand, fps=80.0, window_sec=3.0):
    n = min(int(window_sec * fps), hand.shape[0]); h = hand[:n]
    ftw = fingertip_wrist_distances_normalized(h); ps = palm_span(h)
    mcp = joint_angles(h, MCP_JOINT_TRIPLETS); pip = joint_angles(h, PIP_JOINT_TRIPLETS)
    return {
        "open_fingertip_wrist_norm": float(np.nanmedian(ftw)),
        "open_palm_span_mm": float(np.nanmedian(ps)),
        "open_mcp_angle_mean_rad": float(np.nanmedian(mcp.mean(axis=1))),
        "open_pip_angle_mean_rad": float(np.nanmedian(pip.mean(axis=1))),
    }

def assess_movement_quality(ftw_norm, mcp_angles, fps):
    lo, hi = np.nanpercentile(ftw_norm, [5, 95])
    raw_range = float(hi - lo)
    mcp_lo, mcp_hi = np.nanpercentile(mcp_angles.mean(axis=1), [5, 95])
    mcp_range_deg = float(np.degrees(mcp_hi - mcp_lo))
    snr = float("nan")
    T = len(ftw_norm)
    if T > 16:
        sig = np.nan_to_num(ftw_norm - np.nanmean(ftw_norm))
        fft_mag = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(T, d=1.0/fps)
        power_low = np.sum(fft_mag[freqs <= 5.0] ** 2)
        power_high = np.sum(fft_mag[freqs > 5.0] ** 2) + 1e-12
        snr = float(power_low / power_high)
    if raw_range > 0.6 and mcp_range_deg > 15: mc = "full"
    elif raw_range > 0.15 or mcp_range_deg > 5: mc = "reduced"
    else: mc = "minimal"
    return {"raw_range_ratio": raw_range, "mcp_range_deg": mcp_range_deg, "snr_distance": snr, "movement_class": mc}

def build_closure_scores(hand, fps=80.0, smooth_ms=100, cal_sec=3.0):
    ref = open_hand_reference(hand, fps, cal_sec)
    ftw = fingertip_wrist_distances_normalized(hand)
    mcp = joint_angles(hand, MCP_JOINT_TRIPLETS)
    pip = joint_angles(hand, PIP_JOINT_TRIPLETS)
    mq = assess_movement_quality(ftw, mcp, fps)
    return ClosureResult(
        subject="", activity="",
        distance_score=closure_score_distance(hand, fps, smooth_ms),
        angle_score=closure_score_angle(hand, fps, smooth_ms),
        fingertip_wrist_norm=ftw, palm_span_mm=palm_span(hand),
        mcp_angles_rad=mcp, pip_angles_rad=pip,
        open_ref=ref, movement_quality=mq,
    )


# ============================================================
# I/O
# ============================================================
def _load_pkl(path):
    with open(path, "rb") as f: arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.float64)
    assert arr.ndim == 3 and arr.shape[2] == 3, f"Bad shape {arr.shape} from {path}"
    return arr

def discover_subjects(activity):
    side = C.ACTIVITY_SIDES[activity]
    hand_key = "lefthand" if side == "Left" else "righthand"
    d = os.path.join(C.POSE3D_ROOT, activity)
    subjects = []
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if not os.path.isdir(full): continue
        ok = (os.path.isfile(os.path.join(full, "pose3d_canonical_body.pkl")) and
              os.path.isfile(os.path.join(full, f"pose3d_canonical_{hand_key}.pkl")))
        if ok: subjects.append(name)
        else: logger.warning(f"Skipping {name}: missing pkl")
    return subjects

def load_subject(activity, subject, fps=80.0):
    side = C.ACTIVITY_SIDES[activity]
    mh_key = "lefthand" if side == "Left" else "righthand"
    sh_key = "righthand" if side == "Left" else "lefthand"
    base = os.path.join(C.POSE3D_ROOT, activity, subject)
    body = _load_pkl(os.path.join(base, "pose3d_canonical_body.pkl"))
    mainhand = _load_pkl(os.path.join(base, f"pose3d_canonical_{mh_key}.pkl"))
    sub_path = os.path.join(base, f"pose3d_canonical_{sh_key}.pkl")
    subhand = _load_pkl(sub_path) if os.path.isfile(sub_path) else None
    T = min(body.shape[0], mainhand.shape[0])
    body, mainhand = body[:T], mainhand[:T]
    if subhand is not None: subhand = subhand[:min(T, subhand.shape[0])]
    assert body.shape[1] == 33 and mainhand.shape[1] == 21
    return body, mainhand, subhand

def load_labels():
    labels = {}
    with open(C.CSV_PATH, "r") as f:
        for row in csv.DictReader(f):
            subj = row["subject_name"].strip()
            entry = {}
            for col in ["making_a_fist_Left", "making_a_fist_Right"]:
                val = row.get(col, "").strip()
                entry[col] = int(float(val)) if val and val.lower() != "nan" else None
            labels[subj] = entry
    return labels


# ============================================================
# Sanity checks
# ============================================================
def check_hand(hand, subject="", fps=80.0):
    tag = f"[{subject}] " if subject else ""
    issues = []
    T = hand.shape[0]
    nan_pct = np.isnan(hand).mean() * 100
    if nan_pct > 0: issues.append(f"{tag}{nan_pct:.1f}% NaN")
    ps = palm_span(hand); ps_cv = np.nanstd(ps) / (np.nanmean(ps) + 1e-12)
    if ps_cv > 0.5: issues.append(f"{tag}Palm span CV={ps_cv:.2f}")
    if T > 1:
        speed = np.linalg.norm(np.diff(hand[:, MIDDLE_TIP], axis=0), axis=1) * fps
        if (speed > 5000).mean() > 0.01: issues.append(f"{tag}{(speed>5000).mean()*100:.1f}% fast frames")
    return issues

def check_closure(cr):
    tag = f"[{cr.subject}] " if cr.subject else ""
    issues = []
    for name, sc in [("dist", cr.distance_score), ("angle", cr.angle_score)]:
        if np.std(sc) < 0.01: issues.append(f"{tag}{name} flat (std={np.std(sc):.4f})")
    if len(cr.distance_score) > 10:
        r = np.corrcoef(cr.distance_score, cr.angle_score)[0, 1]
        if r < 0.5: issues.append(f"{tag}D-A corr={r:.2f}")
    mq = cr.movement_quality
    mc = mq.get("movement_class", "")
    if mc == "minimal": issues.append(f"{tag}MINIMAL movement (range={mq['raw_range_ratio']:.2f})")
    elif mc == "reduced": issues.append(f"{tag}REDUCED movement (range={mq['raw_range_ratio']:.2f})")
    snr = mq.get("snr_distance", float("nan"))
    if np.isfinite(snr) and snr < 5.0: issues.append(f"{tag}Low SNR={snr:.1f}")
    return issues


# ============================================================
# Main processing
# ============================================================
def process_activity(activity):
    subjects = discover_subjects(activity)
    labels = load_labels()
    fps = C.FPS
    clip_s = int(C.CLIP_START_SEC * fps)
    clip_len = int(C.ANALYSIS_DURATION * fps)

    results = []; all_issues = []; n_skip = 0

    for subj in subjects:
        final_neurips_subjects = C.FINALFINAL_NEURIPS_56SUBJECTS
        if subj not in final_neurips_subjects:
            print(f"                            [SKIP] This subject is not in our Final Neurips 2026 paper: {subj}")
            continue

        updrs = labels.get(subj, {}).get(activity)
        if updrs is None: n_skip += 1; continue

        try:
            body, mainhand, subhand = load_subject(activity, subj, fps)
        except Exception as e:
            all_issues.append(f"[{subj}] LOAD: {e}"); continue

        all_issues.extend(check_hand(mainhand, subj, fps))
        cr = build_closure_scores(mainhand, fps, C.SMOOTH_MS, C.CLIP_START_SEC)
        cr.subject = subj; cr.activity = activity
        all_issues.extend(check_closure(cr))

        clip_e = min(clip_s + clip_len, mainhand.shape[0])
        if clip_e - clip_s < int(5 * fps): all_issues.append(f"[{subj}] Too short"); continue

        results.append({
            "subject": subj, "activity": activity, "side": C.ACTIVITY_SIDES[activity],
            "updrs_score": updrs, "closure": cr,
            "hand": mainhand[clip_s:clip_e], "body": body[clip_s:clip_e],
            "subhand": subhand[clip_s:clip_e] if subhand is not None else None,
            "clip_frames": (clip_s, clip_e), "full_n_frames": mainhand.shape[0],
        })

    # Summary
    print(f"\n{'='*50}\n  Step A: {activity}\n{'='*50}")
    print(f"  Found: {len(subjects)}, Loaded: {len(results)}, Skipped (no label): {n_skip}")
    scores = [r["updrs_score"] for r in results]
    if scores:
        vals, cnts = np.unique(scores, return_counts=True)
        print(f"  UPDRS: {', '.join(f'{v}:{c}' for v,c in zip(vals,cnts))}")
    mq_dist = Counter(r["closure"].movement_quality.get("movement_class","?") for r in results)
    print(f"  Movement: {dict(mq_dist)}")
    if all_issues:
        print(f"  Warnings ({len(all_issues)}):")
        for w in all_issues[:15]: print(f"    ! {w}")
        if len(all_issues) > 15: print(f"    ... +{len(all_issues)-15} more")
    print(f"{'='*50}\n")
    return results


def generate_report(results, activity):
    """Save per-subject CSV report."""
    rows = []
    for r in results:
        cr = r["closure"]; mq = cr.movement_quality
        s, e = r["clip_frames"]
        d, a = cr.distance_score[s:e], cr.angle_score[s:e]
        da_corr = np.corrcoef(d, a)[0,1] if len(d) > 10 else float("nan")
        ps = cr.palm_span_mm[s:e]
        rows.append({
            "subject": r["subject"], "updrs": r["updrs_score"],
            "movement_class": mq.get("movement_class","?"),
            "raw_range": f"{mq.get('raw_range_ratio',0):.3f}",
            "mcp_range_deg": f"{mq.get('mcp_range_deg',0):.1f}",
            "snr": f"{mq.get('snr_distance',0):.1f}",
            "dist_std": f"{np.std(d):.3f}", "angle_std": f"{np.std(a):.3f}",
            "da_corr": f"{da_corr:.3f}",
            "palm_span": f"{np.nanmean(ps):.1f}",
            "duration_s": f"{(e-s)/C.FPS:.1f}",
        })
    path = os.path.join(C.OUTPUT_DIR, f"stepa_report_{activity}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader(); w.writerows(rows)
    print(f"  Report: {path}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    for activity in C.ACTIVITIES:
        results = process_activity(activity)
        generate_report(results, activity)

        out = C.closure_pkl(activity)
        with open(out, "wb") as f: pickle.dump(results, f, protocol=4)
        print(f"  Saved {len(results)} -> {out}")
