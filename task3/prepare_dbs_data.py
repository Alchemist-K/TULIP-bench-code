# task3_prepare_dbs_data.py
"""
Task 3: Prepare DBS OFF/ON paired dataset.
Uses Neurips2026_DBS_Gait_video_ratings_Categorized.csv with pre-parsed side columns.
"""

import os, csv, pickle, re

import config
import numpy as np

GAIT_PKL_PATH = os.path.join(config.DBS_WORK_DIR, "stride_18features.pkl")
LABELS_CSV_PATH = config.GAIT_DBS_LABELS_CSV
OUTPUT_DIR = config.DBS_WORK_DIR

FEATURE_NAMES = [
    "stride_length", "stride_time", "step_length_asy", "step_time_asy",
    "velocity", "cadance", "hip_ROM_asy", "knee_ROM_asy",
    "elbow_ROM_asy", "shoulder_ROM_asy", "arm2arm_ROM", "leg2leg_ROM",
    "step_length", "step_time", "hip_ROM", "knee_ROM", "elbow_ROM", "shoulder_ROM"
]

SEVERITY_MAP = {
    "normal": 0, "none": 0,
    "reduced (slight)": 1, "flexed (slight)": 1,
    "reduced (mild)": 2, "flexed (mild)": 2,
    "reduced (moderate)": 3, "flexed (moderate)": 3,
    "reduced (severe)": 4, "absent (severe)": 4,
}
DYSKINESIA_TERMS = ["dyskinesia", "excessive movement"]

def parse_severity(val):
    if not val or not val.strip(): return None
    v = val.strip().lower()
    if v in SEVERITY_MAP: return SEVERITY_MAP[v]
    for k, s in SEVERITY_MAP.items():
        if k in v: return s
    if any(d in v for d in DYSKINESIA_TERMS): return None
    return None

def has_dyskinesia(val):
    return bool(val and any(d in val.lower() for d in DYSKINESIA_TERMS))

def parse_responder(raw):
    if not raw or not raw.strip(): return None
    r = raw.lower().strip()
    if "major response" in r: return 1
    if "moderate response" in r: return 0
    if r.startswith("n"): return 0
    if r.startswith("y"): return 1
    return None

def match_pkl_to_csv(pkl_keys, csv_names):
    """Match pose records to label rows on (subject id, stimulation state).

    Released recordings are named Subject_<N>_<STATE>, where STATE is OFF or ON.
    Subject identifiers are de-identified and scoped to the DBS cohort.
    """
    def parse(name):
        subject = re.search(r"(Subject[_-]?\d+)", name, re.IGNORECASE)
        state = "OFF" if "OFF" in name.upper() else ("ON" if "ON" in name.upper() else "UNK")
        return (subject.group(1).lower().replace("-", "_") if subject else "", state)

    index = {}
    for pk in pkl_keys:
        index.setdefault(parse(pk), pk)

    mapping = {}
    for cn in csv_names:
        key = parse(cn)
        if key in index:
            mapping[cn] = index[key]
    return mapping

def build_dataset(feat_data, label_rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = match_pkl_to_csv(list(feat_data.keys()), [r["subject_name"] for r in label_rows])
    print(f"  Matched {len(mapping)}/{len(label_rows)} rows")

    subjects = {}
    for lr in label_rows:
        cn = lr["subject_name"]; state = "OFF" if "OFF" in cn else "ON"
        base = re.sub(r'_DBS(OFF|ON)_', '_', cn)
        if base not in subjects: subjects[base] = {}

        pk = mapping.get(cn); feat_avg = {}
        if pk and pk in feat_data:
            for fn in FEATURE_NAMES:
                if fn in feat_data[pk]:
                    vals = [float(v) for v in feat_data[pk][fn] if v is not None and np.isfinite(float(v))]
                    feat_avg[fn] = float(np.mean(vals)) if vals else float("nan")
                else: feat_avg[fn] = float("nan")
        else: feat_avg = {fn: float("nan") for fn in FEATURE_NAMES}

        arm_L = parse_severity(lr.get("Descriptions1_armswing_left_side", ""))
        arm_R = parse_severity(lr.get("Descriptions1_armswing_right_side", ""))
        stride_L = parse_severity(lr.get("Descriptions2_stride_left_side", ""))
        stride_R = parse_severity(lr.get("Descriptions2_stride_right_side", ""))
        elbow_L = parse_severity(lr.get("Descriptions3_flexed_elbows_left_side", ""))
        elbow_R = parse_severity(lr.get("Descriptions3_flexed_elbows_right_side", ""))
        dys_L = has_dyskinesia(lr.get("Descriptions1_armswing_left_side", ""))
        dys_R = has_dyskinesia(lr.get("Descriptions1_armswing_right_side", ""))

        e = {"state": state, "features": feat_avg,
             "gait_updrs": int(float(lr.get("Gait",0) or 0)),
             "posture_updrs": int(float(lr.get("Posture",0) or 0)),
             "freezing_updrs": int(float(lr.get("Freezing",0) or 0)),
             "responder_raw": lr.get("Significant responder (Y/N)","").strip(),
             "bp_armswing_left": arm_L if arm_L is not None else (0 if not dys_L else None),
             "bp_armswing_right": arm_R if arm_R is not None else (0 if not dys_R else None),
             "bp_stride_left": stride_L, "bp_stride_right": stride_R,
             "bp_elbow_left": elbow_L, "bp_elbow_right": elbow_R,
             "bp_has_dyskinesia": 1 if (dys_L or dys_R) else 0}
        al, ar = e["bp_armswing_left"] or 0, e["bp_armswing_right"] or 0
        e["bp_armswing_max"] = max(al, ar); e["bp_armswing_asy"] = abs(al - ar)
        sl, sr = e["bp_stride_left"] or 0, e["bp_stride_right"] or 0
        e["bp_stride_max"] = max(sl, sr)
        el, er = e["bp_elbow_left"] or 0, e["bp_elbow_right"] or 0
        e["bp_elbow_max"] = max(el, er); e["bp_elbow_asy"] = abs(el - er)
        e["combined_updrs"] = e["gait_updrs"] + e["posture_updrs"] + e["freezing_updrs"]
        e["responder"] = parse_responder(e["responder_raw"])
        subjects[base][state] = e

    bp_fields = ["bp_armswing_left","bp_armswing_right","bp_armswing_max","bp_armswing_asy",
                 "bp_stride_left","bp_stride_right","bp_stride_max",
                 "bp_elbow_left","bp_elbow_right","bp_elbow_max","bp_elbow_asy","bp_has_dyskinesia"]
    paired, off_rows, on_rows = [], [], []
    for base in sorted(subjects):
        if "OFF" not in subjects[base] or "ON" not in subjects[base]: continue
        oe, ne = subjects[base]["OFF"], subjects[base]["ON"]
        p = {"subject": base}
        for c in ["gait_updrs","posture_updrs","freezing_updrs","combined_updrs"]:
            p[f"OFF_{c}"] = oe[c]; p[f"ON_{c}"] = ne[c]; p[f"delta_{c}"] = oe[c] - ne[c]
        p["responder"] = ne["responder"] if ne["responder"] is not None else ""
        p["responder_raw"] = ne["responder_raw"]
        for fn in FEATURE_NAMES:
            ov, nv = oe["features"][fn], ne["features"][fn]
            p[f"OFF_{fn}"] = ov; p[f"ON_{fn}"] = nv
            p[f"delta_{fn}"] = (ov - nv) if np.isfinite(ov) and np.isfinite(nv) else float("nan")
        for bp in bp_fields:
            ov, nv = oe.get(bp), ne.get(bp)
            p[f"OFF_{bp}"] = ov if ov is not None else ""
            p[f"ON_{bp}"] = nv if nv is not None else ""
            p[f"delta_{bp}"] = (int(ov) - int(nv)) if ov is not None and nv is not None else ""
        paired.append(p)
        for entry, tgt in [(oe, off_rows), (ne, on_rows)]:
            row = {"subject": base, "state": entry["state"]}
            for c in ["gait_updrs","posture_updrs","freezing_updrs","combined_updrs"]: row[c] = entry[c]
            row["responder"] = ne["responder"] if ne["responder"] is not None else ""
            for fn in FEATURE_NAMES: row[fn] = entry["features"][fn]
            for bp in bp_fields: row[bp] = entry.get(bp, "")
            tgt.append(row)

    _save(paired, os.path.join(OUTPUT_DIR, "task3_dbs_paired.csv"))
    _save(off_rows, os.path.join(OUTPUT_DIR, "task3_dbs_features_off.csv"))
    _save(on_rows, os.path.join(OUTPUT_DIR, "task3_dbs_features_on.csv"))
    print(f"\n  Paired: {len(paired)} subjects")
    resps = [p["responder"] for p in paired if p["responder"] != ""]
    print(f"  Responders: {sum(r==1 for r in resps)} good, {sum(r==0 for r in resps)} bad")
    return paired

def _save(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else str(v)) for k, v in row.items()})
    print(f"  Saved: {path} ({len(rows)} rows)")

if __name__ == "__main__":
    print(f"\n{'#'*60}\n  Task 3: Prepare DBS Paired Dataset\n{'#'*60}\n")
    with open(GAIT_PKL_PATH, "rb") as f: raw = pickle.load(f)
    feat_data = {k.removesuffix("_gait_pose"): v for k, v in raw.items()}
    print(f"  Loaded {len(feat_data)} pkl entries")
    label_rows = []
    with open(LABELS_CSV_PATH) as f:
        for row in csv.DictReader(f): label_rows.append(dict(row))
    print(f"  Loaded {len(label_rows)} label rows")
    build_dataset(feat_data, label_rows)
