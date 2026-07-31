#!/usr/bin/env python3
import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


SOURCE_DIR_MAP = {
    "sam3d": "SAM3D",
    "videopose3d": "Videopose3D",
}

TASK_TO_GT_PRESET = {
    "making_a_fist_Left": "mfl",
    "making_a_fist_Right": "mfr",
}

TASK_TO_FILE_NAME = {
    "making_a_fist_Left": "pose3d_canonical_lefthand.pkl",
    "making_a_fist_Right": "pose3d_canonical_righthand.pkl",
}

TASK_ORDER = [
    "making_a_fist_Left",
    "making_a_fist_Right",
]

META_COLUMNS = {"subject", "activity", "side", "updrs_score", "n_windows", "window_idx"}
NUMERIC_FEATURE_SUMMARY_COLUMNS = [
    "num_subjects",
    "mean_matched_windows",
    "gt_mean",
    "source_mean",
    "mae",
    "pct_error_mean",
]

DEFAULT_ST_FEATURE = "speed_fingertip_mean"
DEFAULT_KINEMATIC_FEATURE = "amp_mcp_range_rad_mean"
OUTPUT_FIELDNAMES = [
    "group",
    "source",
    "mpjpe_mm",
    "pa_mpjpe_mm",
    "st_err_pct",
    "kinematic_err_pct",
]


def mpjpe_per_frame(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {target.shape}")
    return np.linalg.norm(predicted - target, axis=-1).mean(axis=-1)


def p_mpjpe_per_frame(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {target.shape}")

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0 ** 2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0 ** 2, axis=(1, 2), keepdims=True))

    x0 = x0 / np.maximum(norm_x, 1e-8)
    y0 = y0 / np.maximum(norm_y, 1e-8)

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / np.maximum(norm_y, 1e-8)
    t = mu_x - a * np.matmul(mu_y, r)
    predicted_aligned = a * np.matmul(predicted, r) + t
    return np.linalg.norm(predicted_aligned - target, axis=-1).mean(axis=-1)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def read_csv_rows(path):
    with Path(path).open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return np.nan
    return float(text)


def canonicalize_source(source):
    return SOURCE_DIR_MAP.get(source, source)


def parse_named_path_arg(text):
    if "=" not in text:
        raise ValueError(f"Expected NAME=PATH, got: {text}")
    name, path_text = text.split("=", 1)
    name = canonicalize_source(name.strip())
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Missing source name in: {text}")
    return name, path


def root_center_pose(sequence, root_index=0):
    return sequence - sequence[:, root_index : root_index + 1, :]


def align_frame_count(pred_pose, gt_pose, subject_id, camera_name):
    if pred_pose.ndim != 3 or gt_pose.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays for {subject_id} {camera_name}, got {pred_pose.shape} and {gt_pose.shape}"
        )
    if pred_pose.shape[1:] != gt_pose.shape[1:]:
        raise ValueError(
            f"Joint shape mismatch for {subject_id} {camera_name}: {pred_pose.shape} vs {gt_pose.shape}"
        )
    min_frames = min(pred_pose.shape[0], gt_pose.shape[0])
    return pred_pose[:min_frames], gt_pose[:min_frames]


def load_csv_rows(path):
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def parse_window_idx(value):
    text = str(value).strip()
    if text == "":
        raise ValueError("window_idx is empty")
    return int(float(text))


def build_window_map(rows):
    window_map = {}
    for row in rows:
        subject = row.get("subject")
        if not subject:
            continue
        window_idx = parse_window_idx(row["window_idx"])
        key = (subject, window_idx)
        if key in window_map:
            raise RuntimeError(f"Duplicate subject/window key found: {key}")
        window_map[key] = row
    return window_map


def infer_feature_columns(gt_rows, source_rows):
    gt_cols = set(gt_rows[0].keys()) - META_COLUMNS
    src_cols = set(source_rows[0].keys()) - META_COLUMNS
    return sorted(gt_cols & src_cols)


def compute_task_feature_summary(gt_dir, source_feature_dirs, task_name):
    gt_path = Path(gt_dir) / f"features_window_{task_name}.csv"
    gt_rows = load_csv_rows(gt_path)
    gt_map = build_window_map(gt_rows)

    summary_rows = []
    for source, source_dir in source_feature_dirs.items():
        source_path = Path(source_dir) / f"features_window_{task_name}.csv"
        source_rows = load_csv_rows(source_path)
        source_map = build_window_map(source_rows)
        shared_window_keys = sorted(set(gt_map.keys()) & set(source_map.keys()))
        if not shared_window_keys:
            raise RuntimeError(f"No shared subject/window rows between GT and {source_path}")

        shared_subjects = sorted({subject for subject, _ in shared_window_keys})

        for feature in infer_feature_columns(gt_rows, source_rows):
            feature_subject_rows = []
            for subject in shared_subjects:
                subject_window_keys = [key for key in shared_window_keys if key[0] == subject]
                if not subject_window_keys:
                    continue

                gt_values = []
                src_values = []
                abs_errors = []
                pct_errors = []
                for window_key in subject_window_keys:
                    gt_val = parse_float(gt_map[window_key].get(feature))
                    src_val = parse_float(source_map[window_key].get(feature))
                    if np.isnan(gt_val) or np.isnan(src_val):
                        continue
                    abs_error = abs(src_val - gt_val)
                    pct_error = abs_error / abs(gt_val) * 100.0 if abs(gt_val) > 1e-8 else np.nan

                    gt_values.append(gt_val)
                    src_values.append(src_val)
                    abs_errors.append(abs_error)
                    if np.isfinite(pct_error):
                        pct_errors.append(pct_error)

                if not pct_errors:
                    continue

                feature_subject_rows.append(
                    {
                        "gt_mean": float(np.mean(gt_values)),
                        "source_mean": float(np.mean(src_values)),
                        "mae": float(np.mean(abs_errors)),
                        "pct_error_mean": float(np.mean(pct_errors)),
                        "num_matched_windows": float(len(pct_errors)),
                    }
                )

            if not feature_subject_rows:
                continue

            summary_rows.append(
                {
                    "source": source,
                    "task": task_name,
                    "feature": feature,
                    "num_subjects": float(len(feature_subject_rows)),
                    "mean_matched_windows": float(
                        np.nanmean([row["num_matched_windows"] for row in feature_subject_rows])
                    ),
                    "gt_mean": float(np.nanmean([row["gt_mean"] for row in feature_subject_rows])),
                    "source_mean": float(np.nanmean([row["source_mean"] for row in feature_subject_rows])),
                    "mae": float(np.nanmean([row["mae"] for row in feature_subject_rows])),
                    "pct_error_mean": float(
                        np.nanmean([row["pct_error_mean"] for row in feature_subject_rows])
                    ),
                }
            )
    return summary_rows


def average_hands_feature_summary(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["source"], row["feature"])].append(row)

    averaged_rows = {}
    for key, rows in grouped.items():
        out = {"source": key[0], "feature": key[1]}
        for col in NUMERIC_FEATURE_SUMMARY_COLUMNS:
            out[col] = float(np.nanmean([parse_float(row[col]) for row in rows]))
        averaged_rows[key] = out
    return averaged_rows


def summarize_subject_rows(subject_rows):
    return {
        "mean_mpjpe": float(np.mean([row["mpjpe"] for row in subject_rows])),
        "mean_pa_mpjpe": float(np.mean([row["pa_mpjpe"] for row in subject_rows])),
    }


def compute_pose_task_summary(root_dir, source, task_name, gt_camera, root_center_before_metric):
    source_dir_name = SOURCE_DIR_MAP.get(source.lower(), source)
    pred_root = Path(root_dir) / "predictions" / source_dir_name / task_name
    gt_path = Path(root_dir) / "ground_truth" / TASK_TO_GT_PRESET[task_name] / "all_3d.pkl"
    pred_file_name = TASK_TO_FILE_NAME[task_name]

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction folder not found: {pred_root}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT pickle not found: {gt_path}")

    gt_data = load_pickle(gt_path)
    pred_subject_dirs = sorted(path for path in pred_root.iterdir() if path.is_dir())
    pred_subjects = [path.name for path in pred_subject_dirs]
    gt_subjects = sorted(gt_data.keys())
    shared_subjects = sorted(set(pred_subjects) & set(gt_subjects))
    if not shared_subjects:
        raise RuntimeError(f"No shared subjects for {source} {task_name}")

    subject_rows = []
    for subject_id in shared_subjects:
        pred_path = pred_root / subject_id / pred_file_name
        pred_pose = np.asarray(load_pickle(pred_path), dtype=np.float64)

        cameras_to_use = [gt_camera] if gt_camera is not None else sorted(gt_data[subject_id].keys())
        frame_mpjpe_list = []
        frame_pa_list = []
        total_frames = 0

        for camera_name in cameras_to_use:
            gt_pose = np.asarray(gt_data[subject_id][camera_name], dtype=np.float64)
            pred_aligned, gt_aligned = align_frame_count(pred_pose, gt_pose, subject_id, camera_name)
            if root_center_before_metric:
                pred_aligned = root_center_pose(pred_aligned, root_index=0)
                gt_aligned = root_center_pose(gt_aligned, root_index=0)

            frame_mpjpe_list.append(mpjpe_per_frame(pred_aligned, gt_aligned))
            frame_pa_list.append(p_mpjpe_per_frame(pred_aligned, gt_aligned))
            total_frames += int(gt_aligned.shape[0])

        subject_rows.append(
            {
                "num_frames": total_frames,
                "mpjpe": float(np.mean(np.concatenate(frame_mpjpe_list, axis=0))),
                "pa_mpjpe": float(np.mean(np.concatenate(frame_pa_list, axis=0))),
            }
        )

    return summarize_subject_rows(subject_rows)


def parse_args():
    package_root = Path(__file__).resolve().parents[1]
    inputs_dir = package_root / "inputs" / "fist"

    parser = argparse.ArgumentParser(
        description=(
            "Compute public fist evaluation from raw 3D poses plus GT/source "
            "feature CSV directories, then average Left/Right."
        )
    )
    parser.add_argument("--group", default="release", help="Label written into the output CSV.")
    parser.add_argument(
        "--pose-root-dir",
        type=Path,
        default=inputs_dir / "pose",
        help="Root folder containing organized making-fist public pose inputs.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["sam3d", "videopose3d"],
        help="Prediction sources to evaluate.",
    )
    parser.add_argument(
        "--gt-camera",
        default="Camera5",
        help="GT camera used for pose evaluation. Pass all to average all GT cameras.",
    )
    parser.add_argument(
        "--root-center-before-metric",
        dest="root_center_before_metric",
        action="store_true",
        help="Root-center both prediction and GT before computing pose metrics.",
    )
    parser.add_argument(
        "--no-root-center-before-metric",
        dest="root_center_before_metric",
        action="store_false",
        help="Disable root-centering before pose metrics.",
    )
    parser.set_defaults(root_center_before_metric=True)
    parser.add_argument(
        "--gt-feature-dir",
        type=Path,
        default=inputs_dir / "features" / "window_level" / "GT",
        help="Directory containing GT making-fist window-level feature CSVs for Left/Right.",
    )
    parser.add_argument(
        "--source-feature-dir",
        action="append",
        default=[],
        help="Source window-feature directory in the form SOURCE=/abs/path/source_dir. Repeat for each source.",
    )
    parser.add_argument(
        "--st-feature",
        default=DEFAULT_ST_FEATURE,
        help="Spatiotemporal fist feature name.",
    )
    parser.add_argument(
        "--kinematic-feature",
        default=DEFAULT_KINEMATIC_FEATURE,
        help="Kinematic fist feature name.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=package_root / "outputs" / "fist_public_eval_56sub.csv",
        help="Destination CSV path.",
    )
    return parser.parse_args()


def build_output_rows(args):
    source_feature_dirs = dict(parse_named_path_arg(item) for item in args.source_feature_dir)
    if not source_feature_dirs:
        source_feature_dirs = {
            "SAM3D": Path(__file__).resolve().parents[1] / "inputs" / "fist" / "features" / "window_level" / "SAM3D",
            "Videopose3D": Path(__file__).resolve().parents[1] / "inputs" / "fist" / "features" / "window_level" / "Videopose3D",
        }

    feature_summary_rows = []
    for task_name in TASK_ORDER:
        feature_summary_rows.extend(
            compute_task_feature_summary(args.gt_feature_dir, source_feature_dirs, task_name)
        )
    averaged_feature_rows = average_hands_feature_summary(feature_summary_rows)

    gt_camera = None if str(args.gt_camera).lower() == "all" else args.gt_camera
    rows = []
    for source_name in args.sources:
        source = canonicalize_source(source_name)
        left_pose = compute_pose_task_summary(
            args.pose_root_dir,
            source_name,
            "making_a_fist_Left",
            gt_camera,
            args.root_center_before_metric,
        )
        right_pose = compute_pose_task_summary(
            args.pose_root_dir,
            source_name,
            "making_a_fist_Right",
            gt_camera,
            args.root_center_before_metric,
        )

        st_key = (source, args.st_feature)
        kin_key = (source, args.kinematic_feature)
        if st_key not in averaged_feature_rows or kin_key not in averaged_feature_rows:
            raise ValueError(
                f"Missing selected averaged feature rows for source {source}. "
                f"Need {args.st_feature} and {args.kinematic_feature}."
            )

        rows.append(
            {
                "group": args.group,
                "source": source,
                "mpjpe_mm": float(np.mean([left_pose["mean_mpjpe"], right_pose["mean_mpjpe"]])),
                "pa_mpjpe_mm": float(np.mean([left_pose["mean_pa_mpjpe"], right_pose["mean_pa_mpjpe"]])),
                "st_err_pct": averaged_feature_rows[st_key]["pct_error_mean"],
                "kinematic_err_pct": averaged_feature_rows[kin_key]["pct_error_mean"],
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = build_output_rows(args)
    write_csv(args.output_csv, rows)
    print(f"Saved fist evaluation CSV to: {args.output_csv}")
    for row in rows:
        print(
            f"{row['source']}: "
            f"MPJPE={row['mpjpe_mm']:.4f}, "
            f"PA-MPJPE={row['pa_mpjpe_mm']:.4f}, "
            f"ST_ERR={row['st_err_pct']:.4f}, "
            f"KIN_ERR={row['kinematic_err_pct']:.4f}"
        )


if __name__ == "__main__":
    main()
