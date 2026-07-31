#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


SOURCE_NAME_MAP = {
    "sam3d": "SAM3D",
    "wham": "WHAM",
}

DEFAULT_ST_FEATURE = "stride_length"
DEFAULT_KINEMATIC_FEATURE = "arm2arm_ROM"
OUTPUT_FIELDNAMES = [
    "group",
    "source",
    "mpjpe_mm",
    "pa_mpjpe_mm",
    "st_err_pct",
    "kinematic_err_pct",
]
CSV_META_COLUMNS = {"subject", "stride_idx"}


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


def canonicalize_source(source):
    return SOURCE_NAME_MAP.get(source, source)


def parse_named_path_arg(text):
    if "=" not in text:
        raise ValueError(f"Expected NAME=PATH, got: {text}")
    name, path_text = text.split("=", 1)
    name = canonicalize_source(name.strip())
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Missing source name in: {text}")
    return name, path


def infer_pose_source_name(file_name):
    stem = Path(file_name).stem
    prefix = "56sub_"
    suffix = "_gait_pose_cam1"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return canonicalize_source(stem)


def summarize_pose_subject_rows(subject_rows):
    return {
        "mean_mpjpe": float(np.mean([row["mpjpe"] for row in subject_rows])),
        "mean_pa_mpjpe": float(np.mean([row["pa_mpjpe"] for row in subject_rows])),
    }


def compute_pose_metrics(input_dir, gt_file):
    gt_path = Path(input_dir) / gt_file
    if not gt_path.exists():
        raise FileNotFoundError(f"GT pose file not found: {gt_path}")

    gt_data = load_pickle(gt_path)
    if not isinstance(gt_data, dict):
        raise ValueError(f"Expected GT pose pickle to be a dict, got {type(gt_data).__name__}")

    gt_subjects = sorted(gt_data.keys())
    candidate_paths = sorted(path for path in Path(input_dir).glob("*.pkl") if path.name != gt_file)
    if not candidate_paths:
        raise RuntimeError(f"No comparison pose pickle files found under {input_dir}")

    pose_map = {}
    for pred_path in candidate_paths:
        pred_data = load_pickle(pred_path)
        if not isinstance(pred_data, dict):
            raise ValueError(f"Expected {pred_path.name} to be a dict, got {type(pred_data).__name__}")
        if sorted(pred_data.keys()) != gt_subjects:
            raise ValueError(f"Subject mismatch between GT and {pred_path.name}")

        source = infer_pose_source_name(pred_path.name)
        subject_rows = []
        for subject_id in gt_subjects:
            gt_pose = np.asarray(gt_data[subject_id], dtype=np.float64)
            pred_pose = np.asarray(pred_data[subject_id], dtype=np.float64)
            if gt_pose.shape != pred_pose.shape:
                raise ValueError(
                    f"Shape mismatch for {source} {subject_id}: {pred_pose.shape} vs {gt_pose.shape}"
                )
            subject_rows.append(
                {
                    "mpjpe": float(np.mean(mpjpe_per_frame(pred_pose, gt_pose))),
                    "pa_mpjpe": float(np.mean(p_mpjpe_per_frame(pred_pose, gt_pose))),
                }
            )

        pose_map[source] = summarize_pose_subject_rows(subject_rows)
    return pose_map


def parse_stride_idx(value):
    text = str(value).strip()
    if text == "":
        raise ValueError("stride_idx is empty")
    return int(float(text))


def load_stride_feature_csv(path):
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def build_stride_feature_map(rows):
    stride_map = {}
    for row in rows:
        subject = row.get("subject")
        if not subject:
            continue
        stride_idx = parse_stride_idx(row["stride_idx"])
        key = (subject, stride_idx)
        if key in stride_map:
            raise RuntimeError(f"Duplicate subject/stride key found: {key}")
        stride_map[key] = row
    return stride_map


def infer_stride_feature_columns(gt_rows, source_rows):
    gt_cols = set(gt_rows[0].keys()) - CSV_META_COLUMNS
    source_cols = set(source_rows[0].keys()) - CSV_META_COLUMNS
    return sorted(gt_cols & source_cols)


def compute_feature_summary_from_csvs(gt_feature_csv, source_feature_csvs):
    gt_rows = load_stride_feature_csv(gt_feature_csv)
    gt_map = build_stride_feature_map(gt_rows)

    feature_summary_map = {}
    for source, source_csv in source_feature_csvs.items():
        source_rows = load_stride_feature_csv(source_csv)
        source_map = build_stride_feature_map(source_rows)
        shared_stride_keys = sorted(set(gt_map.keys()) & set(source_map.keys()))
        if not shared_stride_keys:
            raise RuntimeError(f"No shared subject/stride rows between GT and {source_csv}")

        shared_subjects = sorted({subject for subject, _ in shared_stride_keys})
        for feature_name in infer_stride_feature_columns(gt_rows, source_rows):
            subject_mapes = []
            for subject in shared_subjects:
                subject_stride_keys = [key for key in shared_stride_keys if key[0] == subject]
                if not subject_stride_keys:
                    continue

                gt_values = []
                pred_values = []
                for stride_key in subject_stride_keys:
                    gt_values.append(float(gt_map[stride_key][feature_name]))
                    pred_values.append(float(source_map[stride_key][feature_name]))

                gt_arr = np.asarray(gt_values, dtype=np.float64)
                pred_arr = np.asarray(pred_values, dtype=np.float64)
                valid = np.abs(gt_arr) > 1e-8
                if not np.any(valid):
                    continue
                subject_mape = np.mean(np.abs(pred_arr[valid] - gt_arr[valid]) / np.abs(gt_arr[valid])) * 100
                subject_mapes.append(float(subject_mape))

            if subject_mapes:
                feature_summary_map[(source, feature_name)] = float(np.mean(subject_mapes))
    return feature_summary_map


def parse_args():
    package_root = Path(__file__).resolve().parents[1]
    inputs_dir = package_root / "inputs" / "gait"
    parser = argparse.ArgumentParser(
        description=(
            "Compute public gait evaluation from raw 3D pose pickles and precomputed "
            "stride-level gait feature CSVs."
        )
    )
    parser.add_argument("--group", default="release", help="Label written into the output CSV.")
    parser.add_argument(
        "--pose-input-dir",
        type=Path,
        default=inputs_dir / "pose_sequences",
        help="Folder containing GT and predicted gait pose pickles.",
    )
    parser.add_argument(
        "--gt-pose-file",
        default="56sub_GT_gait_pose_cam1.pkl",
        help="Ground-truth pose pickle filename inside pose-input-dir.",
    )
    parser.add_argument(
        "--gt-feature-csv",
        type=Path,
        default=inputs_dir / "feature_csvs" / "gait" / "GT" / "features_stride.csv",
        help="GT stride-level gait feature CSV.",
    )
    parser.add_argument(
        "--source-feature-csv",
        action="append",
        default=[],
        help="Source gait feature CSV in the form SOURCE=/abs/path/source_stride_features.csv. Repeat for each source.",
    )
    parser.add_argument(
        "--st-feature",
        default=DEFAULT_ST_FEATURE,
        help="Spatiotemporal gait feature name.",
    )
    parser.add_argument(
        "--kinematic-feature",
        default=DEFAULT_KINEMATIC_FEATURE,
        help="Kinematic gait feature name.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=package_root / "outputs" / "gait_public_eval_56sub.csv",
        help="Destination CSV path.",
    )
    return parser.parse_args()


def build_output_rows(args):
    source_feature_csvs = dict(parse_named_path_arg(item) for item in args.source_feature_csv)
    if not source_feature_csvs:
        feature_root = Path(__file__).resolve().parents[1] / "inputs" / "gait" / "feature_csvs"
        source_feature_csvs = {
            "MAGFPre": feature_root / "MAGFPre_stride_features.csv",
            "MAGFTulip": feature_root / "MAGFTulip_stride_features.csv",
            "SAM3D": feature_root / "SAM3D_stride_features.csv",
            "WHAM": feature_root / "WHAM_stride_features.csv",
        }

    pose_map = compute_pose_metrics(args.pose_input_dir, args.gt_pose_file)
    feature_map = compute_feature_summary_from_csvs(args.gt_feature_csv, source_feature_csvs)

    missing_feature_sources = sorted(set(pose_map.keys()) - set(source_feature_csvs.keys()))
    if missing_feature_sources:
        raise ValueError(f"Missing feature CSVs for pose sources: {missing_feature_sources}")

    rows = []
    for source in sorted(pose_map):
        st_key = (source, args.st_feature)
        kin_key = (source, args.kinematic_feature)
        if st_key not in feature_map or kin_key not in feature_map:
            raise ValueError(
                f"Missing selected feature rows for source {source}. "
                f"Need {args.st_feature} and {args.kinematic_feature}."
            )
        rows.append(
            {
                "group": args.group,
                "source": source,
                "mpjpe_mm": pose_map[source]["mean_mpjpe"],
                "pa_mpjpe_mm": pose_map[source]["mean_pa_mpjpe"],
                "st_err_pct": feature_map[st_key],
                "kinematic_err_pct": feature_map[kin_key],
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
    print(f"Saved gait evaluation CSV to: {args.output_csv}")
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
