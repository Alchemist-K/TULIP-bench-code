#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""


"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slice full-length 2D pose sequences into stride segments using saved frame ranges.

For each subject frame-range file under RIGHT_STRIDE_FRAME_NUMS_DIR, this script:
1. Loads the chosen camera from the 17-joint raw 2D pose pickle.
2. Loads the same camera from the 17-joint reprojected 2D pose pickle.
3. Slices both full sequences with each (start_frame, end_frame) tuple.
4. Saves one subject-level pickle per source, where the contents are:
   list[np.ndarray], each array shaped (time_points, 17, 2)
"""


import pickle
from pathlib import Path
from typing import Iterable

import os
import numpy as np

# ===== CONFIGURATION: edit for your environment =====
DATA_ROOT = "./data"


# ======== Change Parameters Here ========
PROJECT_ROOT = Path(DATA_ROOT)
CAMERA_NAME = "Camera1"

RIGHT_STRIDE_FRAME_NUMS_DIR = (
    PROJECT_ROOT
    / "feature_extraction_pipeline"
    / "normal_pd_features"
    / "right_strides"
    / "right_strides_frame_nums_cleaned"
)

RAW_2D_POSES_PATH = (
    PROJECT_ROOT
    / "2d_lifting"
    / "organize_poses"
    / "normalPD_data"
    / "17_version_keypoints"
    / "neruips_2d_poses.pkl"
)

REPROJECTED_2D_POSES_PATH = (
    PROJECT_ROOT
    / "2d_lifting"
    / "organize_poses"
    / "normalPD_data"
    / "17_version_keypoints"
    / "neruips_reprojected_2d_poses.pkl"
)

RAW_OUTPUT_DIR = (
    PROJECT_ROOT
    / "feature_extraction_pipeline"
    / "normal_pd_features"
    / "raw_2d_pose_strides"
)

REPROJECTED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "feature_extraction_pipeline"
    / "normal_pd_features"
    / "reprojected_2d_pose_strides"
)

PROCESS_ONLY_SUBJECTS = None
OVERWRITE_EXISTING = True
# =======================================


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, data) -> None:
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def get_subjects_to_process(frame_dir: Path) -> list[str]:
    subject_names = sorted(path.stem for path in frame_dir.glob("*.pkl"))
    if PROCESS_ONLY_SUBJECTS is None:
        return subject_names

    requested = list(PROCESS_ONLY_SUBJECTS)
    available = set(subject_names)
    missing = [subject for subject in requested if subject not in available]
    if missing:
        raise KeyError(
            "These subjects are missing from the stride frame folder: "
            + ", ".join(missing)
        )
    return requested


def validate_pose_array(array: np.ndarray, subject: str, source_name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[1:] != (17, 2):
        raise ValueError(
            f"{source_name} for {subject} must have shape (T, 17, 2), "
            f"but got {array.shape}."
        )
    return array


def extract_raw_camera_sequence(raw_poses: dict, subject: str, camera_name: str) -> np.ndarray:
    if subject not in raw_poses:
        raise KeyError(f"{subject} is missing from raw 2D poses.")
    if camera_name not in raw_poses[subject]:
        raise KeyError(f"{camera_name} is missing for {subject} in raw 2D poses.")

    camera_payload = raw_poses[subject][camera_name]
    if "keypoints" not in camera_payload:
        raise KeyError(
            f"raw 2D poses for {subject} / {camera_name} do not contain 'keypoints'."
        )
    return validate_pose_array(camera_payload["keypoints"], subject, "Raw 2D poses")


def extract_reprojected_camera_sequence(
    reprojected_poses: dict,
    subject: str,
    camera_name: str,
) -> np.ndarray:
    if subject not in reprojected_poses:
        raise KeyError(f"{subject} is missing from reprojected 2D poses.")
    if camera_name not in reprojected_poses[subject]:
        raise KeyError(
            f"{camera_name} is missing for {subject} in reprojected 2D poses."
        )

    return validate_pose_array(
        reprojected_poses[subject][camera_name],
        subject,
        "Reprojected 2D poses",
    )


def validate_frame_ranges(
    frame_ranges: Iterable[tuple[int, int]],
    num_frames: int,
    subject: str,
) -> list[tuple[int, int]]:
    normalized = []
    for stride_idx, frame_range in enumerate(frame_ranges):
        if not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2:
            raise ValueError(
                f"{subject} stride {stride_idx} must be a (start, end) pair, "
                f"but got {frame_range!r}."
            )

        start, end = int(frame_range[0]), int(frame_range[1])
        if start < 0 or end < 0:
            raise ValueError(
                f"{subject} stride {stride_idx} has negative frame numbers: {(start, end)}"
            )
        if end <= start:
            raise ValueError(
                f"{subject} stride {stride_idx} must satisfy end > start, "
                f"but got {(start, end)}."
            )
        if end > num_frames:
            raise ValueError(
                f"{subject} stride {stride_idx} ends at {end}, beyond sequence "
                f"length {num_frames}."
            )
        normalized.append((start, end))
    return normalized


def slice_sequence_into_strides(
    sequence: np.ndarray,
    frame_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    return [np.asarray(sequence[start:end], dtype=np.float32) for start, end in frame_ranges]


def main() -> None:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPROJECTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_poses = load_pickle(RAW_2D_POSES_PATH)
    reprojected_poses = load_pickle(REPROJECTED_2D_POSES_PATH)
    subjects = get_subjects_to_process(RIGHT_STRIDE_FRAME_NUMS_DIR)

    print(f"Camera: {CAMERA_NAME}")
    print(f"Subjects to process: {len(subjects)}")

    for subject in subjects:
        raw_output_path = RAW_OUTPUT_DIR / f"{subject}.pkl"
        reprojected_output_path = REPROJECTED_OUTPUT_DIR / f"{subject}.pkl"

        if (
            not OVERWRITE_EXISTING
            and raw_output_path.exists()
            and reprojected_output_path.exists()
        ):
            print(f"Skipping {subject}: outputs already exist.")
            continue

        frame_ranges = load_pickle(RIGHT_STRIDE_FRAME_NUMS_DIR / f"{subject}.pkl")
        raw_sequence = extract_raw_camera_sequence(raw_poses, subject, CAMERA_NAME)
        reprojected_sequence = extract_reprojected_camera_sequence(
            reprojected_poses,
            subject,
            CAMERA_NAME,
        )

        if raw_sequence.shape[0] != reprojected_sequence.shape[0]:
            raise ValueError(
                f"{subject} has mismatched frame counts between raw and reprojected "
                f"2D poses: {raw_sequence.shape[0]} vs {reprojected_sequence.shape[0]}."
            )

        validated_ranges = validate_frame_ranges(
            frame_ranges,
            raw_sequence.shape[0],
            subject,
        )

        raw_stride_list = slice_sequence_into_strides(raw_sequence, validated_ranges)
        reprojected_stride_list = slice_sequence_into_strides(
            reprojected_sequence,
            validated_ranges,
        )

        save_pickle(raw_output_path, raw_stride_list)
        save_pickle(reprojected_output_path, reprojected_stride_list)

        print(
            f"Saved {subject}: {len(validated_ranges)} strides "
            f"to {raw_output_path.name} and {reprojected_output_path.name}"
        )


if __name__ == "__main__":
    main()
