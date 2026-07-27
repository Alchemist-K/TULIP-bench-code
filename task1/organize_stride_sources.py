#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slice full-length root-centered gait 3D poses into stride segments and rotate each
stride into a canonical gait-aligned coordinate system.

Output format matches the existing stride slicing convention:
one pickle per subject, contents = list[np.ndarray], each shaped (T, 17, 3).

For each stride:
- x positive: estimated walking direction
- z positive: estimated body-up direction
- y: completes a right-handed frame
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import os
import numpy as np

# ===== CONFIGURATION: edit for your environment =====
DATA_ROOT = "./data"


# ======== Change Parameters Here ========
PROJECT_ROOT = Path(DATA_ROOT)
BASE_DIR = (
    PROJECT_ROOT
    / "3d_gait_different_source"
    / "organized_3dposes_4sources"
)

RIGHT_STRIDE_FRAME_NUMS_DIR = (
    PROJECT_ROOT
    / "feature_extraction_pipeline"
    / "normal_pd_features"
    / "right_strides"
    / "right_strides_frame_nums_cleaned"
)

OUTPUT_ROOT = BASE_DIR / "[lifting poses for classification - GAIT]"

SOURCES = [
    {
        "name": "MAGFPre",
        "input_path": BASE_DIR / "56sub_MAGFPre_gait_pose_cam1.pkl",
        "output_dir": OUTPUT_ROOT / "MAGFPre",
    },
    {
        "name": "MAGFTulip",
        "input_path": BASE_DIR / "56sub_MAGFTulip_gait_pose_cam1.pkl",
        "output_dir": OUTPUT_ROOT / "MAGFTulip",
    },
    {
        "name": "sam3d",
        "input_path": BASE_DIR / "56sub_sam3d_gait_pose_cam1.pkl",
        "output_dir": OUTPUT_ROOT / "sam3d",
    },
    {
        "name": "wham",
        "input_path": BASE_DIR / "56sub_wham_gait_pose_cam1.pkl",
        "output_dir": OUTPUT_ROOT / "wham",
    },
]

PROCESS_ONLY_SUBJECTS = None
OVERWRITE_EXISTING = True
SAVE_ROTATION_METADATA = True
MIN_STRIDE_FRAMES = 8

# H36M-17 indices
ROOT = 0
RIGHT_HIP = 1
RIGHT_KNEE = 2
RIGHT_ANKLE = 3
LEFT_HIP = 4
LEFT_KNEE = 5
LEFT_ANKLE = 6
SPINE = 7
NECK = 8
NOSE = 9
HEAD = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 14
# =======================================


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros(3, dtype=np.float32)
    return vector / norm


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
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError(
            f"{source_name} for {subject} must have shape (T, 17, 3), "
            f"but got {array.shape}."
        )
    return array.astype(np.float32, copy=False)


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


def estimate_body_up(stride: np.ndarray) -> np.ndarray:
    hip_center = 0.5 * (stride[:, RIGHT_HIP] + stride[:, LEFT_HIP])
    shoulder_center = 0.5 * (stride[:, LEFT_SHOULDER] + stride[:, RIGHT_SHOULDER])

    candidate_vectors = [
        stride[:, NECK] - stride[:, ROOT],
        stride[:, HEAD] - stride[:, ROOT],
        shoulder_center - hip_center,
        stride[:, HEAD] - hip_center,
    ]
    stacked = np.concatenate(candidate_vectors, axis=0)
    stacked = stacked[np.linalg.norm(stacked, axis=1) > 1e-6]
    if len(stacked) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    up = normalize_vector(np.mean(stacked, axis=0))
    if np.dot(up, np.mean(stride[:, HEAD] - stride[:, ROOT], axis=0)) < 0:
        up = -up
    return up


def remove_component(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = normalize_vector(axis)
    projection = np.sum(points * axis, axis=-1, keepdims=True)
    return points - projection * axis


def principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return normalize_vector(vh[0])


def estimate_facing_direction(stride: np.ndarray, up_axis: np.ndarray) -> np.ndarray:
    shoulder_center = 0.5 * (stride[:, LEFT_SHOULDER] + stride[:, RIGHT_SHOULDER])
    face_candidates = [
        stride[:, NOSE] - stride[:, ROOT],
        stride[:, HEAD] - shoulder_center,
        stride[:, NOSE] - shoulder_center,
    ]
    face = np.mean(np.concatenate(face_candidates, axis=0), axis=0)
    face = remove_component(face[None, :], up_axis)[0]
    return normalize_vector(face)


def estimate_walking_direction(stride: np.ndarray, up_axis: np.ndarray) -> np.ndarray:
    limb_tracks = np.concatenate(
        [
            stride[:, RIGHT_ANKLE],
            stride[:, LEFT_ANKLE],
            stride[:, RIGHT_KNEE],
            stride[:, LEFT_KNEE],
        ],
        axis=0,
    )
    horizontal_tracks = remove_component(limb_tracks, up_axis)
    direction = principal_axis(horizontal_tracks)
    if np.linalg.norm(direction) < 1e-6:
        direction = estimate_facing_direction(stride, up_axis)

    face_direction = estimate_facing_direction(stride, up_axis)
    if np.linalg.norm(face_direction) > 1e-6 and np.dot(direction, face_direction) < 0:
        direction = -direction

    if np.linalg.norm(direction) < 1e-6:
        raise ValueError("Failed to estimate walking direction.")
    return normalize_vector(direction)


def build_stride_rotation(stride: np.ndarray) -> tuple[np.ndarray, dict]:
    up_axis = estimate_body_up(stride)
    x_axis = estimate_walking_direction(stride, up_axis)
    x_axis = normalize_vector(remove_component(x_axis[None, :], up_axis)[0])
    if np.linalg.norm(x_axis) < 1e-6:
        raise ValueError("Walking direction collapsed after orthogonalization.")

    y_axis = normalize_vector(np.cross(up_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-6:
        raise ValueError("Failed to construct lateral axis.")
    z_axis = normalize_vector(np.cross(x_axis, y_axis))

    rotation_basis = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    metadata = {
        "x_axis_forward": x_axis.astype(np.float32),
        "y_axis_lateral": y_axis.astype(np.float32),
        "z_axis_up": z_axis.astype(np.float32),
        "rotation_basis": rotation_basis,
    }
    return rotation_basis, metadata


def rotate_stride_to_canonical(stride: np.ndarray) -> tuple[np.ndarray, dict]:
    if stride.shape[0] < MIN_STRIDE_FRAMES:
        raise ValueError(
            f"Stride is too short to estimate canonical axes reliably: {stride.shape[0]} frames."
        )

    rotation_basis, metadata = build_stride_rotation(stride)
    rotated = np.asarray(stride @ rotation_basis, dtype=np.float32)
    return rotated, metadata


def slice_and_rotate_sequence(
    sequence: np.ndarray,
    frame_ranges: list[tuple[int, int]],
) -> tuple[list[np.ndarray], list[dict]]:
    stride_list = []
    metadata_list = []
    for stride_idx, (start, end) in enumerate(frame_ranges):
        stride = np.asarray(sequence[start:end], dtype=np.float32)
        rotated_stride, metadata = rotate_stride_to_canonical(stride)
        metadata.update(
            {
                "stride_index": int(stride_idx),
                "start_frame": int(start),
                "end_frame": int(end),
                "num_frames": int(end - start),
            }
        )
        stride_list.append(rotated_stride)
        metadata_list.append(metadata)
    return stride_list, metadata_list


def process_source(source_config: dict, subjects: list[str]) -> None:
    source_name = source_config["name"]
    input_path = source_config["input_path"]
    output_dir = source_config["output_dir"]
    metadata_dir = output_dir / "_rotation_metadata"

    pose_dict = load_pickle(input_path)
    if not isinstance(pose_dict, dict):
        raise ValueError(f"{source_name} pose file must contain a dict, got {type(pose_dict).__name__}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_ROTATION_METADATA:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSource: {source_name}")
    print(f"Input: {input_path}")

    for subject in subjects:
        output_path = output_dir / f"{subject}.pkl"
        metadata_path = metadata_dir / f"{subject}.pkl"

        if not OVERWRITE_EXISTING and output_path.exists():
            print(f"Skipping {source_name} / {subject}: output already exists.")
            continue

        if subject not in pose_dict:
            raise KeyError(f"{subject} is missing from {source_name} poses.")

        sequence = validate_pose_array(pose_dict[subject], subject, source_name)
        frame_ranges = load_pickle(RIGHT_STRIDE_FRAME_NUMS_DIR / f"{subject}.pkl")
        validated_ranges = validate_frame_ranges(frame_ranges, sequence.shape[0], subject)

        stride_list, metadata_list = slice_and_rotate_sequence(sequence, validated_ranges)
        save_pickle(output_path, stride_list)
        if SAVE_ROTATION_METADATA:
            save_pickle(metadata_path, metadata_list)

        print(f"Saved {source_name} / {subject}: {len(stride_list)} strides")


def main() -> None:
    subjects = get_subjects_to_process(RIGHT_STRIDE_FRAME_NUMS_DIR)
    print(f"Subjects to process: {len(subjects)}")
    print(f"Output root: {OUTPUT_ROOT}")

    for source_config in SOURCES:
        process_source(source_config, subjects)


if __name__ == "__main__":
    main()
