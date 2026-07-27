"""
Dataset layout and shared constants for the TULIP-Bench release.

Edit DATA_ROOT to point at the extracted dataset. All scripts in this
repository read paths from this module; no command-line arguments are used.

Expected layout after extracting the released archives into DATA_ROOT:

    DATA_ROOT/
      Fist_Video/                     Face-blurred MP4, FistL and FistR,
                                      Cameras 1-6, 56 subjects, 30 s at 80 fps
      Fist_Pose_3D/                   Triangulated hand poses, (2400, 21, 3) mm,
                                      MediaPipe Hand 21-keypoint convention
      Fist_Camera_parameters/         Per-subject 3x4 projection matrices
                                      P = K @ [R|t], Cameras 1-6
      Fist_labels.csv                 MDS-UPDRS Item 3.5 (UPDRS_Fist_L, UPDRS_Fist_R)

      Gait_Videos/
        Gait/                         Observational cohort, 56 subjects
        Gait_DBS/                     DBS cohort, 10 subjects, 20 recordings
                                      (paired OFF/ON), Cameras 1, 2, 3, 6,
                                      90 s at 80 fps
      Gait_Pose_3D/                   Triangulated body poses, (7200, 33, 3) mm;
                                      the 17-keypoint subset used in the paper is
                                      selected via GAIT_KEYPOINT_SUBSET below
      Gait_Camera_parameters/         Per-subject 3x4 projection matrices, all six
                                      cameras stored (Cameras 4 and 5 are present
                                      in the pickles but their gait videos are not
                                      part of the release)
      Gait_normalPD_labels.csv        MDS-UPDRS gait scores, observational cohort
      Gait_DBS_labels.csv             MDS-UPDRS gait scores and body-part severity
                                      ratings, DBS cohort

Subject identifiers in the release are de-identified and take the form
Subject_1 ... Subject_56 for the observational cohort and Subject_1 ...
Subject_10 for the DBS cohort. The two cohorts are disjoint: identifiers are
scoped per cohort and do not refer to the same individuals.
"""

import os

# =============================================================================
# CONFIGURATION -- edit these paths for your environment
# =============================================================================
DATA_ROOT = "./data"
OUTPUT_ROOT = "./outputs"

# =============================================================================
# Released archives
# =============================================================================
FIST_VIDEO_DIR = os.path.join(DATA_ROOT, "Fist_Video")
FIST_POSE_DIR = os.path.join(DATA_ROOT, "Fist_Pose_3D")
FIST_CAMERA_DIR = os.path.join(DATA_ROOT, "Fist_Camera_parameters")
FIST_LABELS_CSV = os.path.join(DATA_ROOT, "Fist_labels.csv")

GAIT_VIDEO_DIR = os.path.join(DATA_ROOT, "Gait_Videos", "Gait")
GAIT_DBS_VIDEO_DIR = os.path.join(DATA_ROOT, "Gait_Videos", "Gait_DBS")
GAIT_POSE_DIR = os.path.join(DATA_ROOT, "Gait_Pose_3D")
GAIT_CAMERA_DIR = os.path.join(DATA_ROOT, "Gait_Camera_parameters")
GAIT_LABELS_CSV = os.path.join(DATA_ROOT, "Gait_normalPD_labels.csv")
GAIT_DBS_LABELS_CSV = os.path.join(DATA_ROOT, "Gait_DBS_labels.csv")

# =============================================================================
# Intermediate outputs produced by this pipeline
# =============================================================================
FIST_WORK_DIR = os.path.join(OUTPUT_ROOT, "fist")
GAIT_WORK_DIR = os.path.join(OUTPUT_ROOT, "gait")
DBS_WORK_DIR = os.path.join(OUTPUT_ROOT, "gait_dbs")

# =============================================================================
# Recording constants
# =============================================================================
FPS = 80
FIST_FRAMES = 2400              # 30 s at 80 fps
GAIT_FRAMES = 7200              # 90 s at 80 fps
FIST_RAISE_SECONDS = 3.0        # discarded hand-raising phase
HAND_KEYPOINTS = 21             # MediaPipe Hand convention
BODY_KEYPOINTS_RELEASED = 33    # MediaPipe Pose convention as released

# 17-keypoint subset used for the gait analyses in the paper. Thirteen joints map
# one-to-one from the released 33-keypoint skeleton; pelvis, thorax, spine, and
# head are derived geometrically (see the supplementary material).
GAIT_KEYPOINT_SUBSET = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
    "pelvis", "thorax", "spine", "head",
]

# Cameras with released video, by activity.
FIST_CAMERAS = [1, 2, 3, 4, 5, 6]
GAIT_CAMERAS = [1, 2, 3, 6]

# Reproducibility
SEED = 42
SEEDS = [42, 7, 123, 256, 2024]


def require(path, description):
    """Fail loudly and specifically when an expected input is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            f"Set DATA_ROOT in config.py to the directory containing the "
            f"extracted TULIP-Bench archives, or check that this archive has "
            f"been extracted."
        )
    return path


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# Subject naming
# =============================================================================
# Released subjects are de-identified as Neurips_Sub{n} for the observational
# cohort and Neurips_DBS_Sub{n} for the DBS cohort. All scripts derive subject
# ids from directory or pickle-key contents rather than embedding any names, so
# these prefixes are only used where an id must be constructed or displayed.
SUBJECT_PREFIX = "Neurips_Sub"
DBS_SUBJECT_PREFIX = "Neurips_DBS_Sub"

# For DBS OFF/ON recordings, the state is encoded in the recording id as one of
# these suffixes (e.g. Neurips_DBS_Sub1_OFF). Adjust only if the released files
# use a different convention.
DBS_STATE_SUFFIXES = {"OFF": "OFF", "ON": "ON"}
