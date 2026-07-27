"""
Monocular 3D hand pose inference for the fist activities.

Runs an off-the-shelf hand pose estimator over the fist videos and writes
canonicalised 3D poses in the layout used by the evaluation code.

Hand crops
    HaMeR and WiLoR expect a tight hand box. The supplied bounding boxes cover
    the whole person, which after resizing leaves the hand a few percent of the
    input, so they are not used directly for cropping. Boxes are instead derived
    per frame from the triangulated ground truth: the 21 ground-truth keypoints
    are projected into the evaluation camera with that subject's projection
    matrix, and the box is their extent plus padding.

    This mirrors the protocol already used for temporal segmentation, where all
    pose sources share timing derived from the triangulated ground truth so that
    pose quality is isolated from segmentation quality. Here every estimator
    receives an identical crop, so pose quality is isolated from detection
    quality.

    Projection needs ground truth carrying global position. The script checks
    this on load and falls back in order when it is unavailable:

        gt_projection      keypoints projected with the camera matrix (preferred)
        detector_fallback  hand located with MediaPipe inside the person box
        person_box_region  upper part of the person box, where a raised hand sits
        full_frame         whole frame, when nothing better exists

    The mode used is recorded per clip in the metadata, and the run summary
    reports the breakdown. Anything other than gt_projection is worth stating
    when the numbers are reported.

Inputs
    {ROOT}/Videos/{ACTIVITY}/{SUBJECT}/Camera{N}.mp4
    {ROOT}/Pose_3D/{ACTIVITY}/{SUBJECT}/pose3d_{bodypart}.pkl
    {ROOT}/Camera_parameters/{SUBJECT}_camera_parameters.pkl

    where ROOT = {NEURIPS_MOTHER}/Dataset1_Fist, ACTIVITY is FistL or FistR, and
    SUBJECT follows Neurips_Sub{n}. Camera parameters are a dictionary keyed
    "Camera_{n}" holding 3x4 projection matrices P = K [R|t].

    Each activity directory carries only the hand being examined, so no
    left/right disambiguation is needed beyond the activity name.

Output
    {OUT_ROOT}/{method}/{ACTIVITY}/{SUBJECT}/pose3d_canonical_{bodypart}.pkl

    One subtree per estimator, so METHODS can list several and each resumes
    independently. A method whose dependencies are missing is reported and
    skipped; the rest continue.

    ACTIVITY = FistL -> bodypart = lefthand
    ACTIVITY = FistR -> bodypart = righthand

Each output pickle holds:
    pose3d        (T, 21, 3) float32, millimetres, wrist-centred, MediaPipe order.
                  With FRAME_STRIDE > 1 and INTERPOLATE_TO_FULL_RATE, T matches
                  the full frame rate and rows between inferred frames are
                  interpolated, so downstream code needs no change.
    valid         (T,) bool, False where no pose could be established
    inferred      (T,) bool, True only where the model actually ran; the
                  complement of valid & inferred is interpolated
    frame_indices (int64) original video frame indices that were run through the
                  model, for auditing or for evaluating on inferred rows alone
    blur_frac     fraction of frames whose hand region appears de-identified
    meta          method, activity, subject, camera, frame alignment, stride,
                  inference and output rates, dt_seconds, crop mode

The script is restartable: completed clips are skipped, so an interrupted run
loses at most the clip in progress.
"""

import os
import re
import sys
import pickle
import time
import traceback

import numpy as np
import cv2


# =============================================================================
# CONFIGURATION
# =============================================================================

NEURIPS_MOTHER = "./data"
OUT_ROOT = "./outputs/monocular_hand"

FIST_ROOT = os.path.join(NEURIPS_MOTHER, "Dataset1_Fist")
VIDEO_ROOT = os.path.join(FIST_ROOT, "Videos")
POSE_ROOT = os.path.join(FIST_ROOT, "Pose_3D")
CAMERA_ROOT = os.path.join(FIST_ROOT, "Camera_parameters")

# Person boxes, used when ground-truth projection is unavailable. Leave as None
# to skip them.
BBOX_ROOT = None

# Bounding-box files follow a different naming scheme from the released subject
# directories, so a matching table maps between them. Set to None if the names
# already agree.
BBOX_MATCH_CSV = None
BBOX_COL_SOURCE = "original_subject_name"
BBOX_COL_NEURIPS = "DeID_subject_name"

# Activity directory names used inside BBOX_ROOT, keyed by released activity.
BBOX_ACTIVITY_MAP = {
    "FistL": "making_a_fist_Left",
    "FistR": "making_a_fist_Right",
}

# Where the bounding-box pickle sits, relative to BBOX_ROOT. Available fields:
# {activity} (mapped), {source} (source-scheme subject name).
BBOX_PATH_TEMPLATE = "{activity}/{source}/{source}.pkl"

# Enlarge a person box to a hand-sized crop when no finer localisation is
# available. A person box fed straight to HaMeR leaves the hand a few percent of
# the input, so the upper portion is taken, where a raised hand sits.
PERSON_BOX_HAND_REGION = (0.0, 0.0, 1.0, 0.55)   # left, top, right, bottom

# Estimators to run, in order. Each writes to its own output subtree, so a list
# can be left running unattended and individual methods resumed independently.
# See ESTIMATOR_REGISTRY below for what is available.
METHODS = ["hamer", "wilor", "hamba", "mediapipe"]

# Repository roots for estimators that are cloned rather than pip-installed.
# The path is prepended to sys.path before that estimator is imported, so a
# checkout sitting beside this script needs no installation.
REPO_PATHS = {
    "wilor": "./WiLoR",
    "hamba": "./Hamba",
    "meshgraphormer": "./MeshGraphormer",
}

# Checkpoints for estimators whose loader takes explicit paths. Relative paths
# resolve against that estimator's repository root.
CHECKPOINTS = {
    "wilor": {
        "checkpoint_path": "./pretrained_models/wilor_final.ckpt",
        "cfg_path": "./pretrained_models/model_config.yaml",
    },
}

# Rendering backend for packages that import pyrender at module load. These
# estimators only need the numeric head, but the import chain pulls in OpenGL,
# which fails on a headless machine unless a backend is chosen. "egl" uses the
# GPU; "osmesa" needs libosmesa6 installed.
PYOPENGL_PLATFORM = "egl"

# Hamba's VMamba backbone calls a compiled CUDA kernel, selective_scan_cuda_oflex.
# When that extension is absent, enabling this substitutes an equivalent PyTorch
# implementation so the model still runs. It is markedly slower because the state
# recurrence is evaluated stepwise, so compile the kernel when possible and treat
# this as the fallback (see the run guide).
HAMBA_TORCH_SELECTIVE_SCAN = True

# Front-facing camera approximating a mobile-phone viewpoint. Task 2 evaluates
# every monocular source on this single view.
CAMERA = 5

ACTIVITIES = ["FistL", "FistR"]

# Restrict to specific subject names (e.g. ["Neurips_Sub1"]) for a short
# validation run. Empty list means every subject present.
SUBJECT_SUBSET = []

# Timing. Videos are 80 fps; the first three seconds are the hand-raising phase
# and are dropped from both the prediction and the ground truth.
FPS = 80
SKIP_SECONDS = 3.0

# Crop geometry around the projected keypoints, as a fraction of the box side.
CROP_PAD = 0.35

# Temporal subsampling. Fist repetitions run at roughly 1-3 Hz, so every tenth
# frame at 80 fps leaves 8 Hz, which resolves the open-close cycle. Set to 1 to
# process every frame.
#
# Rate-dependent features (fingertip speed, jerk) must use the real interval
# between retained frames, dt = FRAME_STRIDE / FPS, and must be computed from
# ground truth sampled on the same frames. The retained indices are written to
# the output as "frame_indices" so the evaluation code can subset the ground
# truth identically.
FRAME_STRIDE = 10

# Resample predictions back to the full frame rate when FRAME_STRIDE > 1, so
# that outputs are always (n_aligned - skip, 21, 3) and downstream feature code
# runs unchanged with dt = 1/FPS.
#
# Interpolation reconstructs position faithfully for motion well below the
# sampling limit, but it cannot recover content above it. See the note on
# feature validity in the run guide before choosing a stride for feature-level
# results.
INTERPOLATE_TO_FULL_RATE = True

# Interpolation is not carried across long runs of failed detections. Gaps wider
# than this many original frames are left invalid instead of being invented.
MAX_INTERP_GAP_FRAMES = 40

# "cuda:0" or "cpu". On CPU, lower BATCH_SIZE and set CPU_THREADS below the
# core count to leave headroom.
DEVICE = "cuda:0"
BATCH_SIZE = 16
CPU_THREADS = 16

# De-identification check. Gaussian blurring removes high-frequency content, so
# a low Laplacian variance inside the hand box indicates a blurred hand.
BLUR_CHECK = True
BLUR_VAR_THRESHOLD = 40.0

CHECKPOINT_EVERY = 400


# =============================================================================
# Keypoint conventions
# =============================================================================

# MANO / HaMeR order:
#   0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb,
#   16-20 tips (index, middle, pinky, ring, thumb)
# MediaPipe Hand order (the ground-truth convention):
#   0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
MANO_TO_MEDIAPIPE = [
    0,
    13, 14, 15, 20,
    1, 2, 3, 16,
    4, 5, 6, 17,
    10, 11, 12, 19,
    7, 8, 9, 18,
]

ACTIVITY_TO_BODYPART = {
    "FistL": "lefthand",
    "FistR": "righthand",
}


# =============================================================================
# Helpers
# =============================================================================

def patch_numpy_aliases():
    """Restore the numpy scalar aliases that chumpy imports.

    chumpy does `from numpy import bool, int, float, complex, object, unicode,
    str`, aliases numpy removed in 1.24. Every MANO-based estimator imports
    chumpy through smplx, so without this the checkpoint load fails with
    "cannot import name 'int' from 'numpy'".

    Restoring the aliases is preferable to pinning numpy below 1.24, which would
    hold back the rest of the environment.
    """
    aliases = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "str": str, "unicode": str,
    }
    restored = [name for name, value in aliases.items()
                if not hasattr(np, name) and not setattr(np, name, value)]
    return restored


def configure_headless_rendering():
    """Select an OpenGL backend before any package imports pyrender."""
    os.environ.setdefault("PYOPENGL_PLATFORM", PYOPENGL_PLATFORM)


def add_repo_to_path(method):
    """Prepend a cloned estimator's repository root to sys.path."""
    root = REPO_PATHS.get(method)
    if not root:
        return None
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"repository for '{method}' not found at: {root}\n"
            f"Clone it there, or correct REPO_PATHS in the configuration block."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def resolve_checkpoints(method, root):
    """Absolute checkpoint paths for an estimator, relative to its repo root."""
    entries = CHECKPOINTS.get(method, {})
    resolved = {}
    for key, value in entries.items():
        path = value if os.path.isabs(value) else os.path.join(root or ".", value)
        path = os.path.abspath(path)
        require(path, f"{method} {key}")
        resolved[key] = path
    return resolved


def require(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            f"Check the configuration block at the top of this file."
        )
    return path


def list_subjects(activity):
    """Subjects present for an activity, ordered by their numeric suffix.

    Directory names are already de-identified (Neurips_Sub{n}), so no external
    mapping is required and every subject present is in the cohort.
    """
    activity_dir = os.path.join(VIDEO_ROOT, activity)
    require(activity_dir, f"video directory for {activity}")

    def index_of(name):
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 0

    subjects = sorted(
        (d for d in os.listdir(activity_dir)
         if os.path.isdir(os.path.join(activity_dir, d))),
        key=index_of,
    )
    if SUBJECT_SUBSET:
        subjects = [s for s in subjects if s in set(SUBJECT_SUBSET)]
    return subjects


def load_gt(activity, subject):
    """Load the ground-truth hand poses for this activity.

    Each activity directory holds only the hand under examination, so the file
    name follows directly from the activity. Returns (poses, is_global), where
    is_global reports whether the poses carry world position and can therefore
    be projected into the image to obtain hand boxes.
    """
    bodypart = ACTIVITY_TO_BODYPART[activity]
    path = os.path.join(POSE_ROOT, activity, subject, f"pose3d_{bodypart}.pkl")
    require(path, f"ground-truth pose for {subject} / {activity}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        for key in ("pose3d", "poses", "keypoints_3d", "joints"):
            if key in data:
                data = data[key]
                break
        else:
            raise KeyError(
                f"could not find a pose array in {path}; keys: {list(data.keys())}"
            )

    poses = np.asarray(data, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (21, 3):
        raise ValueError(f"expected (T, 21, 3) in {path}, got {poses.shape}")

    return poses, not is_wrist_centred(poses)


def is_wrist_centred(poses, tolerance_mm=1.0):
    """Ground truth stored wrist-centred cannot be projected into the image."""
    if poses.shape[0] == 0:
        return True
    return float(np.abs(poses[:, 0, :]).max()) < tolerance_mm


def load_projection(subject):
    """Return the 3x4 projection matrix for the evaluation camera, or None.

    Camera files are dictionaries keyed "Camera_{n}"; a few alternative spellings
    are accepted so that a minor naming change does not silently disable
    projection and push every clip onto the detector fallback.
    """
    path = os.path.join(CAMERA_ROOT, f"{subject}_camera_parameters.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        params = pickle.load(f)

    entry = None
    if isinstance(params, dict):
        for key in (f"Camera_{CAMERA}", f"Camera{CAMERA}",
                    f"camera_{CAMERA}", f"camera{CAMERA}", CAMERA, str(CAMERA)):
            if key in params:
                entry = params[key]
                break
    else:
        array = np.asarray(params)
        if array.ndim == 3 and array.shape[0] >= CAMERA:
            entry = array[CAMERA - 1]

    if entry is None:
        return None
    if isinstance(entry, dict):
        for key in ("P", "projection", "proj"):
            if key in entry:
                entry = entry[key]
                break

    matrix = np.asarray(entry, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :]
    if matrix.shape != (3, 4):
        return None
    return matrix


def project_points(points_mm, projection):
    """Project (N, 3) world points in millimetres to (N, 2) pixels."""
    homogeneous = np.concatenate(
        [points_mm.astype(np.float64), np.ones((points_mm.shape[0], 1))], axis=1
    )
    projected = homogeneous @ projection.T
    depth = projected[:, 2:3]
    depth[np.abs(depth) < 1e-9] = 1e-9
    return (projected[:, :2] / depth).astype(np.float32)


def box_from_points(points_2d, width, height, pad=CROP_PAD):
    """Square box covering the projected keypoints, padded and clipped."""
    finite = points_2d[np.isfinite(points_2d).all(axis=1)]
    if finite.shape[0] < 4:
        return None

    x1, y1 = finite.min(axis=0)
    x2, y2 = finite.max(axis=0)
    side = max(x2 - x1, y2 - y1) * (1.0 + pad)
    if side < 8:
        return None

    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    nx1 = int(max(0, cx - side * 0.5))
    ny1 = int(max(0, cy - side * 0.5))
    nx2 = int(min(width, cx + side * 0.5))
    ny2 = int(min(height, cy + side * 0.5))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return (nx1, ny1, nx2, ny2)


_BBOX_NAME_MAP = None


def load_bbox_name_map():
    """Map released subject names to the names used by the bounding-box files.

    Returns an empty dict when no matching table is configured, in which case
    the released name is used directly.
    """
    global _BBOX_NAME_MAP
    if _BBOX_NAME_MAP is not None:
        return _BBOX_NAME_MAP

    if not BBOX_MATCH_CSV:
        _BBOX_NAME_MAP = {}
        return _BBOX_NAME_MAP

    import pandas as pd

    require(BBOX_MATCH_CSV, "bounding-box subject matching table")
    table = pd.read_csv(BBOX_MATCH_CSV)

    for column in (BBOX_COL_SOURCE, BBOX_COL_NEURIPS):
        if column not in table.columns:
            raise KeyError(
                f"column '{column}' missing from {BBOX_MATCH_CSV}; "
                f"found: {list(table.columns)}"
            )

    paired = table[[BBOX_COL_SOURCE, BBOX_COL_NEURIPS]].dropna()
    _BBOX_NAME_MAP = {
        str(row[BBOX_COL_NEURIPS]).strip(): str(row[BBOX_COL_SOURCE]).strip()
        for _, row in paired.iterrows()
        if str(row[BBOX_COL_NEURIPS]).strip()
    }
    print(f"bounding-box name map: {len(_BBOX_NAME_MAP)} subjects")
    return _BBOX_NAME_MAP


def load_person_boxes(activity, subject):
    """Person boxes for this clip, as (T, 4) [x1, y1, x2, y2], or None."""
    if BBOX_ROOT is None:
        return None

    source = load_bbox_name_map().get(subject, subject)
    path = os.path.join(
        BBOX_ROOT,
        BBOX_PATH_TEMPLATE.format(
            activity=BBOX_ACTIVITY_MAP.get(activity, activity),
            source=source,
            subject=subject,
        ),
    )
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        boxes = pickle.load(f)

    if isinstance(boxes, dict):
        for key in (f"Camera_{CAMERA}", f"Camera{CAMERA}", CAMERA, str(CAMERA)):
            if key in boxes:
                boxes = boxes[key]
                break
    try:
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    except (ValueError, TypeError):
        return None


def hand_region_from_person_box(box, width, height):
    """Crop the part of a person box where a raised hand is expected.

    Cruder than a detected hand box, but far better than passing the whole
    person: it keeps the hand at a usable fraction of the model input.
    """
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None

    left, top, right, bottom = PERSON_BOX_HAND_REGION
    rx1, ry1 = x1 + bw * left, y1 + bh * top
    rx2, ry2 = x1 + bw * right, y1 + bh * bottom

    side = max(rx2 - rx1, ry2 - ry1)
    cx, cy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
    nx1 = int(max(0, cx - side * 0.5))
    ny1 = int(max(0, cy - side * 0.5))
    nx2 = int(min(width, cx + side * 0.5))
    ny2 = int(min(height, cy + side * 0.5))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return (nx1, ny1, nx2, ny2)


def crop(frame, box):
    x1, y1, x2, y2 = box
    patch = frame[int(y1):int(y2), int(x1):int(x2)]
    return patch if patch.size else None


def is_blur_affected(patch):
    if patch is None or patch.size == 0:
        return False
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var()) < BLUR_VAR_THRESHOLD


def canonicalise(joints_mm, is_left, keypoint_order="mano"):
    """Wrist-centre, remap to MediaPipe order, mirror left hands.

    Global translation is not comparable across monocular methods, so it is
    removed here rather than in the evaluation code, matching the ground-truth
    convention.
    """
    if joints_mm is None:
        return None
    joints = np.asarray(joints_mm, dtype=np.float32)
    if joints.shape[0] == 21 and keypoint_order == "mano":
        joints = joints[MANO_TO_MEDIAPIPE]
    joints = joints - joints[0:1]
    if is_left:
        joints = joints * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    return joints


# =============================================================================
# Hand localisation fallback
# =============================================================================

def open_mediapipe_hands(**kwargs):
    """Construct a MediaPipe Hands solution, or return None with a reason.

    The legacy solutions API moved across MediaPipe releases and is absent from
    some builds, so this never raises: hand localisation is an optional
    refinement and the run continues without it.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        return None, f"mediapipe not installed ({exc})"

    module_file = getattr(mp, "__file__", "unknown")

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        try:
            from mediapipe.python import solutions as solutions
        except ImportError:
            return None, (
                f"this mediapipe build exposes no solutions API "
                f"(loaded from {module_file}). Check for a local file or "
                f"directory named mediapipe shadowing the package, or install "
                f"a release that includes it: pip install 'mediapipe==0.10.14'"
            )

    hands_module = getattr(solutions, "hands", None)
    if hands_module is None:
        return None, f"mediapipe.solutions has no hands module ({module_file})"

    try:
        return hands_module.Hands(**kwargs), None
    except Exception as exc:
        return None, f"could not construct mediapipe Hands ({exc})"


class MediaPipeHandLocator:
    """Locates the hand when ground-truth projection is unavailable.

    Degrades rather than failing: if MediaPipe cannot be constructed, the
    locator falls back to the expected hand region of the person box, and to the
    whole frame when no box exists.
    """

    def __init__(self):
        self.hands, self.reason = open_mediapipe_hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3
        )
        if self.hands is None:
            print(f"  hand detector unavailable: {self.reason}")
            print("  falling back to person-box hand region")

    @property
    def available(self):
        return self.hands is not None

    def __call__(self, frame, person_box):
        height, width = frame.shape[:2]

        if self.hands is None:
            return hand_region_from_person_box(person_box, width, height)

        search = hand_region_from_person_box(person_box, width, height)
        region = crop(frame, search) if search is not None else frame
        if region is None or region.size == 0:
            region, search = frame, None

        try:
            result = self.hands.process(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        except Exception:
            return hand_region_from_person_box(person_box, width, height)

        if not result.multi_hand_landmarks:
            return hand_region_from_person_box(person_box, width, height)

        rh, rw = region.shape[:2]
        points = np.array(
            [[lm.x * rw, lm.y * rh] for lm in result.multi_hand_landmarks[0].landmark],
            dtype=np.float32,
        )
        if search is not None:
            points += np.array([search[0], search[1]], dtype=np.float32)
        box = box_from_points(points, width, height)
        return box if box is not None else hand_region_from_person_box(
            person_box, width, height
        )


# =============================================================================
# Estimators
# =============================================================================

def configure_torch_threads():
    """Cap CPU threads so a CPU run does not saturate the machine."""
    if not DEVICE.startswith("cpu"):
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(CPU_THREADS)
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
    print(f"CPU mode: torch limited to {CPU_THREADS} threads")


def interpolate_to_full_rate(poses, valid, source_positions, n_out):
    """Resample strided predictions onto every frame.

    poses / valid are indexed by inference slot; source_positions gives each
    slot's row in the full-rate output. Successfully inferred slots are
    interpolated with a cubic spline per coordinate (linear where too few points
    exist), and the result is clamped at the ends rather than extrapolated.

    Returns (poses_full, valid_full, inferred_full), where inferred_full marks
    the rows that came from the model rather than from interpolation.
    """
    from scipy.interpolate import interp1d

    poses_full = np.zeros((n_out, 21, 3), dtype=np.float32)
    valid_full = np.zeros(n_out, dtype=bool)
    inferred_full = np.zeros(n_out, dtype=bool)

    known = source_positions[valid]
    if known.size == 0:
        return poses_full, valid_full, inferred_full

    poses_full[known] = poses[valid]
    inferred_full[known] = True

    if known.size == 1:
        valid_full[known] = True
        return poses_full, valid_full, inferred_full

    targets = np.arange(n_out)
    kind = "cubic" if known.size >= 4 else "linear"
    flat = poses[valid].reshape(known.size, -1)

    interpolator = interp1d(
        known, flat, axis=0, kind=kind,
        bounds_error=False, fill_value=(flat[0], flat[-1]),
    )
    poses_full[:] = interpolator(targets).reshape(n_out, 21, 3).astype(np.float32)

    # Trust interpolation only inside the observed span and only across gaps
    # short enough that the motion cannot have changed direction unseen.
    valid_full[known.min():known.max() + 1] = True
    gaps = np.diff(known)
    for start, gap in zip(known[:-1], gaps):
        if gap > MAX_INTERP_GAP_FRAMES:
            valid_full[start + 1:start + gap] = False

    return poses_full, valid_full, inferred_full


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None,
                         delta_softplus=False):
    """PyTorch equivalent of the VMamba selective-scan CUDA kernel.

    Evaluates the selective state-space recurrence

        x_t = exp(delta_t * A) * x_{t-1} + delta_t * B_t * u_t
        y_t = <C_t, x_t> + D * u_t

    Shapes follow the kernel's convention: u and delta are (batch, dim, L),
    A is (dim, N), B and C are (batch, groups, N, L) with dim divisible by
    groups, and D and delta_bias are (dim,).

    The recurrence runs stepwise over L, so this is far slower than the compiled
    kernel. It exists so that a missing extension degrades performance rather
    than blocking the run.
    """
    import torch
    import torch.nn.functional as F

    dtype_in = u.dtype
    u, delta, A = u.float(), delta.float(), A.float()
    B, C = B.float(), C.float()

    if delta_bias is not None:
        delta = delta + delta_bias.float()[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, length = u.shape
    state = A.shape[1]

    # Broadcast the per-group B and C across the channels of each group.
    if B.dim() == 4:
        groups = B.shape[1]
        per_group = dim // groups
        B = B.unsqueeze(2).expand(batch, groups, per_group, state, length)
        B = B.reshape(batch, dim, state, length)
        C = C.unsqueeze(2).expand(batch, groups, per_group, state, length)
        C = C.reshape(batch, dim, state, length)

    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    x = u.new_zeros((batch, dim, state))
    outputs = []
    for step in range(length):
        x = delta_a[:, :, step] * x + delta_b_u[:, :, step]
        outputs.append(torch.einsum("bdn,bdn->bd", x, C[:, :, :, step]))

    y = torch.stack(outputs, dim=2)
    if D is not None:
        y = y + u * D.float().unsqueeze(-1)
    return y.to(dtype_in)


class _SelectiveScanShim:
    """Stands in for selective_scan_cuda_oflex, matching its fwd signature."""

    @staticmethod
    def fwd(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False,
            nrows=1, oflex=True):
        out = selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus)
        # The caller unpacks `out, x, *rest`; only `out` is used at inference.
        return out, None

    @staticmethod
    def bwd(*args, **kwargs):
        raise NotImplementedError(
            "the PyTorch selective-scan fallback is inference-only"
        )


def install_selective_scan_fallback():
    """Substitute the PyTorch scan wherever the CUDA extension is missing.

    Hamba's csms6s module imports the extension inside a try/except, so an
    absent kernel is only discovered at the first forward pass as a NameError.
    Binding the shim to the same module global keeps the call site unchanged.

    Returns True when a substitution was made.
    """
    try:
        from hamba.models.backbones.vmamba import csms6s
    except ImportError:
        return False

    if getattr(csms6s, "selective_scan_cuda_oflex", None) is not None:
        return False

    csms6s.selective_scan_cuda_oflex = _SelectiveScanShim
    for name in ("selective_scan_cuda_core", "selective_scan_cuda_ndstate",
                 "selective_scan_cuda"):
        if getattr(csms6s, name, None) is None:
            setattr(csms6s, name, _SelectiveScanShim)
    return True


class _HaMeR:
    """HaMeR. Pavlakos et al., Reconstructing Hands in 3D with Transformers,
    CVPR 2024. ViT-H backbone, MANO output in metres.

        git clone --recursive https://github.com/geopavlakos/hamer.git
    """

    keypoint_order = "mano"

    def __init__(self):
        import torch
        from hamer.models import load_hamer, DEFAULT_CHECKPOINT
        self.torch = torch
        original_load = torch.load
        torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": DEVICE})
        try:
            self.model, _ = load_hamer(DEFAULT_CHECKPOINT)
        finally:
            torch.load = original_load
        self.model = self.model.to(DEVICE).eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _prep(self, patch):
        img = cv2.resize(patch, (256, 256))[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return self.torch.from_numpy(img.transpose(2, 0, 1).copy())

    def __call__(self, patches, is_left):
        if not patches:
            return []
        torch = self.torch
        batch = torch.stack([self._prep(p) for p in patches]).to(DEVICE)
        flag = torch.full((len(patches),), 0.0 if is_left else 1.0, device=DEVICE)
        with torch.no_grad():
            out = self.model({"img": batch, "right": flag})
        return list(out["pred_keypoints_3d"].cpu().numpy() * 1000.0)


class _WiLoR:
    """WiLoR. Potamias et al., WiLoR: End-to-end 3D Hand Localization and
    Reconstruction in-the-wild, arXiv:2409.12259. Built for real-time use, so
    markedly cheaper than HaMeR at comparable accuracy. Same MANO topology.

        git clone https://github.com/rolpotamias/WiLoR.git

    No installation is required: set REPO_PATHS["wilor"] to the checkout and its
    root is added to sys.path before import.
    """

    keypoint_order = "mano"

    def __init__(self):
        import torch
        self.torch = torch

        root = add_repo_to_path("wilor")
        paths = resolve_checkpoints("wilor", root)

        original_load = torch.load
        torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": DEVICE})
        try:
            from wilor.models import load_wilor
            loaded = load_wilor(**paths)
        finally:
            torch.load = original_load

        # load_wilor returns (model, cfg).
        self.model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        self.model = self.model.to(DEVICE).eval()

    def _prep(self, patch):
        img = cv2.resize(patch, (256, 256))[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return self.torch.from_numpy(img.transpose(2, 0, 1).copy())

    def __call__(self, patches, is_left):
        if not patches:
            return []
        torch = self.torch
        batch = torch.stack([self._prep(p) for p in patches]).to(DEVICE)
        flag = torch.full((len(patches),), 0.0 if is_left else 1.0, device=DEVICE)
        with torch.no_grad():
            try:
                out = self.model({"img": batch, "right": flag})
            except (TypeError, KeyError):
                out = self.model(batch)
        for key in ("pred_keypoints_3d", "joints3d", "pred_joints"):
            if key in out:
                return list(out[key].cpu().numpy() * 1000.0)
        raise KeyError(f"no keypoint tensor in WiLoR output; keys: {list(out)}")


class _Hamba:
    """Hamba. Dong et al., Single-view 3D Hand Reconstruction with Graph-guided
    Bi-Scanning Mamba, NeurIPS 2024. State-space backbone rather than a
    transformer, so it probes a different architectural family.

        git clone https://github.com/humansensinglab/Hamba.git

    Requires mamba-ssm, which needs a CUDA toolchain matching torch.
    """

    keypoint_order = "mano"

    # Hamba crops 32 pixels from each side of the width before its backbone
    # (x[:, :, :, 32:-32]), so a 256x256 input becomes 256x192, giving
    # 16 x 12 = 192 patches at patch size 16 — the length its position
    # embedding expects. Feeding 224x224 yields 14 x 10 = 140 and fails.
    input_size = 256

    def __init__(self):
        import torch
        self.torch = torch

        # hamba.models imports a pyrender-backed renderer at module load, which
        # pulls in OpenGL and fails on a headless machine unless a backend is
        # selected first. Only the numeric head is used here.
        configure_headless_rendering()
        root = add_repo_to_path("hamba")
        paths = resolve_checkpoints("hamba", root)

        original_load = torch.load
        torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": DEVICE})
        try:
            from hamba.models import load_hamba
            loaded = load_hamba(**paths) if paths else load_hamba()
        finally:
            torch.load = original_load

        # The VMamba backbone calls a compiled CUDA kernel whose absence only
        # surfaces at the first forward pass, as a NameError inside csms6s.
        if HAMBA_TORCH_SELECTIVE_SCAN and install_selective_scan_fallback():
            print("  selective_scan CUDA kernel not found; using the PyTorch "
                  "fallback (slower)")

        self.model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        self.model = self.model.to(DEVICE).eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _prep(self, patch):
        size = self.input_size
        img = cv2.resize(patch, (size, size))[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return self.torch.from_numpy(img.transpose(2, 0, 1).copy())

    def __call__(self, patches, is_left):
        if not patches:
            return []
        torch = self.torch
        batch = torch.stack([self._prep(p) for p in patches]).to(DEVICE)
        with torch.no_grad():
            try:
                out = self.model({"img": batch})
            except NameError as exc:
                if "selective_scan" in str(exc):
                    raise RuntimeError(
                        "Hamba's VMamba backbone needs the selective_scan CUDA "
                        "extension, which is not built in this environment. "
                        "Either compile it (see the run guide) or set "
                        "HAMBA_TORCH_SELECTIVE_SCAN = True to use the PyTorch "
                        f"fallback. Original error: {exc}"
                    ) from exc
                raise
            except RuntimeError as exc:
                if "must match the size of tensor" in str(exc):
                    raise RuntimeError(
                        f"Hamba position-embedding mismatch at input_size="
                        f"{self.input_size}. The backbone crops 32 pixels from "
                        f"each side of the width, so the patch grid must match "
                        f"the checkpoint's embedding length. Adjust "
                        f"_Hamba.input_size (256 suits a 192-token embedding). "
                        f"Original error: {exc}"
                    ) from exc
                raise
        for key in ("pred_keypoints_3d", "joints3d", "pred_joints"):
            if key in out:
                return list(out[key].cpu().numpy() * 1000.0)
        raise KeyError(f"no keypoint tensor in Hamba output; keys: {list(out)}")


class _MeshGraphormer:
    """MeshGraphormer. Lin et al., ICCV 2021. Graph-convolution-reinforced
    transformer; a well-established pre-ViT reference point.

        git clone https://github.com/microsoft/MeshGraphormer.git
    """

    keypoint_order = "mano"

    def __init__(self):
        import torch
        configure_headless_rendering()
        add_repo_to_path("meshgraphormer")
        from src.modeling.bert import Graphormer_Hand_Network
        self.torch = torch
        self.model = Graphormer_Hand_Network().to(DEVICE).eval()

    def _prep(self, patch):
        img = cv2.resize(patch, (224, 224))[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return self.torch.from_numpy(img.transpose(2, 0, 1).copy())

    def __call__(self, patches, is_left):
        if not patches:
            return []
        torch = self.torch
        batch = torch.stack([self._prep(p) for p in patches]).to(DEVICE)
        with torch.no_grad():
            out = self.model(batch)
        joints = out[1] if isinstance(out, (tuple, list)) else out
        return list(joints.cpu().numpy() * 1000.0)


class _MediaPipe:
    """MediaPipe Hands world landmarks. CPU-native and effectively free.

    The triangulated ground truth is built from MediaPipe 2D detections, so this
    shares a front end with the ground-truth pipeline. Report it as a reference
    point, not as an independent monocular baseline.
    """

    keypoint_order = "mediapipe"

    def __init__(self):
        self.hands, reason = open_mediapipe_hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
        )
        if self.hands is None:
            raise RuntimeError(f"mediapipe estimator unavailable: {reason}")

    def __call__(self, patches, is_left):
        results = []
        for patch in patches:
            out = self.hands.process(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            if not out.multi_hand_world_landmarks:
                results.append(None)
                continue
            landmarks = out.multi_hand_world_landmarks[0].landmark
            results.append(
                np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32) * 1000.0
            )
        return results


ESTIMATOR_REGISTRY = {
    "hamer": _HaMeR,
    "wilor": _WiLoR,
    "hamba": _Hamba,
    "meshgraphormer": _MeshGraphormer,
    "mediapipe": _MediaPipe,
}


def build_estimator(method):
    configure_torch_threads()
    patch_numpy_aliases()
    if method not in ESTIMATOR_REGISTRY:
        raise ValueError(
            f"unknown method '{method}'; available: "
            f"{sorted(ESTIMATOR_REGISTRY)}"
        )
    return ESTIMATOR_REGISTRY[method]()


# =============================================================================
# Per-clip inference
# =============================================================================

def process_clip(estimator, get_locator, activity, subject, method):
    video_path = os.path.join(VIDEO_ROOT, activity, subject, f"Camera{CAMERA}.mp4")
    require(video_path, f"video for {subject} / {activity}")

    is_left = activity.endswith("Left")
    skip = int(round(SKIP_SECONDS * FPS))

    gt_poses, gt_is_global = load_gt(activity, subject)
    projection = load_projection(subject)

    capture = cv2.VideoCapture(video_path)
    n_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    n_gt = int(gt_poses.shape[0])

    # The ground truth may cover the untrimmed recording while the released
    # video is shorter. Both streams start at the same instant, so aligning on
    # the common prefix and dropping the same lead-in keeps them in step.
    n_common = min(n_video, n_gt)
    if n_common <= skip:
        capture.release()
        raise ValueError(
            f"{subject}/{activity}: only {n_common} aligned frames "
            f"(video {n_video}, ground truth {n_gt}), fewer than the "
            f"{skip}-frame hand-raising phase"
        )

    use_projection = (projection is not None) and gt_is_global
    person_boxes = None if use_projection else load_person_boxes(activity, subject)

    locator = None
    if use_projection:
        crop_mode = "gt_projection"
    else:
        locator = get_locator()
        if locator.available:
            crop_mode = "detector_fallback"
        elif person_boxes is not None:
            crop_mode = "person_box_region"
        else:
            crop_mode = "full_frame"

    # Frames actually processed, in original video indexing.
    frame_indices = np.arange(skip, n_common, FRAME_STRIDE, dtype=np.int64)
    slot_of = {int(f): i for i, f in enumerate(frame_indices)}

    poses = np.zeros((len(frame_indices), 21, 3), dtype=np.float32)
    valid = np.zeros(len(frame_indices), dtype=bool)
    blur_flags = []

    pending_patches, pending_slots = [], []

    def flush():
        if not pending_patches:
            return
        order = getattr(estimator, "keypoint_order", "mano")
        for slot, joints in zip(pending_slots, estimator(pending_patches, is_left)):
            canonical = canonicalise(joints, is_left, order)
            if canonical is not None:
                poses[slot] = canonical
                valid[slot] = True
        pending_patches.clear()
        pending_slots.clear()

    frame_index = 0
    while frame_index < n_common:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < skip or frame_index not in slot_of:
            frame_index += 1
            continue

        slot = slot_of[frame_index]
        height, width = frame.shape[:2]

        if use_projection:
            box = box_from_points(
                project_points(gt_poses[frame_index], projection), width, height
            )
        else:
            person_box = (
                person_boxes[frame_index]
                if person_boxes is not None and frame_index < len(person_boxes)
                else None
            )
            box = (
                locator(frame, person_box) if locator is not None
                else hand_region_from_person_box(person_box, width, height)
            )

        patch = crop(frame, box) if box is not None else None
        blur_flags.append(is_blur_affected(patch) if BLUR_CHECK else False)

        if patch is not None:
            pending_patches.append(patch)
            pending_slots.append(slot)
        if len(pending_patches) >= BATCH_SIZE:
            flush()

        frame_index += 1
        if frame_index % CHECKPOINT_EVERY == 0:
            print(f"    {frame_index}/{n_common}", flush=True)

    flush()
    capture.release()

    n_full = n_common - skip
    source_positions = frame_indices - skip

    if FRAME_STRIDE > 1 and INTERPOLATE_TO_FULL_RATE:
        poses_out, valid_out, inferred_out = interpolate_to_full_rate(
            poses, valid, source_positions, n_full
        )
        output_rate = "full"
    else:
        poses_out, valid_out = poses, valid
        inferred_out = valid.copy()
        output_rate = "strided" if FRAME_STRIDE > 1 else "full"

    return {
        "pose3d": poses_out,
        "valid": valid_out,
        "inferred": inferred_out,
        "frame_indices": frame_indices,
        "blur_frac": float(np.mean(blur_flags)) if blur_flags else 0.0,
        "meta": {
            "method": method,
            "activity": activity,
            "subject": subject,
            "camera": CAMERA,
            "fps": FPS,
            "frame_stride": FRAME_STRIDE,
            "inference_fps": FPS / float(FRAME_STRIDE),
            "output_rate": output_rate,
            "output_fps": FPS if output_rate == "full" else FPS / float(FRAME_STRIDE),
            "dt_seconds": (1.0 / FPS) if output_rate == "full"
                          else FRAME_STRIDE / float(FPS),
            "interpolated": bool(FRAME_STRIDE > 1 and INTERPOLATE_TO_FULL_RATE),
            "device": DEVICE,
            "crop_mode": crop_mode,
            "gt_is_global": bool(gt_is_global),
            "n_frames_video": n_video,
            "n_frames_gt": n_gt,
            "n_frames_aligned": int(n_common),
            "skipped_frames": skip,
            "frames_written": int(poses_out.shape[0]),
            "frames_inferred": int(inferred_out.sum()),
            "detection_rate": float(valid.mean()) if valid.size else 0.0,
        },
    }


# =============================================================================
# Entry point
# =============================================================================

def run_method(method, jobs, get_locator):
    """Run one estimator over every clip. Returns a per-clip summary."""
    print(f"\n{'=' * 62}\n{method}\n{'=' * 62}")

    pending = []
    for activity, subject in jobs:
        bodypart = ACTIVITY_TO_BODYPART[activity]
        out_dir = os.path.join(OUT_ROOT, method, activity, subject)
        target = os.path.join(out_dir, f"pose3d_canonical_{bodypart}.pkl")
        if not os.path.exists(target):
            pending.append((activity, subject, out_dir, target))

    done = len(jobs) - len(pending)
    if done:
        print(f"{done} clips already complete, {len(pending)} remaining")
    if not pending:
        print("nothing to do")
        return []

    try:
        estimator = build_estimator(method)
    except Exception as exc:
        # A missing optional dependency should not stop the other methods.
        print(f"could not initialise {method}: {exc}")
        traceback.print_exc()
        return []

    summary, started = [], time.time()
    for n, (activity, subject, out_dir, target) in enumerate(pending, start=1):
        print(f"[{n}/{len(pending)}] {activity}/{subject}")
        try:
            record = process_clip(estimator, get_locator, activity, subject, method)
        except FileNotFoundError:
            raise
        except Exception:
            print(f"    FAILED: {activity}/{subject}")
            traceback.print_exc()
            continue

        os.makedirs(out_dir, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(record, f)

        meta = record["meta"]
        summary.append((meta["detection_rate"], record["blur_frac"], meta["crop_mode"]))
        print(
            f"    frames={meta['frames_written']} "
            f"(video {meta['n_frames_video']}, gt {meta['n_frames_gt']}, "
            f"aligned {meta['n_frames_aligned']}, inferred {meta['frames_inferred']})  "
            f"crop={meta['crop_mode']}  "
            f"detected={meta['detection_rate']:.1%}  "
            f"blurred={record['blur_frac']:.1%}",
            flush=True,
        )

    elapsed = (time.time() - started) / 60.0
    print(f"\n{method}: {len(summary)} clips in {elapsed:.1f} min")
    return summary


def report(method, summary):
    if not summary:
        return
    detection = float(np.mean([s[0] for s in summary]))
    blurred = float(np.mean([s[1] for s in summary]))
    modes = {}
    for _, _, mode in summary:
        modes[mode] = modes.get(mode, 0) + 1

    print(f"  detection rate : {detection:.1%}")
    print(f"  blur-affected  : {blurred:.1%}")
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(modes.items()))
    print(f"  crop modes     : {breakdown}")

    fallback = sum(v for k, v in modes.items() if k != "gt_projection")
    if fallback:
        print(
            f"  {fallback}/{len(summary)} clips did not use ground-truth "
            f"projection; state this when reporting the numbers."
        )


def main():
    require(VIDEO_ROOT, "video root")
    require(POSE_ROOT, "ground-truth pose root")
    os.makedirs(OUT_ROOT, exist_ok=True)

    unknown = [m for m in METHODS if m not in ESTIMATOR_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown methods {unknown}; available: {sorted(ESTIMATOR_REGISTRY)}"
        )

    restored = patch_numpy_aliases()
    if restored:
        print(f"restored numpy aliases for chumpy: {', '.join(restored)}")
    configure_headless_rendering()

    print(f"methods={METHODS}  camera={CAMERA}  device={DEVICE}")
    print(f"stride={FRAME_STRIDE}  interpolate={INTERPOLATE_TO_FULL_RATE}")

    jobs = [(a, s) for a in ACTIVITIES for s in list_subjects(a)]
    for activity in ACTIVITIES:
        print(f"  {activity}: {len(list_subjects(activity))} subjects")
    print(f"{len(jobs)} clips per method")

    # Built once on first use and shared across methods; clips that project the
    # ground truth successfully never construct a detector at all.
    locator_cache = {}

    def get_locator():
        if "locator" not in locator_cache:
            locator_cache["locator"] = MediaPipeHandLocator()
        return locator_cache["locator"]

    results, started = {}, time.time()
    for method in METHODS:
        results[method] = run_method(method, jobs, get_locator)

    print(f"\n{'=' * 62}\nsummary ({(time.time() - started) / 60.0:.1f} min total)\n{'=' * 62}")
    for method in METHODS:
        summary = results.get(method) or []
        if not summary:
            print(f"\n{method}: no clips written")
            continue
        print(f"\n{method}: {len(summary)} clips")
        report(method, summary)

    blur = [s[1] for summary in results.values() for s in summary]
    if blur and float(np.mean(blur)) > 0.05:
        print(
            "\nDe-identification blur overlaps the hand in some clips. The "
            "distribution is typically bimodal, so identify the affected "
            "subjects and report a sensitivity analysis excluding them rather "
            "than a single global caveat."
        )


if __name__ == "__main__":
    main()
