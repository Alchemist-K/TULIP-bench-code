"""
Restore original subject and activity names for feature extraction.

The monocular inference writes predictions under de-identified names:

    {MOTHER}/{ACTIVITY_SHORT}/{DeID_subject_name}/...

The feature extraction pipeline expects the original naming:

    {MOTHER}/for_feature_extraction/{ACTIVITY}/{original_subject_name}/...

This copies each subject folder across, translating both the activity directory
(FistL -> making_a_fist_Left, FistR -> making_a_fist_Right) and the subject name
via the matching table. The source tree is left untouched.

Set the paths in the configuration block and run:

    python restore_subject_names.py
"""

import os
import pickle
import shutil

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a passthrough
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# CONFIGURATION
# =============================================================================

# Root holding the DeID-named activity folders (the inference OUT_ROOT, or a
# method subtree inside it, e.g. ".../monocular_hand/wilor").
MOTHER = "./outputs/monocular_hand/wilor"

# Matching table with the two name columns.
MATCH_CSV = "./data/Neurips2026_DeID_total_labels.csv"
COL_ORIGINAL = "original_subject_name"
COL_DEID = "DeID_subject_name"

# Where the renamed copies are written, relative to MOTHER.
OUTPUT_SUBDIR = "for_feature_extraction"

# Short activity directory -> full activity directory.
ACTIVITY_MAP = {
    "FistL": "making_a_fist_Left",
    "FistR": "making_a_fist_Right",
}

# Overwrite a destination folder if it already exists. When False, existing
# destinations are skipped so an interrupted run can resume.
OVERWRITE = False

# After copying, verify each destination against its source and check for
# collisions and duplicate mappings.
SANITY_CHECK = True

# What to write at the destination:
#   "copy"    reproduce the source folder unchanged (full dict pickles)
#   "array"   for each pose pickle, save only the bare pose3d numpy array, so
#             the feature extractor can load it directly with pickle.load
# In "array" mode, non-pose files are copied across untouched.
OUTPUT_MODE = "array"

# The key holding the pose array inside each dict pickle.
POSE_KEY = "pose3d"

# Only files whose name contains this substring are treated as pose pickles in
# "array" mode. Everything else is copied verbatim.
POSE_FILENAME_HINT = "pose3d"


# =============================================================================
# Helpers
# =============================================================================

def require(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            f"Check the configuration block at the top of this file."
        )
    return path


def load_deid_to_original():
    """Return {DeID_subject_name: original_subject_name} from the matching table."""
    require(MATCH_CSV, "subject matching table")
    table = pd.read_csv(MATCH_CSV)

    for column in (COL_ORIGINAL, COL_DEID):
        if column not in table.columns:
            raise KeyError(
                f"column '{column}' missing from {MATCH_CSV}; "
                f"found: {list(table.columns)}"
            )

    paired = table[[COL_ORIGINAL, COL_DEID]].dropna()
    mapping = {}
    for _, row in paired.iterrows():
        deid = str(row[COL_DEID]).strip()
        original = str(row[COL_ORIGINAL]).strip()
        if deid and original:
            mapping[deid] = original

    if not mapping:
        raise ValueError(f"no paired names found in {MATCH_CSV}")
    return mapping


def extract_pose_array(obj):
    """Return the bare pose array from a loaded pickle, or None if absent.

    Accepts either a dict carrying POSE_KEY (the run_monocular_hand output) or an
    array that is already bare (so re-running is idempotent).
    """
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict) and POSE_KEY in obj:
        return np.asarray(obj[POSE_KEY])
    return None


def is_pose_file(name):
    return name.endswith(".pkl") and POSE_FILENAME_HINT in name


def write_folder_as_arrays(source, destination):
    """Copy a folder, replacing each pose pickle with its bare pose array.

    Pose pickles are rewritten to contain only the numpy array; every other file
    is copied byte for byte. Returns (n_pose_written, n_files_copied).
    """
    n_pose = n_copied = 0
    for directory, _, files in os.walk(source):
        rel_dir = os.path.relpath(directory, source)
        out_dir = os.path.join(destination, rel_dir) if rel_dir != "." else destination
        os.makedirs(out_dir, exist_ok=True)

        for name in files:
            src_path = os.path.join(directory, name)
            dst_path = os.path.join(out_dir, name)

            if is_pose_file(name):
                with open(src_path, "rb") as f:
                    obj = pickle.load(f)
                array = extract_pose_array(obj)
                if array is None:
                    raise ValueError(
                        f"no '{POSE_KEY}' array found in {src_path}; its keys are "
                        f"{list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
                    )
                with open(dst_path, "wb") as f:
                    pickle.dump(np.asarray(array), f)
                n_pose += 1
            else:
                shutil.copy2(src_path, dst_path)
                n_copied += 1
    return n_pose, n_copied


def copy_folder(source, destination):
    if os.path.exists(destination):
        if not OVERWRITE:
            return "skipped (exists)"
        shutil.rmtree(destination)

    if OUTPUT_MODE == "array":
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        n_pose, _ = write_folder_as_arrays(source, destination)
        if n_pose == 0:
            return "copied (no pose file found)"
        return "copied (arrays)"

    if OUTPUT_MODE == "copy":
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copytree(source, destination)
        return "copied"

    raise ValueError(f"unknown OUTPUT_MODE: {OUTPUT_MODE!r} (use 'array' or 'copy')")


def relative_file_index(root):
    """Map each file under root to its byte size, keyed by relative path."""
    index = {}
    for directory, _, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            index[os.path.relpath(path, root)] = os.path.getsize(path)
    return index


def compare_folders(source, destination):
    """Return a list of discrepancies between source and destination.

    In "array" mode, pose pickles are expected to differ on disk (dict vs bare
    array), so each is verified by loading both sides and checking that the
    destination array equals the source's pose array. Non-pose files, and all
    files in "copy" mode, are checked by presence and byte size.
    """
    if not os.path.isdir(destination):
        return [f"destination missing: {destination}"]

    problems = []
    src_files, dst_files = set(), set()

    for directory, _, files in os.walk(source):
        for name in files:
            src_files.add(os.path.relpath(os.path.join(directory, name), source))
    for directory, _, files in os.walk(destination):
        for name in files:
            dst_files.add(os.path.relpath(os.path.join(directory, name), destination))

    only_src = sorted(src_files - dst_files)
    only_dst = sorted(dst_files - src_files)
    if only_src:
        problems.append(f"{len(only_src)} file(s) not copied, e.g. {only_src[0]}")
    if only_dst:
        problems.append(f"{len(only_dst)} extra file(s) in destination, e.g. {only_dst[0]}")

    for rel in sorted(src_files & dst_files):
        src_path = os.path.join(source, rel)
        dst_path = os.path.join(destination, rel)
        name = os.path.basename(rel)

        if OUTPUT_MODE == "array" and is_pose_file(name):
            try:
                with open(src_path, "rb") as f:
                    src_array = extract_pose_array(pickle.load(f))
                with open(dst_path, "rb") as f:
                    dst_obj = pickle.load(f)
            except Exception as exc:
                problems.append(f"could not verify {rel}: {exc}")
                continue

            if not isinstance(dst_obj, np.ndarray):
                problems.append(f"{rel}: destination is not a bare array "
                                f"({type(dst_obj).__name__})")
            elif src_array is None:
                problems.append(f"{rel}: no pose array in source to compare")
            elif dst_obj.shape != src_array.shape:
                problems.append(f"{rel}: shape {dst_obj.shape} vs source "
                                f"{src_array.shape}")
            elif not np.array_equal(np.asarray(dst_obj), np.asarray(src_array)):
                problems.append(f"{rel}: array values differ from source")
        else:
            if os.path.getsize(src_path) != os.path.getsize(dst_path):
                problems.append(f"size mismatch on {rel}: "
                                f"{os.path.getsize(src_path)} vs "
                                f"{os.path.getsize(dst_path)} bytes")

    return problems


# =============================================================================
# Entry point
# =============================================================================

def preflight(name_map):
    """Catch mapping problems before any copying begins.

    A duplicate DeID key means the table is malformed; two DeID names mapping to
    the same original name means distinct source folders would collide at the
    same destination and silently overwrite each other.
    """
    problems = []

    reverse = {}
    for deid, original in name_map.items():
        reverse.setdefault(original, []).append(deid)
    for original, deids in reverse.items():
        if len(deids) > 1:
            problems.append(
                f"original name '{original}' is mapped from multiple DeID names "
                f"{sorted(deids)}; their folders would collide at one destination"
            )
    return problems


def main():
    require(MOTHER, "prediction root (MOTHER)")
    name_map = load_deid_to_original()
    output_root = os.path.join(MOTHER, OUTPUT_SUBDIR)
    print(f"matching table: {len(name_map)} subjects")
    print(f"output mode: {OUTPUT_MODE}"
          + ("  (pose pickles -> bare arrays)" if OUTPUT_MODE == "array" else ""))
    print(f"writing to: {output_root}\n")

    for problem in preflight(name_map):
        print(f"WARNING: {problem}")

    copied = skipped = 0
    missing_activities = []
    unmatched_subjects = []
    copied_pairs = []          # (source, destination) for the sanity check

    for activity_short, activity_full in ACTIVITY_MAP.items():
        source_activity = os.path.join(MOTHER, activity_short)
        if not os.path.isdir(source_activity):
            missing_activities.append(activity_short)
            continue

        dest_activity = os.path.join(output_root, activity_full)
        deid_subjects = sorted(
            d for d in os.listdir(source_activity)
            if os.path.isdir(os.path.join(source_activity, d))
        )

        for deid_name in tqdm(deid_subjects,
                              desc=f"{activity_short} -> {activity_full}",
                              unit="subj"):
            original_name = name_map.get(deid_name)
            if original_name is None:
                unmatched_subjects.append((activity_short, deid_name))
                continue

            source = os.path.join(source_activity, deid_name)
            destination = os.path.join(dest_activity, original_name)
            result = copy_folder(source, destination)
            if result.startswith("skipped"):
                skipped += 1
            else:
                copied += 1
                if result == "copied (no pose file found)":
                    print(f"    note: no pose file in {deid_name} "
                          f"(only non-pose files copied)")
            # Verify every destination that exists, whether freshly copied or
            # already present from an earlier run.
            copied_pairs.append((source, destination))

    print(f"\ncopied {copied}, skipped {skipped}")

    if missing_activities:
        print(f"activity folders not found under MOTHER: {missing_activities}")
    if unmatched_subjects:
        print(f"\n{len(unmatched_subjects)} folders had no match in the table "
              f"and were not copied:")
        for activity_short, deid_name in unmatched_subjects:
            print(f"    {activity_short}/{deid_name}")

    if SANITY_CHECK and copied_pairs:
        print("\nsanity check: verifying copied folders...")
        failures = 0
        for source, destination in tqdm(copied_pairs, unit="folder"):
            problems = compare_folders(source, destination)
            if problems:
                failures += 1
                print(f"  MISMATCH {os.path.relpath(destination, output_root)}")
                for problem in problems:
                    print(f"      {problem}")
        if failures == 0:
            print(f"  all {len(copied_pairs)} folders match their sources "
                  f"(file names and sizes)")
        else:
            print(f"  {failures}/{len(copied_pairs)} folders had discrepancies "
                  f"(see above)")

    expected = len(copied_pairs) + len(unmatched_subjects)
    total_source = sum(
        len([d for d in os.listdir(os.path.join(MOTHER, a))
             if os.path.isdir(os.path.join(MOTHER, a, d))])
        for a in ACTIVITY_MAP if os.path.isdir(os.path.join(MOTHER, a))
    )
    if expected != total_source:
        print(f"\nWARNING: accounted for {expected} source folders but found "
              f"{total_source}; some were neither copied nor reported.")


if __name__ == "__main__":
    main()
