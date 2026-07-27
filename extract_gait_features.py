#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


extract 20 features from each stride

All of them are right strides
1. stride length + time 2
2. left+right step length+time + assymetery 6
# 3. right swing ratio, stance ratio, left swing ratio, swing ratio assymetery 4
4. velocity and cadance 2
5. Hip/Knee/Elbow/shoulder + assymetery 12
6. arm-to-arm, leg-to-leg 2


Intotal, 28-4=24 features, calculate the box plot so that I can get the sense which feature show big differences


"""
import pickle
import os
import numpy as np

# ===== CONFIGURATION: edit for your environment =====
DATA_ROOT = "./data"

def remove_outlier_strides(features_dict):
    """
    Detect outlier strides based on stride length using IQR method,
    remove all feature values for those strides, and record which were removed.

    Args:
        features_dict: dict of feature_name -> list of per-stride values
                       (as stored in all_subs_features[sub])
    Returns:
        features_dict: cleaned dict with outlier strides removed
        outlier_indices: list of original stride indices that were removed
    """
    stride_lengths = np.array(features_dict['stride_length'])

    # IQR-based outlier detection on stride length
    Q1  = np.percentile(stride_lengths, 25)
    Q3  = np.percentile(stride_lengths, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_indices = [i for i, v in enumerate(stride_lengths)
                       if v < lower or v > upper]
    keep_indices    = [i for i in range(len(stride_lengths))
                       if i not in set(outlier_indices)]

    # Remove outlier strides from every feature
    for key in features_dict:
        features_dict[key] = [features_dict[key][i] for i in keep_indices]

    return features_dict, outlier_indices

def cal_bilateral_ROM(stride, joint_a1, joint_a2, joint_b1, joint_b2):
    """
    Calculate ROM of the angle between two limb vectors across a stride.

    Computes the angle between vector a1->a2 and vector b1->b2 at each frame,
    then returns the range (max - min) across the stride.

    Use cases:
        arm2arm: a1=LSHOULDER, a2=LWRIST, b1=RSHOULDER, b2=RWRIST
        leg2leg: a1=LHIP,      a2=LFOOT,  b1=RHIP,      b2=RFOOT

    Args:
        stride:   np.ndarray of shape (frames, J, 3)
        joint_a1: int, proximal joint of limb A
        joint_a2: int, distal joint of limb A
        joint_b1: int, proximal joint of limb B
        joint_b2: int, distal joint of limb B

    Returns:
        rom:          float, max - min angle across the stride (degrees)
        angle_series: np.ndarray of shape (frames,), per-frame angles
    """
    a1 = stride[:, joint_a1, :]  # (frames, 3)
    a2 = stride[:, joint_a2, :]
    b1 = stride[:, joint_b1, :]
    b2 = stride[:, joint_b2, :]

    # Limb vectors
    vec_a = a2 - a1  # (frames, 3)
    vec_b = b2 - b1  # (frames, 3)

    # Vectorized 3D angle between the two vectors
    dot   = np.sum(vec_a * vec_b, axis=1)                                          # (frames,)
    norm  = np.linalg.norm(vec_a, axis=1) * np.linalg.norm(vec_b, axis=1) + 1e-8  # (frames,)
    cos_ang      = np.clip(dot / norm, -1.0, 1.0)
    angle_series = np.degrees(np.arccos(cos_ang))                                  # (frames,)

    rom = np.max(angle_series) - np.min(angle_series)
    return rom, angle_series

def cal_shoulder_sagittal_ROM(stride, shoulder_joint, elbow_joint):
    """
    Calculate shoulder flexion/extension ROM in the sagittal plane.

    Shoulder flexion angle = signed angle between upper arm vector (shoulder->elbow)
    and the vertical Z axis.

    Sign convention:
        positive = elbow in FRONT of shoulder (flexion)
        negative = elbow BEHIND shoulder (extension)

    Args:
        stride:         np.ndarray of shape (frames, J, 3)
        shoulder_joint: int, shoulder joint index (RSHOULDER=14 or LSHOULDER=11)
        elbow_joint:    int, elbow joint index   (RELBOW=15   or LELBOW=12)

    Returns:
        rom:          float, max - min angle across the stride (degrees)
        angle_series: np.ndarray of shape (frames,), signed per-frame angles
    """
    shoulder = stride[:, shoulder_joint, :]  # (frames, 3)
    elbow    = stride[:, elbow_joint,    :]  # (frames, 3)

    # Upper arm vector in XZ plane (shoulder -> elbow)
    arm_x = elbow[:, 0] - shoulder[:, 0]  # (frames,)
    arm_z = elbow[:, 2] - shoulder[:, 2]  # (frames,)

    # Reference: vertical axis pointing downward [0, -1] in XZ
    ref_x = np.zeros(len(stride))
    ref_z = -np.ones(len(stride))

    # Unsigned angle between upper arm and vertical
    dot  = arm_x * ref_x + arm_z * ref_z
    norm = np.sqrt(arm_x**2 + arm_z**2) * np.sqrt(ref_x**2 + ref_z**2) + 1e-8
    cos_ang      = np.clip(dot / norm, -1.0, 1.0)
    angle_series = np.degrees(np.arccos(cos_ang))

    # Sign: positive if elbow is in front of shoulder (arm_x > 0 = flexion)
    #       negative if elbow is behind shoulder      (arm_x < 0 = extension)
    angle_series = np.where(arm_x >= 0, angle_series, -angle_series)

    rom = np.max(angle_series) - np.min(angle_series)
    return rom, angle_series


def cal_hip_sagittal_ROM(stride, hip_joint, knee_joint):
    """
    Calculate hip flexion/extension ROM in the sagittal plane.

    Hip flexion angle = signed angle between thigh vector (hip->knee)
    and the vertical Z axis.

    Sign convention:
        positive = leg in FRONT of vertical (flexion)
        negative = leg BEHIND vertical (extension)

    Args:
        stride:     np.ndarray of shape (frames, J, 3)
        hip_joint:  int, hip joint index (RHIP or LHIP)
        knee_joint: int, knee joint index (RKNEE or LKNEE)

    Returns:
        rom:          float, max - min angle across the stride (degrees)
        angle_series: np.ndarray of shape (frames,), signed per-frame angles
    """
    hip  = stride[:, hip_joint,  :]  # (frames, 3)
    knee = stride[:, knee_joint, :]  # (frames, 3)

    # Thigh vector in XZ plane (hip -> knee)
    thigh_x = knee[:, 0] - hip[:, 0]  # (frames,)
    thigh_z = knee[:, 2] - hip[:, 2]  # (frames,)

    # Z axis reference vector pointing downward (hip -> ground direction)
    # i.e. [0, -1] in XZ plane
    ref_x = np.zeros(len(stride))
    ref_z = -np.ones(len(stride))

    # Unsigned angle between thigh and vertical
    dot  = thigh_x * ref_x + thigh_z * ref_z                                        # (frames,)
    norm = np.sqrt(thigh_x**2 + thigh_z**2) * np.sqrt(ref_x**2 + ref_z**2) + 1e-8  # (frames,)
    cos_ang      = np.clip(dot / norm, -1.0, 1.0)
    angle_series = np.degrees(np.arccos(cos_ang))  # always positive so far

    # Apply sign: positive if knee is in front of hip (thigh_x > 0 = flexion)
    #             negative if knee is behind hip      (thigh_x < 0 = extension)
    angle_series = np.where(thigh_x >= 0, angle_series, -angle_series)

    rom = np.max(angle_series) - np.min(angle_series)
    return rom, angle_series


def cal_sagittal_ROM(stride, joint_a, joint_b, joint_c):
    """
    Calculate the sagittal plane ROM across a stride.
    Angle is measured at joint_b, formed by rays b->a and b->c.
    Sagittal plane = XZ (walking direction X, vertical Z).

    Args:
        stride:   np.ndarray of shape (frames, J, 3)
        joint_a:  int, proximal joint index
        joint_b:  int, vertex joint index (angle measured here)
        joint_c:  int, distal joint index

    Returns:
        rom:          float, max - min angle across the stride (degrees)
        angle_series: np.ndarray of shape (frames,), per-frame angles
    """
    a = stride[:, joint_a, :]  # (frames, 3)
    b = stride[:, joint_b, :]  # (frames, 3)
    c = stride[:, joint_c, :]  # (frames, 3)

    # XZ plane only (X = walking direction, Z = vertical)
    ba = np.stack([a[:, 0] - b[:, 0], a[:, 2] - b[:, 2]], axis=1)  # (frames, 2)
    bc = np.stack([c[:, 0] - b[:, 0], c[:, 2] - b[:, 2]], axis=1)  # (frames, 2)

    # Vectorized dot product and norms across all frames at once
    dot     = np.sum(ba * bc, axis=1)                                        # (frames,)
    norm    = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8 # (frames,)
    cos_ang = np.clip(dot / norm, -1.0, 1.0)                                 # (frames,)

    angle_series = np.degrees(np.arccos(cos_ang))                            # (frames,)
    rom = np.max(angle_series) - np.min(angle_series)

    return rom, angle_series


def cal_symmetry(number1, number2):
    denom = max(abs(number1), abs(number2))
    if denom < 1e-8:
        return 0.0                          # ← add this guard
    return abs(number1 - number2) / denom

def cal_stride_length(point1, point2):
    """
    Calculate stride length between two 3D points projected onto the X axis.

    Args:
        point1: array-like of shape (3,) — (x, y, z)
        point2: array-like of shape (3,) — (x, y, z)

    Returns:
        float: absolute distance along the X axis
    """
    p1 = np.asarray(point1)
    p2 = np.asarray(point2)
    return abs(p2[0] - p1[0])

def detect_right_stride(stride):
    hip_pos  = stride[:, 0,   :]
    lfoot_pos = stride[:, 6, :]
    dist = lfoot_pos[:,0] - hip_pos[:,0]
    heel_strike_frame = np.argmax(dist)


    return heel_strike_frame


# Subjects discovered from the stride directory (no identifiers in code).
sub_list = sorted(
    os.path.splitext(f)[0]
    for f in os.listdir(base_path)
    if f.endswith('.pkl')
)


base_path = os.path.join(DATA_ROOT, 'feature_extraction_pipeline/dbs_gait_features/right_strides_17_global')
all_subs_features = {}
outlier_log = {}

for sub in sub_list:

    temp_path = base_path + '/{}.pkl'.format(sub)
    all_subs_features[sub] = {
        'stride_length': [], 'stride_time': [],
        'lstep_length': [], 'rstep_length': [], 'step_length_asy': [],
        'lstep_time': [], 'rstep_time': [], 'step_time_asy': [],
        # 'rswing_ratio': [], 'rstance_ratio': [], 'lswing_ratio': [], 'swing_ratio_asy': [],
        'velocity': [], 'cadance': [],
        'lhip_ROM': [], 'rhip_ROM': [], 'hip_ROM_asy': [],
        'lknee_ROM': [], 'rknee_ROM': [], 'knee_ROM_asy': [],
        'lelbow_ROM': [], 'relbow_ROM': [], 'elbow_ROM_asy': [],
        'lshoulder_ROM': [], 'rshoulder_ROM': [], 'shoulder_ROM_asy': [],
        'arm2arm_ROM': [], 'leg2leg_ROM': []
        }
    with open (temp_path, 'rb') as f:
        temp_sub_stride_list = pickle.load(f)
    # break

    for stride in temp_sub_stride_list:
        # Should take the x axis as the walking direction since we use the bout to split the direction
        # right strides so the order is right heel-strike, left heel-strike, right heel-stride

        right_heel_strike_start = stride[0,3,:]
        right_heel_strike_end = stride[-1,3,:]

        stride_length = cal_stride_length(right_heel_strike_start,right_heel_strike_end)
        all_subs_features[sub]['stride_length'].append(stride_length)

        stride_time = len(stride)/80
        all_subs_features[sub]['stride_time'].append(stride_time)

        velocity = stride_length/stride_time
        all_subs_features[sub]['velocity'].append(velocity)

        cadance = (2 / stride_time) * 60
        all_subs_features[sub]['cadance'].append(cadance)

        left_heel_strike_frame = detect_right_stride(stride)
        left_heel_strike = stride[left_heel_strike_frame,6,:]

        lstep_length = abs(left_heel_strike[0] - right_heel_strike_start[0])
        all_subs_features[sub]['lstep_length'].append(lstep_length)
        all_subs_features[sub]['lstep_time'].append(left_heel_strike_frame/80)

        rstep_length = abs(right_heel_strike_end[0] - left_heel_strike[0])
        all_subs_features[sub]['rstep_length'].append(rstep_length)
        all_subs_features[sub]['rstep_time'].append((len(stride)-left_heel_strike_frame)/80)

        step_length_asy = cal_symmetry(lstep_length,rstep_length)
        step_time_asy = cal_symmetry(left_heel_strike_frame/80,(len(stride)-left_heel_strike_frame)/80)
        all_subs_features[sub]['step_length_asy'].append(step_length_asy)
        all_subs_features[sub]['step_time_asy'].append(step_time_asy)

        rhip_ROM,      rhip_series      = cal_hip_sagittal_ROM(stride,1,2)
        lhip_ROM,      lhip_series      = cal_hip_sagittal_ROM(stride,4,5)
        rknee_ROM,     rknee_series     = cal_sagittal_ROM(stride,1,2,3)
        lknee_ROM,     lknee_series     = cal_sagittal_ROM(stride,4,5,6)
        rshoulder_ROM, rshoulder_series = cal_shoulder_sagittal_ROM(stride, 14, 15)
        lshoulder_ROM, lshoulder_series = cal_shoulder_sagittal_ROM(stride, 11, 12)
        relbow_ROM,    relbow_series    = cal_sagittal_ROM(stride,14,15,16)
        lelbow_ROM,    lelbow_series    = cal_sagittal_ROM(stride,11,12,13)

        all_subs_features[sub]['rhip_ROM'].append(rhip_ROM)
        all_subs_features[sub]['lhip_ROM'].append(lhip_ROM)
        all_subs_features[sub]['hip_ROM_asy'].append(cal_symmetry(rhip_ROM, lhip_ROM))

        all_subs_features[sub]['rknee_ROM'].append(rknee_ROM)
        all_subs_features[sub]['lknee_ROM'].append(lknee_ROM)
        all_subs_features[sub]['knee_ROM_asy'].append(cal_symmetry(rknee_ROM, lknee_ROM))

        all_subs_features[sub]['rshoulder_ROM'].append(rshoulder_ROM)
        all_subs_features[sub]['lshoulder_ROM'].append(lshoulder_ROM)
        all_subs_features[sub]['shoulder_ROM_asy'].append(cal_symmetry(rshoulder_ROM, lshoulder_ROM))

        all_subs_features[sub]['relbow_ROM'].append(relbow_ROM)
        all_subs_features[sub]['lelbow_ROM'].append(lelbow_ROM)
        all_subs_features[sub]['elbow_ROM_asy'].append(cal_symmetry(relbow_ROM, lelbow_ROM))

        arm2arm_ROM, arm2arm_series = cal_bilateral_ROM(stride, 11, 13, 14, 16)  # L/R shoulder->wrist
        leg2leg_ROM, leg2leg_series = cal_bilateral_ROM(stride,  4,  6,  1,  3)  # L/R hip->foot
        all_subs_features[sub]['arm2arm_ROM'].append(arm2arm_ROM)
        all_subs_features[sub]['leg2leg_ROM'].append(leg2leg_ROM)
        # break


    # then inside the for sub loop, replace the print line with:
    all_subs_features[sub], outlier_idx = remove_outlier_strides(all_subs_features[sub])
    outlier_log[sub] = outlier_idx
    print(f'{sub}: removed {len(outlier_idx)} outlier strides at indices {outlier_idx}')
    #     break


for sub in sub_list:
    temp_feature_dict = all_subs_features[sub]
    print('{}: {} strides left'.format(sub,len(temp_feature_dict['stride_length'])))

with open(os.path.join(DATA_ROOT, 'feature_extraction_pipeline/dbs_gait_features/stride_features.pkl'),'wb') as f:
    pickle.dump(all_subs_features,f)
with open(os.path.join(DATA_ROOT, 'feature_extraction_pipeline/dbs_gait_features/outlier_dict.pkl'),'wb') as f:
    pickle.dump(outlier_log,f)
