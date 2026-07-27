import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks


def cal_distance_2d(x1,x2):
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def cal_distance_2d_with_sign(x1,x2,direction):
    # x2 is always hip
    if direction == 'negativeX':# based on camera2 view
        if x1[0]>x2[0]:
            return -np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        else:
            return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
    elif direction =='positiveX':
        if x1[0]>x2[0]:
            return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        else:
            return -np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
    elif direction == 'same': # x1 is hip and x2 is left heel
        if x1[0]>x2[0]:
            return -np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        else:
            return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)


# def cal_distance_2d_with_sign(x1,x2): # x1 is hip and x2 is left heel
#     if x1[0]>x2[0]:
#         return -np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
#     else:
#         return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def cal_projected_length(p1,p2,slope,intercept,direction='none'):
    slope_p1 = np.array([0,intercept])
    slope_p2 = np.array([2000,float(2000 * slope + intercept)])
    length = np.sum((slope_p1-slope_p2)**2)

    t1 = np.sum((p1 - slope_p1) * (slope_p2 - slope_p1)) / length
    projection_p1 = slope_p1 + t1 * (slope_p2 - slope_p1)

    t2 = np.sum((p2 - slope_p1) * (slope_p2 - slope_p1)) / length
    projection_p2 = slope_p1 + t2 * (slope_p2 - slope_p1)

    if direction == 'none':
        return cal_distance_2d(projection_p1, projection_p2)
    else:
        # return cal_distance_2d_with_sign(projection_p1, projection_p2,direction)
        return cal_distance_2d_with_sign(projection_p1, projection_p2,direction)

def normalize_walking_direction(stride_np_array):
    """
    Use linear regression on both heel trajectories to find the true walking
    direction, then rotate the entire sequence so that direction aligns with +X axis.

    Args:
        stride_np_array: np.ndarray of shape (frames, 33, 3)
                         Convention: Z is vertical, X and Y are ground plane
                         Joint 27 = left heel, 28 = right heel

    Returns:
        np.ndarray of shape (frames, 33, 3) with walking direction aligned to +X
    """

    # Extract both heel trajectories and combine: shape (2*frames, 3)
    left_heel  = stride_np_array[:, 27,  :]
    right_heel = stride_np_array[:, 28, :]
    heels      = np.concatenate([left_heel, right_heel], axis=0)  # (2*frames, 3)

    # Regression in XY ground plane (Z is vertical, so ignore it)
    t = np.concatenate([np.arange(len(left_heel)),
                        np.arange(len(right_heel))], axis=0)  # (2*frames,)

    x_coef = np.polyfit(t, heels[:, 0], 1)
    y_coef = np.polyfit(t, heels[:, 1], 1)

    # Walking direction vector in XY plane
    direction = np.array([x_coef[0], y_coef[0]])
    direction = direction / np.linalg.norm(direction)

    # Angle between walking direction and +X axis
    angle = np.arctan2(direction[1], direction[0])

    # Rotation around Z axis (vertical) by -angle to align walking dir to +X
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [    0,      0, 1]  # Z unchanged
    ])

    rotated = np.einsum('ij,fkj->fki', R_z, stride_np_array)

    # Sanity check: ensure walking direction is +X not -X
    hip_start = rotated[0,  0, 0]
    hip_end   = rotated[-1, 0, 0]
    if hip_end < hip_start:
        R_flip = np.array([[-1, 0, 0],
                           [ 0, -1, 0],
                           [ 0,  0, 1]])  # Z unchanged
        rotated = np.einsum('ij,fkj->fki', R_flip, rotated)

    return rotated


def necessary_rotate(stride_np_array): ## ?,33,3
    # detect the direction of the stride and rotate it if necessary
    # we can simply compare the x position of the hip
    hip_start = stride_np_array[0,0,0] # the x axis of the hip_start
    hip_end = stride_np_array[-1,0,0] # the x axis of the hip_end
    if hip_start > hip_end: # walking in the opposite direction of the x-axis
        # print('yes')
        R_z_180 = np.array([[-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]])

        stride_np_array = stride_np_array.transpose(2,0,1) # 160,26,3 -> 3,160,26


        # Step 1: Iterate over the frames and keypoints and apply the rotation
        rotated_data = np.einsum('ij,jfl->ifl', R_z_180, stride_np_array)
        # print(rotated_data.shape) #3,160,26

        stride_np_array = rotated_data.transpose(1,2,0) # 3,160,26 -> 160,26,3
    return stride_np_array

def move_to_center(stride_np_array): ## 160,26,3
    # temp_original_point = stride_np_array[0,19,:] # the hip point in the first frame, shape is (1,3)
    temp_original_point = (stride_np_array[0,23,:] + stride_np_array[0,24,:] ) / 2
    stride_np_array = stride_np_array - temp_original_point
    return stride_np_array


def normalize_time(input_array,target_length=100):

    num_frames, num_keypoints, num_dimensions = input_array.shape

    interpolated_data = np.zeros((target_length, num_keypoints, num_dimensions))

    original_indices = np.linspace(0, num_frames - 1, num=num_frames)
    target_indices = np.linspace(0, num_frames - 1, num=target_length)

    for kp in range(num_keypoints):
        for dim in range(num_dimensions):
            interpolated_data[:, kp, dim] = np.interp(
                target_indices, original_indices, input_array[:, kp, dim]
            )

    return interpolated_data


def step_width_from_line(p1, p2, slope):
    # distance-difference formula (b cancels anyway)
    num = slope*(p1[0] - p2[0]) - (p1[1] - p2[1])
    den = np.sqrt(slope**2 + 1)
    return abs(num) / den
