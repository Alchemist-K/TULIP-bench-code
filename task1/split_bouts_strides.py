'''
This script is trying to splitting whole gait sequence into each bouts and strides
The input shape should be (7200,33,3)
There should be two outputs
1. bouts dict, which will be used to calculate synchronzation
2. right stride dict, which will be used to calculate right stride length/time, right swing/stance ratio, double support ratio,
cadance, walking speed, step width, PCI, left/right step length/time
3. left stride dict, which will be used to calculate left stride length/time, left swing/stance ratio

For each subject, there are 3 dicts, which contains all usable bounts, left and right strides
'''


import os
import copy
import pickle
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, savgol_filter, find_peaks
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from stride_utils import cal_projected_length, necessary_rotate, move_to_center, normalize_time
import pywt

# =============================================================================
# CONFIGURATION -- edit for your environment
# =============================================================================
DATA_ROOT = "./data"

# Optional per-subject segmentation index overrides, keyed by de-identified
# subject id. Left empty by default; populate only if a subject's automatic
# bout detection needs manual correction.
SEGMENTATION_OVERRIDES = {}
# =============================================================================

# dbs_gait_poses_path = '{DATA_ROOT}'
gait_poses_path = os.path.join(DATA_ROOT, 'dataset/poses/neruips_3d_poses.pkl')
# dbs_gait_poses_path = '{DATA_ROOT}'
with open(gait_poses_path,'rb') as f:
    gait_poses = pickle.load(f)


# def smooth_sig(arr, window=31, polyorder=10):
#     arr = np.asarray(arr)

#     if window >= arr.shape[0]:
#         window = arr.shape[0] - 1
#         if window % 2 == 0:
#             window -= 1

#     smoothed = savgol_filter(arr, window_length=window, polyorder=polyorder, axis=0)
#     return smoothed

def smooth_sig(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, mode='smooth')
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-1]))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    filtered = pywt.waverec(coeffs, wavelet, mode='smooth')
    return filtered[:len(signal)]


for sub_name in gait_poses:

    os.makedirs("stride_sanity_check/{}".format(sub_name), exist_ok=True)

    #     continue

    example1 = gait_poses[sub_name] # shape 7200, 33, 3

    example1[:,:,0] = -example1[:,:,0]
    example1[:,:,2] = -example1[:,:,2]


    # split bouts ###################################################################
    left_hip_tracking = np.squeeze(example1[:,23,:])
    right_hip_tracking = np.squeeze(example1[:,24,:])
    mid_hip_tracking = (left_hip_tracking+right_hip_tracking) / 2 # shape is (7200,3)
    mid_hip_tracking = np.squeeze(mid_hip_tracking[:,0])

    # filter
    plt.plot(mid_hip_tracking)
    mid_hip_tracking = savgol_filter(np.array(mid_hip_tracking),101,2)
    plt.plot(mid_hip_tracking)

    # find peaks and valleys (turning points)
    peaks, properties = find_peaks(mid_hip_tracking, height=30, distance=500)
    valleys, properties = find_peaks(-mid_hip_tracking, height=30, distance=500)
    all_indices = np.sort(np.concatenate((peaks,valleys)))


    # Per-subject segmentation overrides, if any, come from a config
    # mapping rather than being hardcoded. Keys are subject ids in the
    # released (de-identified) naming.
    if sub_name in SEGMENTATION_OVERRIDES:
        all_indices = SEGMENTATION_OVERRIDES[sub_name]


#%%


    # save all bouts into one list
    temp_bout_sequence_list = []
    for indice_id, indice in enumerate(all_indices):
        if indice_id == 0:
            if indice <= 200:
                continue
            elif indice > 200:
                temp_duration_list = [0,indice]
        else:
            temp_duration_list = [all_indices[indice_id-1],indice]

        # print(temp_duration_list)
        temp_bout_sequence_list.append(example1[temp_duration_list[0]:temp_duration_list[1],:,:])
        # print(temp_duration_list)

    if (7200-indice) >= 200:
        temp_bout_sequence_list.append(example1[indice:7200,:,:])
        # print([indice,7200])

    # save the bouts list
    with open(os.path.join(DATA_ROOT, 'stride_splitting/bouts_strides/bouts/{}.pkl').format(sub_name),'wb') as f:
        pickle.dump(temp_bout_sequence_list,f)


    # split strides ###################################################################
    fps = 80
    peak_detection_distance = 40

    example_gait_event_dict = {}
    for bout_id, walking_period in enumerate(temp_bout_sequence_list):
        example_gait_event_dict[str(bout_id)] = {}
        example_gait_event_dict[str(bout_id)]['event_name'] = []

        # detect walking direction
        nose_start, nose_end = walking_period[0,0,0], walking_period[-1,0,0]
        if nose_start > nose_end:
            direction = 'negativeX'
        elif nose_start < nose_end:
            direction = 'positiveX'

        # calculate the walking direction using the ankle points
        real_walking = walking_period[fps:-fps,:,:]
        during_walking_leftankle = real_walking[:,27,:]
        during_walking_rightankle = real_walking[:,28,:]
        x_coordinates_heels = np.hstack((during_walking_leftankle[:,0],during_walking_rightankle[:,0]))
        y_coordinates_heels = np.hstack((during_walking_leftankle[:,1],during_walking_rightankle[:,1]))
        linear_model = LinearRegression().fit(x_coordinates_heels.reshape(-1,1),y_coordinates_heels)
        slope = linear_model.coef_
        intercept = linear_model.intercept_

        # calculate the differences between hip and left/right toe/heel
        xdiff_lefttoe_hip = []
        xdiff_righttoe_hip = []
        xdiff_leftheel_hip = []
        xdiff_rightheel_hip = []

        for frame_num in range(np.shape(walking_period)[0]):
            left_toe = walking_period[frame_num,31,:2]
            left_heel = walking_period[frame_num,29,:2]
            right_toe = walking_period[frame_num,32,:2]
            right_heel = walking_period[frame_num,30,:2]
            hip = (walking_period[frame_num,23,:2] + walking_period[frame_num,24,:2] ) / 2

            xdiff_lefttoe_hip.append(cal_projected_length(left_toe,hip,slope,intercept,direction))
            xdiff_leftheel_hip.append(cal_projected_length(left_heel,hip,slope,intercept,direction))
            xdiff_righttoe_hip.append(cal_projected_length(right_toe,hip,slope,intercept,direction))
            xdiff_rightheel_hip.append(cal_projected_length(right_heel,hip,slope,intercept,direction))


        xdiff_lefttoe_hip = np.array(xdiff_lefttoe_hip)
        xdiff_leftheel_hip = np.array(xdiff_leftheel_hip)
        xdiff_righttoe_hip = np.array(xdiff_righttoe_hip)
        xdiff_rightheel_hip = np.array(xdiff_rightheel_hip)

        #     xdiff_lefttoe_hip = smooth_sig(xdiff_lefttoe_hip)
        #     xdiff_leftheel_hip = smooth_sig(xdiff_leftheel_hip)
        #     xdiff_righttoe_hip = smooth_sig(xdiff_righttoe_hip)
        #     xdiff_rightheel_hip = smooth_sig(xdiff_rightheel_hip)


        left_heel_peaks, _ = find_peaks(xdiff_leftheel_hip, prominence=20,distance=peak_detection_distance)
        left_toe_valleys, _ = find_peaks(-xdiff_lefttoe_hip, prominence=20,distance=peak_detection_distance)
        right_heel_peaks, _ = find_peaks(xdiff_rightheel_hip, prominence=20,distance=peak_detection_distance)
        right_toe_valleys, _ = find_peaks(-xdiff_righttoe_hip, prominence=20,distance=peak_detection_distance)

        #     left_heel_peaks, _ = find_peaks(xdiff_leftheel_hip, prominence=20,distance=90)
        #     left_toe_valleys, _ = find_peaks(-xdiff_lefttoe_hip, prominence=20,distance=80)
        #     right_heel_peaks, _ = find_peaks(xdiff_rightheel_hip, prominence=20,distance=90)
        #     right_toe_valleys, _ = find_peaks(-xdiff_righttoe_hip, prominence=20,distance=peak_detection_distance)

        example_gait_event_dict[str(bout_id)]['event_timepoints'] = \
            np.sort(np.hstack((left_heel_peaks,left_toe_valleys,right_heel_peaks,right_toe_valleys)))
        for timepoint in example_gait_event_dict[str(bout_id)]['event_timepoints']:
            if timepoint in left_heel_peaks:
                example_gait_event_dict[str(bout_id)]['event_name'].append('l_heel_strike')
            elif timepoint in left_toe_valleys:
                example_gait_event_dict[str(bout_id)]['event_name'].append('l_toe_off')
            elif timepoint in right_heel_peaks:
                example_gait_event_dict[str(bout_id)]['event_name'].append('r_heel_strike')
            elif timepoint in right_toe_valleys:
                example_gait_event_dict[str(bout_id)]['event_name'].append('r_toe_off')

        example_gait_event_dict[str(bout_id)]['walking_period_data'] = walking_period


        # plt sanity check
        plt.title(bout_id)

        plt.plot(xdiff_leftheel_hip, c='blue')
        plt.plot(xdiff_lefttoe_hip)
        plt.plot(xdiff_rightheel_hip,c='green')
        plt.plot(xdiff_righttoe_hip)

        plt.scatter(left_heel_peaks,xdiff_leftheel_hip[left_heel_peaks],c='r')
        plt.scatter(left_toe_valleys,xdiff_lefttoe_hip[left_toe_valleys],c='r')
        plt.scatter(right_heel_peaks,xdiff_rightheel_hip[right_heel_peaks],c='r')
        plt.scatter(right_toe_valleys,xdiff_righttoe_hip[right_toe_valleys],c='r')

        plt.savefig('stride_sanity_check/{}/{}.jpg'.format(sub_name,bout_id))
        plt.clf()


    # Step3, split right strides and left strides and save all information into dict
    example_strides = {}
    delete_turning_around = 1

    for bout_count, walking_period_id in enumerate(example_gait_event_dict):
        example_strides[walking_period_id] = {}
        example_strides[walking_period_id]['right_stride'] = []
        example_strides[walking_period_id]['left_stride'] = []

        walking_period = example_gait_event_dict[walking_period_id]['walking_period_data'] # shape [n_frames,26,3]
        if delete_turning_around != 0:
            event_timepoints = example_gait_event_dict[walking_period_id]['event_timepoints'][delete_turning_around:-delete_turning_around]
            event_name = example_gait_event_dict[walking_period_id]['event_name'][delete_turning_around:-delete_turning_around]
        elif delete_turning_around == 0:
            event_timepoints = example_gait_event_dict[walking_period_id]['event_timepoints']
            event_name = example_gait_event_dict[walking_period_id]['event_name']


        for timepoint_id, event in enumerate(event_name):

            # event is r-contact ############################################################################
            if event == 'r_heel_strike':
                # r-contact to r-contact: r stride
                if timepoint_id+4 < len(event_name):
                    if event_name[timepoint_id+4] == 'r_heel_strike':
                        # test more on other 3 gait events
                        if event_name[timepoint_id+1]=='l_toe_off' and event_name[timepoint_id+2]=='l_heel_strike' \
                        and event_name[timepoint_id+3]=='r_toe_off':
                            example_strides[walking_period_id]['right_stride'].append(walking_period[event_timepoints[timepoint_id]:event_timepoints[timepoint_id+4],:,:])

            if event == 'l_heel_strike':
                # l-contact to l-contact: l stride
                if timepoint_id+4 < len(event_name):
                    if event_name[timepoint_id+4] == 'l_heel_strike':
                        # test more on other 3 gait events
                        if event_name[timepoint_id+1]=='r_toe_off' and event_name[timepoint_id+2]=='r_heel_strike' \
                        and event_name[timepoint_id+3]=='l_toe_off':
                            example_strides[walking_period_id]['left_stride'].append(walking_period[event_timepoints[timepoint_id]:event_timepoints[timepoint_id+4],:,:])


    #                     else:
    #                         sub_error += 1
    #                 else:
    #                     # print('{}: something wrong with walking period {}'.format(sub_name,walking_period_id))
    #                     sub_error += 1
    # print('{} has {} wrong gait events'.format(sub_name,sub_error))


    # Step4, some polishment and save both strides dicts
    example_strides1 = copy.deepcopy(example_strides)
    example_right_strides = []
    example_left_strides = []

    for bout_id in example_strides1:
        temp_right_strides = example_strides1[bout_id]['right_stride']
        for each_strides in temp_right_strides:
            this_strides = each_strides
            #rotate x&z
            # this_strides[:,:,0] = -this_strides[:,:,0]
            # this_strides[:,:,2] = -this_strides[:,:,2]

            this_strides = necessary_rotate(each_strides)
            this_strides = move_to_center(this_strides)
            # this_strides = normalize_time(this_strides) # normalize into 100
            example_right_strides.append(this_strides)

        temp_left_strides = example_strides1[bout_id]['left_stride']
        for each_strides in temp_left_strides:
            this_strides = each_strides
            #rotate x&z
            # this_strides[:,:,0] = -this_strides[:,:,0]
            # this_strides[:,:,2] = -this_strides[:,:,2]

            this_strides = necessary_rotate(each_strides)
            this_strides = move_to_center(this_strides)
            # this_strides = normalize_time(this_strides) # normalize into 100
            example_left_strides.append(this_strides)


    # save the strides list
    with open(os.path.join(DATA_ROOT, 'stride_splitting/bouts_strides/left_strides/{}.pkl').format(sub_name),'wb') as f:
        pickle.dump(example_left_strides,f)
    with open(os.path.join(DATA_ROOT, 'stride_splitting/bouts_strides/right_strides/{}.pkl').format(sub_name),'wb') as f:
        pickle.dump(example_right_strides,f)

    print('{}, {} right strides, {} left strides'.format(sub_name, len(example_right_strides),len(example_left_strides)))

    # Step5, some polishment and save both normalized strides np array
    example_strides2 = copy.deepcopy(example_strides)
    example_nor_right_strides = []
    example_nor_left_strides = []

    for bout_id in example_strides2:
        temp_right_strides = example_strides2[bout_id]['right_stride']
        for each_strides in temp_right_strides:
            this_strides = each_strides
            #rotate x&z
            # this_strides[:,:,0] = -this_strides[:,:,0]
            # this_strides[:,:,2] = -this_strides[:,:,2]

            this_strides = necessary_rotate(each_strides)
            this_strides = move_to_center(this_strides)
            this_strides = normalize_time(this_strides) # normalize into 100
            example_nor_right_strides.append(this_strides)

        temp_left_strides = example_strides2[bout_id]['left_stride']
        for each_strides in temp_left_strides:
            this_strides = each_strides
            #rotate x&z
            # this_strides[:,:,0] = -this_strides[:,:,0]
            # this_strides[:,:,2] = -this_strides[:,:,2]

            this_strides = necessary_rotate(each_strides)
            this_strides = move_to_center(this_strides)
            this_strides = normalize_time(this_strides) # normalize into 100
            example_nor_left_strides.append(this_strides)


    example_nor_right_strides = np.array(example_nor_right_strides)
    np.save(os.path.join(DATA_ROOT, 'stride_splitting/bouts_strides/nor_right_strides/{}.npy').format(sub_name),example_nor_right_strides)
    example_nor_left_strides = np.array(example_nor_left_strides)
    np.save(os.path.join(DATA_ROOT, 'stride_splitting/bouts_strides/nor_left_strides/{}.npy').format(sub_name),example_nor_left_strides)


######################## vis for saneity check ########################


# import copy
# import pickle
# import numpy as np
# from scipy.signal import find_peaks
# from scipy.signal import argrelextrema, savgol_filter, find_peaks
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from dbs_utils import cal_projected_length, necessary_rotate, move_to_center, normalize_time

# dbs_gait_poses_path = '{DATA_ROOT}'
# with open(dbs_gait_poses_path,'rb') as f:
#     dbs_gait_poses = pickle.load(f)


# example1 = dbs_gait_poses[sub_name] # shape 7200, 33, 3

# example1[:,:,0] = -example1[:,:,0]
# example1[:,:,2] = -example1[:,:,2]


# # split bouts ###################################################################
# left_hip_tracking = np.squeeze(example1[:,23,:])
# right_hip_tracking = np.squeeze(example1[:,24,:])
# mid_hip_tracking = (left_hip_tracking+right_hip_tracking) / 2 # shape is (7200,3)
# mid_hip_tracking = np.squeeze(mid_hip_tracking[:,0])

# # filter
# plt.plot(mid_hip_tracking)
# mid_hip_tracking = savgol_filter(np.array(mid_hip_tracking),101,2)
# plt.plot(mid_hip_tracking)

# # find peaks and valleys (turning points)
# peaks, properties = find_peaks(mid_hip_tracking, height=30, distance=500)
# valleys, properties = find_peaks(-mid_hip_tracking, height=30, distance=500)
# all_indices = np.sort(np.concatenate((peaks,valleys)))

# # save sanity check
# plt.plot(all_indices, mid_hip_tracking[all_indices], "xr")
## plt.clf()

# # save all bouts into one list
# temp_bout_sequence_list = []
# for indice_id, indice in enumerate(all_indices):
#     if indice_id == 0:
#         if indice <= 200:
#             continue
#         elif indice > 200:
#             temp_duration_list = [0,indice]
#     else:
#         temp_duration_list = [all_indices[indice_id-1],indice]

#     # print(temp_duration_list)
#     temp_bout_sequence_list.append(example1[temp_duration_list[0]:temp_duration_list[1],:,:])
#     print(temp_duration_list)

# if (7200-indice) >= 200:
#     temp_bout_sequence_list.append(example1[indice:7200,:,:])
#     print([indice,7200])

# # save the bouts list
# with open('bouts_strides/bouts/{}.pkl'.format(sub_name),'wb') as f:
#     pickle.dump(temp_bout_sequence_list,f)


# # split strides ###################################################################
# fps = 80
# peak_detection_distance = 40

# example_gait_event_dict = {}
# for bout_id, walking_period in enumerate(temp_bout_sequence_list):
#     example_gait_event_dict[str(bout_id)] = {}
#     example_gait_event_dict[str(bout_id)]['event_name'] = []

#     # detect walking direction
#     nose_start, nose_end = walking_period[0,0,0], walking_period[-1,0,0]
#     if nose_start > nose_end:
#         direction = 'negativeX'
#     elif nose_start < nose_end:
#         direction = 'positiveX'

#     # calculate the walking direction using the ankle points
#     real_walking = walking_period[fps:-fps,:,:]
#     during_walking_leftankle = real_walking[:,27,:]
#     during_walking_rightankle = real_walking[:,28,:]
#     x_coordinates_heels = np.hstack((during_walking_leftankle[:,0],during_walking_rightankle[:,0]))
#     y_coordinates_heels = np.hstack((during_walking_leftankle[:,1],during_walking_rightankle[:,1]))
#     linear_model = LinearRegression().fit(x_coordinates_heels.reshape(-1,1),y_coordinates_heels)
#     slope = linear_model.coef_
#     intercept = linear_model.intercept_

#     # calculate the differences between hip and left/right toe/heel
#     xdiff_lefttoe_hip = []
#     xdiff_righttoe_hip = []
#     xdiff_leftheel_hip = []
#     xdiff_rightheel_hip = []

#     for frame_num in range(np.shape(walking_period)[0]):
#         left_toe = walking_period[frame_num,31,:2]
#         left_heel = walking_period[frame_num,29,:2]
#         right_toe = walking_period[frame_num,32,:2]
#         right_heel = walking_period[frame_num,30,:2]
#         hip = (walking_period[frame_num,23,:2] + walking_period[frame_num,24,:2] ) / 2

#         xdiff_lefttoe_hip.append(cal_projected_length(left_toe,hip,slope,intercept,direction))
#         xdiff_leftheel_hip.append(cal_projected_length(left_heel,hip,slope,intercept,direction))
#         xdiff_righttoe_hip.append(cal_projected_length(right_toe,hip,slope,intercept,direction))
#         xdiff_rightheel_hip.append(cal_projected_length(right_heel,hip,slope,intercept,direction))


#     xdiff_lefttoe_hip = np.array(xdiff_lefttoe_hip)
#     xdiff_leftheel_hip = np.array(xdiff_leftheel_hip)
#     xdiff_righttoe_hip = np.array(xdiff_righttoe_hip)
#     xdiff_rightheel_hip = np.array(xdiff_rightheel_hip)


#     left_heel_peaks, _ = find_peaks(xdiff_leftheel_hip, prominence=20,distance=peak_detection_distance)
#     left_toe_valleys, _ = find_peaks(-xdiff_lefttoe_hip, prominence=20,distance=peak_detection_distance)
#     right_heel_peaks, _ = find_peaks(xdiff_rightheel_hip, prominence=20,distance=peak_detection_distance)
#     right_toe_valleys, _ = find_peaks(-xdiff_righttoe_hip, prominence=20,distance=peak_detection_distance)

#     example_gait_event_dict[str(bout_id)]['event_timepoints'] = \
#         np.sort(np.hstack((left_heel_peaks,left_toe_valleys,right_heel_peaks,right_toe_valleys)))
#     for timepoint in example_gait_event_dict[str(bout_id)]['event_timepoints']:
#         if timepoint in left_heel_peaks:
#             example_gait_event_dict[str(bout_id)]['event_name'].append('l_heel_strike')
#         elif timepoint in left_toe_valleys:
#             example_gait_event_dict[str(bout_id)]['event_name'].append('l_toe_off')
#         elif timepoint in right_heel_peaks:
#             example_gait_event_dict[str(bout_id)]['event_name'].append('r_heel_strike')
#         elif timepoint in right_toe_valleys:
#             example_gait_event_dict[str(bout_id)]['event_name'].append('r_toe_off')

#     example_gait_event_dict[str(bout_id)]['walking_period_data'] = walking_period


# # Step3, split right strides and left strides and save all information into dict
# example_strides = {}
# delete_turning_around = 1

# for bout_count, walking_period_id in enumerate(example_gait_event_dict):
#     example_strides[walking_period_id] = {}
#     example_strides[walking_period_id]['right_stride'] = []
#     example_strides[walking_period_id]['left_stride'] = []

#     walking_period = example_gait_event_dict[walking_period_id]['walking_period_data'] # shape [n_frames,26,3]
#     if delete_turning_around != 0:
#         event_timepoints = example_gait_event_dict[walking_period_id]['event_timepoints'][delete_turning_around:-delete_turning_around]
#         event_name = example_gait_event_dict[walking_period_id]['event_name'][delete_turning_around:-delete_turning_around]
#     elif delete_turning_around == 0:
#         event_timepoints = example_gait_event_dict[walking_period_id]['event_timepoints']
#         event_name = example_gait_event_dict[walking_period_id]['event_name']


#     for timepoint_id, event in enumerate(event_name):

#         # event is r-contact ############################################################################
#         if event == 'r_heel_strike':
#             # r-contact to r-contact: r stride
#             if timepoint_id+4 < len(event_name):
#                 if event_name[timepoint_id+4] == 'r_heel_strike':
#                     # test more on other 3 gait events
#                     if event_name[timepoint_id+1]=='l_toe_off' and event_name[timepoint_id+2]=='l_heel_strike' \
#                     and event_name[timepoint_id+3]=='r_toe_off':
#                         example_strides[walking_period_id]['right_stride'].append(walking_period[event_timepoints[timepoint_id]:event_timepoints[timepoint_id+4],:,:])

#         if event == 'l_heel_strike':
#             # l-contact to l-contact: l stride
#             if timepoint_id+4 < len(event_name):
#                 if event_name[timepoint_id+4] == 'l_heel_strike':
#                     # test more on other 3 gait events
#                     if event_name[timepoint_id+1]=='r_toe_off' and event_name[timepoint_id+2]=='r_heel_strike' \
#                     and event_name[timepoint_id+3]=='l_toe_off':
#                         example_strides[walking_period_id]['left_stride'].append(walking_period[event_timepoints[timepoint_id]:event_timepoints[timepoint_id+4],:,:])


# #                     else:
# #                         sub_error += 1
# #                 else:
# #                     # print('{}: something wrong with walking period {}'.format(sub_name,walking_period_id))
# #                     sub_error += 1
# # print('{} has {} wrong gait events'.format(sub_name,sub_error))


# # Step4, some polishment and save both strides dicts
# example_strides1 = copy.deepcopy(example_strides)
# example_right_strides = []
# example_left_strides = []

# for bout_id in example_strides1:
#     temp_right_strides = example_strides1[bout_id]['right_stride']
#     for each_strides in temp_right_strides:
#         this_strides = each_strides
#         #rotate x&z
#         # this_strides[:,:,0] = -this_strides[:,:,0]
#         # this_strides[:,:,2] = -this_strides[:,:,2]

#         this_strides = necessary_rotate(each_strides)
#         this_strides = move_to_center(this_strides)
#         # this_strides = normalize_time(this_strides) # normalize into 100
#         example_right_strides.append(this_strides)

#     temp_left_strides = example_strides1[bout_id]['left_stride']
#     for each_strides in temp_left_strides:
#         this_strides = each_strides
#         #rotate x&z
#         # this_strides[:,:,0] = -this_strides[:,:,0]
#         # this_strides[:,:,2] = -this_strides[:,:,2]

#         this_strides = necessary_rotate(each_strides)
#         this_strides = move_to_center(this_strides)
#         # this_strides = normalize_time(this_strides) # normalize into 100
#         example_left_strides.append(this_strides)


# # save the strides list
# with open('bouts_strides/left_strides/{}.pkl'.format(sub_name),'wb') as f:
#     pickle.dump(example_left_strides,f)
# with open('bouts_strides/right_strides/{}.pkl'.format(sub_name),'wb') as f:
#     pickle.dump(example_right_strides,f)

# print('{} right strides, {} left strides'.format(len(example_right_strides),len(example_left_strides)))

# # Step5, some polishment and save both normalized strides np array
# example_strides2 = copy.deepcopy(example_strides)
# example_nor_right_strides = []
# example_nor_left_strides = []

# for bout_id in example_strides2:
#     temp_right_strides = example_strides2[bout_id]['right_stride']
#     for each_strides in temp_right_strides:
#         this_strides = each_strides
#         #rotate x&z
#         # this_strides[:,:,0] = -this_strides[:,:,0]
#         # this_strides[:,:,2] = -this_strides[:,:,2]

#         this_strides = necessary_rotate(each_strides)
#         this_strides = move_to_center(this_strides)
#         this_strides = normalize_time(this_strides) # normalize into 100
#         example_nor_right_strides.append(this_strides)

#     temp_left_strides = example_strides2[bout_id]['left_stride']
#     for each_strides in temp_left_strides:
#         this_strides = each_strides
#         #rotate x&z
#         # this_strides[:,:,0] = -this_strides[:,:,0]
#         # this_strides[:,:,2] = -this_strides[:,:,2]

#         this_strides = necessary_rotate(each_strides)
#         this_strides = move_to_center(this_strides)
#         this_strides = normalize_time(this_strides) # normalize into 100
#         example_nor_left_strides.append(this_strides)


# example_nor_right_strides = np.array(example_nor_right_strides)
# np.save('bouts_strides/nor_right_strides/{}.npy'.format(sub_name),example_nor_right_strides)
# example_nor_left_strides = np.array(example_nor_left_strides)
# np.save('bouts_strides/nor_left_strides/{}.npy'.format(sub_name),example_nor_left_strides)


# ######################## vis for saneity check ########################
