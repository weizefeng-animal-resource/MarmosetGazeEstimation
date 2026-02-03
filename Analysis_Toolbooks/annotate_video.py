import os

import cv2
import numpy as np
import pandas as pd
import pickle


def annotate_video(
        csv_file_name,
        marmoset_video_name,
        stimuli_video_name,
        save_file_name,
        frame_delay,
        accuracy_threshold):
    frame_delay = int(frame_delay)
    assert frame_delay >= 0
    print('processing ' + csv_file_name + ' ... ', end='')
    try:
        data_frame = pd.read_csv(os.path.join('DeepLabCutOutputs', csv_file_name), header=1)
        facial_landmark_coords = np.float32(data_frame[[
            'Amount', 'Amount.1', 'Nose', 'Nose.1',
            'Right corner of eye', 'Right corner of eye.1', 'Left corner of eye', 'Left corner of eye.1',
            'Mouth', 'Mouth.1', 'Right corner of mouth', 'Right corner of mouth.1',
            'Left corner of mouth', 'Left corner of mouth.1']][1:])
        facial_landmark_coords[:, 0::2] = facial_landmark_coords[:, 0::2] + 460
        facial_landmark_coords[:, 1::2] = facial_landmark_coords[:, 1::2] + 20
        facial_landmark_coords = np.int64(facial_landmark_coords)
        is_accurate = np.float32(data_frame[[
            'Amount.2', 'Nose.2', 'Right corner of eye.2', 'Left corner of eye.2',
            'Mouth.2', 'Right corner of mouth.2', 'Left corner of mouth.2']][1:])
        is_accurate = is_accurate > accuracy_threshold
    except Exception as e:
        raise Exception('failed, error message: ' + str(e))
    try:
        data_frame = pd.read_csv(os.path.join('CalibratedResults', csv_file_name), header=0)
        face_tvec = np.float32(data_frame[['face transfer x', 'face transfer y', 'face transfer z']])
        face_rvec = np.float32(data_frame[['face rotation alpha', 'face rotation beta', 'face rotation gamma']])
        gaze_points = np.float32(data_frame[['calibrated gaze point x (pixel)', 'calibrated gaze point y (pixel)']])
        gaze_points = np.int64(gaze_points)
    except Exception as e:
        raise Exception('failed, error message: ' + str(e))
    print('successful')
    print('processing ' + marmoset_video_name + ' ... ', end='')
    try:
        marmoset_video = cv2.VideoCapture(marmoset_video_name)
        assert marmoset_video.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
        assert marmoset_video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
        assert marmoset_video.get(cv2.CAP_PROP_FPS) == 25
        total_frame_number = int(marmoset_video.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception as e:
        raise Exception('failed, error message: ' + str(e))
    print('successful')
    print('processing ' + stimuli_video_name + ' ... ', end='')
    try:
        stimuli_video = cv2.VideoCapture(stimuli_video_name)
        assert stimuli_video.get(cv2.CAP_PROP_FRAME_WIDTH) == 1920
        assert stimuli_video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1080
        assert stimuli_video.get(cv2.CAP_PROP_FPS) == 25
        assert stimuli_video.get(cv2.CAP_PROP_FRAME_COUNT) == total_frame_number
    except Exception as e:
        raise Exception('failed, error message: ' + str(e))
    print('successful')
    if 'camera1.pickle' not in os.listdir():
        raise Exception('camera1.pickle is not found.')
    with open('camera1.pickle', 'rb') as f:
        _, camera_matrix, distortion_coefficients = pickle.load(f)
    save_file_path = os.path.join('AnnotatedVideos', save_file_name)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(save_file_path, fourcc, 25, (1280, 720))
    marmoset_video.set(cv2.CAP_PROP_POS_FRAMES, frame_delay)
    for i in range(frame_delay, total_frame_number):
        print(i)
        _, frame_marmoset = marmoset_video.read()
        _, frame_stimuli = stimuli_video.read()
        if is_accurate[i][0]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[0, 1]], 3, (0, 0, 255), -1)
        if is_accurate[i][1]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[2, 3]], 3, (0, 165, 255), -1)
        if is_accurate[i][2]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[4, 5]], 3, (0, 255, 255), -1)
        if is_accurate[i][3]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[6, 7]], 3, (0, 255, 0), -1)
        if is_accurate[i][4]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[8, 9]], 3, (255, 255, 0), -1)
        if is_accurate[i][5]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[10, 11]], 3, (255, 0, 0), -1)
        if is_accurate[i][6]:
            cv2.circle(frame_marmoset, facial_landmark_coords[i][[12, 13]], 3, (255, 0, 255), -1)
        if np.sum(is_accurate[i]) > 5:
            axes = np.float32([[0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -2]])
            axes, _ = cv2.projectPoints(axes, face_rvec[i], face_tvec[i], camera_matrix, distortion_coefficients)
            axes = np.reshape(axes, (len(axes), 2))
            cv2.line(frame_marmoset, np.int64(axes[0]), np.int64(axes[1]), (0, 0, 255), 3)
            cv2.line(frame_marmoset, np.int64(axes[0]), np.int64(axes[2]), (0, 255, 0), 3)
            cv2.line(frame_marmoset, np.int64(axes[0]), np.int64(axes[3]), (255, 0, 0), 3)
            cv2.circle(frame_stimuli, gaze_points[i], 15, (0, 0, 255), -1)
        frame_stimuli = cv2.resize(frame_stimuli, (480, 270))
        frame_marmoset[-270:, (1280 - 480) // 2:(1280 + 480) // 2] = frame_stimuli
        writer.write(frame_marmoset)
    marmoset_video.release()
    stimuli_video.release()
    writer.release()
