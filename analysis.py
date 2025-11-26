import os

import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import pickle
from scipy.signal import medfilt

import lmfit


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


def calibrate_camera(dir_name, pattern_size=8, calibrate_distortion=False):
    camera_size = None
    img_list = []
    for file_name in os.listdir(dir_name):
        print('loading ' + file_name + ' ... ', end='')
        file_path = os.path.join(dir_name, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('failed.')
        else:
            print('successful.')
        if camera_size is None:
            camera_size = img.shape
        elif camera_size[0] != img.shape[0] or camera_size[1] != img.shape[1]:
            raise Exception('all images must be the same size.')
        img_list.append(img)
    img_points = []
    for i in range(len(img_list)):
        img = img_list[i]
        flag, corners = cv2.findChessboardCorners(img, (pattern_size, pattern_size))
        if not flag:
            print('warning: chessboard is not detected.')
            continue
        corners = corners.reshape(-1, 2)
        img_points.append(corners)
    if len(img_points) == 0:
        raise Exception('no chessboard is detected.')
    img_points = np.float32(img_points)
    obj_points = np.float32([[np.array([i, j, 0]) for i in range(pattern_size)] for j in range(pattern_size)])
    obj_points = obj_points.reshape(-1, 3)
    obj_points = np.float32([obj_points for _ in range(len(img_list))])
    _, camera_matrix, distortion_coefficients, _, _ = \
        cv2.calibrateCamera(obj_points, img_points, camera_size, None, None)
    if not calibrate_distortion:
        distortion_coefficients = np.zeros((5, 1))
    return camera_size, camera_matrix, distortion_coefficients


class CalibrateEstimator:
    def __init__(self):
        self.affine_transform = None

    def calibrate_gaze_points(self, gaze_points, affine_transform=None):
        if affine_transform is None:
            if self.affine_transform is None:
                raise Exception('estimator is not calibrated.')
            affine_transform = self.affine_transform
        gaze_points = np.hstack((gaze_points, np.ones((len(gaze_points), 1))))
        calibrated_gaze_points = np.einsum('jk,ik->ij', affine_transform, gaze_points)
        calibrated_gaze_points = calibrated_gaze_points[:, :-1]
        return calibrated_gaze_points

    def calc_loss(self, actual_points, gaze_points, a11, a12, a21, a22, tx, ty):
        affine_transform = np.array([[a11, a12, tx], [a21, a22, ty], [0, 0, 1]])
        calibrated_gaze_points = self.calibrate_gaze_points(gaze_points, affine_transform)
        return actual_points - calibrated_gaze_points

    def estimate_affine_matrix(self, actual_points, gaze_points):
        model = lmfit.Model(self.calc_loss, independent_vars=['actual_points', 'gaze_points'])
        parameters = model.make_params()
        default_affine_transform = {'a11': 1, 'a12': 0, 'a21': 0, 'a22': 1, 'tx': 0, 'ty': 0}
        for key in parameters.keys():
            parameters[key].set(value=default_affine_transform[key], vary=True, min=-np.inf, max=np.inf)
        result = model.fit(
            actual_points=actual_points,
            gaze_points=gaze_points,
            data=np.zeros(gaze_points.shape),
            params=parameters)
        self.affine_transform = np.float32([
            [result.values['a11'], result.values['a12'], result.values['tx']],
            [result.values['a21'], result.values['a22'], result.values['ty']],
            [0, 0, 1]])


def calibrate_estimator():
    data_frame = pd.read_csv('calibration_configs.csv', header=0)
    video_size = np.float32(data_frame[['video width (pixel)', 'video height (pixel)']])
    actual_points = np.float32(data_frame[['solid white circle x (pixel)', 'solid white circle y (pixel)']])
    actual_points = actual_points / video_size - 0.5
    file_name_list = data_frame['result csv file']
    frame_number_list = data_frame['frame number']
    gaze_points = []
    flags = []
    for i in range(len(file_name_list)):
        file_name = file_name_list[i]
        frame_number = frame_number_list[i]
        if file_name not in os.listdir('Results'):
            flags.append(False)
            print(file_name + ' not found.')
            continue
        data_frame = pd.read_csv(os.path.join('Results', file_name), header=0)
        gaze_point = np.float32(data_frame[['gaze point x', 'gaze point y']])[frame_number]
        if np.isnan(gaze_point).any():
            flags.append(False)
            print('gaze point is invalid at frame ' + str(frame_number) + ' of ' + file_name + '.')
            continue
        gaze_points.append(gaze_point)
        flags.append(True)
    if len(gaze_points) == 0:
        raise Exception('no gaze point is available.')
    gaze_points = np.array(gaze_points, dtype='float32')
    calibrator = CalibrateEstimator()
    calibrator.estimate_affine_matrix(actual_points[flags], gaze_points)
    return calibrator


def calibrate_results():
    try:
        data_frame = pd.read_csv('configs.csv', header=0)
        video_width = float(data_frame['video width (pixel)'][0])
        video_height = float(data_frame['video height (pixel)'][0])
    except Exception:
        raise Exception('configs.csv is not found or is invalid.')
    if 'calibrator.pickle' not in os.listdir():
        raise Exception('calibrator.pickle is not found.')
    with open('calibrator.pickle', 'rb') as f:
        calibrator = pickle.load(f)
    for file_name in os.listdir('Results'):
        print('processing ' + file_name + ' ... ', end='')
        try:
            data_frame = pd.read_csv(os.path.join('Results', file_name), header=0)
            gaze_points = np.float32(data_frame[['gaze point x', 'gaze point y']])
        except Exception as e:
            print('failed, error message: ' + str(e))
            continue
        calibrated_gaze_points = calibrator.calibrate_gaze_points(gaze_points)
        data_frame['calibrated gaze point x (pixel)'] = (calibrated_gaze_points[:, 0] + 0.5) * video_width
        data_frame['calibrated gaze point y (pixel)'] = (calibrated_gaze_points[:, 1] + 0.5) * video_height
        save_file_path = os.path.join('CalibratedResults', file_name)
        data_frame.to_csv(save_file_path, index=False)
        print('successful.')


def create_directories():
    os.makedirs('Camera1/', exist_ok=True)
    os.makedirs('Camera2/', exist_ok=True)
    os.makedirs('Setup/', exist_ok=True)
    os.makedirs('DeepLabCutOutputs/', exist_ok=True)
    os.makedirs('Videos/', exist_ok=True)
    os.makedirs('AnnotatedVideos/', exist_ok=True)
    os.makedirs('Results/', exist_ok=True)
    os.makedirs('CalibratedResults/', exist_ok=True)
    data_frame = pd.DataFrame([[1920, 1080]], columns=['video width (pixel)', 'video height (pixel)'])
    data_frame.to_csv('configs.csv', index=False)
    data_frame = pd.DataFrame(columns=[
        'result csv file', 'frame number', 'video width (pixel)', 'video height (pixel)',
        'solid white circle x (pixel)', 'solid white circle y (pixel)'])
    data_frame.to_csv('calibration_configs.csv', index=False)
    face_model = np.float32([
        [0, -0.436, -1.165], [0, 0.68, -1.54], [-1.325, 0, 0], [1.325, 0, 0],
        [0, 1.41, -1.613], [-1.058, 1.695, -0.381], [1.058, 1.695, -0.381]]).T
    data_frame = pd.DataFrame(face_model, columns=[
        'Amount', 'Nose', 'Right corner of eye', 'Left corner of eye',
        'Mouth', 'Right corner of mouth', 'Left corner of mouth'], index=['x', 'y', 'z'])
    data_frame.to_csv('face_model.csv')


def decide_gaze_point(face_tvec, gaze_vec, camera_position, monitor_position):
    camera_tvec, camera_xvec, camera_yvec, camera_zvec = camera_position
    monitor_tvec, monitor_xvec, monitor_yvec, monitor_zvec, monitor_xlen, monitor_ylen = monitor_position
    face_tvec = camera_tvec + face_tvec[0] * camera_xvec + face_tvec[1] * camera_yvec + face_tvec[2] * camera_zvec
    gaze_vec = gaze_vec[0] * camera_xvec + gaze_vec[1] * camera_yvec + gaze_vec[2] * camera_zvec
    gaze_tvec = face_tvec + np.dot(monitor_tvec - face_tvec, monitor_zvec) / np.dot(gaze_vec, monitor_zvec) * gaze_vec
    gaze_x = np.dot(gaze_tvec - monitor_tvec, monitor_xvec) / monitor_xlen
    gaze_y = np.dot(gaze_tvec - monitor_tvec, monitor_yvec) / monitor_ylen
    return np.array([gaze_x, gaze_y])


class EstimateFacePosition:
    def __init__(self, face_model, camera_matrix):
        self.face_model = face_model
        self.camera_matrix = camera_matrix

    def calc_image_coords(self, rvec, tvec, face_model):
        camera_coords = np.einsum('jk,ik->ij', to_rotation_matrix(rvec), face_model) + tvec
        image_coords = np.einsum('jk,ik->ij', self.camera_matrix, camera_coords)
        image_coords = (image_coords.T[:-1] / image_coords.T[-1]).T
        return image_coords

    def calc_loss(self, image_coords, face_model, r_x, r_y, r_z, t_x, t_y, t_z):
        rvec = np.array([r_x, r_y, r_z])
        tvec = np.array([t_x, t_y, t_z])
        return image_coords - self.calc_image_coords(rvec, tvec, face_model)

    def estimate_face_position(self, facial_landmark_coords, is_accurate):
        model = lmfit.Model(self.calc_loss, independent_vars=['image_coords', 'face_model'])
        parameters = model.make_params()
        for key in parameters.keys():
            if key == 't_z':
                value = 25
            else:
                value = 0
            parameters[key].set(value=value, vary=True, min=-np.inf, max=np.inf)
        result = model.fit(
            image_coords=facial_landmark_coords[is_accurate],
            face_model=self.face_model[is_accurate],
            data=np.zeros(facial_landmark_coords[is_accurate].shape),
            params=parameters)
        rvec = np.array([result.values[key] for key in ['r_x', 'r_y', 'r_z']])
        tvec = np.array([result.values[key] for key in ['t_x', 't_y', 't_z']])
        return rvec, tvec


def estimate_gaze_point(accuracy_threshold, medfilt_kernel_size=7):
    try:
        data_frame = pd.read_csv('face_model.csv', header=0, index_col=0)
        face_model = np.float32(data_frame).T
        assert face_model.shape == (7, 3) and not np.isnan(face_model).any()
    except Exception:
        raise Exception('face_model.csv is not found or is invalid.')
    try:
        data_frame = pd.read_csv('configs.csv', header=0)
        video_width = float(data_frame['video width (pixel)'][0])
        video_height = float(data_frame['video height (pixel)'][0])
    except Exception:
        raise Exception('configs.csv is not found or is invalid.')
    if 'camera1.pickle' not in os.listdir():
        raise Exception('camera1.pickle is not found.')
    with open('camera1.pickle', 'rb') as f:
        _, camera_matrix, _ = pickle.load(f)
    if 'setup.pickle' not in os.listdir():
        raise Exception('setup.pickle is not found.')
    with open('setup.pickle', 'rb') as f:
        camera_position, monitor_position = pickle.load(f)
    estimator = EstimateFacePosition(face_model, camera_matrix)
    for file_name in os.listdir('DeepLabCutOutputs'):
        print('processing ' + file_name + ' ... ', end='')
        try:
            data_frame = pd.read_csv(os.path.join('DeepLabCutOutputs', file_name), header=1)
            facial_landmark_coords = np.array(data_frame[[
                'Amount', 'Amount.1', 'Nose', 'Nose.1',
                'Right corner of eye', 'Right corner of eye.1', 'Left corner of eye', 'Left corner of eye.1',
                'Mouth', 'Mouth.1', 'Right corner of mouth', 'Right corner of mouth.1',
                'Left corner of mouth', 'Left corner of mouth.1']][1:], dtype='float32')
            is_accurate = np.float32(data_frame[[
                'Amount.2', 'Nose.2', 'Right corner of eye.2', 'Left corner of eye.2',
                'Mouth.2', 'Right corner of mouth.2', 'Left corner of mouth.2']][1:])
            is_accurate = is_accurate > accuracy_threshold
        except Exception as e:
            print('failed, error message: ' + str(e))
            continue
        for i in range(facial_landmark_coords.shape[1]):
            facial_landmark_coords[:, i] = medfilt(facial_landmark_coords[:, i], medfilt_kernel_size)
        facial_landmark_coords = np.reshape(facial_landmark_coords, (len(facial_landmark_coords), -1, 2))
        face_rvec_list = []
        face_tvec_list = []
        gaze_point_list = []
        for i in range(len(facial_landmark_coords)):
            if np.sum(is_accurate[i]) < 6:
                face_rvec_list.append(np.array([np.nan, np.nan, np.nan]))
                face_tvec_list.append(np.array([np.nan, np.nan, np.nan]))
                gaze_point_list.append(np.array([np.nan, np.nan]))
                continue
            face_rvec, face_tvec = estimator.estimate_face_position(facial_landmark_coords[i], is_accurate[i])
            gaze_vec = np.dot(to_rotation_matrix(face_rvec), np.array([0., 0., -1.]))
            gaze_vec /= np.linalg.norm(gaze_vec)
            gaze_point = decide_gaze_point(face_tvec, gaze_vec, camera_position, monitor_position)
            face_rvec_list.append(face_rvec)
            face_tvec_list.append(face_tvec)
            gaze_point_list.append(gaze_point)
        face_rvec_list = np.array(face_rvec_list, dtype='float32')
        face_tvec_list = np.array(face_tvec_list, dtype='float32')
        gaze_point_list = np.array(gaze_point_list, dtype='float32')
        data_frame = pd.DataFrame()
        data_frame['face transfer x'] = face_tvec_list[:, 0]
        data_frame['face transfer y'] = face_tvec_list[:, 1]
        data_frame['face transfer z'] = face_tvec_list[:, 2]
        data_frame['face rotation alpha'] = face_rvec_list[:, 0]
        data_frame['face rotation beta'] = face_rvec_list[:, 1]
        data_frame['face rotation gamma'] = face_rvec_list[:, 2]
        data_frame['gaze point x'] = gaze_point_list[:, 0]
        data_frame['gaze point y'] = gaze_point_list[:, 1]
        data_frame['gaze point x (pixel)'] = (gaze_point_list[:, 0] + 0.5) * video_width
        data_frame['gaze point y (pixel)'] = (gaze_point_list[:, 1] + 0.5) * video_height
        save_file_path = os.path.join('Results', file_name)
        data_frame.to_csv(save_file_path)
        print('successful.')


def find_aruco_marker(img, camera_parameters, aruco_marker_size, aruco_dict):
    _, camera_matrix, distortion_coefficients = camera_parameters
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco_dict), parameters)
    corners, ids, rejected = detector.detectMarkers(img)
    object_points = np.array([[-0.5, -0.5, 0.], [0.5, -0.5, 0.], [0.5, 0.5, 0.], [-0.5, 0.5, 0.]])
    object_points *= aruco_marker_size
    rvec_list = {}
    tvec_list = {}
    for i in range(len(ids)):
        flag, rvec, tvec = cv2.solvePnP(object_points, corners[i], camera_matrix, distortion_coefficients)
        if not flag:
            raise Exception('solvePnP failed; marker index: ' + str(i) + ', ArUco_dict: ' + str(aruco_dict) + '.')
        rvec_list[int(ids[i])] = np.reshape(rvec, 3)
        tvec_list[int(ids[i])] = np.reshape(tvec, 3)
    img_show = img.copy()
    aruco.drawDetectedMarkers(img, rejected, borderColor=(100, 100, 255))
    if ids is not None:
        aruco.drawDetectedMarkers(img_show, corners, ids)
    for i in ids:
        i = int(i)
        cv2.drawFrameAxes(img_show, camera_matrix, distortion_coefficients, rvec_list[i], tvec_list[i], 1)
    cv2.imwrite('ArUco_marker_{0}.png'.format(aruco_dict), img_show)
    return rvec_list, tvec_list


def find_camera(img, camera_parameters):
    rvec_list_4x4, tvec_list_4x4 = find_aruco_marker(img, camera_parameters, 2, aruco.DICT_4X4_50)
    for i in range(4):
        if i not in tvec_list_4x4.keys():
            raise Exception('marker 4x4 not found, index: ' + str(i))
    tvec = (tvec_list_4x4[0] + tvec_list_4x4[1] + tvec_list_4x4[2] + tvec_list_4x4[3]) / 4
    xvec = (tvec_list_4x4[0] - tvec_list_4x4[2] + tvec_list_4x4[1] - tvec_list_4x4[3]) / 2
    xvec /= np.linalg.norm(xvec)
    yvec = (tvec_list_4x4[1] - tvec_list_4x4[0] + tvec_list_4x4[3] - tvec_list_4x4[2]) / 2
    yvec /= np.linalg.norm(yvec)
    zvec = np.cross(xvec, yvec)
    zvec /= np.linalg.norm(zvec)
    return tvec, xvec, yvec, zvec


def find_monitor(img, camera_parameters):
    aruco_marker_size = 4.9
    rvec_list_5x5, tvec_list_5x5 = find_aruco_marker(img, camera_parameters, aruco_marker_size, aruco.DICT_5X5_50)
    for i in range(4):
        if i not in tvec_list_5x5.keys():
            raise Exception('marker 5x5 not found, index: ' + str(i))
    tvec = (tvec_list_5x5[0] + tvec_list_5x5[1] + tvec_list_5x5[2] + tvec_list_5x5[3]) / 4
    xvec = (tvec_list_5x5[1] - tvec_list_5x5[0] + tvec_list_5x5[3] - tvec_list_5x5[2]) / 2
    xlen = np.linalg.norm(xvec)
    xvec /= xlen
    xlen += aruco_marker_size
    yvec = (tvec_list_5x5[2] - tvec_list_5x5[0] + tvec_list_5x5[3] - tvec_list_5x5[1]) / 2
    ylen = np.linalg.norm(yvec)
    yvec /= ylen
    ylen += aruco_marker_size
    zvec = np.cross(xvec, yvec)
    zvec /= np.linalg.norm(zvec)
    return tvec, xvec, yvec, zvec, xlen, ylen


def find_camera_and_monitor():
    if 'camera2.pickle' not in os.listdir():
        raise Exception('camera2.pickle is not found.')
    with open('camera2.pickle', 'rb') as f:
        camera2_parameters = pickle.load(f)
    img = None
    for file_name in os.listdir('Setup'):
        print('loading ' + file_name + ' ... ', end='')
        file_path = os.path.join('Setup', file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('failed.')
        else:
            print('successful.')
            break
    if img is None:
        raise Exception('no image is found in Setup.')
    camera_position = find_camera(img, camera2_parameters)
    monitor_position = find_monitor(img, camera2_parameters)
    return camera_position, monitor_position


def to_rotation_matrix(rvec):
    c_x = np.cos(rvec[0])
    s_x = np.sin(rvec[0])
    rotation_matrix_x = np.array([[1., 0., 0.], [0., c_x, -s_x], [0., s_x, c_x]])
    c_y = np.cos(rvec[1])
    s_y = np.sin(rvec[1])
    rotation_matrix_y = np.array([[c_y, 0., s_y], [0., 1., 0.], [-s_y, 0., c_y]])
    c_z = np.cos(rvec[2])
    s_z = np.sin(rvec[2])
    rotation_matrix_z = np.array([[c_z, -s_z, 0.], [s_z, c_z, 0.], [0., 0., 1.]])
    return np.dot(np.dot(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)


def main(args):
    if args.task == 'createDirs':
        create_directories()
    elif args.task == 'calCamera1':
        camera_parameters = calibrate_camera('Camera1')
        with open('camera1.pickle', 'wb') as f:
            pickle.dump(camera_parameters, f)
        print('camera1 is calibrated successfully.')
    elif args.task == 'calCamera2':
        camera_parameters = calibrate_camera('Camera2')
        with open('camera2.pickle', 'wb') as f:
            pickle.dump(camera_parameters, f)
        print('camera2 is calibrated successfully.')
    elif args.task == 'locateSetup':
        setup = find_camera_and_monitor()
        with open('setup.pickle', 'wb') as f:
            pickle.dump(setup, f)
        print('camera1 and display are located successfully.')
    elif args.task == 'estimateGazePoints':
        if args.accuracy is None:
            raise Exception('Threshold for accuracy of facial landmark detection is not specified.')
        estimate_gaze_point(args.accuracy)
    elif args.task == 'calEstimator':
        calibrator = calibrate_estimator()
        with open('calibrator.pickle', 'wb') as f:
            pickle.dump(calibrator, f)
        print('estimator is calibrated successfully.')
    elif args.task == 'calResults':
        calibrate_results()
    elif args.task == 'annotateVideo':
        if args.accuracy is None:
            raise Exception('accuracy is not confirmed or is invalid.')
        if args.csv is None:
            raise Exception('csv file name is not confirmed or is invalid.')
        if args.marmoset_video is None:
            raise Exception('marmoset video name is not confirmed or is invalid.')
        if args.stimuli_video is None:
            raise Exception('stimuli video name is not confirmed or is invalid.')
        if args.output is None:
            raise Exception('output file name is not confirmed or is invalid.')
        if args.frame_delay is None:
            raise Exception('frame number of delay is not confirmed or is invalid.')
        annotate_video(args.csv, args.marmoset_video, args.stimuli_video, args.output, args.frame_delay, args.accuracy)
    else:
        raise Exception('task name should be one of the follows: createDirs calCamera1 calCamera2 locateSetup' +
                        ' estimateGazePoints calEstimator calResults annotateVideo')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True, help='Task name')
    parser.add_argument('-a', '--accuracy', type=float, help='Accuracy threshold')
    parser.add_argument('-c', '--csv', type=str, help='.csv file name')
    parser.add_argument('-r', '--marmoset_video', type=str, help='Marmoset video name')
    parser.add_argument('-p', '--stimuli_video', type=str, help='Stimuli video name')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('-f', '--frame_delay', type=int, help='Frame number of the delay')
    main(parser.parse_args())
