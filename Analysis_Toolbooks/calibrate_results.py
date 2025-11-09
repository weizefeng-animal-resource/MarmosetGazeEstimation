import os

import numpy as np
import pandas as pd
import pickle
import lmfit


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
