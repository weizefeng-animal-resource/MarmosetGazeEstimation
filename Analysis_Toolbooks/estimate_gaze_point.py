import os

import numpy as np
import pandas as pd
import pickle
import lmfit
from scipy.signal import medfilt


def decide_gaze_point(face_tvec, gaze_vec, camera_position, screen_position):
    camera_tvec, camera_xvec, camera_yvec, camera_zvec = camera_position
    screen_tvec, screen_xvec, screen_yvec, screen_zvec, screen_xlen, screen_ylen = screen_position
    face_tvec = camera_tvec + face_tvec[0] * camera_xvec + face_tvec[1] * camera_yvec + face_tvec[2] * camera_zvec
    gaze_vec = gaze_vec[0] * camera_xvec + gaze_vec[1] * camera_yvec + gaze_vec[2] * camera_zvec
    gaze_tvec = face_tvec + np.dot(screen_tvec - face_tvec, screen_zvec) / np.dot(gaze_vec, screen_zvec) * gaze_vec
    gaze_x = np.dot(gaze_tvec - screen_tvec, screen_xvec) / screen_xlen
    gaze_y = np.dot(gaze_tvec - screen_tvec, screen_yvec) / screen_ylen
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
    
    
def estimate_gaze_points(accuracy_threshold=0.96, medfilt_kernel_size=7):
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
        camera_position, screen_position = pickle.load(f)
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
            gaze_point = decide_gaze_point(face_tvec, gaze_vec, camera_position, screen_position)
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
