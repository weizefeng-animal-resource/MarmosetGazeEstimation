import os

import cv2
import numpy as np


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
