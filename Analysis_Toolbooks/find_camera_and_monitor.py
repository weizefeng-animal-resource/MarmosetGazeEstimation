import os

import cv2
import cv2.aruco as aruco
import numpy as np
import pickle


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
