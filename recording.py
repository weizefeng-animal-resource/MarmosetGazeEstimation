import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import os
import pandas as pd
import time
import tkinter as tk


def create_directories():
    os.makedirs('Results', exist_ok=True)
    os.makedirs('Videos', exist_ok=True)
    data_frame = pd.DataFrame([[25, 1920, 1080, 0]], columns=['video fps', 'video frame width (pixel)', 'video frame height (pixel)', 'camera index'])
    data_frame.to_csv('configs.csv', index=False)


def detect_delays(parameters, video_name):
    video = cv2.VideoCapture(os.path.join('Videos', video_name))
    if not video.isOpened():
        print('Failed to load video file {0}.'.format(video_name))
        return
    video_width = parameters['video_width']
    video_height = parameters['video_height']
    try:
        assert int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == video_width
        assert int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == video_height
    except AssertionError:
        print('Video resolution does not match the value specified in configs.csv.')
        return
    try:
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) > 500
    except AssertionError:
        print('Video should have more than 200 frames.')
        return
    camera_index = parameters['camera_index']
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print('Failed to open camera {0}.'.format(camera_index))
        return
    save_file_path = 'temp.mp4'
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = parameters['fps']
    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(save_file_path, four_cc, fps, (camera_width, camera_height))
    if not writer.isOpened():
        print('Failed to create output file.')
        return

    root = tk.Tk()
    main_screen_width = root.winfo_screenwidth()
    root.quit()
    root.destroy()
    cv2.namedWindow('stimuli presentation', cv2.WINDOW_NORMAL)
    background = np.zeros((video_height, video_width, 3), dtype='uint8')
    cv2.imshow('stimuli presentation', background)
    cv2.waitKey(1)
    cv2.moveWindow('stimuli presentation', main_screen_width, 0)
    cv2.moveWindow('stimuli presentation', main_screen_width, 0)
    cv2.resizeWindow('stimuli presentation', video_width, video_height)
    cv2.namedWindow('monitoring')
    cv2.waitKey(1)
    cv2.moveWindow('monitoring', 0, 0)
    cv2.moveWindow('monitoring', 0, 0)

    black_screen = np.zeros((video_height, video_width, 3), dtype='uint8')
    white_screen = np.ones((video_height, video_width, 3), dtype='uint8') * 255
    img_recorded = camera.read()[1]
    executor = ThreadPoolExecutor(max_workers=2)
    interval = 1 / fps
    next_loop = time.perf_counter() + interval
    count = 0
    while True:
        future = executor.submit(lambda: camera.read()[1])
        flag, current_frame = video.read()
        if not flag or count > 400:
            future.result()
            break
        if count < 200:
            cv2.imshow('stimuli presentation', black_screen[28:])
        else:
            cv2.imshow('stimuli presentation', white_screen[28:])
        count += 1
        cv2.imshow('monitoring', img_recorded)
        if cv2.waitKey(1) == 27:
            future.result()
            break
        img_recorded = future.result()
        writer.write(img_recorded)
        sleep_time = next_loop - time.perf_counter()
        next_loop += interval
        if sleep_time > 0:
            time.sleep(sleep_time)
    video.release()
    camera.release()
    writer.release()
    cv2.destroyAllWindows()

    video = cv2.VideoCapture('temp.mp4')
    intensity = []
    while True:
        flag, current_frame = video.read()
        if not flag:
            break
        intensity.append(np.mean(current_frame))
    video.release()
    intensity = np.array(intensity, dtype='float32')
    mean_black = np.mean(intensity[50:150])
    std_black = np.std(intensity[50:150])
    mean_white = np.mean(intensity[-150:-50])
    std_white = np.std(intensity[-150:-50])
    if mean_white - mean_black < (std_white + std_black) * 3:
        print('Failed.')
        return
    delay = np.where(intensity[150:] > mean_black + std_black * 3)[0][0] + 150 - 200
    print('Delay is {0} frames.'.format(delay))


def load_parameters():
    try:
        data_frame = pd.read_csv('configs.csv', header=0)
        fps = float(data_frame['video fps'][0])
        video_width = int(data_frame['video frame width (pixel)'][0])
        video_height = int(data_frame['video frame height (pixel)'][0])
        camera_index = int(data_frame['camera index'][0])
    except Exception as e:
        print('Failed to load parameters from config.csv:', e)
        exit()
    parameters = {'fps': fps, 'video_width': video_width, 'video_height': video_height, 'camera_index': camera_index}
    return parameters


def play_and_record(parameters, output_file_name, video_name):
    if output_file_name in os.listdir('Results'):
        print('Interrupted because {0} exists in Results folder.'.format(output_file_name))
    video = cv2.VideoCapture(os.path.join('Videos', video_name))
    if not video.isOpened():
        print('Failed to load video file {0}.'.format(video_name))
        return
    video_width = parameters['video_width']
    video_height = parameters['video_height']
    try:
        assert int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == video_width
        assert int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == video_height
    except AssertionError:
        print('Video resolution does not match the value specified in configs.csv.')
        return
    camera_index = parameters['camera_index']
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print('Failed to open camera {0}.'.format(camera_index))
        return
    save_file_path = os.path.join('Results', output_file_name)
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = parameters['fps']
    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(save_file_path, four_cc, fps, (camera_width, camera_height))
    if not writer.isOpened():
        print('Failed to create output file.')
        return

    root = tk.Tk()
    main_screen_width = root.winfo_screenwidth()
    root.quit()
    root.destroy()
    cv2.namedWindow('stimuli presentation', cv2.WINDOW_NORMAL)
    background = np.zeros((video_height, video_width, 3), dtype='uint8')
    cv2.imshow('stimuli presentation', background)
    cv2.waitKey(1)
    cv2.moveWindow('stimuli presentation', main_screen_width, 0)
    cv2.moveWindow('stimuli presentation', main_screen_width, 0)
    cv2.resizeWindow('stimuli presentation', video_width, video_height)
    cv2.namedWindow('monitoring')
    cv2.waitKey(1)
    cv2.moveWindow('monitoring', 0, 0)
    cv2.moveWindow('monitoring', 0, 0)
    while True:
        img_recorded = camera.read()[1]
        cv2.imshow('monitoring', img_recorded)
        cv2.imshow('stimuli presentation', background[28:])
        if cv2.waitKey(1) == 13:
            break
    executor = ThreadPoolExecutor(max_workers=2)
    interval = 1 / fps
    next_loop = time.perf_counter() + interval
    while True:
        future = executor.submit(lambda: camera.read()[1])
        flag, current_frame = video.read()
        if not flag:
            future.result()
            break
        cv2.imshow('stimuli presentation', current_frame[28:])
        cv2.imshow('monitoring', img_recorded)
        if cv2.waitKey(1) == 27:
            future.result()
            break
        img_recorded = future.result()
        writer.write(img_recorded)
        sleep_time = next_loop - time.perf_counter()
        next_loop += interval
        if sleep_time > 0:
            time.sleep(sleep_time)
    while True:
        img_recorded = camera.read()[1]
        cv2.imshow('monitoring', img_recorded)
        cv2.imshow('stimuli presentation', background[28:])
        if cv2.waitKey(1) == 13:
            break
    video.release()
    camera.release()
    writer.release()
    cv2.destroyAllWindows()


def main(args):
    match args.task:
        case 'createDirs':
            create_directories()
        case 'detectDelays':
            if args.video_name is None:
                print('Required inputs are incomplete.')
                return
            parameters = load_parameters()
            detect_delays(parameters, args.video_name)
        case 'run':
            if args.output_file_name is None or args.video_name is None:
                print('Required inputs are incomplete.')
                return
            parameters = load_parameters()
            play_and_record(parameters, args.output_file_name, args.video_name)
        case _:
            print('Task name should be one of the follows: createDirs detectDelays run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file_name', type=str, help='Output File Name')
    parser.add_argument('-t', '--task', type=str, required=True, help='Task Name')
    parser.add_argument('-v', '--video_name', type=str, help='Video Name')
    main(parser.parse_args())
