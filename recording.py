import os

import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import tkinter as tk


def create_directories():
    os.makedirs('Results', exist_ok=True)
    os.makedirs('Videos', exist_ok=True)


def play_and_record(video_name, camera_index, save_file_name):
    video = cv2.VideoCapture(os.path.join('Videos', video_name))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    assert video_width == 1920
    assert video_height == 1080
    assert video_fps == 25
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
    save_file_path = os.path.join('Results', save_file_name)
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(save_file_path, four_cc, video_fps, (video_width, video_height))
    background = np.zeros((video_height, video_width, 3), dtype='uint8')
    root = tk.Tk()
    main_screen_width = root.winfo_screenwidth()
    root.quit()
    root.destroy()
    cv2.namedWindow('stimuli presentation', cv2.WINDOW_NORMAL)
    cv2.imshow('stimuli presentation', background)
    cv2.waitKey(1)
    cv2.moveWindow('stimuli presentation', main_screen_width, 0)
    cv2.moveWindow('stimuli presentation', main_screen_width, 0)
    cv2.resizeWindow('stimuli presentation', 1920, 1080)
    cv2.namedWindow('monitoring')
    cv2.waitKey(1)
    cv2.moveWindow('monitoring', 0, 0)
    cv2.moveWindow('monitoring', 0, 0)
    while True:
        img_recorded = capture.read()[1]
        cv2.imshow('monitoring', img_recorded)
        cv2.imshow('stimuli presentation', background[28:])
        if cv2.waitKey(1) == 13:
            break
    executor = ThreadPoolExecutor(max_workers=2)
    import time
    start = time.time()
    while True:
        future = executor.submit(lambda: capture.read()[1])
        flag, frame = video.read()
        if not flag:
            break
        cv2.imshow('stimuli presentation', frame[28:])
        cv2.imshow('monitoring', img_recorded)
        if cv2.waitKey(1) == 27:
            break
        img_recorded = future.result()
        writer.write(img_recorded)
    writer.release()
    print(time.time() - start)
    while True:
        img_recorded = capture.read()[1]
        cv2.imshow('monitoring', img_recorded)
        cv2.imshow('stimuli presentation', background[28:])
        if cv2.waitKey(1) == 13:
            break
    capture.release()
    cv2.destroyAllWindows()


def main(args):
    if args.task == 'createDirs':
        create_directories()
    elif args.task == 'run':
        video_name = args.video
        camera_index = args.camera
        save_file_name = args.output
        if video_name is None or camera_index is None or save_file_name is None:
            raise Exception('video name, camera index or output file name is not confirmed or is invalid.')
        if save_file_name in os.listdir('Results'):
            raise Exception(save_file_name + ' exists in Results folder.')
        play_and_record(video_name, camera_index, save_file_name)
    else:
        raise Exception('task name should be one of the follows: createDirs run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', type=int, help='Camera index')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('-t', '--task', type=str, required=True, help='Task name')
    parser.add_argument('-v', '--video', type=str, help='Video name')
    main(parser.parse_args())
