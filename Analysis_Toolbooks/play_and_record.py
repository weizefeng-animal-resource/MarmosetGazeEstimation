import os

from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import tkinter as tk


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
