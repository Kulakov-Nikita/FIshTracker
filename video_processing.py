import cv2
import numpy as np


def read_video(path_to_input_video: str, video_resolution: tuple = (1280, 720), video_duration_sec: int = 300,
               fps: int = 30) -> np.ndarray:
    video_duration_frames: int = video_duration_sec * fps
    frames = np.zeros((video_duration_frames, video_resolution[1], video_resolution[0], 3), dtype='uint8')

    cap = cv2.VideoCapture(path_to_input_video)
    counter: int = 0
    if not cap.isOpened():
        raise Exception("Error opening video file: " + path_to_input_video)
    while cap.isOpened() and counter < video_duration_frames:
        ret, frame = cap.read()
        if ret:
            frames[counter] = frame
            counter += 1
            if counter % (video_duration_frames // 100) == 0:
                print(f"{counter // (video_duration_frames // 100)}%")
        else:
            cap.release()

    return frames
