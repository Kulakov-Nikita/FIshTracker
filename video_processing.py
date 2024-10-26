import cv2
import numpy as np
import torch


class VideoReader:
    def __init__(self, path_to_input_video: str, image_shape: tuple[int, int], batch_size: int):
        self.path_to_input_video = path_to_input_video
        self.image_shape: tuple[int, int] = image_shape
        self.batch_size: int = batch_size

        self.video_capture = cv2.VideoCapture(self.path_to_input_video)

        if not self.video_capture.isOpened():
            raise Exception("Error opening video file: " + path_to_input_video)

    def read_batch(self) -> torch.Tensor:
        frames = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], 3), dtype=np.int8)
        counter: int = 0
        while self.video_capture.isOpened() and counter < self.batch_size:
            ret, frame = self.video_capture.read()
            if ret:
                frames[counter] = frame
                counter += 1
            else:
                self.release()

            if counter == self.batch_size:
                yield frames
                counter = 0

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()



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


# def write_video(frames: np.ndarray, path_to_output_video: str, fps: int = 30) -> None:
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     out = cv2.VideoWriter(path_to_output_video, fourcc, fps, (frames.shape[2], frames.shape[1]))
#     print(frames.shape, frames.dtype)
#     for frame in frames:
#         out.write(frame)
#     out.release()

