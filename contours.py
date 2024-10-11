import cv2
import numpy as np


def draw_contours(background_frames: np.ndarray, mask_frames: np.ndarray, min_contour_area: int = 10) -> np.ndarray:
    for mask, background in zip(mask_frames, background_frames):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_contours = [cnt for cnt in contours if
                          cv2.contourArea(cnt) > min_contour_area]

        if len(large_contours) > 0:
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                background = cv2.rectangle(background, (x, y), (x + w, y + h), (0, 0, 200), 3)

            frame_out = cv2.drawContours(background, large_contours, -1, (0, 255, 0), thickness=2)

    return background_frames
