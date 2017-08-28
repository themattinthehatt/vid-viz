"""Contains OpenCV-specific utility fucntions"""

import numpy as np
import cv2


def resize(frame, frame_width, frame_height, interpolation=cv2.INTER_LINEAR):
    """
    Function to resize a given frame while maintaining the original aspect 
    ratio

    ARGS:
        frame (numpy array) - frame to be resized
        frame_width (int) - desired frame width
        frame_height (int) - desired frame height
        interpolation (OpenCV option)
            cv2.INTER_LINEAR | cv2.INTER_CUBIC

    RETURNS:
        frame (numpy array) - updated frame

    """

    dims_og = frame.shape

    # crop current frame so that it has desired aspect ratio
    aspect_og = dims_og[1] / dims_og[0]
    aspect_final = frame_width / frame_height

    if aspect_final >= aspect_og:
        # frame is too tall and skinny; keep width
        # get target height
        height_new = int(dims_og[1] * frame_height / frame_width)
        frame = cv2.getRectSubPix(
            frame,
            (dims_og[1], height_new),
            (dims_og[1] / 2, dims_og[0] / 2))
    else:
        # frame is too short and fat; keep height
        # get target width
        width_new = int(dims_og[0] * frame_width / frame_height)
        frame = cv2.getRectSubPix(
            frame,
            (width_new, dims_og[0]),
            (dims_og[1] / 2, dims_og[0] / 2))

    # resize frame
    frame = cv2.resize(
        frame,
        (frame_width, frame_height),
        interpolation=interpolation)

    return frame

