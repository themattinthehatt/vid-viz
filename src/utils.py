from __future__ import division

import numpy as np
import cv2


class SmoothNoise(object):

    def __init__(self, num_samples=10, num_channels=1):
        """
        Returns values from a Gaussian noise process that is smoothed by 
        averaging over previous random values (moving average filter); values
        will lie approximately between -1 and 1
        
        ARGS:
            num_samples - number of samples used in moving average filter
            num_channels - number of independent noise processes
            
        """

        self.num_samples = num_samples
        self.num_channels = num_channels
        self.noise_index = 0                # counter in noise array
        self.noise_samples = np.random.randn(self.num_samples,
                                             self.num_channels)

    def reinitialize(self):
        """Reinitialize noise samples"""

        self.noise_samples = np.random.randn(self.num_samples,
                                             self.num_channels)

    def get_next_vals(self):
        """Update noise_samples and return values for each channel"""

        self.noise_samples[self.noise_index, :] = np.random.randn(
            1, self.num_channels)
        self.noise_index = (self.noise_index + 1) % self.num_samples

        return np.sqrt(self.num_samples) / 3 * \
            np.mean(self.noise_samples, axis=0)


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


def get_sources():
    # file_type options : 'cam' | 'video' | 'image' | 'auto'
    source_list = []
    # source_list.append({
    #     'file_loc': 'hueswirlchain',
    #     'file_type': 'auto'})
    # source_list.append({
    #     'file_loc': '/media/data/Dropbox/Git/vid-viz/data/snowflake-02.mp4',
    #     'file_type': 'video'})
    source_list.append({
        'file_loc': '/media/data/Dropbox/Git/vid-viz/data/tree-00.jpg',
        'file_type': 'image'})
    source_list.append({
        'file_loc': '/media/data/Dropbox/Git/vid-viz/data/honeycomb-00.jpg',
        'file_type': 'image'})
    source_list.append({
        'file_loc': None,
        'file_type': 'cam'})
    # source_list.append({
    #     'file_loc': '/media/data/Dropbox/Git/vid-viz/data/waves-00.jpg',
    #     'file_type': 'image'})
    # source_list.append({
    #     'file_loc': '/media/data/Dropbox/Git/vid-viz/data/waves-01.jpg',
    #     'file_type': 'image'})
    # source_list.append({
    #     'file_loc': '/media/data/Dropbox/Git/vid-viz/data/waves-02.jpg',
    #     'file_type': 'image'})
    # source_list.append({
    #     'file_loc': '/media/data/Dropbox/Git/vid-viz/data/waves-03.jpg',
    #     'file_type': 'image'})
    source_list.append({
        'file_loc': '/media/data/Dropbox/Git/vid-viz/data/waves-04.jpg',
        'file_type': 'image'})
    # source_list.append({
    #     'file_loc': '/media/data/Dropbox/Git/vid-viz/data/waves-05.jpg',
    #     'file_type': 'image'})
    return source_list
