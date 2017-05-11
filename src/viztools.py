from __future__ import print_function
from __future__ import division

import cv2
import numpy as np

import utils as util

"""TODO
- add print statements
"""

INF = 100000
NONE = {
    'DESC': '',
    'NAME': '',
    'VAL': 0,
    'INIT': 0,
    'MIN': 0,
    'MAX': 1,
    'MOD': INF,
    'STEP': 1,
    'INC': False,
    'DEC': False}


class Effect(object):
    """Base class for vid-viz effects"""
    def __init__(self):
        # set attributes common to all effects
        self.PROPS = None
        self.MAX_NUM_STYLES = 1
        self.auto_play = False
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=1,
            num_channels=self.chan_vec_pos.size)

    def _process_io(self, key_list):

        if key_list[ord('-')]:
            key_list[ord('-')] = False
            self.PROPS[0]['DEC'] = True
        elif key_list[ord('=')]:
            key_list[ord('=')] = False
            self.PROPS[0]['INC'] = True
        elif key_list[ord('[')]:
            key_list[ord('[')] = False
            self.PROPS[1]['DEC'] = True
        elif key_list[ord(']')]:
            key_list[ord(']')] = False
            self.PROPS[1]['INC'] = True
        elif key_list[ord(';')]:
            key_list[ord(';')] = False
            self.PROPS[2]['DEC'] = True
        elif key_list[ord('\'')]:
            key_list[ord('\'')] = False
            self.PROPS[2]['INC'] = True
        elif key_list[ord(',')]:
            key_list[ord(',')] = False
            self.PROPS[3]['DEC'] = True
        elif key_list[ord('.')]:
            key_list[ord('.')] = False
            self.PROPS[3]['INC'] = True
        elif key_list[ord('R')]:
            key_list[ord('R')] = False
            self.PROPS[4]['DEC'] = True
        elif key_list[ord('T')]:
            key_list[ord('T')] = False
            self.PROPS[4]['INC'] = True
        elif key_list[ord('Q')]:
            key_list[ord('Q')] = False
            self.PROPS[5]['DEC'] = True
        elif key_list[ord('S')]:
            key_list[ord('S')] = False
            self.PROPS[5]['INC'] = True
        elif key_list[ord('/')]:
            key_list[ord('/')] = False
            self.reinitialize = True
        elif key_list[ord('t')]:
            key_list[ord('t')] = False
            self.style = (self.style + 1) % self.MAX_NUM_STYLES
            # self.reinitialize = True
        elif key_list[ord('a')]:
            key_list[ord('a')] = False
            self.auto_play = not self.auto_play
        elif key_list[ord('w')]:
            key_list[ord('w')] = False
            self.random_walk = not self.random_walk
            self.chan_vec_pos = np.zeros(self.chan_vec_pos.shape)
            self.noise.reinitialize()

        # process options
        for index, _ in enumerate(self.PROPS):
            if self.PROPS[index]['DEC']:
                self.PROPS[index]['DEC'] = False
                self.PROPS[index]['VAL'] -= self.PROPS[index]['STEP']
            if self.PROPS[index]['INC']:
                self.PROPS[index]['INC'] = False
                self.PROPS[index]['VAL'] += self.PROPS[index]['STEP']
            self.PROPS[index]['VAL'] = np.mod(
                self.PROPS[index]['VAL'],
                self.PROPS[index]['MOD'])
            self.PROPS[index]['VAL'] = np.clip(
                self.PROPS[index]['VAL'],
                self.PROPS[index]['MIN'],
                self.PROPS[index]['MAX'])

    def process(self, frame, key_list):
        raise NotImplementedError

    def reset(self):
        for index, _ in enumerate(self.PROPS):
            self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']
        self.style = 0
        self.auto_play = False
        self.reinitialize = False
        self.chan_vec_pos = np.zeros(self.chan_vec_pos.shape)
        self.noise.reinitialize()


class Border(Effect):
    """
    Manipulate image borders

    KEYBOARD INPUTS:
        t - toggle between border styles
        -/+ - decrease/increase border padding
        [/] - decrease/increase zoom
        ;/' - rotate image left/right
        ,/. - None
        lrud arrows - translate image
        backspace - quit border effect
    """

    def __init__(self):

        super(Border, self).__init__()

        # user option constants
        MULT_FACTOR = {
            'NAME': 'mult_factor',
            'VAL': 1.0,
            'INIT': 1.0,
            'MIN': 0.01,
            'MAX': 1.0,
            'MOD': INF,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        ZOOM_FACTOR = {
            'NAME': 'zoom_factor',
            'VAL': 1.0,
            'INIT': 1.0,
            'MIN': 1.0,
            'MAX': 10.0,
            'MOD': INF,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        ROT_ANGLE = {
            'NAME': 'rot_angle',
            'VAL': 0,
            'INIT': 0,
            'MIN': -INF,
            'MAX': INF,
            'MOD': 360,
            'STEP': 5,
            'INC': False,
            'DEC': False}
        SHIFT_PIX_VERT = {
            'NAME': 'shift_vert',
            'VAL': 0,
            'INIT': 0,
            'MIN': -500,
            'MAX': 500,
            'MOD': INF,
            'STEP': 10,
            'INC': False,
            'DEC': False}
        SHIFT_PIX_HORZ = {
            'NAME': 'shift_horz',
            'VAL': 0,
            'INIT': 0,
            'MIN': -500,
            'MAX': 500,
            'MOD': INF,
            'STEP': 10,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            MULT_FACTOR,
            ZOOM_FACTOR,
            ROT_ANGLE,
            NONE,
            SHIFT_PIX_VERT,
            SHIFT_PIX_HORZ]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        mult_factor = self.PROPS[0]['VAL']
        zoom_factor = self.PROPS[1]['VAL']
        rot_angle = self.PROPS[2]['VAL']
        shift_vert = self.PROPS[4]['VAL']
        shift_horz = self.PROPS[5]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # rotate
        if rot_angle is not 0:
            rot_mat = cv2.getRotationMatrix2D(
                (im_width / 2, im_height / 2),
                rot_angle,
                1.0)
            frame = cv2.warpAffine(
                frame,
                rot_mat,
                (im_width, im_height))

        # translate
        if shift_horz is not 0 or shift_vert is not 0:
            frame = cv2.warpAffine(
                frame,
                np.float32([[1, 0, shift_horz],
                            [0, 1, shift_vert]]),
                (im_width, im_height))

        # zoom
        if zoom_factor > 1.0:
            frame = cv2.getRectSubPix(
                frame,
                (int(im_width / zoom_factor),
                 int(im_height / zoom_factor)),
                (im_width / 2, im_height / 2))
            frame = cv2.resize(frame, (im_width, im_height))

        # add borders
        if self.style == 1:
            # resize frame
            frame = cv2.resize(
                frame, None,
                fx=mult_factor,
                fy=mult_factor,
                interpolation=cv2.INTER_LINEAR)
            if mult_factor < 1.0:
                # top, bottom, left, right
                frame = cv2.copyMakeBorder(
                    frame,
                    int(im_height * (1.0 - mult_factor) / 2),
                    int(im_height * (1.0 - mult_factor) / 2),
                    int(im_width * (1.0 - mult_factor) / 2),
                    int(im_width * (1.0 - mult_factor) / 2),
                    cv2.BORDER_WRAP)
        elif self.style == 2:
            # resize frame
            frame = cv2.resize(
                frame, None,
                fx=mult_factor,
                fy=mult_factor,
                interpolation=cv2.INTER_LINEAR)
            if mult_factor < 1.0:
                # top, bottom, left, right
                frame = cv2.copyMakeBorder(
                    frame,
                    int(im_height * (1.0 - mult_factor) / 2),
                    int(im_height * (1.0 - mult_factor) / 2),
                    int(im_width * (1.0 - mult_factor) / 2),
                    int(im_width * (1.0 - mult_factor) / 2),
                    cv2.BORDER_REFLECT)

        return frame


class PostProcess(Effect):
    """
    Apply post-processing (image filtering) to final image

    KEYBOARD INPUTS:
        -/+ - decrease/increase gaussian smoothing kernel
        [/] - decrease/increase median filtering kernel
        ;/' - None
        ,/. - None
        lrud arrows - translate image
    """

    def __init__(self):

        super(PostProcess, self).__init__()

        GAUSSIAN_KERN = {
            'DESC': 'kernel size for gaussian smoothing',
            'NAME': 'gauss_kern',
            'VAL': 7,
            'INIT': 7,
            'MIN': 3,
            'MAX': 75,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        MEDIAN_KERN = {
            'DESC': 'kernel size for median filtering',
            'NAME': 'median_kern',
            'VAL': 7,
            'INIT': 7,
            'MIN': 3,
            'MAX': 75,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            GAUSSIAN_KERN,
            MEDIAN_KERN,
            NONE,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        gauss_kern = self.PROPS[0]['VAL']
        median_kern = self.PROPS[1]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # rotate
        if self.style == 1:
            frame_blurred = cv2.GaussianBlur(frame, (gauss_kern, gauss_kern), 0)
            # frame = cv2.addWeighted(
            #     frame, 0.5,
            #     frame_blurred, 0.5, 0)
            frame = cv2.bitwise_or(frame, frame_blurred)
        elif self.style == 2:
            frame = cv2.medianBlur(frame, median_kern)

        return frame


class Threshold(Effect):
    """
    Threshold individual channels in RGB frame
    
    KEYBOARD INPUTS:
        t - toggle between threshold types
        -/+ - decrease/increase adaptive threshold kernel size
        [/] - decrease/increase adaptive threshold offset value
        ;/' - None
        ,/. - None
        r/g/b - select red/green/blue channel for further processing
        0 - selected channel uses original values
        1 - selected channel uses all pixels as 0
        2 - selected channel uses all pixels as 255
        3 - selected channel uses threshold effect
        / - reset parameters
        q - quit threshold effect
    """

    def __init__(self):

        super(Threshold, self).__init__()

        # user option constants
        THRESH_KERNEL = {
            'DESC': 'kernel size for adaptive thresholding',
            'NAME': 'kernel_size',
            'VAL': 21,
            'INIT': 21,
            'MIN': 3,
            'MAX': 71,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        THRESH_OFFSET = {
            'DESC': 'offset constant for adaptive thresholding',
            'NAME': 'offset',
            'VAL': 4,
            'INIT': 4,
            'MIN': 0,
            'MAX': 30,
            'MOD': INF,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            THRESH_KERNEL,
            THRESH_OFFSET,
            NONE,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # other user options
        self.optimize = 1                       # skips a smoothing step
        self.use_chan = [False, False, False]   # rgb channel selector
        self.chan_style = [0, 0, 0]             # effect selector for each chan
        self.MAX_NUM_CHAN_STYLES = 4

        # opencv parameters
        self.THRESH_TYPE = cv2.THRESH_BINARY
        self.ADAPTIVE_THRESH_TYPE = cv2.ADAPTIVE_THRESH_MEAN_C

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)
            if key_list[ord('b')]:
                key_list[ord('b')] = False
                self.use_chan[0] = True
                self.use_chan[1] = False
                self.use_chan[2] = False
            elif key_list[ord('g')]:
                key_list[ord('g')] = False
                self.use_chan[0] = False
                self.use_chan[1] = True
                self.use_chan[2] = False
            elif key_list[ord('r')]:
                key_list[ord('r')] = False
                self.use_chan[0] = False
                self.use_chan[1] = False
                self.use_chan[2] = True

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        kernel_size = self.PROPS[0]['VAL']
        offset = self.PROPS[1]['VAL']

        for chan in range(3):
            if self.use_chan[chan]:
                for chan_style in range(self.MAX_NUM_CHAN_STYLES):
                    if key_list[ord(str(chan_style))]:
                        self.chan_style[chan] = chan_style
                        key_list[ord(str(chan_style))] = False

        # process image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.medianBlur(frame_gray, 11)
        frame_thresh = cv2.adaptiveThreshold(
            frame_gray,
            255,
            self.ADAPTIVE_THRESH_TYPE,
            self.THRESH_TYPE,
            kernel_size,
            offset)

        if self.style == 0:
            self.THRESH_TYPE = cv2.THRESH_BINARY
        elif self.style == 1:
            self.THRESH_TYPE = cv2.THRESH_BINARY_INV

        for chan in range(3):
            if self.chan_style[chan] == 1:
                frame[:, :, chan] = 0
            elif self.chan_style[chan] == 2:
                frame[:, :, chan] = 255
            elif self.chan_style[chan] == 3:
                frame[:, :, chan] = frame_thresh

        if not self.optimize:
            frame = cv2.medianBlur(frame, 11)

        return frame

    def reset(self):
        super(Threshold, self).reset()
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]            # effect selector for each chan


class Alien(Effect):
    """
    Effect found in GIMP:Colors > Map > Alien Map

    KEYBOARD INPUTS:
        -/+ - decrease/increase sine frequency
        [/] - decrease/increase sine phase
        ;/' - None
        ,/. - None
        r/g/b - select red/green/blue channel for further processing
        0 - selected channel uses original values
        1 - selected channel uses all pixels as 0
        2 - selected channel uses all pixels as 255
        3 - selected channel uses alien effect
        / - reset parameters
        q - quit alien effect
    """

    def __init__(self):

        super(Alien, self).__init__()

        # user option constants
        FREQ = {
            'DESC': 'frequency of sine function',
            'NAME': 'freq',
            'VAL': 0.2,
            'INIT': 0.2,
            'MIN': 0.0,
            'MAX': 10.0,
            'MOD': INF,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        PHASE = {
            'DESC': 'phase of sine function',
            'NAME': 'phase',
            'VAL': 0,
            'INIT': 0,
            'MIN': -INF,
            'MAX': INF,
            'MOD': 2 * np.pi,
            'STEP': np.pi / 10.0,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            FREQ,
            PHASE,
            NONE,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # other user options
        self.optimize = 1                       # skips a smoothing step
        self.use_chan = [False, False, False]   # rgb channel selector
        self.chan_style = [0, 0, 0]             # effect selector for each chan
        self.chan_freq = [                      # current freq for each chan
            self.PROPS[0]['INIT'],
            self.PROPS[0]['INIT'],
            self.PROPS[0]['INIT']]
        self.chan_phase = [                     # current phase for each chan
            self.PROPS[1]['INIT'],
            self.PROPS[1]['INIT'],
            self.PROPS[1]['INIT']]
        self.MAX_NUM_CHAN_STYLES = 4

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            if key_list[ord('b')]:
                key_list[ord('b')] = False
                self.use_chan[0] = True
                self.use_chan[1] = False
                self.use_chan[2] = False
                self.PROPS[0]['VAL'] = self.chan_freq[0]
                self.PROPS[1]['VAL'] = self.chan_phase[0]
            elif key_list[ord('g')]:
                key_list[ord('g')] = False
                self.use_chan[0] = False
                self.use_chan[1] = True
                self.use_chan[2] = False
                self.PROPS[0]['VAL'] = self.chan_freq[1]
                self.PROPS[1]['VAL'] = self.chan_phase[1]
            elif key_list[ord('r')]:
                key_list[ord('r')] = False
                self.use_chan[0] = False
                self.use_chan[1] = False
                self.use_chan[2] = True
                self.PROPS[0]['VAL'] = self.chan_freq[2]
                self.PROPS[1]['VAL'] = self.chan_phase[2]
            self._process_io(key_list)

        # process options
        for chan in range(3):
            if self.use_chan[chan]:
                # update channel style
                for chan_style in range(self.MAX_NUM_CHAN_STYLES):
                    if key_list[ord(str(chan_style))]:
                        self.chan_style[chan] = chan_style
                        key_list[ord(str(chan_style))] = False
                # update channel params
                self.chan_freq[chan] = self.PROPS[0]['VAL']
                self.chan_phase[chan] = self.PROPS[1]['VAL']

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']
            for chan in range(3):
                self.chan_freq[chan] = self.PROPS[0]['INIT']
                self.chan_phase[chan] = self.PROPS[1]['INIT']

        # process image
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        frame_orig = frame
        frame_orig = frame_orig.astype('float16')
        frame = frame.astype('float16')

        for chan in range(3):
            if self.chan_style[chan] == 0:
                frame[:, :, chan] = frame_orig[:, :, chan]
            elif self.chan_style[chan] == 1:
                frame[:, :, chan] = 0
            elif self.chan_style[chan] == 2:
                frame[:, :, chan] = 255
            elif self.chan_style[chan] == 3:
                frame[:, :, chan] = 128 + 127 * np.cos(frame[:, :, chan] *
                                                       self.chan_freq[chan] +
                                                       self.chan_phase[chan])

        frame = frame.astype('uint8')

        return frame

    def reset(self):
        super(Alien, self).reset()
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]  # effect selector for each chan
        self.chan_freq = [  # current freq for each chan
            self.PROPS[0]['INIT'],
            self.PROPS[0]['INIT'],
            self.PROPS[0]['INIT']]
        self.chan_phase = [  # current phase for each chan
            self.PROPS[1]['INIT'],
            self.PROPS[1]['INIT'],
            self.PROPS[1]['INIT']]


class RGBWalk(Effect):
    """
    Use outline from grayscale thresholding in each of randomly drifting color
    channels

    KEYBOARD INPUTS:
        t - toggle between effect types 
        w - reset random walk
        -/+ - None
        [/] - None
        ;/' - None
        ,/. - decrease/increase random walk step size
        / - reset random walk
        q - quit rgbwalk effect
    """

    def __init__(self):

        super(RGBWalk, self).__init__()

        # user option constants
        STEP_SIZE = {
            'DESC': 'step size that scales random walk',
            'NAME': 'step_size',
            'VAL': 5.0,
            'INIT': 5.0,
            'MIN': 0.5,
            'MAX': 15.0,
            'MOD': INF,
            'STEP': 1.0,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            NONE,
            NONE,
            NONE,
            STEP_SIZE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False           # reset random walk
        self.random_walk = True
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()

        # human-readable names
        step_size = self.PROPS[3]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.adaptiveThreshold(
            frame_gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 5)
        frame_gray = cv2.medianBlur(frame_gray, 7)

        # update noise values
        self.chan_vec_pos += np.reshape(
            step_size * self.noise.get_next_vals(), (3, 2))

        # translate channels
        if self.style == 0:

            for chan in range(3):
                frame[:, :, chan] = cv2.warpAffine(
                    frame_gray,
                    np.float32([[1, 0, self.chan_vec_pos[chan, 0]],
                                [0, 1, self.chan_vec_pos[chan, 1]]]),
                    (im_width, im_height))

        elif self.style == 1:

            x_dir = self.chan_vec_pos[0, 1]
            y_dir = self.chan_vec_pos[1, 1]
            norm_dirs = [x_dir, y_dir] / np.linalg.norm([x_dir, y_dir])
            for chan in range(3):
                step_len = 0.1 * step_size * self.chan_vec_pos[chan, 0]
                frame[:, :, chan] = cv2.warpAffine(
                    frame_gray,
                    np.float32([[1, 0, x_dir + step_len * norm_dirs[0]],
                                [0, 1, y_dir + step_len * norm_dirs[1]]]),
                    (im_width, im_height))

        return frame


class RGBBurst(Effect):
    """
    Expand outline from grayscale thresholding differently in each color 
    channel

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        -/+ - decrease/increase interval at which burst takes place
        [/] - decrease/increase frame decay rate (apparent speed)
        ;/' - decrease/increase frame expansion rate (tail length)
        ,/. - decrease/increase random walk step size
        / - reset parameters
        q - quit rgbburst effect
    """

    def __init__(self):

        super(RGBBurst, self).__init__()

        # user option constants
        FRAME_INT = {
            'DESC': 'interval at which burst takes place',
            'NAME': 'frame_interval',
            'VAL': 10,
            'INIT': 10,
            'MIN': 1,
            'MAX': 100,
            'MOD': INF,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        FRAME_DECAY = {
            'DESC': 'decay rate for background frame luminance',
            'NAME': 'frame_decay',
            'VAL': 0.8,
            'INIT': 0.8,
            'MIN': 0.3,
            'MAX': 0.99,
            'MOD': INF,
            'STEP': 0.01,
            'INC': False,
            'DEC': False}
        EXP_RATE = {
            'DESC': 'expansion rate of background frame',
            'NAME': 'frame_expansion_rate',
            'VAL': 1.1,
            'INIT': 1.1,
            'MIN': 1.01,
            'MAX': 2.0,
            'MOD': INF,
            'STEP': 0.01,
            'INC': False,
            'DEC': False}
        STEP_SIZE = {
            'DESC': 'step size that scales random walk',
            'NAME': 'step_size',
            'VAL': 5.0,
            'INIT': 5.0,
            'MIN': 0.5,
            'MAX': 15.0,
            'MOD': INF,
            'STEP': 1.0,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            FRAME_INT,
            FRAME_DECAY,
            EXP_RATE,
            STEP_SIZE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # background frame parameters
        self.frame_cnt = 0                              # frame counter
        self.frame = None                               # background frame

    def process(self, frame, key_list, key_lock=False):

        # update frame info
        if self.frame_cnt is 0:
            self.frame = frame
        self.frame_cnt += 1

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        frame_interval = self.PROPS[0]['VAL']
        frame_decay = self.PROPS[1]['VAL']
        frame_expansion_rate = self.PROPS[2]['VAL']
        step_size = self.PROPS[3]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # random walk
        if self.random_walk:
            frame = cv2.warpAffine(
                frame,
                np.float32([[1, 0, self.chan_vec_pos[0, 0]],
                            [0, 1, self.chan_vec_pos[0, 1]]]),
                (im_width, im_height))
            # update noise values
            self.chan_vec_pos += np.reshape(
                step_size * self.noise.get_next_vals(), (3, 2))

        if self.style == 0:

            # update background frame
            frame_exp = cv2.resize(
                self.frame,
                None,
                fx=frame_expansion_rate,
                fy=frame_expansion_rate,
                interpolation=cv2.INTER_LINEAR)
            [im_exp_height, im_exp_width, _] = frame_exp.shape
            self.frame = cv2.getRectSubPix(
                frame_exp,
                (im_width, im_height),
                (im_exp_width / 2, im_exp_height / 2))
            self.frame = cv2.addWeighted(
                0, 1.0-frame_decay,
                self.frame, frame_decay, 0)

        elif self.style == 1:

            # same as style 0, but channels are differentially modulated

            # update background frame
            rate_diff = [1, 1.05, 1.1]
            for chan in range(3):
                frame_exp = cv2.resize(
                    self.frame[:, :, chan],
                    None,
                    fx=frame_expansion_rate * rate_diff[chan],
                    fy=frame_expansion_rate * rate_diff[chan],
                    interpolation=cv2.INTER_LINEAR)
                [im_exp_height, im_exp_width] = frame_exp.shape
                self.frame[:, :, chan] = cv2.getRectSubPix(
                    frame_exp,
                    (im_width, im_height),
                    (im_exp_width / 2,
                     im_exp_height / 2))

            self.frame = cv2.addWeighted(0, 1.0 - frame_decay,
                                         self.frame, frame_decay, 0)

        elif self.style == 2:

            # same as style 1, but channels are differentially modulated using
            # grayscale image

            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

            # update background frame
            rate_diff = [1, 1.05, 1.1]
            for chan in range(3):
                frame_exp = cv2.resize(
                    frame_gray,
                    None,
                    fx=frame_expansion_rate * rate_diff[chan],
                    fy=frame_expansion_rate * rate_diff[chan],
                    interpolation=cv2.INTER_LINEAR)
                [im_exp_height, im_exp_width] = frame_exp.shape
                self.frame[:, :, chan] = cv2.getRectSubPix(
                    frame_exp,
                    (im_width, im_height),
                    (im_exp_width / 2,
                     im_exp_height / 2))

            self.frame = cv2.addWeighted(0, 1.0 - frame_decay,
                                         self.frame, frame_decay, 0)

        frame_ret = cv2.bitwise_or(self.frame, frame)

        # add new frame periodically
        if self.frame_cnt % frame_interval == 0:
            self.frame += frame

        return frame_ret

    def reset(self):
        super(RGBBurst, self).reset()
        self.frame_cnt = 0  # frame counter
        self.frame = None   # background frame


class HueBloom(Effect):
    """
    Create bloom effect around thresholded version of input frame

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        -/+ - decrease/increase random matrix size
        [/] - decrease/increase bloom size
        ;/' - None
        ,/. - None
        / - reset parameters
        q - quit huebloom effect
    """

    def __init__(self):

        super(HueBloom, self).__init__()

        # user option constants
        DIM_SIZE = {
            'DESC': 'dimension of random matrix height/width',
            'NAME': 'dim_size',
            'VAL': 10,
            'INIT': 10,
            'MIN': 2,
            'MAX': 100,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        BLUR_KERNEL = {
            'DESC': 'size of background blur kernel',
            'NAME': 'blur_kernel',
            'VAL': 3,
            'INIT': 3,
            'MIN': 3,
            'MAX': 51,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        PULSE_FREQ = {
            'DESC': 'frequency of blur_kernal sine modulator',
            'NAME': 'pulse_freq',
            'VAL': 0.1,
            'INIT': 0.1,
            'MIN': 0.05,
            'MAX': 5,
            'MOD': INF,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            DIM_SIZE,
            BLUR_KERNEL,
            PULSE_FREQ,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # frame parameters
        self.frame_count = 0
        self.prev_dim_size = self.PROPS[0]['INIT']
        self.frame = np.ones(
            (self.PROPS[0]['INIT'], self.PROPS[0]['INIT'], 3))
        self.frame[:, :, 0] = np.random.rand(
            self.PROPS[0]['INIT'], self.PROPS[0]['INIT'])

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        dim_size = self.PROPS[0]['VAL']
        blur_kernel = self.PROPS[1]['VAL']
        pulse_freq = self.PROPS[2]['VAL']

        # update blur kernel size
        val = 0.5 + 0.5 * np.sin(self.frame_count * pulse_freq)
        self.frame_count += 1
        blur_kernel = int(val * (self.PROPS[1]['MAX'] - self.PROPS[1]['MIN']) +
                          self.PROPS[1]['MIN'])

        # make kernel odd-size
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # create new random matrix if necessary
        if int(dim_size) is not int(self.prev_dim_size):
            self.prev_dim_size = dim_size
            self.frame = np.ones((dim_size, dim_size, 3))
            self.frame[:, :, 0] = np.random.rand(dim_size, dim_size)

        # get background
        frame_back = cv2.resize(
            self.frame,
            (im_width, im_height),
            interpolation=cv2.INTER_CUBIC)
        frame_back = 179.0 * frame_back
        frame_back = frame_back.astype('uint8')
        frame_back = cv2.cvtColor(frame_back, cv2.COLOR_HSV2BGR)

        # get mask
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.medianBlur(frame_gray, 5)
        frame_mask = cv2.adaptiveThreshold(
            frame_gray,
            255,  # thresh ceil
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,   # thresh block
            4     # thresh bias
        )

        # get blurred, masked background
        for chan in range(3):
            frame_back[:, :, chan] = cv2.bitwise_and(
                frame_back[:, :, chan],
                frame_mask)
        frame_back = cv2.GaussianBlur(
            frame_back,
            (blur_kernel, blur_kernel),
            0)

        # remask blurred background
        for chan in range(3):
            frame[:, :, chan] = cv2.bitwise_and(
                frame_back[:, :, chan],
                255 - frame_mask)

        return frame

    def reset(self):
        super(HueBloom, self).reset()
        self.frame_count = 0
        self.prev_dim_size = self.PROPS[0]['INIT']
        self.frame = np.ones(
            (self.PROPS[0]['INIT'], self.PROPS[0]['INIT'], 3))
        self.frame[:, :, 0] = np.random.rand(
            self.PROPS[0]['INIT'], self.PROPS[0]['INIT'])


class HueSwirl(Effect):
    """
    Create bloom effect around thresholded version of input frame then melt it
    with iteratively applying blur kernels. Iterative blur occurs 
    automatically until reaching a stopping point defined in ITER_INDEX dict,
    then reverses until no blur occurs. Then an automatic change in the mask 
    blur kernel size occurs, and the iterative blur kernel operation continues
    again. Once the entire loop has gone back and forth in blur kernel size 
    space the blur type is switched from median to Gaussian or vice-versa.
    Note: This effect is a bit messy now, needs cleaning up

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        a - toggle automatic behavior (vs keyboard input)
        -/+ - decrease/increase random matrix size
        [/] - decrease/increase bloom size
        ;/' - decrease/increase mask blur kernel
        ,/. - decrease/increase final masking offset value
        ud arrows - walk through blur iterations
        lr arrows - decrease/increase offset in background huespace
        / - reset parameters
        q - quit hueswirl effect
    """

    def __init__(self):

        super(HueSwirl, self).__init__()

        # user option constants
        DIM_SIZE = {
            'DESC': 'dimension of background random matrix height/width',
            'NAME': 'dim_size',
            'VAL': 2,
            'INIT': 2,
            'MIN': 2,
            'MAX': 100,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        BACKGROUND_BLUR_KERNEL = {
            'DESC': 'kernel size for Gaussian blur that produces bloom',
            'NAME': 'back_blur',
            'VAL': 19,
            'INIT': 19,
            'MIN': 3,
            'MAX': 31,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        MASK_BLUR_KERNEL = {
            'DESC': 'kernel size for Gauss/med blur that acts on mask',
            'NAME': 'mask_blur',
            'VAL': 5,
            'INIT': 5,
            'MIN': 5,
            'MAX': 31,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        ITER_INDEX = {
            'DESC': 'index into blurring iterations',
            'NAME': 'iter_index',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 75,
            'MOD': INF,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        FINAL_MASK_OFFSET = {
            'DESC': 'mask is subtracted from this value before final masking',
            'NAME': 'mask_offset',
            'VAL': 255,
            'INIT': 255,
            'MIN': -INF,
            'MAX': INF,
            'MOD': 255,
            'STEP': 5,
            'INC': False,
            'DEC': False}
        HUE_OFFSET = {
            'NAME': 'hue value offset for background frame',
            'VAL': 0,
            'INIT': 0,
            'MIN': -INF,
            'MAX': INF,
            'MOD': 180,
            'STEP': 5,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            DIM_SIZE,
            BACKGROUND_BLUR_KERNEL,
            MASK_BLUR_KERNEL,
            FINAL_MASK_OFFSET,
            ITER_INDEX,
            HUE_OFFSET]

        # user options
        self.style = 0
        self.auto_play = True
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # frame parameters
        self.prev_mask_blur = 0  # to initialize frame_mask_list
        self.prev_hue_offset = self.PROPS[5]['INIT']
        self.prev_dim_size = self.PROPS[0]['INIT']
        self.frame_back_0 = np.ones(
            (self.PROPS[0]['INIT'], self.PROPS[0]['INIT'], 3))
        self.frame_back_0[:, :, 0] = \
            np.random.rand(self.PROPS[0]['INIT'], self.PROPS[0]['INIT'])
        self.frame_back = None
        self.frame_mask_list = [None for _ in
                                range(self.PROPS[4]['MAX'] + 1)]
        self.frame_mask = None

        # control parameters
        self.increase_index = True
        self.increase_meta_index = True

    def process(self, frame, key_list, key_lock=False):

        # update if blur kernel toggled
        if key_list[ord('t')]:
            reset_iter_seq = True
        else:
            reset_iter_seq = False

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']
            self.frame_mask_list = \
                [None for _ in range(self.PROPS[4]['MAX'] + 1)]
            self.increase_index = True
            self.increase_meta_index = True

        # control parameters
        if self.auto_play:
            if self.increase_index:
                self.PROPS[4]['VAL'] += 1
            else:
                self.PROPS[4]['VAL'] -= 1
            if self.PROPS[4]['VAL'] == self.PROPS[4]['MAX']:
                self.increase_index = False
            if self.PROPS[4]['VAL'] == self.PROPS[4]['MIN']:
                self.increase_index = True
                # change meta index
                if self.increase_meta_index:
                    self.PROPS[2]['VAL'] += self.PROPS[2]['STEP']
                else:
                    self.PROPS[2]['VAL'] -= self.PROPS[2]['STEP']
                if self.PROPS[2]['VAL'] == self.PROPS[2]['MAX']:
                    self.increase_meta_index = False
                if self.PROPS[2]['VAL'] == self.PROPS[2]['MIN']:
                    self.increase_meta_index = True
                    self.style = (self.style + 1) % self.MAX_NUM_STYLES

        # human-readable names
        dim_size = self.PROPS[0]['VAL']
        back_blur = self.PROPS[1]['VAL']
        mask_blur = self.PROPS[2]['VAL']
        final_offset = self.PROPS[3]['VAL']
        iter_index = self.PROPS[4]['VAL']
        hue_offset = self.PROPS[5]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # create new random matrix if necessary
        if int(dim_size) is not int(self.prev_dim_size):
            self.prev_dim_size = dim_size
            self.frame_back_0 = np.ones((dim_size, dim_size, 3))
            self.frame_back_0[:, :, 0] = np.random.rand(dim_size, dim_size)
            self.frame_back = None

        # create background frame if necessary
        if self.frame_back is None:
            # get resized background
            self.frame_back = cv2.resize(
                self.frame_back_0,
                (im_width, im_height),
                interpolation=cv2.INTER_CUBIC)
            self.frame_back[:, :, 0] = 179.0 * self.frame_back[:, :, 0]
            self.frame_back[:, :, 1:3] = 255.0 * self.frame_back[:, :, 1:3]
            self.frame_back = self.frame_back.astype('uint8')
            self.frame_back = cv2.cvtColor(self.frame_back,
                                           cv2.COLOR_HSV2BGR)

        # update background frame if necessary
        if int(hue_offset) is not int(self.prev_hue_offset):
            self.frame_back = cv2.cvtColor(self.frame_back,
                                           cv2.COLOR_BGR2HSV)
            # uint8s don't play nice with subtraction
            self.frame_back[:, :, 0] += abs(
                int(hue_offset - self.prev_hue_offset))
            self.frame_back[:, :, 0] = np.mod(self.frame_back[:, :, 0],
                                              180)
            self.frame_back = cv2.cvtColor(self.frame_back,
                                           cv2.COLOR_HSV2BGR)
            self.prev_hue_offset = hue_offset

        # get mask if necessary
        if int(mask_blur) is not int(
                self.prev_mask_blur) or reset_iter_seq:
            # blur kernel changed; restart iteration sequence
            self.PROPS[4]['VAL'] = self.PROPS[4]['INIT']
            iter_index = self.PROPS[4]['VAL']
            self.increase_index = True
            self.frame_mask_list = \
                [None for _ in range(self.PROPS[4]['MAX'] + 1)]
            # get new mask
            self.frame_mask = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.style == 0:
                self.frame_mask = cv2.medianBlur(
                    self.frame_mask,
                    mask_blur)
            elif self.style == 1:
                self.frame_mask = cv2.GaussianBlur(
                    self.frame_mask,
                    (mask_blur, mask_blur),
                    0)
            self.frame_mask_list[0] = self.frame_mask
            self.prev_mask_blur = mask_blur

        # update mask if necessary
        if self.frame_mask_list[iter_index] is not None:
            self.frame_mask = self.frame_mask_list[iter_index]
        elif (self.frame_mask_list[iter_index] is None) and \
                (iter_index % 2 == 0):
            # need to update frame_mask
            if self.style == 0:
                # two blur passes from previously stored mask
                self.frame_mask = cv2.medianBlur(
                    self.frame_mask_list[iter_index - 2],
                    mask_blur)
                self.frame_mask = cv2.medianBlur(
                    self.frame_mask,
                    mask_blur)
            elif self.style == 1:
                self.frame_mask = cv2.GaussianBlur(
                    self.frame_mask_list[iter_index - 2],
                    (mask_blur, mask_blur),
                    0)
                self.frame_mask = cv2.GaussianBlur(
                    self.frame_mask,
                    (mask_blur, mask_blur),
                    0)
            # update frame_mask_list
            self.frame_mask_list[iter_index] = self.frame_mask
        elif (self.frame_mask_list[iter_index] is None) and \
                (iter_index % 2 == 1):
            # need to update frame_mask_list
            if self.style == 0:
                self.frame_mask = cv2.medianBlur(
                    self.frame_mask_list[iter_index - 1],
                    mask_blur)
            elif self.style == 1:
                self.frame_mask = cv2.GaussianBlur(
                    self.frame_mask_list[iter_index - 1],
                    (mask_blur, mask_blur),
                    0)

        # get masked then blurred background
        frame_back_blurred = np.zeros(self.frame_back.shape, dtype='uint8')
        for chan in range(3):
            frame_back_blurred[:, :, chan] = cv2.bitwise_and(
                self.frame_back[:, :, chan],
                self.frame_mask)
        frame_back_blurred = cv2.GaussianBlur(
            frame_back_blurred,
            (back_blur, back_blur),
            0)

        # remask blurred background
        frame = np.zeros(self.frame_back.shape, dtype='uint8')
        for chan in range(3):
            frame[:, :, chan] = cv2.bitwise_and(
                frame_back_blurred[:, :, chan],
                final_offset - self.frame_mask)

        return frame

    def reset(self):
        super(HueSwirl, self).reset()
        self.auto_play = True
        # frame parameters
        self.prev_mask_blur = 0  # to initialize frame_mask_list
        self.prev_dim_size = self.PROPS[0]['INIT']
        self.frame_back_0 = np.ones(
            (self.PROPS[0]['INIT'], self.PROPS[0]['INIT'], 3))
        self.frame_back_0[:, :, 0] = \
            np.random.rand(self.PROPS[0]['INIT'], self.PROPS[0]['INIT'])
        self.frame_back = None
        self.frame_mask_list = [None for _ in range(self.PROPS[4]['MAX'] + 1)]
        self.frame_mask = None
        # control parameters
        self.increase_index = True
        self.increase_meta_index = True


class HueSwirlMover(Effect):
    """
    Performs same processing steps as HueSwirl, but adds in a fading background 
    so that video can be used as input

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        a - toggle automatic behavior (vs keyboard input)
        -/+ - decrease/increase random matrix size
        [/] - decrease/increase bloom size
        ;/' - decrease/increase mask blur kernel
        ,/. - decrease/increase final masking offset value
        ud arrows - walk through blur iterations
        lr arrows - decrease/increase offset in background huespace
        / - reset parameters
        q - quit hueswirl effect
    """

    def __init__(self):

        super(HueSwirlMover, self).__init__()

        # user option constants
        DIM_SIZE = {
            'DESC': 'dimension of background random matrix height/width',
            'NAME': 'dim_size',
            'VAL': 2,
            'INIT': 2,
            'MIN': 2,
            'MAX': 100,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        BACKGROUND_BLUR_KERNEL = {
            'DESC': 'kernel size for Gaussian blur that produces bloom',
            'NAME': 'back_blur',
            'VAL': 19,
            'INIT': 19,
            'MIN': 3,
            'MAX': 31,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        MASK_BLUR_KERNEL = {
            'DESC': 'kernel size for Gauss/med blur that acts on mask',
            'NAME': 'mask_blur',
            'VAL': 5,
            'INIT': 5,
            'MIN': 5,
            'MAX': 31,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        ITER_INDEX = {
            'DESC': 'index into blurring iterations',
            'NAME': 'iter_index',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 75,
            'MOD': INF,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        BLEND_PROP = {
            'DESC': 'modulates weighting between new and previous frame',
            'NAME': 'blend_prop',
            'VAL': 0.5,
            'INIT': 0.5,
            'MIN': 0.05,
            'MAX': 1.0,
            'MOD': INF,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        HUE_OFFSET = {
            'NAME': 'hue value offset for background frame',
            'VAL': 0,
            'INIT': 0,
            'MIN': -INF,
            'MAX': INF,
            'MOD': 180,
            'STEP': 5,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            DIM_SIZE,
            BACKGROUND_BLUR_KERNEL,
            MASK_BLUR_KERNEL,
            BLEND_PROP,
            ITER_INDEX,
            HUE_OFFSET]

        # user options
        self.style = 0
        self.auto_play = False
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # frame parameters
        self.prev_hue_offset = self.PROPS[5]['INIT']
        self.prev_dim_size = self.PROPS[0]['INIT']
        self.frame_back_0 = \
            np.random.rand(self.PROPS[0]['INIT'], self.PROPS[0]['INIT'])
        self.frame_back = None
        self.frame_mask_blurred = None

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        dim_size = self.PROPS[0]['VAL']
        back_blur = self.PROPS[1]['VAL']
        mask_blur = self.PROPS[2]['VAL']
        blend_prop = self.PROPS[3]['VAL']
        hue_offset = self.PROPS[5]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # create new random matrix if necessary
        if int(dim_size) is not int(self.prev_dim_size):
            self.prev_dim_size = dim_size
            self.frame_back_0 = np.random.rand(dim_size, dim_size)
            self.frame_back = None

        # create background frame if necessary
        if self.frame_back is None:
            # get resized background
            self.frame_back = cv2.resize(
                self.frame_back_0,
                (im_width, im_height),
                interpolation=cv2.INTER_CUBIC)
            self.frame_back = 179.0 * self.frame_back
            self.frame_back = self.frame_back.astype('uint8')

        # update background frame if necessary
        if int(hue_offset) is not int(self.prev_hue_offset):
            # uint8s don't play nice with subtraction
            self.frame_back += abs(
                int(hue_offset - self.prev_hue_offset))
            self.frame_back = np.mod(self.frame_back, 180)
            self.frame_back = self.frame_back.astype('uint8')
            self.prev_hue_offset = hue_offset

        if self.style == 0:

            # get mask
            frame_mask = cv2.adaptiveThreshold(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                255,  # thresh ceil
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                21,  # thresh block
                4)  # thresh bias
            # reduce noise in mask
            frame_mask = cv2.medianBlur(frame_mask, 9)
            # blur mask
            if self.style == 0:
                frame_mask = cv2.medianBlur(
                    frame_mask,
                    mask_blur)
            elif self.style == 1:
                frame_mask = cv2.GaussianBlur(
                    frame_mask,
                    (mask_blur, mask_blur),
                    0)
            # add to previous masks
            if self.frame_mask_blurred is not None:
                self.frame_mask_blurred = cv2.addWeighted(
                    frame_mask,
                    blend_prop,
                    self.frame_mask_blurred,
                    1.0 - blend_prop,
                    0)
            else:
                self.frame_mask_blurred = frame_mask
            # make new mask stand out since it was weighted w/ previous mask
            frame_mask = cv2.bitwise_or(
                self.frame_mask_blurred,
                frame_mask)

            # mask background
            frame[:, :, 0] = self.frame_back
            frame[:, :, 1] = 255
            frame[:, :, 2] = frame_mask
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        elif self.style == 1:

            # get mask
            frame_mask = cv2.adaptiveThreshold(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                255,  # thresh ceil
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                21,  # thresh block
                4)  # thresh bias
            frame_mask = cv2.medianBlur(frame_mask, 9)  # noise reduction
            if self.style == 0:
                frame_mask = cv2.medianBlur(
                    frame_mask,
                    mask_blur)
            elif self.style == 1:
                frame_mask = cv2.GaussianBlur(
                    frame_mask,
                    (mask_blur, mask_blur),
                    0)
            if self.frame_mask_blurred is not None:
                self.frame_mask_blurred = cv2.addWeighted(
                    frame_mask,
                    blend_prop,
                    self.frame_mask_blurred,
                    1.0 - blend_prop,
                    0)
            else:
                self.frame_mask_blurred = frame_mask

            # get masked then blurred background
            frame_back_blurred = np.zeros(self.frame_back.shape, dtype='uint8')
            for chan in range(3):
                frame_back_blurred[:, :, chan] = cv2.bitwise_and(
                    self.frame_back[:, :, chan],
                    frame_mask)
            frame_back_blurred = cv2.GaussianBlur(
                frame_back_blurred,
                (back_blur, back_blur),
                0)

            # remask blurred background
            frame = np.zeros(self.frame_back.shape, dtype='uint8')
            for chan in range(3):
                frame[:, :, chan] = cv2.bitwise_and(
                    frame_back_blurred[:, :, chan],
                    255 - self.frame_mask_blurred)

        return frame

    def reset(self):
        super(HueSwirlMover, self).reset()
        self.auto_play = False
        # frame parameters
        self.prev_hue_offset = self.PROPS[5]['INIT']
        self.prev_dim_size = self.PROPS[0]['INIT']
        self.frame_back_0 = \
            np.random.rand(self.PROPS[0]['INIT'], self.PROPS[0]['INIT'])
        self.frame_back = None
        self.frame_mask_blurred = None


class HueCrusher(Effect):
    """
    Takes an input frame and partitions the hue values into a discrete number 
    of (equally spaced) continuous chunks. Can modify the center of the chunks 
    as well as their width.

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        -/+ - decrease/increase number of discrete chunks that divide huespace
        [/] - decrease/increase offset to all chunk centers
        ;/' - decrease/increase width of chunks
        ,/. - None
        lrud arrows - None
        / - reset parameters
        q - quit huecrusher effect
    """

    def __init__(self):

        super(HueCrusher, self).__init__()

        # user option constants
        NUM_CHUNKS = {
            'DESC': 'number of chunks that divide huespace',
            'NAME': 'num_chunks',
            'VAL': 3,
            'INIT': 3,
            'MIN': 1,
            'MAX': 10,
            'MOD': INF,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        CENTER_OFFSET = {
            'DESC': 'offset value to apply to chunk centers',
            'NAME': 'center_offset',
            'VAL': 0,
            'INIT': 0,
            'MIN': -INF,
            'MAX': INF,
            'MOD': 255,
            'STEP': 5,
            'INC': False,
            'DEC': False}
        CHUNK_WIDTH = {
            'DESC': 'width of each chunk in huespace',
            'NAME': 'chunk_width',
            'VAL': 16,
            'INIT': 16,
            'MIN': 3,
            'MAX': 180,
            'MOD': INF,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 1

        # combine dicts into a list for easy general access
        self.PROPS = [
            NUM_CHUNKS,
            CENTER_OFFSET,
            CHUNK_WIDTH,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # chunk params
        num_chunks = self.PROPS[0]['INIT']
        self.num_chunks_prev = num_chunks

        self.chunk_widths_og = int(180 / num_chunks)
        self.chunk_range_mins = np.zeros((num_chunks, 1), 'uint8')
        self.chunk_range_maxs = np.zeros((num_chunks, 1), 'uint8')
        self.chunk_centers = np.zeros((num_chunks, 1), 'uint8')
        for chunk in range(num_chunks):
            self.chunk_range_mins[chunk] = \
                chunk * self.chunk_widths_og
            self.chunk_range_maxs[chunk] = \
                (chunk + 1) * self.chunk_widths_og - 1
            self.chunk_centers[chunk] = self.chunk_range_mins[chunk] + \
                int(self.chunk_widths_og / 2)
        self.chunk_range_maxs[-1] = 179

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        # # mod out offset so it can circle around hue-space
        # self.PROPS[1]['VAL'] = self.PROPS[1]['VAL'] % 180

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.PROPS):
                self.PROPS[index]['VAL'] = self.PROPS[index]['INIT']

        # human-readable names
        num_chunks = self.PROPS[0]['VAL']
        center_offset = self.PROPS[1]['VAL']
        chunk_width = self.PROPS[2]['VAL']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # update params if number of chunks has been changed
        if int(num_chunks) is not int(self.num_chunks_prev):
            self.num_chunks_prev = num_chunks
            self.chunk_widths_og = int(180 / num_chunks)
            self.chunk_range_mins = np.zeros((num_chunks, 1), 'uint8')
            self.chunk_range_maxs = np.zeros((num_chunks, 1), 'uint8')
            self.chunk_centers = np.zeros((num_chunks, 1), 'uint8')
            for chunk in range(num_chunks):
                self.chunk_range_mins[chunk] = \
                    chunk * self.chunk_widths_og
                self.chunk_range_maxs[chunk] = \
                    (chunk + 1) * self.chunk_widths_og - 1
                self.chunk_centers[chunk] = self.chunk_range_mins[chunk] + \
                    int(self.chunk_widths_og / 2)
            self.chunk_range_maxs[-1] = 179

        # extract hue values
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_final = np.zeros((im_height, im_width), dtype='uint8')

        # scale whole image
        scale = chunk_width / self.chunk_widths_og
        frame_scaled = cv2.addWeighted(0, 1.0 - scale, frame, scale, 0)

        for chunk in range(num_chunks):

            chunk_center = self.chunk_centers[chunk]

            # get rid of values outside bounds of chunk
            hue_mask = cv2.inRange(frame[:, :, 0],
                                   self.chunk_range_mins[chunk],
                                   self.chunk_range_maxs[chunk])
            # hue_mask is now a mask for the location of values in a specific
            # hue band;
            hue = np.mod(frame_scaled[:, :, 0] + chunk_center -
                         int(chunk_width / 2) + center_offset, 180)
            hue = hue.astype('uint8')
            hue = cv2.bitwise_and(hue, hue_mask)

            # put into final hue array (bitwise or takes all nonzero vals)
            hue_final = cv2.bitwise_or(hue_final, hue)

        # return to bgr format
        frame[:, :, 0] = hue_final
        # frame[:, :, 1] = 255
        # frame[:, :, 2] = 255
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        return frame

    def reset(self):
        super(HueCrusher, self).reset()
        # chunk params
        num_chunks = self.PROPS[0]['INIT']
        self.num_chunks_prev = num_chunks
        self.chunk_widths_og = int(180 / num_chunks)
        self.chunk_range_mins = np.zeros((num_chunks, 1), 'uint8')
        self.chunk_range_maxs = np.zeros((num_chunks, 1), 'uint8')
        self.chunk_centers = np.zeros((num_chunks, 1), 'uint8')
        for chunk in range(num_chunks):
            self.chunk_range_mins[chunk] = \
                chunk * self.chunk_widths_og
            self.chunk_range_maxs[chunk] = \
                (chunk + 1) * self.chunk_widths_og - 1
            self.chunk_centers[chunk] = self.chunk_range_mins[chunk] + \
                int(self.chunk_widths_og / 2)
        self.chunk_range_maxs[-1] = 179
