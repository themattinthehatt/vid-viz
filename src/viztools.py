from __future__ import print_function

import cv2
import numpy as np

import utils as util

"""TODO
- classes that need refactoring with list/list
    threshold
    alien
"""


class Effect(object):
    """Base class for vid-viz effects"""

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
            self.PROPS[index]['VAL'] = np.clip(
                self.PROPS[index]['VAL'],
                self.PROPS[index]['MIN'],
                self.PROPS[index]['MAX'])

    def process(self, frame, key_list):
        raise NotImplementedError


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

        # user option constants
        MULT_FACTOR = {
            'NAME': 'mult_factor',
            'VAL': 1.0,
            'INIT': 1.0,
            'MIN': 0.01,
            'MAX': 1.0,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        ZOOM_FACTOR = {
            'NAME': 'zoom_factor',
            'VAL': 1.0,
            'INIT': 1.0,
            'MIN': 1.0,
            'MAX': 10.0,
            'STEP': 0.05,
            'INC': False,
            'DEC': False}
        ROT_ANGLE = {
            'NAME': 'rot_angle',
            'VAL': 0,
            'INIT': 0,
            'MIN': -360,
            'MAX': 360,
            'STEP': 5,
            'INC': False,
            'DEC': False}
        SHIFT_PIX_VERT = {
            'NAME': 'shift_vert',
            'VAL': 0,
            'INIT': 0,
            'MIN': -500,
            'MAX': 500,
            'STEP': 10,
            'INC': False,
            'DEC': False}
        SHIFT_PIX_HORZ = {
            'NAME': 'shift_horz',
            'VAL': 0,
            'INIT': 0,
            'MIN': -500,
            'MAX': 500,
            'STEP': 10,
            'INC': False,
            'DEC': False}
        NONE = {
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            MULT_FACTOR,
            ZOOM_FACTOR,
            ROT_ANGLE,
            NONE,
            SHIFT_PIX_HORZ,
            SHIFT_PIX_VERT]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(num_samples=10,
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
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
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
        if self.style is 1:
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
        elif self.style is 2:
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

        GAUSSIAN_KERN = {
            'DESC': 'kernel size for gaussian smoothing',
            'NAME': 'gauss_kern',
            'VAL': 7,
            'INIT': 7,
            'MIN': 3,
            'MAX': 75,
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
            'STEP': 2,
            'INC': False,
            'DEC': False}
        NONE = {
            'DESC': '',
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
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
        self.noise = util.SmoothNoise(num_samples=10,
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
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
            [im_height, im_width] = frame.shape

        # rotate
        if self.style is 1:
            frame_blurred = cv2.GaussianBlur(frame, (gauss_kern, gauss_kern), 0)
            # frame = cv2.addWeighted(
            #     frame, 0.5,
            #     frame_blurred, 0.5, 0)
            frame = cv2.bitwise_or(frame, frame_blurred)
        elif self.style is 2:
            frame = cv2.medianBlur(frame, median_kern)

        return frame


class Threshold(Effect):
    """
    Threshold individual channels in RGB frame
    
    KEYBOARD INPUTS:
        r/g/b - select red/green/blue channel for further processing
        0 - selected channel uses original values
        1 - selected channel uses all pixels as 0
        2 - selected channel uses all pixels as 255
        3 - selected channel uses threshold effect
        t - toggle between threshold styles
        q - quit threshold effect
    """

    def __init__(self):

        # user option constants
        self.MAX_NUM_THRESH_STYLES = 2
        self.MAX_NUM_CHAN_STYLES = 4

        # user options
        self.style = 0                          # maps to THRESH_TYPE
        self.optimize = 1                       # skips a smoothing step
        self.use_chan = [False, False, False]   # rgb channel selector
        self.chan_style = [0, 0, 0]             # effect selector for each chan

        # opencv parameters
        self.THRESH_TYPE = cv2.THRESH_BINARY
        self.ADAPTIVE_THRESH_TYPE = cv2.ADAPTIVE_THRESH_MEAN_C
        self.THRESH_CEIL = 255
        self.THRESH_BLOCK = 21
        self.THRESH_C = 4

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
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
            elif key_list[ord('t')]:
                key_list[ord('t')] = False
                self.style = (self.style + 1) % self.MAX_NUM_THRESH_STYLES

        # process options
        if self.style is 0:
            self.THRESH_TYPE = cv2.THRESH_BINARY
        elif self.style is 1:
            self.THRESH_TYPE = cv2.THRESH_BINARY_INV

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
            self.THRESH_CEIL,
            self.ADAPTIVE_THRESH_TYPE,
            self.THRESH_TYPE,
            self.THRESH_BLOCK,
            self.THRESH_C)
        for chan in range(3):
            if self.chan_style[chan] is 1:
                frame[:, :, chan] = 0
            elif self.chan_style[chan] is 2:
                frame[:, :, chan] = 255
            elif self.chan_style[chan] is 3:
                frame[:, :, chan] = frame_thresh

        if not self.optimize:
            frame = cv2.medianBlur(frame, 11)

        return frame


class Alien(Effect):
    """
    Effect found in GIMP:Colors > Map > Alien Map

    KEYBOARD INPUTS:
        r/g/b - select red/green/blue channel for further processing
        0 - selected channel uses original values
        1 - selected channel uses all pixels as 0
        2 - selected channel uses all pixels as 255
        3 - selected channel uses alien effect
        -/+ - decrease/increase frequency
        DOWN/UP arrow - decrease/increase phase
        / - reset frequency and phase values across all channels
        q - quit alien effect
    """

    def __init__(self):

        # user option constants
        self.FREQ_MIN = 0
        self.FREQ_MAX = 10
        self.FREQ_INC = 0.05
        self.PHASE_MIN = 0
        self.PHASE_MAX = 2 * np.pi
        self.PHASE_INC = np.pi / 10.0
        self.MAX_NUM_CHAN_STYLES = 4

        # user options
        self.style = 0                          # maps to THRESH_TYPE
        self.optimize = 1                       # skips a smoothing step
        self.use_chan = [False, False, False]   # rgb channel selector
        self.chan_style = [0, 0, 0]             # effect selector for each chan
        self.chan_freq = [0.2, 0.2, 0.2]        # current freq for each chan
        self.chan_phase = [0, 0, 0]             # current phase for each chan

        # key press parameters
        self.INC0 = False
        self.DEC0 = False
        self.INC1 = False
        self.DEC1 = False

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
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
            elif key_list[ord('/')]:
                key_list[ord('/')] = False
                # reset parameters
                self.chan_freq = [0.2, 0.2, 0.2]
                self.chan_phase = [0, 0, 0]
            elif key_list[ord('-')]:
                key_list[ord('-')] = False
                self.DEC0 = True
            elif key_list[ord('=')]:
                key_list[ord('=')] = False
                self.INC0 = True
            elif key_list[ord('T')]:
                key_list[ord('T')] = False
                self.DEC1 = True
            elif key_list[ord('R')]:
                key_list[ord('R')] = False
                self.INC1 = True

        # process options
        for chan in range(3):
            if self.use_chan[chan]:
                # update channel style
                for chan_style in range(self.MAX_NUM_CHAN_STYLES):
                    if key_list[ord(str(chan_style))]:
                        self.chan_style[chan] = chan_style
                        key_list[ord(str(chan_style))] = False
                # update channel freq
                if self.DEC0:
                    self.DEC0 = False
                    self.chan_freq[chan] -= self.FREQ_INC
                if self.INC0:
                    self.INC0 = False
                    self.chan_freq[chan] += self.FREQ_INC
                self.chan_freq[chan] = np.clip(self.chan_freq[chan],
                                               self.FREQ_MIN,
                                               self.FREQ_MAX)
                # update channel phase
                if self.DEC1:
                    self.DEC1 = False
                    self.chan_phase[chan] = (self.chan_phase[chan] -
                                             self.PHASE_INC) % self.PHASE_MAX
                if self.INC1:
                    self.INC1 = False
                    self.chan_phase[chan] = (self.chan_phase[chan] +
                                             self.PHASE_INC) % self.PHASE_MAX

        # process image
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        frame_orig = frame
        frame_orig.astype('float16')
        frame.astype('float16')

        for chan in range(3):
            if self.chan_style[chan] is 0:
                frame[:, :, chan] = frame_orig[:, :, chan]
            elif self.chan_style[chan] is 1:
                frame[:, :, chan] = 0
            elif self.chan_style[chan] is 2:
                frame[:, :, chan] = 255
            elif self.chan_style[chan] is 3:
                frame[:, :, chan] = 128 + 127 * np.cos(frame[:, :, chan] *
                                                       self.chan_freq[chan] +
                                                       self.chan_phase[chan])
                # if THRESH:
                #     frame[:, :, chan] = np.floor(frame[:, :, chan] / 255.0 *
                #                                NUM_LEVELS) / NUM_LEVELS*255.0

        frame.astype('uint8')

        # if THRESH:
        #     frame = cv2.medianBlur(frame, 9)
        # else:
        #     frame = cv2.GaussianBlur(frame, (9, 9), 0)
        #
        # # threshold to clean up effect
        # if THRESH:
        #     if THRESH_STYLE is 0:
        #         THRESH_TYPE = cv2.THRESH_BINARY
        #     elif THRESH_STYLE is 1:
        #         THRESH_TYPE = cv2.THRESH_BINARY_INV
        #
        #     # MEAN_C | GAUSSIAN_C
        #     ADAPTIVE_THRESH_TYPE = cv2.ADAPTIVE_THRESH_MEAN_C
        #     THRESH_CEIL = 255
        #     THRESH_BLOCK = 15
        #     THRESH_C = 5
        #
        #     if ALIEN_R_CH_STYLE is 3:
        #         frame[:, :, 2] = cv2.adaptiveThreshold(frame[:, :, 2],
        #                                                THRESH_CEIL,
        #                                                ADAPTIVE_THRESH_TYPE,
        #                                                THRESH_TYPE,
        #                                                THRESH_BLOCK, THRESH_C)
        #     if ALIEN_G_CH_STYLE is 3:
        #         frame[:, :, 1] = cv2.adaptiveThreshold(frame[:, :, 1],
        #                                                THRESH_CEIL,
        #                                                ADAPTIVE_THRESH_TYPE,
        #                                                THRESH_TYPE,
        #                                                THRESH_BLOCK, THRESH_C)
        #     if ALIEN_B_CH_STYLE is 3:
        #         frame[:, :, 0] = cv2.adaptiveThreshold(frame[:, :, 0],
        #                                                THRESH_CEIL,
        #                                                ADAPTIVE_THRESH_TYPE,
        #                                                THRESH_TYPE,
        #                                                THRESH_BLOCK, THRESH_C)
        #     frame = cv2.medianBlur(frame, 11)
        # else:
        #     frame = cv2.GaussianBlur(frame, (9, 9), 0)
        #
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.medianBlur(frame_gray, 11)
        # frame_thresh = cv2.adaptiveThreshold(frame_gray,
        #                                      self.THRESH_CEIL,
        #                                      self.ADAPTIVE_THRESH_TYPE,
        #                                      self.THRESH_TYPE,
        #                                      self.THRESH_BLOCK,
        #                                      self.THRESH_C)

        return frame


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

        # user option constants
        STEP_SIZE = {
            'DESC': 'step size that scales random walk',
            'NAME': 'step_size',
            'VAL': 5.0,
            'INIT': 5.0,
            'MIN': 0.5,
            'MAX': 15.0,
            'STEP': 1.0,
            'INC': False,
            'DEC': False}
        NONE = {
            'DESC': '',
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
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
        self.noise = util.SmoothNoise(num_samples=10,
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
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
            [im_height, im_width] = frame.shape

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.adaptiveThreshold(
            frame_gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 5)
        frame_gray = cv2.medianBlur(frame_gray, 7)

        # update noise values
        self.chan_vec_pos += np.reshape(
            step_size * self.noise.get_next_vals(), (3, 2))

        # translate channels
        if self.style is 0:

            for chan in range(3):
                frame[:, :, chan] = cv2.warpAffine(
                    frame_gray,
                    np.float32([[1, 0, self.chan_vec_pos[chan, 0]],
                                [0, 1, self.chan_vec_pos[chan, 1]]]),
                    (im_width, im_height))

        elif self.style is 1:

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

        # user option constants
        FRAME_INT = {
            'DESC': 'interval at which burst takes place',
            'NAME': 'frame_interval',
            'VAL': 10,
            'INIT': 10,
            'MIN': 1,
            'MAX': 100,
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
            'STEP': 1.0,
            'INC': False,
            'DEC': False}
        NONE = {
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
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
        self.noise = util.SmoothNoise(num_samples=10,
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
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
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

        if self.style is 0:

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

        elif self.style is 1:

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

        elif self.style is 2:

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
        if self.frame_cnt % frame_interval is 0:
            self.frame += frame

        return frame_ret


class Mask(Effect):
    """
    Manipulate mask on image

    KEYBOARD INPUTS:
        t - toggle between mask and inverse
        w - toggle random walk
        -/+ - decrease/increase adaptive threshold kernel size
        [/] - decrease/increase median blur kernel size
        ;/' - None
        ,/. - decrease/increase random walk step size
        / - reset parameters
        backspace - quit mask effect
    """

    def __init__(self):

        # user option constants
        THRESH_KERN = {
            'DESC': 'kernel size for thresholding operation',
            'NAME': 'thresh_kern',
            'VAL': 15,
            'INIT': 15,
            'MIN': 3,
            'MAX': 75,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        MEDIAN_KERN = {
            'DESC': 'kernel size for median filtering after threshold',
            'NAME': 'median_kern',
            'VAL': 7,
            'INIT': 7,
            'MIN': 3,
            'MAX': 75,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        STEP_SIZE = {
            'DESC': 'step size that scales random walk',
            'NAME': 'step_size',
            'VAL': 5.0,
            'INIT': 5.0,
            'MIN': 0.5,
            'MAX': 15.0,
            'STEP': 1.0,
            'INC': False,
            'DEC': False}
        NONE = {
            'DESC': '',
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            THRESH_KERN,
            MEDIAN_KERN,
            NONE,
            STEP_SIZE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(num_samples=10,
                                      num_channels=self.chan_vec_pos.size)

    def process(self, frame_orig, key_list, key_lock=False):

        frame = np.copy(frame_orig)

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
        thresh_kern = self.PROPS[0]['VAL']
        median_kern = self.PROPS[1]['VAL']
        step_size = self.PROPS[3]['VAL']

        # process image
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
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

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.style is 0:
            frame = cv2.adaptiveThreshold(
                frame_gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                thresh_kern,
                5)
        elif self.style is 1:
            frame = cv2.adaptiveThreshold(
                frame_gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                thresh_kern,
                5)
        frame = cv2.medianBlur(frame, median_kern)

        return frame


class HueCloud(Effect):
    """
    Use random values to modulate hue channel

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        -/+ - decrease/increase random matrix size
        [/] - None
        ;/' - None
        ,/. - None
        / - reset parameters
        q - quit huecloud effect
    """

    def __init__(self):

        # user option constants
        DIM_SIZE = {
            'DESC': 'dimension of random matrix height/width',
            'NAME': 'dim_size',
            'VAL': 40,
            'INIT': 40,
            'MIN': 2,
            'MAX': 200,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        NONE = {
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            DIM_SIZE,
            NONE,
            NONE,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(num_samples=10,
                                      num_channels=self.chan_vec_pos.size)

        # frame parameters
        self.use_hsv = 1
        self.frame_dim = 500
        self.frame = np.ones((DIM_SIZE['VAL'], DIM_SIZE['VAL'], 3))
        self.frame[:, :, 0] = np.random.rand(DIM_SIZE['VAL'], DIM_SIZE['VAL'])

    def process(self, frame, key_list, key_lock=False):

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
        dim_size = self.PROPS[0]['VAL']

        # process image
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
            [im_height, im_width] = frame.shape

        if self.use_hsv:

            # create new random matrix if necessary
            if im_height is not int(dim_size):
                self.frame = np.ones((dim_size, dim_size, 3))
                self.frame[:, :, 0] = np.random.rand(dim_size, dim_size)

            # transform random matrix into image
            frame_hsv = np.copy(self.frame)
            frame_hsv = 255.0 * frame_hsv
            frame_hsv = frame_hsv.astype('uint8')

            if self.style is 0:

                # update background frame
                frame = cv2.resize(
                    frame_hsv,
                    (self.frame_dim, self.frame_dim),
                    interpolation=cv2.INTER_LINEAR)

            elif self.style is 1:

                # same as style 0, but uses bicubic interpolation

                frame = cv2.resize(
                    frame_hsv,
                    (self.frame_dim, self.frame_dim),
                    interpolation=cv2.INTER_CUBIC)

            elif self.style is 2:

                # same as style 0, but uses bicubic interpolation

                frame = cv2.resize(
                    frame_hsv,
                    (self.frame_dim, self.frame_dim),
                    interpolation=cv2.INTER_LANCZOS4)

            # convert from hsv to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        else:

            # create new random matrix if necessary
            if im_height is not int(dim_size):
                self.frame = np.random.rand(dim_size, dim_size, 3)

            # transform random matrix into image
            frame = np.copy(self.frame)
            frame = 255.0 * frame
            frame = frame.astype('uint8')

            if self.style is 0:

                # update background frame
                frame = cv2.resize(
                    frame,
                    (self.frame_dim, self.frame_dim),
                    interpolation=cv2.INTER_LINEAR)

            elif self.style is 1:

                # same as style 0, but uses bicubic interpolation

                frame = cv2.resize(
                    frame,
                    (self.frame_dim, self.frame_dim),
                    interpolation=cv2.INTER_CUBIC)

            elif self.style is 2:

                # same as style 0, but uses bicubic interpolation

                frame = cv2.resize(
                    frame,
                    (self.frame_dim, self.frame_dim),
                    interpolation=cv2.INTER_LANCZOS4)

        return frame


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

        # user option constants
        DIM_SIZE = {
            'DESC': 'dimension of random matrix height/width',
            'NAME': 'dim_size',
            'VAL': 10,
            'INIT': 10,
            'MIN': 2,
            'MAX': 100,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        BACKGROUND_SCALE = {
            'DESC': 'reduction parameter for bloom substrate',
            'NAME': 'background_scale',
            'VAL': 0.10,
            'INIT': 0.10,
            'MIN': 0.01,
            'MAX': 1.0,
            'STEP': 0.01,
            'INC': False,
            'DEC': False}
        NONE = {
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 3

        # combine dicts into a list for easy general access
        self.PROPS = [
            DIM_SIZE,
            BACKGROUND_SCALE,
            NONE,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(num_samples=10,
                                      num_channels=self.chan_vec_pos.size)

        # frame parameters
        self.prev_dim_size = DIM_SIZE['VAL']
        self.frame = np.ones((DIM_SIZE['VAL'], DIM_SIZE['VAL'], 3))
        self.frame[:, :, 0] = np.random.rand(DIM_SIZE['VAL'], DIM_SIZE['VAL'])

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
        # back_scale = self.PROPS[1]['VAL']

        # change scale
        int_val = np.floor(80.0 * self.noise.get_next_vals()) / 1000.0
        back_scale = 0.15 + int_val

        # process image
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
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
        frame_back = 255.0 * frame_back
        frame_back = frame_back.astype('uint8')
        frame_back = cv2.cvtColor(frame_back, cv2.COLOR_HSV2BGR)

        # get mask
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.medianBlur(frame_gray, 11)
        frame_mask = cv2.adaptiveThreshold(
            frame_gray,
            255,  # thresh ceil
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,   # thresh block
            4     # thresh bias
        )
        frame_mask2 = cv2.adaptiveThreshold(
            frame_gray,
            255,  # thresh ceil
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            21,   # thresh block
            4     # thresh bias
        )

        # get blurred, masked background
        frame_back2 = frame_back
        for chan in range(3):
            frame_back2[:, :, chan] = cv2.bitwise_and(
                frame_back[:, :, chan],
                frame_mask)
        frame_back2 = cv2.resize(
            frame_back2,
            None,
            fx=back_scale,
            fy=back_scale,
            interpolation=cv2.INTER_LINEAR)
        frame_back2 = cv2.GaussianBlur(frame_back2, (5, 5), 0)
        frame_back2 = cv2.resize(
            frame_back2,
            (im_width, im_height),
            interpolation=cv2.INTER_LINEAR)

        # remask blurred background
        frame = frame_back2
        for chan in range(3):
            frame[:, :, chan] = cv2.bitwise_and(
                frame_back2[:, :, chan],
                frame_mask2)

        return frame


class HueSwirl(Effect):
    """
    Create bloom effect around thresholded version of input frame then melt it
    with iteratively applying blur kernels

    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        -/+ - decrease/increase random matrix size
        [/] - decrease/increase bloom size
        ;/' - None
        ,/. - None
        / - reset parameters
        q - quit hueswirleffect
    """

    def __init__(self):

        # user option constants
        DIM_SIZE = {
            'DESC': 'dimension of background random matrix height/width',
            'NAME': 'dim_size',
            'VAL': 2,
            'INIT': 2,
            'MIN': 2,
            'MAX': 100,
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
            'STEP': 2,
            'INC': False,
            'DEC': False}
        MASK_BLUR_KERNEL = {
            'DESC': 'kernel size for Gauss/med blur that acts on mask',
            'NAME': 'mask_blur',
            'VAL': 5,
            'INIT': 5,
            'MIN': 1,
            'MAX': 31,
            'STEP': 2,
            'INC': False,
            'DEC': False}
        NONE = {
            'NAME': '',
            'VAL': 0,
            'INIT': 0,
            'MIN': 0,
            'MAX': 1,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        self.MAX_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.PROPS = [
            DIM_SIZE,
            BACKGROUND_BLUR_KERNEL,
            MASK_BLUR_KERNEL,
            NONE,
            NONE,
            NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(num_samples=10,
                                      num_channels=self.chan_vec_pos.size)

        # frame parameters
        self.prev_mask_blur = MASK_BLUR_KERNEL['VAL']
        self.prev_dim_size = DIM_SIZE['VAL']
        self.frame_back_0 = np.ones((DIM_SIZE['VAL'], DIM_SIZE['VAL'], 3))
        self.frame_back_0[:, :, 0] = \
            np.random.rand(DIM_SIZE['VAL'], DIM_SIZE['VAL'])
        self.frame_back = None
        self.frame_mask = None

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

        # process image
        if len(frame.shape) is 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) is 2:
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
            self.frame_back = 255.0 * self.frame_back
            self.frame_back = self.frame_back.astype('uint8')
            self.frame_back = cv2.cvtColor(self.frame_back, cv2.COLOR_HSV2BGR)

        # get mask if necessary
        if self.frame_mask is None or \
                int(mask_blur) is not int(self.prev_mask_blur):
            self.frame_mask = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame_gray = cv2.medianBlur(frame_gray, 11)
            # self.frame_mask = cv2.adaptiveThreshold(
            #     frame_gray,
            #     255,  # thresh ceil
            #     cv2.ADAPTIVE_THRESH_MEAN_C,
            #     cv2.THRESH_BINARY_INV,
            #     21,  # thresh block
            #     4)  # thresh bias
            self.prev_mask_blur = mask_blur
        elif self.style is 0:
            self.frame_mask = cv2.medianBlur(self.frame_mask, mask_blur)
        elif self.style is 1:
            self.frame_mask = cv2.GaussianBlur(
                self.frame_mask,
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
                255 - self.frame_mask)

        return frame
