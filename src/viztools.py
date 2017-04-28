from __future__ import print_function

import cv2
import numpy as np

import utils as util


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
        q - quit border effect
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
        self.PROPS = [MULT_FACTOR,
                      ZOOM_FACTOR,
                      ROT_ANGLE,
                      NONE,
                      SHIFT_PIX_HORZ,
                      SHIFT_PIX_VERT]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
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
        [im_height, im_width, _] = frame.shape

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
            frame = cv2.resize(frame, None,
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
            frame = cv2.resize(frame, None,
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
        if self.style == 0:
            self.THRESH_TYPE = cv2.THRESH_BINARY
        elif self.style == 1:
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
        frame_thresh = cv2.adaptiveThreshold(frame_gray,
                                             self.THRESH_CEIL,
                                             self.ADAPTIVE_THRESH_TYPE,
                                             self.THRESH_TYPE,
                                             self.THRESH_BLOCK,
                                             self.THRESH_C)
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
        #     if THRESH_STYLE == 0:
        #         THRESH_TYPE = cv2.THRESH_BINARY
        #     elif THRESH_STYLE == 1:
        #         THRESH_TYPE = cv2.THRESH_BINARY_INV
        #
        #     # MEAN_C | GAUSSIAN_C
        #     ADAPTIVE_THRESH_TYPE = cv2.ADAPTIVE_THRESH_MEAN_C
        #     THRESH_CEIL = 255
        #     THRESH_BLOCK = 15
        #     THRESH_C = 5
        #
        #     if ALIEN_R_CH_STYLE == 3:
        #         frame[:, :, 2] = cv2.adaptiveThreshold(frame[:, :, 2],
        #                                                THRESH_CEIL,
        #                                                ADAPTIVE_THRESH_TYPE,
        #                                                THRESH_TYPE,
        #                                                THRESH_BLOCK, THRESH_C)
        #     if ALIEN_G_CH_STYLE == 3:
        #         frame[:, :, 1] = cv2.adaptiveThreshold(frame[:, :, 1],
        #                                                THRESH_CEIL,
        #                                                ADAPTIVE_THRESH_TYPE,
        #                                                THRESH_TYPE,
        #                                                THRESH_BLOCK, THRESH_C)
        #     if ALIEN_B_CH_STYLE == 3:
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
        / - reset random walk
        t - toggle between effect types 
        -/+ - decrease/increase random walk step size
        q - quit rgbwalk effect
    """

    def __init__(self):

        # user option constants
        self.STEP_SIZE_MIN = 0.5
        self.STEP_SIZE_MAX = 15.0
        self.STEP_SIZE_INC = 1.0
        self.MAX_NUM_STYLES = 2

        # user options
        self.style = 0
        self.step_size = 5.0                # step size of random walk (pixels)
        self.reinitialize = False           # reset random walk
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(num_samples=10, num_channels=6)

        # key press parameters
        self.INC0 = False
        self.DEC0 = False
        self.INC1 = False
        self.DEC1 = False

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            if key_list[ord('-')]:
                key_list[ord('-')] = False
                self.DEC0 = True
            elif key_list[ord('=')]:
                key_list[ord('=')] = False
                self.INC0 = True
            elif key_list[ord('/')]:
                key_list[ord('/')] = False
                self.reinitialize = True
            elif key_list[ord('t')]:
                key_list[ord('t')] = False
                self.style = (self.style + 1) % self.MAX_NUM_STYLES
                self.reinitialize = True

        # process options
        if self.DEC0:
            self.DEC0 = False
            self.step_size -= self.STEP_SIZE_INC
        if self.INC0:
            self.INC0 = False
            self.step_size += self.STEP_SIZE_INC
        self.step_size = np.clip(self.step_size,
                                 self.STEP_SIZE_MIN, self.STEP_SIZE_MAX)
        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()

        # process image
        [im_height, im_width, _] = frame.shape
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.adaptiveThreshold(frame_gray, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 5)
        frame_gray = cv2.medianBlur(frame_gray, 7)

        # update noise values
        self.chan_vec_pos += np.reshape(
            self.step_size * self.noise.get_next_vals(), (3, 2))

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
                step_len = 0.1 * self.step_size * self.chan_vec_pos[chan, 0]
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
        ,/. - None
        / - reset parameters
        q - quit rgbburst effect
    """

    def __init__(self):

        # user option constants
        FRAME_INT = {
            'NAME': 'frame_interval',
            'VAL': 10,
            'INIT': 10,
            'MIN': 1,
            'MAX': 100,
            'STEP': 1,
            'INC': False,
            'DEC': False}
        FRAME_DECAY = {
            'NAME': 'frame_decay',
            'VAL': 0.8,
            'INIT': 0.8,
            'MIN': 0.3,
            'MAX': 0.99,
            'STEP': 0.01,
            'INC': False,
            'DEC': False}
        EXP_RATE = {
            'NAME': 'frame_expansion_rate',
            'VAL': 1.1,
            'INIT': 1.1,
            'MIN': 1.01,
            'MAX': 2.0,
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
        self.PROPS = [FRAME_INT,
                      FRAME_DECAY,
                      EXP_RATE,
                      NONE,
                      NONE,
                      NONE]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.step_size = 3.0  # multiplier on random walk step size
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(num_samples=10,
                                      num_channels=self.chan_vec_pos.size)

        # background frame parameters
        self.frame_cnt = 0                              # frame counter
        self.frame = None                               # background frame

    # def _process_io(self, key_list):
    #
    #     if key_list[ord('-')]:
    #         key_list[ord('-')] = False
    #         self.DEC0 = True
    #     elif key_list[ord('=')]:
    #         key_list[ord('=')] = False
    #         self.INC0 = True
    #     elif key_list[ord('[')]:
    #         key_list[ord('[')] = False
    #         self.DEC1 = True
    #     elif key_list[ord(']')]:
    #         key_list[ord(']')] = False
    #         self.INC1 = True
    #     elif key_list[ord(';')]:
    #         key_list[ord(';')] = False
    #         self.DEC2 = True
    #     elif key_list[ord('\'')]:
    #         key_list[ord('\'')] = False
    #         self.INC2 = True
    #     elif key_list[ord('/')]:
    #         key_list[ord('/')] = False
    #         self.reinitialize = True
    #     elif key_list[ord('t')]:
    #         key_list[ord('t')] = False
    #         self.style = (self.style + 1) % self.MAX_NUM_STYLES
    #         self.reinitialize = True
    #     elif key_list[ord('w')]:
    #         key_list[ord('w')] = False
    #         self.random_walk = not self.random_walk
    #         self.chan_vec_pos = np.zeros((3, 2))
    #         self.noise.reinitialize()
    #
    #     # process options
    #     if self.DEC0:
    #         self.DEC0 = False
    #         self.frame_int -= self.FRAME_INT_INC
    #     if self.INC0:
    #         self.INC0 = False
    #         self.frame_int += self.FRAME_INT_INC
    #     self.frame_int = np.clip(self.frame_int,
    #                              self.FRAME_INT_MIN,
    #                              self.FRAME_INT_MAX)
    #
    #     if self.DEC1:
    #         self.DEC1 = False
    #         self.frame_decay -= self.FRAME_DECAY_INC
    #     if self.INC1:
    #         self.INC1 = False
    #         self.frame_decay += self.FRAME_DECAY_INC
    #     self.frame_decay = np.clip(self.frame_decay,
    #                                self.FRAME_DECAY_MIN,
    #                                self.FRAME_DECAY_MAX)
    #
    #     if self.DEC2:
    #         self.DEC2 = False
    #         self.frame_exp_rate -= self.FRAME_EXP_RATE_INC
    #     if self.INC2:
    #         self.INC2 = False
    #         self.frame_exp_rate += self.FRAME_EXP_RATE_INC
    #     self.frame_exp_rate = np.clip(self.frame_exp_rate,
    #                                   self.FRAME_EXP_RATE_MIN,
    #                                   self.FRAME_EXP_RATE_MAX)

    def process(self, frame, key_list, key_lock=False):

        # update frame info
        if self.frame_cnt == 0:
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

        # process image
        [im_height, im_width, _] = frame.shape

        # random walk
        if self.random_walk:
            frame = cv2.warpAffine(
                frame,
                np.float32([[1, 0, self.chan_vec_pos[0, 0]],
                            [0, 1, self.chan_vec_pos[0, 1]]]),
                (im_width, im_height))
            # update noise values
            self.chan_vec_pos += np.reshape(
                self.step_size * self.noise.get_next_vals(), (3, 2))

        if self.style == 0:

            # update background frame
            frame_exp = cv2.resize(self.frame,
                                   None,
                                   fx=frame_expansion_rate,
                                   fy=frame_expansion_rate,
                                   interpolation=cv2.INTER_LINEAR)
            [im_exp_height, im_exp_width, _] = frame_exp.shape
            self.frame = cv2.getRectSubPix(
                frame_exp,
                (im_width, im_height),
                (im_exp_width / 2, im_exp_height / 2))
            self.frame = cv2.addWeighted(0, 1.0-frame_decay,
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


class Mask(Effect):
    """
    Manipulate mask on image

    KEYBOARD INPUTS:
        t - toggle between mask and inverse
        w - toggle random walk
        -/+ - decrease/increase adaptive threshold kernel size
        [/] - decrease/increase median blur kernel size
        ;/' - decrease/increase ?
        / - reset parameters
        q - quit mask effect
    """

    def __init__(self):

        # background frame constants
        self.THRESH_KERN_INIT = 15
        self.THRESH_KERN_MIN = 3
        self.THRESH_KERN_MAX = 75
        self.THRESH_KERN_INC = 2
        self.MEDIAN_KERN_INIT = 7
        self.MEDIAN_KERN_MIN = 3
        self.MEDIAN_KERN_MAX = 75
        self.MEDIAN_KERN_INC = 2
        # self.FRAME_EXP_RATE_INIT = 1.1
        # self.FRAME_EXP_RATE_MIN = 1.01
        # self.FRAME_EXP_RATE_MAX = 2.0
        # self.FRAME_EXP_RATE_INC = 0.01
        self.MAX_NUM_STYLES = 2

        # user options
        self.style = 0
        self.thresh_kern = self.THRESH_KERN_INIT
        self.median_kern = self.MEDIAN_KERN_INIT
        self.step_size = 3.0                # step size of random walk (pixels)
        self.reinitialize = False           # reset random walk
        self.random_walk = False
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(num_samples=10, num_channels=6)

        # key press parameters
        self.INC0 = False
        self.DEC0 = False
        self.INC1 = False
        self.DEC1 = False
        self.INC2 = False
        self.DEC2 = False

    def _process_io(self, key_list):

        if key_list[ord('-')]:
            key_list[ord('-')] = False
            self.DEC0 = True
        elif key_list[ord('=')]:
            key_list[ord('=')] = False
            self.INC0 = True
        elif key_list[ord('[')]:
            key_list[ord('[')] = False
            self.DEC1 = True
        elif key_list[ord(']')]:
            key_list[ord(']')] = False
            self.INC1 = True
        elif key_list[ord(';')]:
            key_list[ord(';')] = False
            self.DEC2 = True
        elif key_list[ord('\'')]:
            key_list[ord('\'')] = False
            self.INC2 = True
        elif key_list[ord('/')]:
            key_list[ord('/')] = False
            self.reinitialize = True
        elif key_list[ord('t')]:
            key_list[ord('t')] = False
            self.style = (self.style + 1) % self.MAX_NUM_STYLES
            self.reinitialize = True
        elif key_list[ord('w')]:
            key_list[ord('w')] = False
            self.random_walk = not self.random_walk
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()

        # process options
        if self.DEC0:
            self.DEC0 = False
            self.thresh_kern -= self.THRESH_KERN_INC
        if self.INC0:
            self.INC0 = False
            self.thresh_kern += self.THRESH_KERN_INC
        self.thresh_kern = np.clip(self.thresh_kern,
                                   self.THRESH_KERN_MIN,
                                   self.THRESH_KERN_MAX)

        if self.DEC1:
            self.DEC1 = False
            self.median_kern -= self.MEDIAN_KERN_INC
        if self.INC1:
            self.INC1 = False
            self.median_kern += self.MEDIAN_KERN_INC
        self.median_kern = np.clip(self.median_kern,
                                   self.MEDIAN_KERN_MIN,
                                   self.MEDIAN_KERN_MAX)

        # if self.DEC2:
        #     self.DEC2 = False
        #     self.frame_exp_rate -= self.FRAME_EXP_RATE_INC
        # if self.INC2:
        #     self.INC2 = False
        #     self.frame_exp_rate += self.FRAME_EXP_RATE_INC
        # self.frame_exp_rate = np.clip(self.frame_exp_rate,
        #                               self.FRAME_EXP_RATE_MIN,
        #                               self.FRAME_EXP_RATE_MAX)

    def process(self, frame_orig, key_list, key_lock=False):

        frame = np.copy(frame_orig)

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.thresh_kern = self.THRESH_KERN_INIT
            self.median_kern = self.MEDIAN_KERN_INIT
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()

        # process image
        [im_height, im_width, _] = frame.shape

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.style == 0:
            frame = cv2.adaptiveThreshold(frame_gray,
                                          255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          self.thresh_kern,
                                          5)
        elif self.style == 1:
            frame = cv2.adaptiveThreshold(frame_gray,
                                          255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY,
                                          self.thresh_kern,
                                          5)
        frame = cv2.medianBlur(frame, self.median_kern)

        # random walk
        if self.random_walk:
            frame = cv2.warpAffine(
                frame,
                np.float32([[1, 0, self.chan_vec_pos[0, 0]],
                            [0, 1, self.chan_vec_pos[0, 1]]]),
                (im_width, im_height))

        return frame
