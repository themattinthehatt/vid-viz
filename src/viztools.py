from __future__ import print_function

import cv2
import numpy as np

import utils as util


KEY_BS = 8


class Effect(object):
    """Base class for vid-viz effects"""

    def process(self, frame, key_list):
        raise NotImplementedError


class Border(Effect):
    """
    Manipulate image borders

    KEYBOARD INPUTS:
        0-9 select border effect
        q - quit border effect
    """

    def __init__(self):

        # user option constants
        self.MAX_NUM_BORDER_STYLES = 3
        self.height_mult = 0.5
        self.width_mult = 0.5
        self.mult_inc = 0.05

        # user options
        self.style = 0

        # key press parameters
        self.INC0 = False
        self.DEC0 = False
        self.INC1 = False
        self.DEC1 = False

    def process(self, frame, key_list):

        # process keyboard input
        if key_list[ord('n')]:
            key_list[ord('n')] = False
            self.style = (self.style + 1) % self.MAX_NUM_BORDER_STYLES
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
        if self.DEC0:
            self.DEC0 = False
            self.height_mult -= self.mult_inc
            if self.height_mult < 0:
                self.height_mult = 0
            self.width_mult -= self.mult_inc
            if self.width_mult < 0:
                self.width_mult = 0
        if self.INC0:
            self.INC0 = False
            self.height_mult += self.mult_inc
            self.width_mult += self.width_mult

        # process image
        [im_height, im_width, im_channels] = frame.shape

        if self.style == 1:
            # top, bottom, left, right
            frame = cv2.copyMakeBorder(frame,
                                       int(im_height * self.height_mult),
                                       int(im_height * self.height_mult),
                                       int(im_width * self.width_mult),
                                       int(im_width * self.width_mult),
                                       cv2.BORDER_WRAP)
        elif self.style == 2:
            # top, bottom, left, right
            frame = cv2.copyMakeBorder(frame,
                                       int(im_height * 2 * self.height_mult),
                                       0,
                                       int(im_width * 2 * self.width_mult),
                                       0,
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

    def process(self, frame, key_list):

        # process keyboard input
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

    def process(self, frame, key_list):

        # process keyboard input
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

    def process(self, frame, key_list):

        # process keyboard input
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
        [im_height, im_width, im_channels] = frame.shape
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
        w - togglel random walk
        -/+ - decrease/increase interval at which burst takes place
        [/] - decrease/increase frame decay rate (apparent speed)
        ;/' - decrease/increase frame expansion rate (tail length)
        / - reset parameters
        q - quit rgbburst effect
    """

    def __init__(self):

        # background frame constants
        self.FRAME_INT_INIT = 10
        self.FRAME_INT_MIN = 1
        self.FRAME_INT_MAX = 100
        self.FRAME_INT_INC = 1
        self.FRAME_DECAY_INIT = 0.8
        self.FRAME_DECAY_MIN = 0.3
        self.FRAME_DECAY_MAX = 0.99
        self.FRAME_DECAY_INC = 0.01
        self.FRAME_EXP_RATE_INIT = 1.1
        self.FRAME_EXP_RATE_MIN = 1.01
        self.FRAME_EXP_RATE_MAX = 2.0
        self.FRAME_EXP_RATE_INC = 0.01
        self.MAX_NUM_STYLES = 2

        # background frame parameters
        self.frame_cnt = 0                              # frame counter
        self.frame_int = self.FRAME_INT_INIT            # rgb burst interval
        self.frame = None                               # background frame
        self.frame_decay = self.FRAME_DECAY_INIT        # background decay rate
        self.frame_exp_rate = self.FRAME_EXP_RATE_INIT  # background expansion

        # user options
        self.style = 0
        self.step_size = 3.0                # step size of random walk (pixels)
        self.reinitialize = False           # reset random walk
        self.random_walk = True
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
            self.frame_int -= self.FRAME_INT_INC
        if self.INC0:
            self.INC0 = False
            self.frame_int += self.FRAME_INT_INC
        self.frame_int = np.clip(self.frame_int,
                                 self.FRAME_INT_MIN,
                                 self.FRAME_INT_MAX)

        if self.DEC1:
            self.DEC1 = False
            self.frame_decay -= self.FRAME_DECAY_INC
        if self.INC1:
            self.INC1 = False
            self.frame_decay += self.FRAME_DECAY_INC
        self.frame_decay = np.clip(self.frame_decay,
                                   self.FRAME_DECAY_MIN,
                                   self.FRAME_DECAY_MAX)

        if self.DEC2:
            self.DEC2 = False
            self.frame_exp_rate -= self.FRAME_EXP_RATE_INC
        if self.INC2:
            self.INC2 = False
            self.frame_exp_rate += self.FRAME_EXP_RATE_INC
        self.frame_exp_rate = np.clip(self.frame_exp_rate,
                                      self.FRAME_EXP_RATE_MIN,
                                      self.FRAME_EXP_RATE_MAX)

    def process(self, frame, key_list):

        # update frame info
        if self.frame_cnt == 0:
            self.frame = frame
        self.frame_cnt += 1

        # process keyboard input
        self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.frame_int = self.FRAME_INT_INIT
            self.frame_decay = self.FRAME_DECAY_INIT
            self.frame_exp_rate = self.FRAME_EXP_RATE_INIT
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()

        # process image
        [im_height, im_width, _] = frame.shape

        # random walk
        if self.random_walk:
            frame = cv2.warpAffine(
                frame,
                np.float32([[1, 0, self.chan_vec_pos[0, 0]],
                            [0, 1, self.chan_vec_pos[0, 1]]]),
                (im_width, im_height))

        if self.style == 0:

            # update noise values
            self.chan_vec_pos += np.reshape(
                self.step_size * self.noise.get_next_vals(), (3, 2))

            # update background frame
            frame_exp = cv2.resize(self.frame,
                                   None,
                                   fx=self.frame_exp_rate,
                                   fy=self.frame_exp_rate,
                                   interpolation=cv2.INTER_LINEAR)
            [im_exp_height, im_exp_width, _] = frame_exp.shape
            self.frame = cv2.getRectSubPix(frame_exp, (im_width, im_height),
                                           (im_exp_width / 2, im_exp_height / 2))
            self.frame = cv2.addWeighted(0, 1.0-self.frame_decay,
                                         self.frame, self.frame_decay, 0)

        elif self.style == 1:
            pass

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, frame_gray = cv2.threshold(frame_gray,
        #                               128, 255,
        #                               cv2.THRESH_BINARY)
        # frame[:, :, 0] = frame_gray
        # frame[:, :, 1] = frame_gray
        # frame[:, :, 2] = frame_gray

        # translate channels
        # for chan in range(3):
        #     frame[:, :, chan] = cv2.warpAffine(
        #         frame_gray,
        #         np.float32([[1, 0, self.chan_vec_pos[chan, 0]],
        #                     [0, 1, self.chan_vec_pos[chan, 1]]]),
        #         (im_width, im_height))

        frame_ret = cv2.bitwise_or(self.frame, frame)

        # add new frame periodically
        if self.frame_cnt % self.frame_int == 0:
            self.frame += frame

        return frame_ret
