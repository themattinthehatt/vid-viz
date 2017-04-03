from __future__ import print_function

import cv2
import numpy as np


KEY_BS = 8


class Threshold(object):
    """
    Threshold individual channels in RGB frame
    
    KEYBOARD INPUTS:
        r/g/b - select red/green/blue channel for further processing
        0 - selected channel uses original values
        1 - selected channel uses all pixels as 0
        2 - selected channel uses all pixels as 255
        3 - selected channel uses threshold effect
        y - toggle between threshold styles
        q - quit threshold effect
    """

    def __init__(self):

        # user options
        self.style = 0                          # maps to THRESH_TYPE
        self.optimize = 1                       # skips a smoothing step
        self.use_chan = [False, False, False]   # rgb channel selector
        self.chan_style = [0, 0, 0]             # effect selector for each chan

        # user option constants
        self.MAX_NUM_THRESH_STYLES = 2
        self.MAX_NUM_CHAN_STYLES = 4

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
        elif key_list[ord('y')]:
            key_list[ord('y')] = False
            self.style = (self.style + 1) % self.MAX_NUM_THRESH_STYLES

        # process options
        if self.style == 0:
            self.THRESH_TYPE = cv2.THRESH_BINARY
        elif self.style == 1:
            self.THRESH_TYPE = cv2.THRESH_BINARY_INV

        """ 
        Channel styles:
            0 - original channel values
            1 - all pixels set to 0
            2 - all pixels set to 255
            3 - thresholded grayscale image
        """
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


class Alien(object):
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

        # user options
        self.style = 0                          # maps to THRESH_TYPE
        self.optimize = 1                       # skips a smoothing step
        self.use_chan = [False, False, False]   # rgb channel selector
        self.chan_style = [0, 0, 0]             # effect selector for each chan
        self.chan_freq = [0.2, 0.2, 0.2]        # current freq for each chan
        self.chan_phase = [0, 0, 0]             # current phase for each chan

        # user option constants
        self.FREQ_MIN = 0
        self.FREQ_MAX = 10
        self.FREQ_INC = 0.05
        self.PHASE_MIN = 0
        self.PHASE_MAX = 2 * np.pi
        self.PHASE_INC = np.pi / 10.0
        self.MAX_NUM_CHAN_STYLES = 4

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

        """ 
        Channel styles:
            0 - original channel values
            1 - all pixels set to 0
            2 - all pixels set to 255
            3 - cos applied to channel
        """
        for chan in range(3):
            if self.use_chan[chan]:
                # update channel style
                for chan_style in range(self.MAX_NUM_CHAN_STYLES):
                    if key_list[ord(str(chan_style))]:
                        self.chan_style[chan] = chan_style
                        key_list[ord(str(chan_style))] = False
                # update channel freq
                if self.DEC0:
                    self.chan_freq[chan] -= self.FREQ_INC
                    self.DEC0 = False
                if self.INC0:
                    self.chan_freq[chan] += self.FREQ_INC
                    self.INC0 = False
                self.chan_freq[chan] = np.clip(self.chan_freq[chan],
                                               self.FREQ_MIN,
                                               self.FREQ_MAX)
                # update channel phase
                if self.DEC1:
                    self.chan_phase[chan] = (self.chan_phase[chan] -
                                             self.PHASE_INC) % self.PHASE_MAX
                    self.DEC1 = False
                if self.INC1:
                    self.chan_phase[chan] = (self.chan_phase[chan] +
                                             self.PHASE_INC) % self.PHASE_MAX
                    self.INC1 = False

        # process image
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        frame_orig = frame
        frame_orig.astype('float16')
        frame.astype('float16')

        for chan in range(3):
            if self.chan_style[chan] == 0:
                frame[:, :, chan] = frame_orig[:, :, chan]
            if self.chan_style[chan] == 1:
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
