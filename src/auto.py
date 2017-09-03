"""
This file contains a library of automated image processing effects that borrow
heavily from the simpler effects found in viztools.py
"""

from __future__ import print_function
from __future__ import division

from os import listdir
from os.path import join, isfile
import cv2
import numpy as np

import viztools as vv
import utils as util
import cvutils as cvutil

INF = 1000
NONE = {
    'desc': 'unassigned',
    'name': '',
    'val': 0,
    'init': 0,
    'min': 0,
    'max': 1,
    'mod': INF,
    'step': 1,
    'inc': False,
    'dec': False}


class HueSwirlChain(vv.Effect):
    """
    Create bloom effect around thresholded version of input frame then melt it
    with iteratively applying blur kernels. Iterative blur occurs 
    automatically until reaching a stopping point defined in ITER_INDEX dict,
    then begins to transition to an unblurred version of a new image. 
    
    KEYBOARD INPUTS:
        t - toggle between effect types
        w - toggle random walk
        a - toggle automatic behavior (vs keyboard input)
        -/+ - decrease/increase random matrix size
        [/] - decrease/increase bloom size
        ;/' - decrease/increase mask blur kernel
        ,/. - decrease/increase final masking offset value
        lr arrows - decrease/increase offset in background huespace
        / - reset parameters
        spacebar - quit hueswirlchain (transition to new input source)
    """

    def __init__(self, frame_width, frame_height):

        super(HueSwirlChain, self).__init__()

        # user option constants
        DIM_SIZE = {
            'desc': 'dimension of background random matrix height/width',
            'name': 'dim_size',
            'val': 1,
            'init': 1,
            'min': 1,
            'max': 100,
            'mod': INF,
            'step': 1,
            'inc': False,
            'dec': False}
        BACKGROUND_BLUR_KERNEL = {
            'desc': 'kernel size for Gaussian blur that produces bloom',
            'name': 'back_blur',
            'val': 19,
            'init': 19,
            'min': 3,
            'max': 31,
            'mod': INF,
            'step': 2,
            'inc': False,
            'dec': False}
        MASK_BLUR_KERNEL = {
            'desc': 'kernel size for Gauss/med blur that acts on mask',
            'name': 'mask_blur',
            'val': 5,
            'init': 5,
            'min': 5,
            'max': 31,
            'mod': INF,
            'step': 2,
            'inc': False,
            'dec': False}
        FINAL_MASK_OFFSET = {
            'desc': 'mask is subtracted from this value before final masking',
            'name': 'mask_offset',
            'val': 235,
            'init': 235,
            'min': -INF,
            'max': INF,
            'mod': 256,
            'step': 5,
            'inc': False,
            'dec': False}
        ITER_INDEX = {
            'desc': '(no user input) index into blurring iterations',
            'name': 'iter_index',
            'val': 0,
            'init': 0,
            'min': 0,
            'max': 75,
            'mod': INF,
            'step': 1,
            'inc': False,
            'dec': False}
        HUE_OFFSET = {
            'desc': '(no user input) ' +
                    'hue value offset for background frame',
            'name': 'hue_offset',
            'val': 0,
            'init': 0,
            'min': -INF,
            'max': INF,
            'mod': 180,
            'step': 5,
            'inc': False,
            'dec': False}
        self.max_NUM_STYLES = 2

        # combine dicts into a list for easy general access
        self.props = [
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

        self.frame_width = frame_width
        self.frame_height = frame_height

        # get source images
        source_dir = '/media/data/Dropbox/Git/vid-viz/data/deep-dream/'
        self.file_list = [join(source_dir, f) for f in listdir(source_dir)
                          if isfile(join(source_dir, f))]
        self.num_files = len(self.file_list)

        # intialize other parameters
        self.reset()

    def reset(self):

        # reset base class attributes
        super(HueSwirlChain, self).reset()
        self.prev_mask_blur = 0  # to initialize frame_mask_list
        self.prev_hue_offset = self.props[5]['init']
        self.prev_dim_size = self.props[0]['init']

        # background frame parameters
        self.frame_back_0 = np.ones(
            (self.props[0]['init'], self.props[0]['init'], 3))
        self.frame_back_0[:, :, 0] = \
            np.random.rand(self.props[0]['init'], self.props[0]['init'])
        self.frame_back = None
        self.auto_play = True

        # frame parameters
        self.curr_frame_index = 0
        self.file_index = 0
        frame_0 = cv2.imread(self.file_list[self.file_index])
        frame_0 = cvutil.resize(frame_0, self.frame_width, self.frame_height)
        self.file_index += 1
        frame_1 = cv2.imread(self.file_list[self.file_index])
        frame_1 = cvutil.resize(frame_1, self.frame_width, self.frame_height)
        self.file_index += 1
        self.frame = [frame_0, frame_1]

        # mask parameters
        self.frame_masks = [None, None]
        self.frame_mask_list = [
            [None for _ in range(self.props[4]['max'] + 1)],
            [None for _ in range(self.props[4]['max'] + 1)]]
        self.num_blend_levels = self.props[4]['max']
        self.curr_blend_level = 0

        # control parameters
        self.increase_blur_index = True
        self.increase_blend_index = True
        self.increase_source_index = False

    def process(self, key_list, key_lock=False):

        # update if blur kernel toggled
        # if key_list[ord('t')]:
        #     reset_iter_seq = True
        # else:
        #     reset_iter_seq = False

        # control parameters (use _process_io for clipping and modding)
        if self.auto_play:
            # self.props[3]['val'] += 1  # final mask offset
            self.props[5]['val'] += 0.1  # hue offset

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            """TODO"""
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']
            self.frame_mask_list = \
                [None for _ in range(self.props[4]['max'] + 1)]
            self.increase_index = True
            self.increase_meta_index = True

        # control parameters - blend
        if self.style == 0:
            # |_|
            # if increasing blur index, use blurred mask from original image
            # if decreasing blur index, use a linear combination of blurred
            # masks from original and new images
            if not self.increase_blur_index:
                # increase blend parameter
                self.curr_blend_level += 1
        elif self.style == 1:
            # |/
            pass
        elif self.style == 2:
            # \/
            pass

        # control parameters - blur
        if self.increase_blur_index:
            self.props[4]['val'] += 1
        else:
            self.props[4]['val'] -= 1
        if self.props[4]['val'] == self.props[4]['max']:
            self.increase_blur_index = False
        if self.props[4]['val'] == self.props[4]['min']:
            self.increase_blur_index = True
            reset_iter_seq = True
        else:
            reset_iter_seq = False

        # control parameters - source
        if reset_iter_seq:
            # reset part of mask list so new image can take over
            self.frame_mask_list[self.curr_frame_index] = \
                [None for _ in range(self.props[4]['max'] + 1)]
            # load new image
            temp_frame = cv2.imread(self.file_list[self.file_index])
            hw = temp_frame.shape
            temp_frame = cv2.getRectSubPix(
                temp_frame,
                (256, 256),
                (hw[1]/2, hw[0]/2))
            self.frame[self.curr_frame_index] = temp_frame

            self.frame[self.curr_frame_index] = cvutil.resize(
                self.frame[self.curr_frame_index],
                self.frame_width,
                self.frame_height)
            self.file_index = (self.file_index + 1) % self.num_files
            # update curr frame index to old image
            self.curr_frame_index = (self.curr_frame_index + 1) % 2
            # reset blending param
            self.curr_blend_level = 0

        # human-readable names
        dim_size = self.props[0]['val']
        back_blur = self.props[1]['val']
        mask_blur = self.props[2]['val']
        final_offset = self.props[3]['val']
        iter_index = self.props[4]['val']
        hue_offset = self.props[5]['val']
        curr_fr_indx = self.curr_frame_index
        next_fr_indx = (curr_fr_indx + 1) % 2

        # create new random matrix if necessary
        if int(dim_size) is not int(self.prev_dim_size):
            self.prev_dim_size = dim_size
            self.frame_back_0 = np.ones((dim_size, dim_size, 3))
            self.frame_back_0[:, :, 2] = np.random.rand(dim_size, dim_size)
            self.frame_back = None

        # create background frame if necessary
        if self.frame_back is None:
            # get resized background
            self.frame_back = cv2.resize(
                self.frame_back_0,
                (self.frame_width, self.frame_height),
                interpolation=cv2.INTER_LINEAR)
            self.frame_back[:, :, 0] = 179.0 * self.frame_back[:, :, 0]
            self.frame_back[:, :, 1:3] = 255.0 * self.frame_back[:, :, 1:3]
            self.frame_back = self.frame_back.astype('uint8')
            self.frame_back = cv2.cvtColor(self.frame_back, cv2.COLOR_HSV2BGR)

        # update background frame if necessary
        if int(hue_offset) is not int(self.prev_hue_offset):
            self.frame_back = cv2.cvtColor(self.frame_back, cv2.COLOR_BGR2HSV)
            # uint8s don't play nice with subtraction
            self.frame_back[:, :, 0] += abs(
                int(hue_offset - self.prev_hue_offset))
            self.frame_back[:, :, 0] = np.mod(self.frame_back[:, :, 0],
                                              180)
            self.frame_back = cv2.cvtColor(self.frame_back, cv2.COLOR_HSV2BGR)
            self.prev_hue_offset = hue_offset

        # get mask if necessary
        if int(mask_blur) is not int(
                self.prev_mask_blur) or reset_iter_seq:
            # blur kernel changed; restart iteration sequence
            # self.props[4]['val'] = self.props[4]['init']
            # iter_index = self.props[4]['val']
            # self.increase_index = True
            # self.frame_mask_list = \
            #     [None for _ in range(self.props[4]['max'] + 1)]
            # get new mask

            frame_gray = cv2.cvtColor(
                self.frame[curr_fr_indx],
                cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.adaptiveThreshold(
                frame_gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                51,
                10)
            self.frame_mask_list[curr_fr_indx][0] = cv2.medianBlur(
                    frame_gray,
                    mask_blur)

            frame_gray = cv2.cvtColor(
                self.frame[next_fr_indx],
                cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.adaptiveThreshold(
                frame_gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                51,
                10)
            self.frame_mask_list[next_fr_indx][0] = cv2.medianBlur(
                frame_gray,
                mask_blur)

            self.prev_mask_blur = mask_blur

        # update masks if necessary
        for fr_indx in range(2):
            if (self.frame_mask_list[fr_indx][iter_index]) is None \
                    and (iter_index % 2 == 0):
                # need to update and store frame mask
                # two blur passes from previously stored mask
                if self.style == 0:
                    frame_mask_temp = cv2.GaussianBlur(
                        self.frame_mask_list[fr_indx][iter_index - 2],
                        (mask_blur, mask_blur),
                        0)
                    frame_mask_temp = cv2.GaussianBlur(
                        frame_mask_temp,
                        (mask_blur, mask_blur),
                        0)
                elif self.style == 1:
                    frame_mask_temp = cv2.medianBlur(
                        self.frame_mask_list[fr_indx][iter_index - 2],
                        mask_blur)
                    frame_mask_temp = cv2.medianBlur(
                        frame_mask_temp,
                        mask_blur)
                self.frame_mask_list[fr_indx][iter_index] = frame_mask_temp
                self.frame_masks[fr_indx] = frame_mask_temp
            elif (self.frame_mask_list[fr_indx][iter_index] is None) and \
                    (iter_index % 2 == 1):
                # need to update but not store frame mask
                if self.style == 0:
                    self.frame_masks[fr_indx] = cv2.GaussianBlur(
                        self.frame_mask_list[fr_indx][iter_index - 1],
                        (mask_blur, mask_blur),
                        0)
                elif self.style == 1:
                    self.frame_masks[fr_indx] = cv2.medianBlur(
                        self.frame_mask_list[fr_indx][iter_index - 1],
                        mask_blur)

        # combine masks
        if self.curr_blend_level is not 0:
            # blend masks
            frame_mask = cv2.addWeighted(
                self.frame_masks[curr_fr_indx],
                1.0 - self.curr_blend_level / self.num_blend_levels,
                self.frame_masks[next_fr_indx],
                self.curr_blend_level / self.num_blend_levels,
                0)
        else:
            frame_mask = self.frame_masks[curr_fr_indx]

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
                final_offset - frame_mask)

        # _, frame = cv2.threshold(frame, 32, 255, cv2.THRESH_BINARY)
        return frame

    def print_update(self):
        """Print effect settings to console if not changed automatically"""
        if (self.update_output != -1) & (self.update_output != 4) & (
                self.update_output != 5):
            super(HueSwirlChain, self).print_update()
