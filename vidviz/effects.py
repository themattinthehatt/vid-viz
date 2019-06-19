"""
Library of simple image processing effects that can be applied to source 
images or video
"""

from __future__ import print_function
from __future__ import division

import cv2
import numpy as np

import utils as util
import cvutils


class Effect(object):
    """Base class for vid-viz effects"""

    def __init__(self, style='effect'):
        """
        Args:
            style (str): 'effect' | 'postproc'
        """

        # set attributes common to all effects
        self.name = None
        self.type = style
        self.props = None
        self.max_num_styles = 1
        self.auto_play = False
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=1,
            num_channels=self.chan_vec_pos.size)
        self.update_output = False  # boolean for updating screen output

        self.inf = 1000
        self.none_dict = {
            'desc': 'unassigned',
            'name': '',
            'val': 0,
            'init': 0,
            'min': 0,
            'max': 1,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}

    def _process_io(self, key_list):

        self.update_output = -1
        if key_list[ord('-')]:
            key_list[ord('-')] = False
            self.props[0]['dec'] = True
            self.update_output = 0
        elif key_list[ord('=')]:
            key_list[ord('=')] = False
            self.props[0]['inc'] = True
            self.update_output = 0
        elif key_list[ord('[')]:
            key_list[ord('[')] = False
            self.props[1]['dec'] = True
            self.update_output = 1
        elif key_list[ord(']')]:
            key_list[ord(']')] = False
            self.props[1]['inc'] = True
            self.update_output = 1
        elif key_list[ord(';')]:
            key_list[ord(';')] = False
            self.props[2]['dec'] = True
            self.update_output = 2
        elif key_list[ord('\'')]:
            key_list[ord('\'')] = False
            self.props[2]['inc'] = True
            self.update_output = 2
        elif key_list[ord(',')]:
            key_list[ord(',')] = False
            self.props[3]['dec'] = True
            self.update_output = 3
        elif key_list[ord('.')]:
            key_list[ord('.')] = False
            self.props[3]['inc'] = True
            self.update_output = 3
        elif key_list[ord('R')]:
            key_list[ord('R')] = False
            self.props[4]['dec'] = True
            self.update_output = 4
        elif key_list[ord('T')]:
            key_list[ord('T')] = False
            self.props[4]['inc'] = True
            self.update_output = 4
        elif key_list[ord('Q')]:
            key_list[ord('Q')] = False
            self.props[5]['dec'] = True
            self.update_output = 5
        elif key_list[ord('S')]:
            key_list[ord('S')] = False
            self.props[5]['inc'] = True
            self.update_output = 5
        elif key_list[ord('/')]:
            key_list[ord('/')] = False
            self.reinitialize = True
        elif key_list[ord('t')]:
            key_list[ord('t')] = False
            self.style = (self.style + 1) % self.max_num_styles
            # self.reinitialize = True
        # elif key_list[ord('a')]:
        #     key_list[ord('a')] = False
        #     self.auto_play = not self.auto_play
        elif key_list[ord('w')]:
            key_list[ord('w')] = False
            self.random_walk = not self.random_walk
            self.chan_vec_pos = np.zeros(self.chan_vec_pos.shape)
            self.noise.reinitialize()

        # process options
        for index, _ in enumerate(self.props):
            if self.props[index]['dec']:
                self.props[index]['dec'] = False
                self.props[index]['val'] -= self.props[index]['step']
            if self.props[index]['inc']:
                self.props[index]['inc'] = False
                self.props[index]['val'] += self.props[index]['step']
            if self.props[index]['mod'] != self.inf:
                self.props[index]['val'] = np.mod(
                    self.props[index]['val'],
                    self.props[index]['mod'])
            self.props[index]['val'] = np.clip(
                self.props[index]['val'],
                self.props[index]['min'],
                self.props[index]['max'])

    def process(self, frame, key_list):
        raise NotImplementedError

    def print_update(self, force=False):
        """Print effect settings to console"""
        if self.update_output > -1 or force:
            print()
            print()
            print()
            print('%s effect settings' % self.name)
            print('keys |  min  |  cur  |  max  |  description')
            print('-----|-------|-------|-------|-------------')
            for index in range(6):
                if index == 0:
                    keys = '-/+'
                elif index == 1:
                    keys = '{/}'
                elif index == 2:
                    keys = ";/'"
                elif index == 3:
                    keys = '</>'
                elif index == 4:
                    keys = 'u/d'
                elif index == 5:
                    keys = 'l/r'
                print(' %s | %5g | %5g | %5g | %s' %
                      (keys,
                       self.props[index]['min'],
                       self.props[index]['val'],
                       self.props[index]['max'],
                       self.props[index]['desc']))

            # print extra updates
            print('t - toggle between effect types')
            print('w - toggle random walk')
            print('/ - reset effect parameters')
            print('q - quit %s effect' % self.name)
            if self.type == 'effect':
                print('~ - enable post-processing edit mode')
                print('spacebar - cycle through sources')
            elif self.type == 'postproc':
                print('tab - reverse processing order')
                print('backspace - quit post-processing edit mode')

    def reset(self):
        for index, _ in enumerate(self.props):
            self.props[index]['val'] = self.props[index]['init']
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
        / - reset parameters
        backspace - quit border effect
    """

    def __init__(self, style='effect'):

        super(Border, self).__init__(style=style)
        self.name = 'border'

        # user option constants
        MULT_FACTOR = {
            'desc': 'shrink frame and fill border',
            'name': 'mult_factor',
            'val': 1.0,
            'init': 1.0,
            'min': 0.01,
            'max': 1.0,
            'mod': self.inf,
            'step': 0.05,
            'inc': False,
            'dec': False}
        ZOOM_FACTOR = {
            'desc': 'zoom on original frame',
            'name': 'zoom_factor',
            'val': 1.0,
            'init': 1.0,
            'min': 1.0,
            'max': 10.0,
            'mod': self.inf,
            'step': 0.05,
            'inc': False,
            'dec': False}
        ROT_ANGLE = {
            'desc': 'rotation on original frame',
            'name': 'rot_angle',
            'val': 0,
            'init': 0,
            'min': -self.inf,
            'max': self.inf,
            'mod': 360,
            'step': 5,
            'inc': False,
            'dec': False}
        SHIFT_PIX_VERT = {
            'desc': 'vertical shift on original frame',
            'name': 'shift_vert',
            'val': 0,
            'init': 0,
            'min': -500,
            'max': 500,
            'mod': self.inf,
            'step': 10,
            'inc': False,
            'dec': False}
        SHIFT_PIX_HORZ = {
            'desc': 'horizontal shift on original frame',
            'name': 'shift_horz',
            'val': 0,
            'init': 0,
            'min': -500,
            'max': 500,
            'mod': self.inf,
            'step': 10,
            'inc': False,
            'dec': False}
        self.max_num_styles = 3

        # combine dicts into a list for easy general access
        self.props = [
            MULT_FACTOR,
            ZOOM_FACTOR,
            ROT_ANGLE,
            self.none_dict,
            SHIFT_PIX_VERT,
            SHIFT_PIX_HORZ]

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        mult_factor = self.props[0]['val']
        zoom_factor = self.props[1]['val']
        rot_angle = self.props[2]['val']
        shift_vert = self.props[4]['val']
        shift_horz = self.props[5]['val']

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


class Cell(object):
    """Helper class for Grating class"""

    def __init__(self, num_pix_cell, num_pix_cell_half, border_prop,
                 center, vel, frame_size=[0, 0], use_full_frame=False):
        self.num_pix_cell = num_pix_cell
        self.num_pix_cell_half = num_pix_cell_half
        self.center = center
        self.vel = vel
        self.num_pix_img_half = None
        self.border_prop = None
        self.update_border_prop(border_prop)
        self.frame_size = frame_size
        self.use_full_frame = use_full_frame

    def update_border_prop(self, border_prop):
        self.border_prop = border_prop
        self.num_pix_img_half = \
            [int((self.num_pix_cell[0] * (1 - self.border_prop) // 2)),
             int((self.num_pix_cell[1] * (1 - self.border_prop) // 2))]

    def update_position_lazy(self):
        self.center[0] += self.vel[0]
        self.center[1] += self.vel[1]
        if self.center[0] + self.num_pix_cell_half[0] >= self.frame_size[0]:
            self.center[0] = self.num_pix_cell_half[0] + 1
        elif self.center[0] - self.num_pix_cell_half[0] <= 0:
            self.center[0] = self.frame_size[0] - self.num_pix_cell_half[0] - 1
        if self.center[1] + self.num_pix_cell_half[1] >= self.frame_size[1]:
            self.center[1] = self.num_pix_cell_half[1] + 1
        elif self.center[1] - self.num_pix_cell_half[1] <= 0:
            self.center[1] = self.frame_size[1] - self.num_pix_cell_half[1] - 1

    def update_position(self):
        self.center[0] += self.vel[0]
        self.center[1] += self.vel[1]
        if self.center[0] + self.num_pix_cell_half[0] >= self.frame_size[0]:
            # set position to border
            self.center[0] = self.frame_size[0] - self.num_pix_cell_half[0]
            # reverse vertical velocity
            self.vel[0] *= -1
        elif self.center[0] - self.num_pix_cell_half[0] <= 0:
            self.center[0] = self.num_pix_cell_half[0] + 1
            self.vel[0] *= -1

        if self.center[1] + self.num_pix_cell_half[1] >= self.frame_size[1]:
            self.center[1] = self.frame_size[1] - self.num_pix_cell_half[1]
            self.vel[1] *= -1
        elif self.center[1] - self.num_pix_cell_half[1] <= 0:
            self.center[1] = self.num_pix_cell_half[1] + 1
            self.vel[1] *= -1

    def draw(self, frame, background):

        if self.use_full_frame:
            # render full frame in cell
            cell = frame
        else:
            # render part of frame in cell
            cell = cv2.getRectSubPix(
                frame,
                (self.num_pix_cell[1], self.num_pix_cell[0]),
                (self.center[1], self.center[0]))
        cell = cv2.resize(
            cell,
            (2 * self.num_pix_img_half[1] + 1,
             2 * self.num_pix_img_half[0] + 1),
            interpolation=cv2.INTER_LINEAR)
        background[
            self.center[0] - self.num_pix_img_half[0]:
            self.center[0] + self.num_pix_img_half[0] + 1,
            self.center[1] - self.num_pix_img_half[1]:
            self.center[1] + self.num_pix_img_half[1] + 1,
            :] = cell

        return background


class Grating(Effect):
    """
    Render image in rectangular cells, the aspect ratio and border thickness of
    which are controllable parameters

    KEYBOARD INPUTS:
        -/+ - decrease/increase border proportion
        [/] - None
        ;/' - None
        ,/. - None
        lrud arrows - decrease/increase number of cells in horz/vert direction
    """

    def __init__(self, style='effect'):

        super(Grating, self).__init__(style=style)
        self.name = 'grating'

        border_prop = {
            'desc': 'proportion of cell used for border',
            'name': 'border_prop',
            'val': 0.1,
            'init': 0.2,
            'min': 0.0,
            'max': 1.0,
            'mod': self.inf,
            'step': 0.02,
            'inc': False,
            'dec': False}
        cells_horz = {
            'desc': 'number of cells in horizontal direction',
            'name': 'num_cells_horz',
            'val': 10,
            'init': 10,
            'min': 1,
            'max': 200,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        cells_vert = {
            'desc': 'number of cells in vertical direction',
            'name': 'num_cells_vert',
            'val': 5,
            'init': 5,
            'min': 1,
            'max': 200,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        self.max_num_styles = 3

        # combine dicts into a list for easy general access
        self.props = [
            border_prop,
            self.none_dict,
            self.none_dict,
            self.none_dict,
            cells_vert,
            cells_horz]

        # user options
        self.prev_border_prop = self.props[0]['val']
        self.cells = []
        self.cell_index = -1  # index into list of cells

    def process(self, frame, key_list, key_lock=False):

        # update if blur kernel toggled
        if key_list[ord('t')]:
            self.reinitialize = True
        else:
            self.reinitialize = False

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']
            self.cells = []

        # human-readable names
        border_prop = self.props[0]['val']
        num_cells = [self.props[4]['val'], self.props[5]['val']]

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        if self.style == -1:
            # original grating style; static vertical and horizontal black bars

            # reinitialize cells if number has changed
            if num_cells[0] * num_cells[1] != len(self.cells):

                # get initial values for cells
                num_pix_cell_half = [(im_height / num_cells[0]) // 2,
                                     (im_width / num_cells[1]) // 2]
                num_pix_cell = [int(2 * num_pix_cell_half[0] + 1),
                                int(2 * num_pix_cell_half[1] + 1)]
                centers = [
                    [int(val * num_pix_cell[0] + num_pix_cell_half[0] + 1)
                        for val in range(num_cells[0])],
                    [int(val * num_pix_cell[1] + num_pix_cell_half[1] + 1)
                        for val in range(num_cells[1])]]
                # shift center points at end
                centers[0][-1] = int(im_height - num_pix_cell_half[0] - 1)
                centers[1][-1] = int(im_width - num_pix_cell_half[1] - 1)

                # build cells
                self.cells = []
                for h in range(num_cells[0]):
                    for w in range(num_cells[1]):
                        self.cells.append(Cell(
                            num_pix_cell, num_pix_cell_half, border_prop,
                            [centers[0][h], centers[1][w]], [0, 0]))

            # update cells if border prop has changed
            if self.prev_border_prop != border_prop:
                self.prev_border_prop = border_prop
                for _, cell in enumerate(self.cells):
                    cell.update_border_prop(border_prop)

            # update background with frame info
            if border_prop == 0.0:
                background = frame
            elif border_prop == 1.0:
                background = np.zeros(shape=frame.shape, dtype=np.uint8)
            else:
                background = np.zeros(shape=frame.shape, dtype=np.uint8)

                # tile background array with image
                for _, cell in enumerate(self.cells):
                    # tile background array with image
                    background = cell.draw(frame, background)

        elif self.style == 0 or self.style == 1 or self.style == 2:
            # random horizontal translation effect

            # add cells if necessary
            while len(self.cells) < int(num_cells[1]):
                # get initial values for cells
                if self.style == 0:
                    # render portion of frame in each cell
                    use_full_frame = False
                    # num_pix_cell_half = [np.random.randint(10, 30),
                    #                      np.random.randint(10, 30)]
                    num_pix_cell_half = [np.random.randint(30, 50),
                                         np.random.randint(30, 50)]
                    # velocity = [0, np.random.randint(2, 20)]
                    velocity = [np.random.randint(-20, 20),
                                np.random.randint(-20, 20)]
                elif self.style == 1:
                    # render full frame in each (larger) cell
                    use_full_frame = True
                    num_pix_cell_half = [np.random.randint(50, 80),
                                         np.random.randint(70, 90)]
                    velocity = [0, np.random.randint(-10, 10)]
                elif self.style == 2:
                    use_full_frame = True
                    # num_pix_cell_half = [np.random.randint(50, 80),
                    #                      np.random.randint(70, 90)]
                    num_pix_cell_half = [100, int(100 * 16 / 9)]
                    # velocity = [np.random.randint(-5, 5),
                    #             np.random.randint(-5, 5)]
                    velocity = [np.random.randint(-8, 8),
                                np.random.randint(-8, 8)]
                num_pix_cell = [int(2 * num_pix_cell_half[0] + 1),
                                int(2 * num_pix_cell_half[1] + 1)]
                # use random portion of frame
                lower_height = num_pix_cell_half[0] + 1;
                upper_height = im_height - num_pix_cell_half[0] - 1
                lower_width = num_pix_cell_half[1] + 1
                upper_width = im_width - num_pix_cell_half[1] - 1
                centers = [np.random.randint(lower_height, upper_height),
                           np.random.randint(lower_width, upper_width)]

                self.cells.append(Cell(
                    num_pix_cell, num_pix_cell_half, border_prop,
                    centers, velocity, frame_size=[im_height, im_width],
                    use_full_frame=use_full_frame))
            # delete cells if necessary
            while len(self.cells) > int(num_cells[1]):
                del self.cells[-1]

            # update cells if border prop has changed
            if self.prev_border_prop != border_prop:
                self.prev_border_prop = border_prop
                for _, cell in enumerate(self.cells):
                    cell.update_border_prop(border_prop)

            # update background with frame info
            background = np.zeros(shape=frame.shape, dtype=np.uint8)

            # tile background array with image
            for _, cell in enumerate(self.cells):
                # update positions of cells
                if self.style == 0 or self.style == 1:
                    cell.update_position_lazy()
                elif self.style == 2:
                    cell.update_position()
                # tile background array with image
                background = cell.draw(frame, background)

        else:
            raise NotImplementedError

        return background

    def reset(self):
        super(Grating, self).reset()
        self.cells = []


class AdaptiveThreshold(Effect):
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

    def __init__(self, style='effect'):

        super(AdaptiveThreshold, self).__init__(style=style)
        self.name = 'threshold'

        # user option constants
        THRESH_KERNEL = {
            'desc': 'kernel size for adaptive thresholding',
            'name': 'kernel_size',
            'val': 21,
            'init': 21,
            'min': 3,
            'max': 71,
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        THRESH_OFFSET = {
            'desc': 'offset constant for adaptive thresholding',
            'name': 'offset',
            'val': 4,
            'init': 4,
            'min': 0,
            'max': 30,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2

        # combine dicts into a list for easy general access
        self.props = [
            THRESH_KERNEL,
            THRESH_OFFSET,
            self.none_dict,
            self.none_dict,
            self.none_dict,
            self.none_dict]

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
        self.max_NUM_CHAN_STYLES = 4

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
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        kernel_size = self.props[0]['val']
        offset = self.props[1]['val']

        for chan in range(3):
            if self.use_chan[chan]:
                for chan_style in range(self.max_NUM_CHAN_STYLES):
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
        super(AdaptiveThreshold, self).reset()
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]            # effect selector for each chan


class SimpleThreshold(Effect):
    """
    Threshold individual channels in RGB frame

    KEYBOARD INPUTS:
        t - toggle threshold type (apply inverse threshold)
        -/+ - decrease/increase threshold
        [/] - None
        ;/' - None
        ,/. - None
        r/g/b/a - select red/green/blue/all channels for further processing
        / - reset parameters
        q - quit soft threshold effect
    """

    def __init__(self, style='effect'):

        super(SimpleThreshold, self).__init__(style=style)
        self.name = 'soft-threshold'

        # user option constants
        THRESHOLD = {
            'desc': 'threshold value',
            'name': 'threshold',
            'val': 128,
            'init': 128,
            'min': 0,
            'max': 255,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2  # thresh_binary, thresh_binary_inv

        # combine dicts into a list for easy general access
        self.props = [
            THRESHOLD,
            self.none_dict,
            self.none_dict,
            self.none_dict,
            self.none_dict,
            self.none_dict]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = False
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # other user options
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]  # effect selector for each chan
        self.max_NUM_CHAN_STYLES = 5

        # opencv parameters
        self.THRESH_TYPE = cv2.THRESH_BINARY

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
            elif key_list[ord('a')]:
                key_list[ord('a')] = False
                self.use_chan[0] = True
                self.use_chan[1] = True
                self.use_chan[2] = True

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        threshold = self.props[0]['val']

        for chan_style in range(self.max_NUM_CHAN_STYLES):
            if key_list[ord(str(chan_style))]:
                for chan in range(3):
                    if self.use_chan[chan]:
                        self.chan_style[chan] = chan_style
                key_list[ord(str(chan_style))] = False

        # process image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.style == 0:
            self.THRESH_TYPE = cv2.THRESH_BINARY
        elif self.style == 1:
            self.THRESH_TYPE = cv2.THRESH_BINARY_INV
        _, frame_thresh = cv2.threshold(
            frame_gray,
            threshold,
            255,
            self.THRESH_TYPE)

        for chan in range(3):
            if self.chan_style[chan] == 1:
                frame[:, :, chan] = 0
            elif self.chan_style[chan] == 2:
                frame[:, :, chan] = 255
            elif self.chan_style[chan] == 3:
                frame[:, :, chan] = frame_thresh
            elif self.chan_style[chan] == 4:
                frame[:, :, chan] = frame_gray

        return frame

    def reset(self):
        super(SimpleThreshold, self).reset()
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]  # effect selector for each chan


class PowerThreshold(Effect):
    """
    Threshold individual channels in RGB frame

    KEYBOARD INPUTS:
        t - toggle threshold type (apply inverse threshold)
        -/+ - decrease/increase threshold
        [/] - None
        ;/' - None
        ,/. - None
        r/g/b/a - select red/green/blue/all channels for further processing
        / - reset parameters
        q - quit soft threshold effect
    """

    def __init__(self, style='effect'):

        super(PowerThreshold, self).__init__(style=style)
        self.name = 'power-threshold'

        # user option constants
        THRESHOLD = {
            'desc': 'threshold value',
            'name': 'threshold',
            'val': 128,
            'init': 128,
            'min': 0,
            'max': 255,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        THRESHOLD_POWER = {
            'desc': 'threshold power',
            'name': 'threshold_power',
            'val': 1,
            'init': 1,
            'min': 1,
            'max': 5,
            'mod': self.inf,
            'step': 0.01,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2  # thresh_binary, thresh_binary_inv

        # combine dicts into a list for easy general access
        self.props = [
            THRESHOLD,
            THRESHOLD_POWER,
            self.none_dict,
            self.none_dict,
            self.none_dict,
            self.none_dict]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = False  # TODO - walk through power space
        self.chan_vec_pos = np.zeros((3, 2))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # other user options
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]  # effect selector for each chan
        self.max_NUM_CHAN_STYLES = 5

        # opencv parameters
        self.THRESH_TYPE = cv2.THRESH_BINARY

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
            elif key_list[ord('a')]:
                key_list[ord('a')] = False
                self.use_chan[0] = True
                self.use_chan[1] = True
                self.use_chan[2] = True

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        threshold = self.props[0]['val']
        power = self.props[1]['val']

        for chan_style in range(self.max_NUM_CHAN_STYLES):
            if key_list[ord(str(chan_style))]:
                for chan in range(3):
                    if self.use_chan[chan]:
                        self.chan_style[chan] = chan_style
                key_list[ord(str(chan_style))] = False

        # process image
        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_, (5, 5), 0)

        # only perform threshold if necessary
        if any([cs == 3 for cs in self.chan_style]):
            if self.style == 0:
                # thresh binary
                pass
            elif self.style == 1:
                frame_gray = 255 - frame_gray

            # convert to image to [0, 1] range
            frame_gray_fl = frame_gray.astype('float16') / 255.0
            frame_thresh = np.copy(frame_gray_fl)

            mask = frame_thresh < (threshold / 255.0)
            # 0-thresh range: [0, thresh]**power
            frame_thresh[mask] = np.power(frame_gray_fl[mask], power)
            # thresh-1 range: 1-(1-[thresh, 1])**power
            frame_thresh[~mask] = 1.0 - np.power(
                1.0 - frame_gray_fl[~mask], power)
            # convert image to [0, 255] range
            frame_thresh *= 255
            frame_thresh = frame_thresh.astype('uint8')
        else:
            frame_thresh = 0

        for chan in range(3):
            if self.chan_style[chan] == 1:
                frame[:, :, chan] = 0
            elif self.chan_style[chan] == 2:
                frame[:, :, chan] = 255
            elif self.chan_style[chan] == 3:
                frame[:, :, chan] = frame_thresh
            elif self.chan_style[chan] == 4:
                frame[:, :, chan] = frame_gray

        return frame

    def reset(self):
        super(PowerThreshold, self).reset()
        self.use_chan = [False, False, False]  # rgb channel selector
        self.chan_style = [0, 0, 0]  # effect selector for each chan


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

    def __init__(self, style='effect'):

        super(Alien, self).__init__(style=style)
        self.name = 'alien'

        # user option constants
        FREQ = {
            'desc': 'frequency of sine function',
            'name': 'freq',
            'val': 0.2,
            'init': 0.2,
            'min': 0.0,
            'max': 10.0,
            'mod': self.inf,
            'step': 0.05,
            'inc': False,
            'dec': False}
        PHASE = {
            'desc': 'phase of sine function',
            'name': 'phase',
            'val': 0,
            'init': 0,
            'min': -self.inf,
            'max': self.inf,
            'mod': 2 * np.pi,
            'step': np.pi / 10.0,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2

        # combine dicts into a list for easy general access
        self.props = [
            FREQ,
            PHASE,
            self.none_dict,
            self.none_dict,
            self.none_dict,
            self.none_dict]

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
            self.props[0]['init'],
            self.props[0]['init'],
            self.props[0]['init']]
        self.chan_phase = [                     # current phase for each chan
            self.props[1]['init'],
            self.props[1]['init'],
            self.props[1]['init']]
        self.max_NUM_CHAN_STYLES = 4

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            if key_list[ord('b')]:
                key_list[ord('b')] = False
                self.use_chan[0] = True
                self.use_chan[1] = False
                self.use_chan[2] = False
                self.props[0]['val'] = self.chan_freq[0]
                self.props[1]['val'] = self.chan_phase[0]
            elif key_list[ord('g')]:
                key_list[ord('g')] = False
                self.use_chan[0] = False
                self.use_chan[1] = True
                self.use_chan[2] = False
                self.props[0]['val'] = self.chan_freq[1]
                self.props[1]['val'] = self.chan_phase[1]
            elif key_list[ord('r')]:
                key_list[ord('r')] = False
                self.use_chan[0] = False
                self.use_chan[1] = False
                self.use_chan[2] = True
                self.props[0]['val'] = self.chan_freq[2]
                self.props[1]['val'] = self.chan_phase[2]
            self._process_io(key_list)

        # process options
        for chan in range(3):
            if self.use_chan[chan]:
                # update channel style
                for chan_style in range(self.max_NUM_CHAN_STYLES):
                    if key_list[ord(str(chan_style))]:
                        self.chan_style[chan] = chan_style
                        key_list[ord(str(chan_style))] = False
                # update channel params
                self.chan_freq[chan] = self.props[0]['val']
                self.chan_phase[chan] = self.props[1]['val']

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((3, 2))
            self.noise.reinitialize()
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']
            for chan in range(3):
                self.chan_freq[chan] = self.props[0]['init']
                self.chan_phase[chan] = self.props[1]['init']

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
            self.props[0]['init'],
            self.props[0]['init'],
            self.props[0]['init']]
        self.chan_phase = [  # current phase for each chan
            self.props[1]['init'],
            self.props[1]['init'],
            self.props[1]['init']]


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

    def __init__(self, style='effect'):

        super(RGBWalk, self).__init__(style=style)
        self.name = 'rgb-walk'

        # user option constants
        step_SIZE = {
            'desc': 'step size that scales random walk',
            'name': 'step_size',
            'val': 5.0,
            'init': 5.0,
            'min': 0.5,
            'max': 15.0,
            'mod': self.inf,
            'step': 1.0,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2

        # combine dicts into a list for easy general access
        self.props = [
            self.none_dict,
            self.none_dict,
            self.none_dict,
            step_SIZE,
            self.none_dict,
            self.none_dict]

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
        step_size = self.props[3]['val']

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

    def __init__(self, style='effect'):

        super(RGBBurst, self).__init__(style=style)
        self.name = 'rgb-burst'

        # user option constants
        FRAME_INT = {
            'desc': 'interval at which burst takes place',
            'name': 'frame_interval',
            'val': 10,
            'init': 10,
            'min': 1,
            'max': 100,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        FRAME_decAY = {
            'desc': 'decay rate for background frame luminance',
            'name': 'frame_decay',
            'val': 0.8,
            'init': 0.8,
            'min': 0.3,
            'max': 0.99,
            'mod': self.inf,
            'step': 0.01,
            'inc': False,
            'dec': False}
        EXP_RATE = {
            'desc': 'expansion rate of background frame',
            'name': 'frame_expansion_rate',
            'val': 1.1,
            'init': 1.1,
            'min': 1.01,
            'max': 2.0,
            'mod': self.inf,
            'step': 0.01,
            'inc': False,
            'dec': False}
        step_SIZE = {
            'desc': 'step size that scales random walk',
            'name': 'step_size',
            'val': 5.0,
            'init': 5.0,
            'min': 0.5,
            'max': 15.0,
            'mod': self.inf,
            'step': 1.0,
            'inc': False,
            'dec': False}
        self.max_num_styles = 3

        # combine dicts into a list for easy general access
        self.props = [
            FRAME_INT,
            FRAME_decAY,
            EXP_RATE,
            step_SIZE,
            self.none_dict,
            self.none_dict]

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
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        frame_interval = self.props[0]['val']
        frame_decay = self.props[1]['val']
        frame_expansion_rate = self.props[2]['val']
        step_size = self.props[3]['val']

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

    def __init__(self, style='effect'):

        super(HueBloom, self).__init__(style=style)
        self.name = 'hue-bloom'

        # user option constants
        DIM_SIZE = {
            'desc': 'dimension of random matrix height/width',
            'name': 'dim_size',
            'val': 10,
            'init': 10,
            'min': 2,
            'max': 100,
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        BLUR_KERNEL = {
            'desc': 'size of background blur kernel',
            'name': 'blur_kernel',
            'val': 3,
            'init': 3,
            'min': 3,
            'max': 51,
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        PULSE_FREQ = {
            'desc': 'frequency of blur_kernal sine modulator',
            'name': 'pulse_freq',
            'val': 0.1,
            'init': 0.1,
            'min': 0.05,
            'max': 5,
            'mod': self.inf,
            'step': 0.05,
            'inc': False,
            'dec': False}
        self.max_num_styles = 3

        # combine dicts into a list for easy general access
        self.props = [
            DIM_SIZE,
            BLUR_KERNEL,
            PULSE_FREQ,
            self.none_dict,
            self.none_dict,
            self.none_dict]

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
        self.prev_dim_size = self.props[0]['init']
        self.frame = np.ones(
            (self.props[0]['init'], self.props[0]['init'], 3))
        self.frame[:, :, 0] = np.random.rand(
            self.props[0]['init'], self.props[0]['init'])

    def process(self, frame, key_list, key_lock=False):

        # process keyboard input
        if not key_lock:
            self._process_io(key_list)

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        dim_size = self.props[0]['val']
        blur_kernel = self.props[1]['val']
        pulse_freq = self.props[2]['val']

        # update blur kernel size
        val = 0.5 + 0.5 * np.sin(self.frame_count * pulse_freq)
        self.frame_count += 1
        blur_kernel = int(val * (self.props[1]['max'] - self.props[1]['min']) +
                          self.props[1]['min'])

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
        frame_back = frame_back.astype(np.uint8)
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
        self.prev_dim_size = self.props[0]['init']
        self.frame = np.ones(
            (self.props[0]['init'], self.props[0]['init'], 3))
        self.frame[:, :, 0] = np.random.rand(
            self.props[0]['init'], self.props[0]['init'])


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

    def __init__(self, style='effect'):

        super(HueSwirl, self).__init__(style=style)
        self.name = 'hue-swirl'

        # user option constants
        DIM_SIZE = {
            'desc': 'dimension of background random matrix height/width',
            'name': 'dim_size',
            'val': 1,
            'init': 1,
            'min': 1,
            'max': 100,
            'mod': self.inf,
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
            'mod': self.inf,
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
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        ITER_INDEX = {
            'desc': 'index into blurring iterations',
            'name': 'iter_index',
            'val': 0,
            'init': 0,
            'min': 0,
            'max': 75,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        FINAL_MASK_OFFSET = {
            'desc': 'mask is subtracted from this value before final masking',
            'name': 'mask_offset',
            'val': 255,
            'init': 255,
            'min': 0,
            'max': 255,
            'mod': 255,
            'step': 5,
            'inc': False,
            'dec': False}
        HUE_OFFSET = {
            'desc': 'hue value offset for background frame',
            'name': 'hue_offset',
            'val': 0,
            'init': 0,
            'min': -self.inf,
            'max': self.inf,
            'mod': 180,
            'step': 5,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2

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

        # frame parameters
        self.prev_mask_blur = 0  # to initialize frame_mask_list
        self.prev_hue_offset = self.props[5]['init']
        self.prev_dim_size = self.props[0]['init']
        self.frame_back_0 = np.ones(
            (self.props[0]['init'], self.props[0]['init'], 3))
        self.frame_back_0[:, :, 0] = \
            np.random.rand(self.props[0]['init'], self.props[0]['init'])
        self.frame_back = None
        self.frame_mask_list = [None for _ in
                                range(self.props[4]['max'] + 1)]
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
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']
            self.frame_mask_list = \
                [None for _ in range(self.props[4]['max'] + 1)]
            self.increase_index = True
            self.increase_meta_index = True

        # control parameters
        if self.auto_play:
            if self.increase_index:
                self.props[4]['val'] += 1
            else:
                self.props[4]['val'] -= 1
            if self.props[4]['val'] == self.props[4]['max']:
                self.increase_index = False
            if self.props[4]['val'] == self.props[4]['min']:
                self.increase_index = True
                # change meta index
                if self.increase_meta_index:
                    self.props[2]['val'] += self.props[2]['step']
                else:
                    self.props[2]['val'] -= self.props[2]['step']
                if self.props[2]['val'] == self.props[2]['max']:
                    self.increase_meta_index = False
                if self.props[2]['val'] == self.props[2]['min']:
                    self.increase_meta_index = True
                    self.style = (self.style + 1) % self.max_num_styles

        # human-readable names
        dim_size = self.props[0]['val']
        back_blur = self.props[1]['val']
        mask_blur = self.props[2]['val']
        final_offset = self.props[3]['val']
        iter_index = self.props[4]['val']
        hue_offset = self.props[5]['val']

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
            self.frame_back = self.frame_back.astype(np.uint8)
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
            self.props[4]['val'] = self.props[4]['init']
            iter_index = self.props[4]['val']
            self.increase_index = True
            self.frame_mask_list = \
                [None for _ in range(self.props[4]['max'] + 1)]
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
        frame_back_blurred = np.zeros(self.frame_back.shape, dtype=np.uint8)
        for chan in range(3):
            frame_back_blurred[:, :, chan] = cv2.bitwise_and(
                self.frame_back[:, :, chan],
                self.frame_mask)
        frame_back_blurred = cv2.GaussianBlur(
            frame_back_blurred,
            (back_blur, back_blur),
            0)

        # remask blurred background
        frame = np.zeros(self.frame_back.shape, dtype=np.uint8)
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
        self.prev_dim_size = self.props[0]['init']
        self.frame_back_0 = np.ones(
            (self.props[0]['init'], self.props[0]['init'], 3))
        self.frame_back_0[:, :, 0] = \
            np.random.rand(self.props[0]['init'], self.props[0]['init'])
        self.frame_back = None
        self.frame_mask_list = [None for _ in range(self.props[4]['max'] + 1)]
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

    def __init__(self, style='effect'):

        super(HueSwirlMover, self).__init__(style=style)
        self.name = 'hue-swirl-mover'

        # user option constants
        DIM_SIZE = {
            'desc': 'dimension of background random matrix height/width',
            'name': 'dim_size',
            'val': 2,
            'init': 2,
            'min': 2,
            'max': 100,
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        BACKGROUND_BLUR_KERNEL = {
            'desc': 'kernel size for Gaussian blur that produces bloom',
            'name': 'back_blur',
            'val': 19,
            'init': 19,
            'min': 3,
            'max': 31,
            'mod': self.inf,
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
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        ITER_INDEX = {
            'desc': 'index into blurring iterations',
            'name': 'iter_index',
            'val': 0,
            'init': 0,
            'min': 0,
            'max': 75,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        BLEND_PROP = {
            'desc': 'modulates weighting between new and previous frame',
            'name': 'blend_prop',
            'val': 0.5,
            'init': 0.5,
            'min': 0.05,
            'max': 1.0,
            'mod': self.inf,
            'step': 0.05,
            'inc': False,
            'dec': False}
        HUE_OFFSET = {
            'desc': 'hue value offset for background frame',
            'name': 'hue_offset',
            'val': 0,
            'init': 0,
            'min': -self.inf,
            'max': self.inf,
            'mod': 180,
            'step': 5,
            'inc': False,
            'dec': False}
        self.max_num_styles = 2

        # combine dicts into a list for easy general access
        self.props = [
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
        self.prev_hue_offset = self.props[5]['init']
        self.prev_dim_size = self.props[0]['init']
        self.frame_back_0 = \
            np.random.rand(self.props[0]['init'], self.props[0]['init'])
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
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        dim_size = self.props[0]['val']
        back_blur = self.props[1]['val']
        mask_blur = self.props[2]['val']
        blend_prop = self.props[3]['val']
        hue_offset = self.props[5]['val']

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
            self.frame_back = self.frame_back.astype(np.uint8)

        # update background frame if necessary
        if int(hue_offset) is not int(self.prev_hue_offset):
            # uint8s don't play nice with subtraction
            self.frame_back += abs(
                int(hue_offset - self.prev_hue_offset))
            self.frame_back = np.mod(self.frame_back, 180)
            self.frame_back = self.frame_back.astype(np.uint8)
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
            frame_back_blurred = np.zeros(self.frame_back.shape, dtype=np.uint8)
            for chan in range(3):
                frame_back_blurred[:, :, chan] = cv2.bitwise_and(
                    self.frame_back[:, :, chan],
                    frame_mask)
            frame_back_blurred = cv2.GaussianBlur(
                frame_back_blurred,
                (back_blur, back_blur),
                0)

            # remask blurred background
            frame = np.zeros(self.frame_back.shape, dtype=np.uint8)
            for chan in range(3):
                frame[:, :, chan] = cv2.bitwise_and(
                    frame_back_blurred[:, :, chan],
                    255 - self.frame_mask_blurred)

        return frame

    def reset(self):
        super(HueSwirlMover, self).reset()
        self.auto_play = False
        # frame parameters
        self.prev_hue_offset = self.props[5]['init']
        self.prev_dim_size = self.props[0]['init']
        self.frame_back_0 = \
            np.random.rand(self.props[0]['init'], self.props[0]['init'])
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

    def __init__(self, style='effect'):

        super(HueCrusher, self).__init__(style=style)
        self.name = 'hue-crusher'

        # user option constants
        NUM_CHUNKS = {
            'desc': 'number of chunks that divide huespace',
            'name': 'num_chunks',
            'val': 3,
            'init': 3,
            'min': 1,
            'max': 10,
            'mod': self.inf,
            'step': 1,
            'inc': False,
            'dec': False}
        CENTER_OFFSET = {
            'desc': 'offset value to apply to chunk centers',
            'name': 'center_offset',
            'val': 0,
            'init': 0,
            'min': -self.inf,
            'max': self.inf,
            'mod': 255,
            'step': 5,
            'inc': False,
            'dec': False}
        CHUNK_WIDTH = {
            'desc': 'width of each chunk in huespace',
            'name': 'chunk_width',
            'val': 16,
            'init': 16,
            'min': 3,
            'max': 180,
            'mod': self.inf,
            'step': 2,
            'inc': False,
            'dec': False}
        self.max_num_styles = 1

        # combine dicts into a list for easy general access
        self.props = [
            NUM_CHUNKS,
            CENTER_OFFSET,
            CHUNK_WIDTH,
            self.none_dict,
            self.none_dict,
            self.none_dict]

        # user options
        self.style = 0
        self.reinitialize = False
        self.random_walk = True
        self.chan_vec_pos = np.zeros((1, 1))
        self.noise = util.SmoothNoise(
            num_samples=10,
            num_channels=self.chan_vec_pos.size)

        # chunk params
        num_chunks = self.props[0]['init']
        self.num_chunks_prev = num_chunks

        self.chunk_widths_og = int(180 / num_chunks)
        self.chunk_range_mins = np.zeros((num_chunks, 1), np.uint8)
        self.chunk_range_maxs = np.zeros((num_chunks, 1), np.uint8)
        self.chunk_centers = np.zeros((num_chunks, 1), np.uint8)
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
        # self.props[1]['val'] = self.props[1]['val'] % 180

        if self.reinitialize:
            self.reinitialize = False
            self.chan_vec_pos = np.zeros((1, 1))
            self.noise.reinitialize()
            for index, _ in enumerate(self.props):
                self.props[index]['val'] = self.props[index]['init']

        # human-readable names
        num_chunks = self.props[0]['val']
        center_offset = self.props[1]['val']
        chunk_width = self.props[2]['val']

        # process image
        if len(frame.shape) == 3:
            [im_height, im_width, _] = frame.shape
        elif len(frame.shape) == 2:
            [im_height, im_width] = frame.shape

        # update params if number of chunks has been changed
        if int(num_chunks) is not int(self.num_chunks_prev):
            self.num_chunks_prev = num_chunks
            self.chunk_widths_og = int(180 / num_chunks)
            self.chunk_range_mins = np.zeros((num_chunks, 1), np.uint8)
            self.chunk_range_maxs = np.zeros((num_chunks, 1), np.uint8)
            self.chunk_centers = np.zeros((num_chunks, 1), np.uint8)
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
        hue_final = np.zeros((im_height, im_width), dtype=np.uint8)

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
            hue = hue.astype(np.uint8)
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
        num_chunks = self.props[0]['init']
        self.num_chunks_prev = num_chunks
        self.chunk_widths_og = int(180 / num_chunks)
        self.chunk_range_mins = np.zeros((num_chunks, 1), np.uint8)
        self.chunk_range_maxs = np.zeros((num_chunks, 1), np.uint8)
        self.chunk_centers = np.zeros((num_chunks, 1), np.uint8)
        for chunk in range(num_chunks):
            self.chunk_range_mins[chunk] = \
                chunk * self.chunk_widths_og
            self.chunk_range_maxs[chunk] = \
                (chunk + 1) * self.chunk_widths_og - 1
            self.chunk_centers[chunk] = self.chunk_range_mins[chunk] + \
                int(self.chunk_widths_og / 2)
        self.chunk_range_maxs[-1] = 179
