import numpy as np
import cv2

import vidviz.effects as effects
import vidviz.utils as utils
import vidviz.cvutils as cvutils
import vidviz.auto as auto


class WindowServer(object):
    """
    WindowServer object controls loading/generation of source frame, 
    processes the frame, and then returns the frame as a numpy array to be 
    rendered in the active window
    """

    def __init__(self, window_width=100, window_height=100, formats='RGB'):

        self.window_width = window_width
        self.window_height = window_height
        self.formats = formats

        # initialize effect objects
        self.effects = [
            effects.SoftThreshold(),
            effects.Alien(),
            effects.RGBWalk(),
            effects.RGBBurst(),
            effects.HueBloom(),
            effects.HueSwirl(),
            effects.HueSwirlMover()
        ]
        self.effect_index = None
        self.num_effects = len(self.effects)

        # initialize post-processing objects
        self.postproc = [
            effects.Border(style='postproc'),
            effects.Grating(style='postproc'),
        ]
        self.postproc_index = None
        self.num_postproc = len(self.postproc)
        self.is_post_process_pre = False  # post-proc pre/proceeds other effects

        self.mode = 'effect'  # define active mode; 'effect' | 'postproc'
        # get source material meta-data
        self.source_index = 0
        self.source_list = utils.get_sources()
        self.source_type = None
        self.num_sources = len(self.source_list)
        self.is_new_source = True

        # keep track of keyboard input
        self.key = 0
        self.key_list = [None for _ in range(256)]

        # misc
        self.fr_count = 0           # for replaying videos
        self.total_frame_count = 0  # for replaying videos
        self.cap = None             # for playing videos and webcam
        self.frame_mask = None      # ?
        self.auto_effect = None     # ?
        self.frame_orig = None      # ?

    def process(self):

        # parse keyboard input
        self._process_io()

        # load new source
        if self.is_new_source:
            self._load_new_source()

        # get frame and relevant info
        if self.source_type is 'cam' or self.source_type is 'video':
            ret, frame = self.cap.read()
        elif self.source_type is 'image':
            frame = np.copy(self.frame_orig)
            # frame = self.frame_orig
        elif self.source_type is 'auto':
            if self.mode == 'postproc':
                # passive auto mode
                frame = self.auto_effect.process(self.key_list, key_lock=True)
            else:
                # active auto mode
                frame = self.auto_effect.process(self.key_list)
                self.auto_effect.print_update()
        else:
            raise ValueError('%s is an invalid source type' % self.source_type)

        # get uniform frame sizes; 'auto' and 'gen' resize on their own
        if self.source_type is not 'auto' and self.source_type is not 'gen':
            if frame is None:
                raise TypeError('Frame is NoneType??')
            frame = cvutils.resize(
                frame, self.window_width, self.window_height)

        # update current effect
        if self.effect_index is None and self.mode == 'effect':
            for num in range(self.num_effects):
                if self.key == ord(str(num)):
                    self.effect_index = num
                    # print first update
                    self.effects[self.effect_index].print_update(force=True)
                    self.key_list[self.key] = False

        # update current post-processing effect
        if self.postproc_index is None and self.mode == 'postproc':
            for num in range(self.num_postproc):
                if self.key == ord(str(num)):
                    self.postproc_index = num
                    # print first update
                    self.postproc[self.postproc_index].print_update(
                        force=True)
                    self.key_list[self.key] = False

        # apply borders before effect
        if self.is_post_process_pre and self.postproc_index is not None:
            if self.mode == 'postproc':
                # active post-process mode
                frame = self.postproc[self.postproc_index].process(
                    frame, self.key_list)
                self.postproc[self.postproc_index].print_update()
            else:
                # passive post-process mode
                frame = self.postproc[self.postproc_index].process(
                    frame, self.key_list, key_lock=True)

        # process frame
        if self.source_type is not 'auto' and self.effect_index is not None:
            if self.mode == 'effect':
                # active process mode
                frame = self.effects[self.effect_index].process(
                    frame, self.key_list)
                self.effects[self.effect_index].print_update()
            else:
                # passive process mode
                frame = self.effects[self.effect_index].process(
                    frame, self.key_list, key_lock=True)

        # apply borders after effect
        if not self.is_post_process_pre and self.postproc_index is not None:
            if self.mode == 'postproc':
                # active post-process mode
                frame = self.postproc[self.postproc_index].process(
                    frame, self.key_list)
                self.postproc[self.postproc_index].print_update()
            else:
                # passive post-process mode
                frame = self.postproc[self.postproc_index].process(
                    frame, self.key_list, key_lock=True)

        # control animation
        self.fr_count += 1
        if self.fr_count == self.total_frame_count:
            # reset frame postion to 1 (not zero so window isn't moved)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            self.fr_count = 1

        self._clear_key_press()

        if self.formats == 'RGB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.formats == 'BGR':
            return frame

    def _process_io(self):

        # process keyboard input
        if self.key_list[ord('q')]:
            # quit current effect
            if self.mode == 'effect' and self.effect_index is not None:
                print('Quitting %s effect\n' %
                      self.effects[self.effect_index].name)
                self._display_options('effect')
                self.effect_index = None
            elif self.mode == 'postproc' and self.postproc_index is not None:
                print('Quitting %s post-processing effect\n' %
                      self.postproc[self.postproc_index].name)
                self._display_options('postproc')
                self.postproc_index = None
        elif self.key_list[ord('\b')]:
            # quit editing post-processing effect
            self.mode = 'effect'
            print('Exiting post-processing mode\n')
            if self.effect_index is not None:
                self.effects[self.effect_index].print_update(force=True)
            else:
                self._display_options('effect')
        elif self.key_list[ord('`')]:
            self.mode = 'postproc'
            print('Entering post-processing mode\n')
            if self.postproc_index is not None:
                self.postproc[self.postproc_index].print_update(force=True)
            else:
                self._display_options('postproc')
        elif self.key_list[ord(' ')]:
            self.source_index = (self.source_index + 1) % self.num_sources
            self.is_new_source = True
        elif self.key_list[ord('\t')]:
            # only change post-processing order if in post-process mode
            if self.mode == 'postproc':
                self.is_post_process_pre = not self.is_post_process_pre

    def _load_new_source(self):

        # reset necessary parameters
        self.is_new_source = False
        self.effect_index = None
        self.fr_count = 0
        for _, effect in enumerate(self.effects):
            effect.reset()
        # for _, effect in enumerate(self.postproc):
        #     effect.reset()

        # free previous resources
        if self.source_type is 'cam' or self.source_type is 'video':
            self.cap.release()

        # load source
        self.source_type = self.source_list[self.source_index]['file_type']
        source_loc = self.source_list[self.source_index]['file_loc']
        print('Loading %s' % source_loc)
        if self.source_type is 'cam':
            self.cap = cv2.VideoCapture(0)
            self.total_frame_count = float('inf')
            self.frame_mask = None
        elif self.source_type is 'video':
            self.cap = cv2.VideoCapture(source_loc)
            self.total_frame_count = \
                int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_mask = None
        elif self.source_type is 'image':
            self.frame_orig = cv2.imread(source_loc)
            print(self.frame_orig.shape)
            self.total_frame_count = float('inf')
            self.frame_mask = np.copy(self.frame_orig)
        elif self.source_type is 'auto':
            if source_loc is 'hueswirlchain':
                self.auto_effect = auto.HueSwirlChain(
                    self.window_width, self.window_height)
                self.auto_effect.update_output = 1  # force output
                self.auto_effect.print_update()
            else:
                print('Invalid auto effect')
                self.total_frame_count = float('inf')
        else:
            raise TypeError('Invalid source_type')
        # display effect options to user
        self._display_options('effect')

    def _display_options(self, style):
        if style == 'effect':
            print('Effect options:')
            for index, effect in enumerate(self.effects):
                print('%i: %s' % (index, effect.name))
        elif style == 'postproc':
            print('Post-processing options:')
            for index, effect in enumerate(self.postproc):
                print('%i: %s' % (index, effect.name))
        else:
            raise NameError

    def update_key_list(self, key):
        self.key = key
        self.key_list[self.key] = True

    def _clear_key_press(self):
        # don't clear escape
        if self.key != 27:
            self.key_list[self.key] = False
            self.key = 0

    def close(self):
        # free previous resources
        if self.source_type is 'cam' or self.source_type is 'video':
            self.cap.release()
