import numpy as np
import cv2

import src.viztools as filters
import src.utils as utils
import src.cvutils as cvutils
import src.auto as auto


class WindowServer(object):
    """WindowServer object controls loading/generation of source frame, 
    processes the frame, and then returns the frame as a numpy array to be 
    rendered in the active window"""

    def __init__(self, window_width=100, window_height=100, formats='RGB'):

        self.window_width = window_width
        self.window_height = window_height
        self.formats = formats

        # initialize processing objects
        self.effects = [
            filters.Threshold(),     # 0
            filters.Alien(),         # 1
            filters.RGBWalk(),       # 2
            filters.RGBBurst(),      # 3
            filters.HueBloom(),      # 4
            filters.HueSwirl(),      # 5
            filters.HueSwirlMover()  # 6
        ]
        self.effect_index = None
        self.num_effects = len(self.effects)

        # set pre/post-processing options
        self.border = filters.Border()
        self.postproc = filters.PostProcess()
        self.post_process = False  # post-proc is active effect
        self.post_process_pre = False  # post-proc pre/proceeds other effects

        # get source material meta-data
        self.source_index = 0
        self.source_list = utils.get_sources()
        self.source_type = None
        self.num_sources = len(self.source_list)
        self.new_source = True

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
        if self.key_list[ord('q')]:
            # quit current effect
            if self.effect_index is not None:
                print('Quitting %s effect' %
                      self.effects[self.effect_index].name)
                print('')
                print('')
            self.effect_index = None
        elif self.key_list[ord('\b')]:
            self.post_process = False
        elif self.key_list[ord('`')]:
            self.post_process = True
        elif self.key_list[ord(' ')]:
            self.source_index = (self.source_index + 1) % self.num_sources
            self.new_source = True
        elif self.key_list[ord('\t')]:
            # only change post-processing order if in post-process mode
            if self.post_process:
                self.post_process_pre = not self.post_process_pre

        # load new source
        if self.new_source:

            # reset necessary parameters
            self.new_source = False
            self.effect_index = None
            self.fr_count = 0
            for _, effect in enumerate(self.effects):
                effect.reset()
            self.border.reset()
            self.postproc.reset()

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
                self.total_frame_count = float('inf')
                self.frame_mask = np.copy(self.frame_orig)
            elif self.source_type is 'auto':
                if source_loc is 'hueswirlchain':
                    self.auto_effect = auto.HueSwirlChain(
                        self.window_width, self.window_height)
                else:
                    print('Invalid auto effect')
                    self.total_frame_count = float('inf')
            else:
                raise TypeError('Invalid source_type')

        # get frame and relevant info
        if self.source_type is 'cam' or self.source_type is 'video':
            ret, frame = self.cap.read()
        elif self.source_type is 'image':
            frame = np.copy(self.frame_orig)
            # frame = self.frame_orig
        elif self.source_type is 'auto':
            frame = self.auto_effect.process(self.key_list)

        if self.source_type is not 'auto':
            if frame is None:
                raise TypeError('Frame is NoneType??')
            # get uniform frame sizes
            frame = cvutils.resize(frame,
                                   self.window_width, self.window_height)

        # update current effect
        if self.effect_index is None:
            for num in range(self.num_effects):
                if self.key == ord(str(num)):
                    self.effect_index = num
                    self.effects[self.effect_index].print_update()
                    self.key_list[self.key] = False

        # apply borders before effect
        if self.post_process_pre:
            if self.post_process:
                frame = self.border.process(frame, self.key_list)
            else:
                frame = self.border.process(frame, self.key_list,
                                            key_lock=True)

        # process frame
        if self.source_type is not 'auto' and self.effect_index is not None:
            if self.post_process:
                frame = self.effects[self.effect_index].process(
                    frame, self.key_list, key_lock=True)
            else:
                frame = self.effects[self.effect_index].process(
                    frame, self.key_list)
                # output info
                if self.effects[self.effect_index].update_output:
                    self.effects[self.effect_index].print_update()

        # apply borders after effect
        if not self.post_process_pre:
            if self.post_process:
                frame = self.border.process(frame, self.key_list)
                #             frame = postproc.process(frame, key_list)
            else:
                frame = self.border.process(frame, self.key_list,
                                            key_lock=True)
                # frame = postproc.process(frame, key_list, key_lock=True)

        # control animation
        self.fr_count += 1
        if self.fr_count == self.total_frame_count:
            # reset frame postion to 1 (not zero so window isn't moved)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            self.fr_count = 1

        self.clear_key_press()

        if self.formats == 'RGB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.formats == 'BGR':
            return frame

    def update_key_list(self, key):
        self.key = key
        self.key_list[self.key] = True

    def clear_key_press(self):
        # don't clear escape
        if self.key != 27:
            self.key_list[self.key] = False
            self.key = 0

    def close(self):
        # free previous resources
        if self.source_type is 'cam' or self.source_type is 'video':
            self.cap.release()
