import numpy as np
import cv2
import pyglet.window.key as key

import src.viztools as vv
import src.utils as utils
import src.auto as auto


class WindowServer(object):
    """WindowServer object controls loading/generation of source frame, 
    process the frame, and then returns the frame as a numpy array to be 
    rendered in the active window"""

    def __init__(self, window_width=100, window_height=100):

        self.window_width = window_width
        self.window_height = window_height

        # initialize processing objects
        self.effects = [
            vv.Threshold(),  # 0
            vv.Alien(),  # 1
            vv.RGBWalk(),  # 2
            vv.RGBBurst(),  # 3
            vv.HueBloom(),  # 4
            vv.HueSwirl(),  # 5
            vv.HueSwirlMover()]  # 6
        self.effect_index = None
        self.num_effects = len(self.effects)

        # set pre/post-processing options
        self.border = vv.Border()
        self.postproc = vv.PostProcess()
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
        self.fr_count = 0
        self.total_frame_count = 0
        self.cap = None
        self.frame_mask = None
        self.auto_effect = None
        self.frame_orig = None

    def process(self):

        # parse keyboard input
        if self.key_list[ord('q')]:
            # quit current effect
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
            if self.source_type is 'cam':
                # pass
                self.cap = cv2.VideoCapture(0)
                self.total_frame_count = float('inf')
                self.frame_mask = None
            elif self.source_type is 'video':
                cap = cv2.VideoCapture(source_loc)
                self.total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                print('Invalid source_type')

        # # get frame and relevant info
        if self.source_type is 'cam' or self.source_type is 'video':
            ret, frame = self.cap.read()
        elif self.source_type is 'image':
            frame = np.copy(self.frame_orig)
        elif self.source_type is 'auto':
            frame = self.auto_effect.process(self.key_list)

        if self.source_type is not 'auto':
            # get uniform frame sizes
            frame = utils.resize(frame, self.window_width, self.window_height)

        # update current effect
        if self.effect_index is None:
            for num in range(10):
                if self.key == ord(str(num)):
                    print('Effect %g' % num)
                    self.effect_index = num
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

        # apply borders after effect
        if not self.post_process_pre:
            if self.post_process:
                frame = self.border.process(frame, self.key_list)
                #             frame = postproc.process(frame, key_list)
            else:
                frame = self.border.process(frame, self.key_list,
                                            key_lock=True)
                # frame = postproc.process(frame, key_list, key_lock=True)

        self.clear_key_press()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_key_list(self, symbol):
        """`symbol` is a pyglet key symbol"""

        if symbol == key.A:
            self.key = ord('a')
        elif symbol == key.B:
            self.key = ord('b')
        elif symbol == key.C:
            self.key = ord('c')
        elif symbol == key.D:
            self.key = ord('d')
        elif symbol == key.E:
            self.key = ord('e')
        elif symbol == key.F:
            self.key = ord('f')
        elif symbol == key.G:
            self.key = ord('g')
        elif symbol == key.H:
            self.key = ord('h')
        elif symbol == key.I:
            self.key = ord('i')
        elif symbol == key.J:
            self.key = ord('j')
        elif symbol == key.K:
            self.key = ord('k')
        elif symbol == key.L:
            self.key = ord('l')
        elif symbol == key.M:
            self.key = ord('m')
        elif symbol == key.N:
            self.key = ord('n')
        elif symbol == key.O:
            self.key = ord('o')
        elif symbol == key.P:
            self.key = ord('p')
        elif symbol == key.Q:
            self.key = ord('q')
        elif symbol == key.R:
            self.key = ord('r')
        elif symbol == key.S:
            self.key = ord('s')
        elif symbol == key.T:
            self.key = ord('t')
        elif symbol == key.U:
            self.key = ord('u')
        elif symbol == key.V:
            self.key = ord('v')
        elif symbol == key.W:
            self.key = ord('w')
        elif symbol == key.X:
            self.key = ord('x')
        elif symbol == key.Y:
            self.key = ord('y')
        elif symbol == key.Z:
            self.key = ord('z')

        elif symbol == key._0:
            self.key = ord('0')
        elif symbol == key._1:
            self.key = ord('1')
        elif symbol == key._2:
            self.key = ord('2')
        elif symbol == key._3:
            self.key = ord('3')
        elif symbol == key._4:
            self.key = ord('4')
        elif symbol == key._5:
            self.key = ord('5')
        elif symbol == key._6:
            self.key = ord('6')
        elif symbol == key._7:
            self.key = ord('7')
        elif symbol == key._8:
            self.key = ord('8')
        elif symbol == key._9:
            self.key = ord('9')

        elif symbol == key.ESCAPE:
            self.key = 27
        elif symbol == key.GRAVE:
            self.key = ord('`')
        elif symbol == key.MINUS:
            self.key = ord('-')
        elif symbol == key.EQUAL:
            self.key = ord('=')
        elif symbol == key.BACKSPACE:
            self.key = ord('\b')
        elif symbol == key.BRACKETLEFT:
            self.key = ord('[')
        elif symbol == key.BRACKETRIGHT:
            self.key = ord(']')
        elif symbol == key.BACKSLASH:
            self.key = ord('\\')
        elif symbol == key.SEMICOLON:
            self.key = ord(';')
        elif symbol == key.APOSTROPHE:
            self.key = ord('\'')
        # elif symbol == key.ENTER or key.RETURN:
            # self.key = ord('\n')
            # print('enter')
        elif symbol == key.COMMA:  # key.COMMA:
            self.key = ord(',')
        elif symbol == key.PERIOD:
            self.key = ord('.')
        elif symbol == key.SLASH:
            self.key = ord('/')
        elif symbol == key.SPACE:
            self.key = ord(' ')
        elif symbol == key.TAB:
            self.key = ord('\t')

        elif symbol == key.LEFT:
            self.key = ord('Q')
        elif symbol == key.RIGHT:
            self.key = ord('S')
        elif symbol == key.UP:
            self.key = ord('T')
        elif symbol == key.DOWN:
            self.key = ord('R')

        else:
            self.key = 0

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
