"""
Contains classes for windows using pyglet or OpenCV; functions initialize
window, render texture to window upon calls to a draw command, and handle 
implementation-specific parsing of user input (just keyboard for now)
"""

import numpy as np
import ctypes
import pyglet
import pyglet.window.key as key
import cv2


class MyWindow(object):

    def set_image_data(self, image):
        raise NotImplementedError

    def on_draw(self):
        raise NotImplementedError

    def on_key_press(self, *args):
        raise NotImplementedError

    def clear_key_press(self):
        raise NotImplementedError

    def return_key(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class MyWindowPyglet(pyglet.window.Window, MyWindow):

    def __init__(self, disp_fps, *args, **kwargs):
        # call pyglet.window.Window constructor
        super(MyWindowPyglet, self).__init__(*args, **kwargs)
        # keep image data clean for now
        self.image_data = None
        # save current key press
        self.pyg_key = 0
        # display fps in window
        if disp_fps:
            self.fps_display = pyglet.clock.ClockDisplay()
        else:
            self.fps_display = None
        self.x = 0
        self.y = 0

    def to_c(self, arr):
        arr = np.swapaxes(np.swapaxes(np.flip(arr, 0), 0, 2), 1, 2)
        # bound = 0.1
        # bound_scale = 1 / (2. * bound)
        # ab = (bound_scale * (arr + bound)).astype('uint8')
        # arr2 = np.copy(arr1)
        # arr = np.array(arr, copy=True)
        return arr.astype('uint8').ctypes.data_as(
            ctypes.POINTER(ctypes.c_ubyte))
        # return ab.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        # return (GLubyte * arr.size)(*arr.ravel().astype('uint8'))

    def set_image_data(self, numpy_image):
        height, width, depth = numpy_image.shape
        if self.image_data is None:
            # self.image_data = ArrayInterfaceImage(numpy_image)
            self.image_data = pyglet.image.ImageData(
                width, height, 'RGB', self.to_c(numpy_image), width * 3)
        else:
            # self.image_data = pyglet.image.ImageData(
            #     width, height, 'RGB', self.to_c(numpy_image), width * 3)
            # self.image_data.view_new_array(numpy_image)
            self.image_data.set_data('RGB', width * 3, self.to_c(numpy_image))
            # self.image_data.blit_to_texture(
            #     self.image_data.texture.target,
            #     self.image_data.texture.level, 0, 0, 0)

    def on_draw(self):
        self.clear()
        if self.image_data is not None:
            self.image_data.blit(self.x, self.y)
        else:
            print('NoneType image data')
        if self.fps_display is not None:
            self.fps_display.draw()
        self.flip()

    def on_key_press(self, symbol, modifiers):
        # update key list
        self.pyg_key = symbol

    def clear_key_press(self):
        self.pyg_key = 0

    def return_key(self):
        """Return ascii value of most recently pressed key"""

        if self.pyg_key == key.A:
            ascii_key = ord('a')
        elif self.pyg_key == key.B:
            ascii_key = ord('b')
        elif self.pyg_key == key.C:
            ascii_key = ord('c')
        elif self.pyg_key == key.D:
            ascii_key = ord('d')
        elif self.pyg_key == key.E:
            ascii_key = ord('e')
        elif self.pyg_key == key.F:
            ascii_key = ord('f')
        elif self.pyg_key == key.G:
            ascii_key = ord('g')
        elif self.pyg_key == key.H:
            ascii_key = ord('h')
        elif self.pyg_key == key.I:
            ascii_key = ord('i')
        elif self.pyg_key == key.J:
            ascii_key = ord('j')
        elif self.pyg_key == key.K:
            ascii_key = ord('k')
        elif self.pyg_key == key.L:
            ascii_key = ord('l')
        elif self.pyg_key == key.M:
            ascii_key = ord('m')
        elif self.pyg_key == key.N:
            ascii_key = ord('n')
        elif self.pyg_key == key.O:
            ascii_key = ord('o')
        elif self.pyg_key == key.P:
            ascii_key = ord('p')
        elif self.pyg_key == key.Q:
            ascii_key = ord('q')
        elif self.pyg_key == key.R:
            ascii_key = ord('r')
        elif self.pyg_key == key.S:
            ascii_key = ord('s')
        elif self.pyg_key == key.T:
            ascii_key = ord('t')
        elif self.pyg_key == key.U:
            ascii_key = ord('u')
        elif self.pyg_key == key.V:
            ascii_key = ord('v')
        elif self.pyg_key == key.W:
            ascii_key = ord('w')
        elif self.pyg_key == key.X:
            ascii_key = ord('x')
        elif self.pyg_key == key.Y:
            ascii_key = ord('y')
        elif self.pyg_key == key.Z:
            ascii_key = ord('z')

        elif self.pyg_key == key._0:
            ascii_key = ord('0')
        elif self.pyg_key == key._1:
            ascii_key = ord('1')
        elif self.pyg_key == key._2:
            ascii_key = ord('2')
        elif self.pyg_key == key._3:
            ascii_key = ord('3')
        elif self.pyg_key == key._4:
            ascii_key = ord('4')
        elif self.pyg_key == key._5:
            ascii_key = ord('5')
        elif self.pyg_key == key._6:
            ascii_key = ord('6')
        elif self.pyg_key == key._7:
            ascii_key = ord('7')
        elif self.pyg_key == key._8:
            ascii_key = ord('8')
        elif self.pyg_key == key._9:
            ascii_key = ord('9')

        elif self.pyg_key == key.ESCAPE:
            ascii_key = 27
        elif self.pyg_key == key.GRAVE:
            ascii_key = ord('`')
        elif self.pyg_key == key.MINUS:
            ascii_key = ord('-')
        elif self.pyg_key == key.EQUAL:
            ascii_key = ord('=')
        elif self.pyg_key == key.BACKSPACE:
            ascii_key = ord('\b')
        elif self.pyg_key == key.BRACKETLEFT:
            ascii_key = ord('[')
        elif self.pyg_key == key.BRACKETRIGHT:
            ascii_key = ord(']')
        elif self.pyg_key == key.BACKSLASH:
            ascii_key = ord('\\')
        elif self.pyg_key == key.SEMICOLON:
            ascii_key = ord(';')
        elif self.pyg_key == key.APOSTROPHE:
            ascii_key = ord('\'')
        # elif self.pyg_key == key.ENTER or key.RETURN:
            # ascii_key = ord('\n')
            # print('enter')
        elif self.pyg_key == key.COMMA:  # key.COMMA:
            ascii_key = ord(',')
        elif self.pyg_key == key.PERIOD:
            ascii_key = ord('.')
        elif self.pyg_key == key.SLASH:
            ascii_key = ord('/')
        elif self.pyg_key == key.SPACE:
            ascii_key = ord(' ')
        elif self.pyg_key == key.TAB:
            ascii_key = ord('\t')

        elif self.pyg_key == key.LEFT:
            ascii_key = ord('Q')
        elif self.pyg_key == key.RIGHT:
            ascii_key = ord('S')
        elif self.pyg_key == key.UP:
            ascii_key = ord('T')
        elif self.pyg_key == key.DOWN:
            ascii_key = ord('R')

        else:
            ascii_key = 0

        return ascii_key


class MyWindowOCV(MyWindow):

    def __init__(self, disp_fps, fullscreen=False, name='frame'):

        # display fps in window
        self.disp_fps = disp_fps
        # fullscreen
        self.fullscreen = fullscreen
        # name of window
        self.name = name
        # keep image data clean for now
        self.image_data = None
        # save current key press
        self.key = 0

    def set_image_data(self, numpy_image):
        self.image_data = numpy_image
        self.on_draw()

    def on_draw(self):
        if self.fullscreen:
            cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.imshow(self.name, self.image_data)
        else:
            cv2.imshow(self.name, self.image_data)
            cv2.moveWindow(self.name, 0, 0)

    def on_key_press(self, symbol):
        # update key list
        self.key = symbol

    def clear_key_press(self):
        self.key = 0

    def return_key(self):
        """Return ascii value of most recently pressed key"""
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.name)
