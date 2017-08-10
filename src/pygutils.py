# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2007 Alex Holkner
# Copyright (c) 2007 Andrew Straw
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import numpy as np
import pyglet
from pyglet.window import key


class NumpyImage(pyglet.image.ImageData):
    def __init__(self, arr, format=None):
        '''Initialize numpy-based image data.

        :Parameters:
            `arr` : array
                numpy array of data. If rank 2, the shape must be
                (height, width). If rank 3, the shape is (depth,
                height, width). Must be C contiguous.
            `format` : str or None
                If specified, a format string describing the numpy
                array ('L' or 'RGB'). Defaults to a format determined
                from the shape of the array.

        '''
        arr = np.asarray(arr)
        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError('numpy array must be C contiguous')
        if len(arr.shape) == 2:
            height, width = arr.shape
            if format is None:
                format = 'L'
        elif len(arr.shape) == 3:
            height, width, depth = arr.shape
            if format is None:
                if depth == 3:
                    format = 'RGB'
                elif depth == 4:
                    format = 'RGBA'
                elif depth == 1:
                    format = 'L'
                else:
                    raise ValueError(
                        "could not determine a format for depth %d" % depth)
        else:
            raise ValueError("array must be rank 2 or rank 3")
        data = None
        pitch = arr.strides[0]
        super(NumpyImage, self).__init__(width, height, format, data,
                                         pitch=pitch)
        self.arr = arr
        self.view_new_array(arr)

    def _convert(self, format, pitch):
        if format == self._current_format and pitch == self._current_pitch:
            return self.numpy_data_ptr
        else:
            raise NotImplementedError(
                "no support for changing numpy format/pitch")

    def _ensure_string_data(self):
        raise RuntimeError(
            "we should never get here - we are trying to avoid data copying")

    def dirty(self):
        '''Force an update of the texture data.
        '''

        texture = self.texture
        internalformat = None
        self.blit_to_texture(texture.target, texture.level, 0, 0, 0,
                             internalformat)

    def view_new_array(self, arr):
        '''View a new numpy array of the same shape.

        The same texture will be kept, but the data from the new array
        will be loaded.

        :Parameters:
            `arr` : array
                numpy array of data. If rank 2, the shape must be
                (height, width). If rank 3, the shape is (depth,
                height, width).
        '''
        arr = np.asarray(arr)
        if arr.shape != self.arr.shape:
            raise ValueError("NumpyImage shape changed!")
        if not arr.dtype == np.uint8:
            raise ValueError("only uint8 numpy arrays supported")
        self.numpy_data_ptr = arr.ctypes.data
        self.arr = arr  # maintain a reference to numpy array so it's not de-allocated
        self.dirty()


class MyWindow(pyglet.window.Window):
    def __init__(self, frame, *args, **kwargs):
        # call pyglet.window.Window constructor
        super(MyWindow, self).__init__(*args, **kwargs)
        # store numpy array
        self.np_array = frame
        # store ctype
        self.np_image = NumpyImage(frame)
        # add key_list attribute
        self.key_list = [False for i in range(256)]
        # add current key attribute
        self.key = 0
        self.x = 0
        self.y = 0

    def on_draw(self):
        self.clear()
        if self.key_list[ord('a')]:
            self.x -= 10
        elif self.key_list[ord('d')]:
            self.x += 10
        self.np_image.texture.blit(self.x, self.y)
        self.clear_key_press()

    def on_key_press(self, symbol, modifiers):
        # update key list
        # if symbol == key.A:
        #     pass
        # elif symbol == key.LEFT:
        #     self.np_image.view_new_array(self.np_array)
        # elif symbol == key.ENTER:
        #     self.np_image.view_new_array(np.zeros_like(self.np_array))
        # elif symbol == key.ESCAPE:
        #     self.key_list[0] = True

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
        elif symbol == key.ENTER or key.RETURN:
            self.key = ord('\n')
        elif symbol == key.COMMA:
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
        # print(self.key)
        # print(self.key_list[self.key])

    def clear_key_press(self):

        self.key_list[self.key] = False
        self.key = 0
