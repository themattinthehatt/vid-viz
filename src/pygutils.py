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
import ctypes
import pyglet


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
                    format = 'G'
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

    # def _convert(self, format, pitch):
    #     if format == self._current_format and pitch == self._current_pitch:
    #         return self.numpy_data_ptr
    #     else:
    #         raise NotImplementedError(
    #             "no support for changing numpy format/pitch")
    #
    # def _ensure_string_data(self):
    #     raise RuntimeError(
    #         "we should never get here - we are trying to avoid data copying")

    def dirty(self):
        '''Force an update of the texture data.
        '''

        texture = self.texture
        internalformat = None
        self.blit_to_texture(texture.target, texture.level, 0, 0, 0,
                             internalformat=internalformat)

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
    def __init__(self, disp_fps, *args, **kwargs):
        # call pyglet.window.Window constructor
        super(MyWindow, self).__init__(*args, **kwargs)
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
        arr1 = np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)
        # noinspection PyUnresolvedReferences
        # ab = (bound_scale * (arr + bound)).astype('uint8')
        return arr1.astype('uint8').ctypes.data_as(
            ctypes.POINTER(ctypes.c_ubyte))

    def set_image_data(self, numpy_image):
        height, width, depth = numpy_image.shape
        if self.image_data is None:
            self.image_data = pyglet.image.ImageData(
                width, height, 'RGB', self.to_c(numpy_image), width * 3)
        else:
            self.image_data.set_data('RGB', width * 3, self.to_c(numpy_image))

    def on_draw(self):
        self.clear()
        if self.image_data is not None:
            self.image_data.blit(self.x, self.y)
        if self.fps_display is not None:
            self.fps_display.draw()

    def on_key_press(self, symbol, modifiers):
        # update key list
        self.pyg_key = symbol

    def clear_key_press(self):
        self.pyg_key = 0
