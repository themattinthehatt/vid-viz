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
from pyglet.gl import *

def is_c_contiguous(inter):
    strides = inter.get('strides')
    shape = inter.get('shape')
    if strides is None:
        return True
    else:
        test_strides = strides[-1]
        N = len(strides)
        for i in range(N-2):
            test_strides *= test_strides * shape[N-i-1]
            if test_strides == strides[N-i-2]:
                continue
            else:
                return False
        return True


def get_stride0(inter):
    strides = inter.get('strides')
    if strides is not None:
        return strides[0]
    else:
        # C contiguous
        shape = inter.get('shape')
        cumproduct = 1
        for i in range(1,len(shape)):
            cumproduct *= shape[i]
        return cumproduct


class ArrayInterfaceImage(pyglet.image.ImageData):
    def __init__(self,arr,format=None,allow_copy=True):
        '''Initialize image data from the numpy array interface
        :Parameters:
            `arr` : array
                data supporting the __array_interface__ protocol. If
                rank 2, the shape must be (height, width). If rank 3,
                the shape is (height, width, depth). Typestr must be
                '|u1' (uint8).
            `format` : str or None
                If specified, a format string describing the data
                format array (e.g. 'L', 'RGB', or 'RGBA'). Defaults to
                a format determined from the shape of the array.
            `allow_copy` : bool
                If False, no copies of the data will be made, possibly
                resulting in exceptions being raised if the data is
                unsuitable. In particular, the data must be C
                contiguous in this case. If True (default), the data
                may be copied to avoid such exceptions.
        '''

        self.inter = arr.__array_interface__
        self.allow_copy = allow_copy
        self.data_ptr = ctypes.c_void_p()
        self.data_ptr.value = 0

        if len(self.inter['shape'])==2:
            height,width = self.inter['shape']
            if format is None:
                format = 'L'
        elif len(self.inter['shape'])==3:
            height,width,depth = self.inter['shape']
            if format is None:
                if depth==3:
                    format = 'RGB'
                elif depth==4:
                    format = 'RGBA'
                elif depth==1:
                    format = 'L'
                else:
                    raise ValueError("could not determine a format for "
                                     "depth %d"%depth)
        else:
            raise ValueError("arr must have 2 or 3 dimensions")
        data = None
        pitch = get_stride0(self.inter)
        super(ArrayInterfaceImage, self).__init__(
            width, height, format, data, pitch=pitch)

        self.view_new_array( arr )

    def get_data(self):
        if self._real_string_data is not None:
            return self._real_string_data

        if not self.allow_copy:
            raise ValueError("cannot get a view of the data without "
                             "allowing copy")

        # create a copy of the data in a Python str
        shape = self.inter['shape']
        nbytes = 1
        for i in range(len(shape)):
            nbytes *= shape[i]
        mydata = ctypes.create_string_buffer( nbytes )
        ctypes.memmove( mydata, self.data_ptr, nbytes)
        return mydata.value

    data = property(get_data,None,"string view of data")

    def _convert(self, format, pitch):
        if format == self._current_format and pitch == self._current_pitch:
            # do something with these values to convert to a ctypes.c_void_p
            if self._real_string_data is None:
                return self.data_ptr
            else:
                # XXX pyglet may copy this to create a pointer to the buffer?
                return self._real_string_data
        else:
            if self.allow_copy:
                raise NotImplementedError("XXX")
            else:
                raise ValueError("cannot convert to desired "
                                 "format/pitch without copying")

    def _ensure_string_data(self):
        if self.allow_copy:
            raise NotImplementedError("XXX")
        else:
            raise ValueError("cannot create string data without copying")

    def dirty(self):
        '''Force an update of the texture data.
        '''

        texture = self.texture
        internalformat = None
        self.blit_to_texture(
            texture.target, texture.level, 0, 0, 0, internalformat )

    def view_new_array(self,arr):
        '''View a new array of the same shape.
        The same texture will be kept, but the data from the new array
        will be loaded.
        :Parameters:
            `arr` : array
                data supporting the __array_interface__ protocol. If
                rank 2, the shape must be (height, width). If rank 3,
                the shape is (height, width, depth). Typestr must be
                '|u1' (uint8).
        '''

        inter = arr.__array_interface__

        if not is_c_contiguous(inter):
            if self.allow_copy:
                # Currently require numpy to deal with this
                # case. POSSIBLY TODO: re-implement copying into
                # string buffer so that numpy is not required.
                import numpy
                arr = numpy.array( arr, copy=True, order='C' )
                inter = arr.__array_interface__
            else:
                raise ValueError('copying is not allowed but data is not '
                                 'C contiguous')

        if inter['typestr'] != '|u1':
            raise ValueError("data is not type uint8 (typestr=='|u1')")

        if inter['shape'] != self.inter['shape']:
            raise ValueError("shape changed!")

        self._real_string_data = None
        self.data_ptr.value = 0

        idata = inter['data']
        if isinstance(idata,tuple):
            data_ptr_int,readonly = idata
            self.data_ptr.value = data_ptr_int
        elif isinstance(idata,str):
            self._real_string_data = idata
        else:
            raise ValueError("__array_interface__ data attribute was not "
                             "tuple or string")

        # maintain references so they're not de-allocated
        self.inter = inter
        self.arr = arr

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
        arr = np.swapaxes(np.swapaxes(np.flip(arr, 0), 0, 2), 1, 2)
        # noinspection PyUnresolvedReferences
        # ab = (bound_scale * (arr + bound)).astype('uint8')
        # arr2 = np.copy(arr1)
        # arr = np.array(arr, copy=True)
        return arr.astype('uint8').ctypes.data_as(
            ctypes.POINTER(ctypes.c_ubyte))
        # return (GLubyte * arr.size)( *arr.ravel().astype('uint8') )

    def set_image_data(self, numpy_image):
        height, width, depth = numpy_image.shape
        if self.image_data is None:
            # self.image_data = ArrayInterfaceImage(numpy_image)
            self.image_data = pyglet.image.ImageData(
                width, height, 'RGB', self.to_c(numpy_image), width * 3)
        else:
            # self.image_data.view_new_array(numpy_image)
            self.image_data.set_data('RGB', width * 3, self.to_c(numpy_image))
            self.image_data.blit_to_texture(
                self.image_data.texture.target,
                self.image_data.texture.level, 0, 0, 0)

    def on_draw(self):
        self.clear()
        if self.image_data is not None:
            self.image_data.texture.blit(self.x, self.y)
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
