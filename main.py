"""vid-viz main loop using pyglet to handle window creation

KEYBOARD INPUTS:
    number keys - choose effect (see viztools.py for detailed keyboard input 
    for each effect)
        0 - thresholding
        1 - alien
        2 - rgbwalk
        3 - rgbburst
        4 - huebloom
        5 - hueswirl
        6 - hueswirlmover
    q - quit effect (then number keys are once again available for choosing 
        a new effect)
    ` - enable toggling of border effects ('t' to toggle, tab to switch 
        processing/border order)
    spacebar - cycle through sources
    backspace - quit border effect editing
    esc - exit loop
"""

from __future__ import division
from __future__ import print_function

import time
import sys
sys.path.append('lib/pyglet-1.2.4-py2.7.egg')
import pyglet

import src.window as win
import src.windowserver as ws


# general user parameters
window_manager = 'opencv'  # 'opencv' | 'pyglet'
disp_fullscreen = True
window_width = int(1980 / 2)
window_height = int(1200 / 2)
# window_width = int(1024)
# window_height = int(768)
disp_fps = True
target_fps = 25.0

# run event loop
if window_manager == 'pyglet':

    # create window with pyglet
    if disp_fullscreen:
        window = win.MyWindowPyglet(
            disp_fps,
            resizable=False,
            vsync=False,
            fullscreen=True)
    else:
        window = win.MyWindowPyglet(
            disp_fps,
            width=window_width,
            height=window_height,
            resizable=False,
            vsync=False,
            fullscreen=False)
        window.set_location(100, 50)

    # create object to serve frames to pyglet window
    server = ws.WindowServer(
        window_width=window_width,
        window_height=window_height,
        formats='RGB')

    # define event loop
    def loop(dt):

        # update keyboard input
        server.update_key_list(window.return_key())

        # generate new frame
        frame = server.process()

        # update image_data to display new frame
        window.set_image_data(frame)

        # clear out key presses
        window.clear_key_press()

        # exit program if desired
        if server.key_list[27]:
            # escape key has been pressed
            server.close()
            window.close()
            pyglet.app.exit()

    # run event loop
    # pyglet.clock.set_fps_limit(target_fps)
    # pyglet.clock.schedule(window.update)
    pyglet.clock.schedule_interval(loop, 1.0 / target_fps)
    pyglet.app.run()

elif window_manager == 'opencv':

    # create window with opencv
    window = win.MyWindowOCV(
        disp_fps,
        fullscreen=disp_fullscreen)

    # create object to serve frames to opencv window
    server = ws.WindowServer(
        window_width=window_width,
        window_height=window_height,
        formats='BGR')

    # define event loop
    def loop():

        # update keyboard input
        server.update_key_list(window.return_key())

        # generate new frame
        frame = server.process()

        # update image_data to display new frame
        window.set_image_data(frame)

        # clear out key presses
        window.clear_key_press()

        # exit program if desired
        if server.key_list[27]:
            # escape key has been pressed
            server.close()
            window.close()
            break_loop = True
        else:
            break_loop = False

        return break_loop

    # run event loop
    while True:

        time_pre = time.time()

        break_loop = loop()

        # calculate, limit and output fps
        time_tot = time.time() - time_pre
        if time_tot < 1 / target_fps:
            time.sleep(1 / target_fps - time_tot)
        time_tot = time.time() - time_pre

        # if disp_fps:
        #     print('\r%03i fps' % (1.0 / time_tot), end='')

        if break_loop:
            break
