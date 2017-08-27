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

from __future__ import print_function
from __future__ import division

import cv2
import sys
sys.path.append('lib/pyglet-1.2.4-py2.7.egg')
import pyglet

import src.pygutils as putil
import src.windowserver as ws


# general user parameters
window_manager = 'pyglet'  # 'opencv' | 'pyglet'
disp_full_screen = False
window_width = int(1980 / 2)
window_height = int(1200 / 2)
# window_width = int(1024)
# window_height = int(768)
disp_fps = True
target_fps = 25.0

if window_manager == 'pyglet':

    # create object to serve frames to pyglet window
    server = ws.WindowServer(window_width=window_width,
                             window_height=window_height,
                             formats='RGB')

    # create window with pyglet
    if disp_full_screen:
        window = putil.MyWindow(
            disp_fps,
            resizable=False,
            vsync=False,
            fullscreen=True)
    else:
        window = putil.MyWindow(
            disp_fps,
            width=window_width,
            height=window_height,
            resizable=False,
            vsync=False,
            fullscreen=False)
        window.set_location(100, 50)


    def loop(dt):

        # update keyboard input
        server.update_key_list(window.pyg_key)

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

    # pyglet.clock.set_fps_limit(target_fps)
    # pyglet.clock.schedule(window.update)
    pyglet.clock.schedule_interval(loop, 1.0 / target_fps)
    pyglet.app.run()

elif window_manager == 'opencv':

    import time

    # create object to serve frames to opencv window
    server = ws.WindowServer(window_width=window_width,
                             window_height=window_height,
                             formats='BGR')

    while True:

        time_pre = time.time()

        # update keyboard input
        server.key = cv2.waitKey(1) & 0xFF
        server.key_list[server.key] = True

        # exit program if desired
        if server.key_list[27]:
            # escape key has been pressed
            server.close()
            cv2.destroyAllWindows()
            break

        # generate new frame
        frame = server.process()

        # display frame
        if disp_full_screen:
            cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.imshow('frame', frame)
        else:
            cv2.imshow('frame', frame)
            if server.fr_count == 0:
                cv2.moveWindow('frame', 0, 0)

        # calculate, limit and output fps
        time_tot = time.time() - time_pre
        if time_tot < 1 / target_fps:
            time.sleep(1 / target_fps - time_tot)
        time_tot = time.time() - time_pre

        if disp_fps:
            print('\r%03i fps' % (1.0 / time_tot), end='')
