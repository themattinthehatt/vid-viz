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
        6 - huecrusher
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

import pyglet

import src.pygutils as putil
import src.windowserver as ws


# general user parameters
disp_full_screen = False
window_width = int(1980 / 2)
window_height = int(1200 / 2)
# window_width = int(1024)
# window_height = int(768)
disp_fps = True
target_fps = 30.0

# create window with pyglet
if disp_full_screen:
    window = putil.MyWindow(
        disp_fps,
        width=window_width,
        height=window_height,
        fullscreen=True)
else:
    window = putil.MyWindow(
        disp_fps,
        width=window_width,
        height=window_height,
        resizable=False)
    window.set_location(100, 800)

# create object to serve frames to pyglet window
server = ws.WindowServer(window_width=window_width,
                         window_height=window_height)


def loop(dt):

    # update keyboard input
    server.update_key_list(window.pyg_key)

    # exit program if desired
    if server.key_list[27]:
        # escape key has been pressed
        window.close()
        pyglet.app.exit()

    # generate new frame
    frame = server.process()

    # update image_data to display new frame
    window.set_image_data(frame)

    # clear out key presses
    window.clear_key_press()

pyglet.clock.set_fps_limit(target_fps)
pyglet.clock.schedule_interval(loop, 1. / target_fps)
pyglet.app.run()
