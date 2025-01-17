import sys

# from gestureAndEyeTracking.Gesture_Control import GestureController

sys.path.append('/usr/local/lib/python3.10/site-packages')
import os
import argparse
import time
import datetime
import cv2
import numpy as np
import pyautogui
# import screeninfo
from gaze_tracker import GazeTracker
from calibration import calibrate
from screen import Screen

# from screeninfo import get_monitors
# for m in get_monitors():


RES_SCREEN = pyautogui.size()  # RES_SCREEN[0] -> width
# RES_SCREEN[1] -> height
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


def nothing(val):
    pass


def main():
    # remote source
    # camera = cv2.VideoCapture("URL")

    # webcam source
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    gaze_tracker = GazeTracker()
    screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)

    cv2.namedWindow("frame")
    cv2.createTrackbar('threshold', 'frame', 0, 255, nothing)
    cv2.setTrackbarPos('threshold', 'frame', 1)

    screen.clean()
    screen.show()

    os.makedirs('images', exist_ok=True)

    while True:

        _, frame = camera.read()

        #        print(frame.shape)

        start = time.time()

        gaze_tracker.update(frame)

        end = time.time()

        cv2.namedWindow("frame")
        dec_frame = gaze_tracker.eye_tracker.decorate_frame()

        dec_frame = cv2.resize(dec_frame, (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)))

        cv2.moveWindow("frame", 0, 0)
        cv2.imshow('frame', dec_frame)

        # gc1 = GestureController()
        # gc1.start()

        try:
            gaze = gaze_tracker.get_gaze()
        except:
            gaze = None
            screen.print_message("CALIBRATION REQUIRED!")
            screen.show()
            print("CALIBRATION REQUIRED!")

        print("GAZE: {}".format(gaze))

        if gaze:
            screen.update(gaze)
            screen.refresh()
            try:
                pyautogui.moveTo(gaze[0] + ((RES_SCREEN[0] - screen.width) // 2), gaze[1] + 25)
            except:
                pass

        print("TIME: {:.3f} ms".format(end * 1000 - start * 1000))

        k = cv2.waitKey(1) & 0xff
        if k == 1048603 or k == 27:  # esc to quit
            break
        if k == ord('c'):  # c to calibrate
            screen.mode = "calibration"
            screen.draw_center()
            calibrate(camera, screen, gaze_tracker)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
