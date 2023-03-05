import time, cv2
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue

from djitellopy import Tello
import threading
import DetectSmile
import led_matrix
import logging

tello = Tello()
tello.LOGGER.setLevel(logging.DEBUG)
run = True

def videoRecorder():
    while run:
        image = tello.get_frame_read().frame

        OUT, (X_bot, X_top, Y_bot, Y_top) = DetectSmile.DetectSMILE(image)
        led_matrix.light_metrix(tello, OUT)

        # some motor controlling xD
        #
        #

        cv2.imshow("output", OUT)
        if cv2.waitKey(1) == ord('q'):
            led_matrix.clear_metrix(tello)
            run = False
            break

tello = Tello()
time.sleep(3)
tello.connect(False)

tello.streamon()

recorder = Thread(target=videoRecorder)
recorder.start()


tello.takeoff()
"""
tello.rotate_counter_clockwise(360)
tello.land()
keepRecording = False
recorder.join()
"""