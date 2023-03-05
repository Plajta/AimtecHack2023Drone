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
#tello.LOGGER.setLevel(logging.DEBUG)

time.sleep(3)
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
deviation = None

def videoRecorder():
    while True:
        image = frame_read.frame

        OUT, deviation = DetectSmile.DetectSMILE(image)
        led_matrix.light_metrix(tello, OUT)

        #compute how to rotate
        #convert pixels to degrees
        #dev_to_deg_Y = (deviation[0] / image.shape[0]) * 180
        dev_to_deg_X = (deviation[1] / image.shape[1]) * 180

        if dev_to_deg_X < 0: tello.rotate_clockwise(dev_to_deg_X)
        else: tello.rotate_counter_clockwise(-dev_to_deg_X)

        cv2.imshow("output", OUT)
        cv2.imshow("output_drone", image)
        if cv2.waitKey(1) == ord('q'):
            led_matrix.clear_metrix(tello)
            run = False
            break

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