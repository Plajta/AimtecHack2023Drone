import time, cv2
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue

from djitellopy import Tello
import threading
import DetectSmile

tello = Tello()

def videoRecorder():
    while True:
        image = tello.get_frame_read().frame

        OUT, (X_bot, X_top, Y_bot, Y_top) = DetectSmile.DetectSMILE(image)

        cv2.imshow("output", OUT)
        cv2.waitKey(1)

def FollowFaces():
    while True:
        image = tello.get_frame_read().frame
        


tello = Tello()

tello.connect()

tello.streamon()

recorder = Thread(target=videoRecorder)
recorder.start()

"""
tello.takeoff()
tello.rotate_counter_clockwise(360)
tello.land()
keepRecording = False
recorder.join()
"""