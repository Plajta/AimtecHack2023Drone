from djitellopy import Tello
import numpy as np
from numpy._typing import NDArray

COLORS = {0: "b", 1: "p", 2: "r"}

def light_metrix(tello: Tello, image: NDArray) -> str:
    
    if image.shape != (8,8,3):
        raise SyntaxError

    for row in image:
        for pix in row:
            i = np.argmax(pix)
            out += "0" if pix[i] < 128 else COLORS[i]

    tello.send_expansion_command("mled g " + out)


def clear_metrix(tello: Tello):
    tello.send_expansion_command("mled g " + "0"*64)
