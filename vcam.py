import time
import threading
import logging

import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

def start(vtuber, size):
    r, c = size
    def q():
        logging.warning("VCam Thread Looping......")
        with pyvirtualcam.Camera(width=r, height=c, fps=30, \
            backend="unitycapture", fmt=PixelFormat('ABGR')) as cam:

            base = np.zeros(shape=(c, r, 4), dtype=np.uint8)
            while True:
                img, alpha = vtuber.get_frame()
                a = np.where(alpha == 0, 1, 0)
                img[a] = 255,255,255,255
                print(img)
                # img = vtuber.get_frame()
                # base[:, (r-c)//2:(r-c)//2+c] = img[:, :, :3]
                cam.send(img)
                time.sleep(0.01)
    t = threading.Thread(target=q)
    t.setDaemon(True)
    t.start()