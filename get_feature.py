import time
import logging
import threading
import numpy as np

import socket
import struct

target_ip = "127.0.0.1"
target_port = 11573

def loop_func():
    global feature
    feature = np.array([0,0,0,0,0,0,0,0,0,0,0])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((target_ip, target_port))

    logging.warning("server start at: %s:%s" % (target_ip, target_port))
    logging.warning("wait for connection......")

    while True:
        indata, addr = sock.recvfrom(2048)
        A = struct.unpack("=di4fB11f",indata[:73])
        B = struct.unpack("14f",indata[-56:])

        rlt_pos = np.array([A[15], A[16]])
        face = np.array([A[17]])

        rot = np.array([A[12],A[13],A[14]])
        eye = np.array([A[4], A[5]])
        brow = np.array([B[2], B[5]])
        mouth = np.array([B[12]])

        feature = np.concatenate([rlt_pos, rot, face, eye, brow, mouth])
    sock.close()

def get_feature():
    return feature

t = threading.Thread(target=loop_func)
t.setDaemon(True)
t.start()
logging.warning("FaceTracter Starting......")

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=400)
if __name__ == "__main__":
    while True:
        time.sleep(0.1)
        # x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
        print(feature)