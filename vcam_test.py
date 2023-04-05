import pyvirtualcam
import numpy as np

with pyvirtualcam.Camera(width=256, height=256, fps=20, \
                         backend="unitycapture") as cam:
    print(f'Using virtual camera: {cam.device}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        frame[:] = cam.frames_sent % 255  # grayscale animation
        cam.send(frame)
        cam.sleep_until_next_frame()