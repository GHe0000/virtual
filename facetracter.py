import time
import logging
import threading

import cv2
import dlib
import numpy as np

# 计算多边形面积
def area(a):
    a = np.array(a)
    x = a[:, 0]
    y = a[:, 1]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


# 提取脸位置
detector = dlib.get_frontal_face_detector()
def get_pos(img):
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))


# 提取面部关键点
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def get_pts(img, pos):
    landmark_shape = predictor(img, pos)
    pts = []
    for i in range(68):
        p = landmark_shape.part(i)
        pts.append(np.array([p.x, p.y], dtype=np.float32))
    return pts


# 计算pitch, yaw, roll
def get_rotation(pts):
    def get_center(indexes):
        return sum([pts[i] for i in indexes]) / len(indexes)
    
    l_brow = [18, 19, 20, 21]
    r_brow = [22, 23, 24, 25]
    chin = [6, 7, 8, 9, 10]
    nose = [29, 30]

    c_brow = get_center(l_brow + r_brow)
    c_chin = get_center(chin)
    c_nose = get_center(nose)

    l1 = c_brow - c_chin
    l2 = c_brow - c_nose
    l1_length = np.linalg.norm(l1)
    yaw = np.cross(l1, l2) / l1_length**2
    pitch = l1 @ l2 / l1_length**2
    roll = np.cross(l1, [0, 1]) / l1_length
    return np.array([yaw, pitch, roll])

# 计算相对位置
def get_rlt_pos(img, pos):
    x = (pos.top() + pos.bottom())/2/img.shape[0]
    y = 1 - (pos.left() + pos.right())/2/img.shape[1]
    return np.array([x, y])

# 计算面部大小
def face_size(pts):
    return np.array([area(pts[0:17])**0.5])

# 计算嘴大小
def mouth_size(pts):
    return np.array([area(pts[48:60]) / area(pts[0:17])])

# 计算眼睛大小
def eye_size(pts):
    l = area(pts[36:42]) / area(pts[0:17])
    r = area(pts[42:48]) / area(pts[0:17])
    return np.array([l,r])

# 计算眉毛高度
def brow_height(pts): 
    l = area([*pts[18:22]]+[pts[38], pts[37]]) / area(pts[0:17])
    r = area([*pts[22:26]]+[pts[44], pts[43]]) / area(pts[0:17])
    return np.array([l, r])

# 初始化
def init_func(img):
    pos = get_pos(img)
    if not pos:
        return None
    pts = get_pts(img, pos)
    rot = get_rotation(pts)
    return np.concatenate([rot])

# 主循环
def loop_func():
    global feature

    feature = np.array([0,0,0,0,0,0,0,0,0,0])
    init_rot = init_func(cv2.imread('std_face.jpg'))
    cap = cv2.VideoCapture(0)
    logging.warning('开始捕捉了！')
    while True:
        ret, img = cap.read()
        pos = get_pos(img)
        if pos is not None:
            pts = get_pts(img, pos)
            rlt_pos = get_rlt_pos(img, pos)
            rot = get_rotation(pts) - init_rot
            eye = eye_size(pts)
            brow = brow_height(pts)
            mouth = mouth_size(pts)
            feature = np.concatenate([rlt_pos, rot, eye, brow, mouth])

            # 绘图监测
            img //= 2
            img[pos.top():pos.bottom(), pos.left():pos.right()] *= 2 
            for i, (px, py) in enumerate(pts):
                cv2.putText(img, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))

        cv2.imshow('', img[:, ::-1])
        cv2.waitKey(1)
        
        time.sleep(1/60)


def get_feature():
    return feature


t = threading.Thread(target=loop_func)
t.setDaemon(True)
t.start()
logging.warning('捕捉线程启动中……')

if __name__ == '__main__':
    while True:
        time.sleep(0.1)
        print(feature)