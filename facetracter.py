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
# detector = dlib.get_frontal_face_detector()
# def get_pos(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     dets = detector(gray, 0)
#     if not dets:
#         return None
#     return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))

from ncnn.model_zoo import get_model
class mnet_detector:
    def __init__(self):
        self.c = 2
        self.net = get_model("retinaface", num_threads=4)
    def get_pos(self, img):
        img = cv2.resize(img, (320,240), interpolation=cv2.INTER_CUBIC)
        faceobjects = self.net(img)
        if not faceobjects:
            return None
        else:
            obj = faceobjects[0]
            if np.isnan(obj.rect.x):
                return None
            x = obj.rect.x * self.c
            y = obj.rect.y * self.c
            w = obj.rect.w * self.c
            h = obj.rect.h * self.c
            return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

# from retinaface import RetinaFaceDetector
# class mnet_detector:
#     def __init__(self):
#         self.retina = RetinaFaceDetector(top_k=40, min_conf=0.2)
#     def get_pos(self, img):
#         faces = self.retina.detect_retina(img)
#         if not faces:
#             return None
#         else:
#             x,y,w,h = faces[0]
#             if np.isnan(x):
#                 return None
#             return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

detector = mnet_detector()

# 提取面部关键点
predictor = dlib.shape_predictor('./files/shape_predictor_68_face_landmarks.dat')
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
    y = (pos.top() + pos.bottom())/2/img.shape[0]
    x = 1 - (pos.left() + pos.right())/2/img.shape[1]
    return np.array([x, y])


# 计算面部大小
def face_size(pts):
    return np.array([area(pts[0:17])**0.5])


# 计算嘴大小
def mouth_size(pts):
    if area(pts[0:17]) == 0:
        return np.array([0.1])
    return np.array([area(pts[48:60]) / area(pts[0:17])])


# 计算眼睛大小
def eye_size(pts):
    l = area(pts[36:42]) / area(pts[0:17])
    r = area(pts[42:48]) / area(pts[0:17])
    return np.array([l,r])

def eye_size2(pts):
    d00 =np.linalg.norm(pts[27]-pts[8]) # Length of face (eyebrow to chin)
    d11 =np.linalg.norm(pts[0]-pts[16]) # width of face
    d_reference = (d00+d11)/2
    # Left eye
    d1 =  np.linalg.norm(pts[37]-pts[41])
    d2 =  np.linalg.norm(pts[38]-pts[40])
    # Right eye
    d3 =  np.linalg.norm(pts[43]-pts[47])
    d4 =  np.linalg.norm(pts[44]-pts[46])
    l_eye = ((d1+d2)/(2*d_reference) - 0.02)*6
    r_eye = ((d3+d4)/(2*d_reference) -0.02)*6
    return np.array([l_eye,r_eye])

# 计算眉毛高度
def brow_height(pts): 
    l = area([*pts[18:22]]+[pts[38], pts[37]]) / area(pts[0:17])
    r = area([*pts[22:26]]+[pts[44], pts[43]]) / area(pts[0:17])
    return np.array([l, r])


class SlidingAverage:
    def __init__(self, num, l):
        self.S = np.zeros((l, num))

    def update(self, new):
        self.S = np.delete(self.S,0,axis=0)
        self.S = np.vstack((self.S,new))
        return np.average(self.S,axis=0)

class SlidingAverageXY:
    def __init__(self, l):
        self.l = l
        self.S = [np.zeros((self.l, 2))] * 68

    def update(self, new):
        rtn = [np.zeros((self.l, 2))] * 68
        for i in range(68):
            self.S[i] = np.delete(self.S[i],0,axis=0)
            self.S[i] = np.vstack((self.S[i],new[i]))
            rtn[i] = np.average(self.S[i],axis=0)
        return rtn

class KalmanFilterSimple:
    def __init__(self):
        self.K = 0
        self.X = 0
        self.P = 0.1
        self.Q = 0.008
        self.R = 0.0005

    def update(self, z):
        self.K = self.P / (self.P + self.R)
        self.X = self.X + self.K * (z - self.X)
        self.P = self.P - self.K * self.P + self.Q
        return self.X

class KalmanFilter:
    def __init__(self, m, Qval, Rval):
        self.K = np.zeros((m,m))
        self.X = np.zeros(m)
        self.P = np.eye(m)
        self.F = np.eye(m)
        self.B = np.eye(m)
        self.H = np.eye(m)
        self.Q = Qval * np.eye(m)
        self.R = Rval * np.eye(m)

    def update(self, uu, zz):
        self.X = self.F @ self.X + self.B @ uu
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.X = self.X + self.K @ (zz - self.H @ self.X)
        self.P = self.P - self.K @ self.H @ self.P
        return self.X

class DynamicsControl:
    def __init__(self, M=1, ALPHA=0.7, KP=0.04, KD=1):
        self.T = 0.1 # time interval
        self.ALPHA = ALPHA # incomplete derivative coefficient
        self.KP = KP
        self.KD = KD
        self.M = M # mass
        self.a = 0 # acceleration
        self.v = 0 # velocity
        self.x = 0 # position
        self.x_d = 0 # desired position
        self.e = 0 # error
        self.e_1 = 0 # last error
        self.de = 0 # derivative of error
        self.p_out = 0 # proportional termd_outd_out_1
        self.d_out = 0 # derivative term
        self.d_out_1 = 0 #  last derivative term 
        self.F = 0 # control force

        self.THRESH = 0.05 # control law changing threshold
    
    def update(self, X):
        self.x_d = self.x
        self.x = X

        self.e = self.x_d - self.x # Update error
        self.de = (self.e - self.e_1)/self.T # Compute the derivative of error
        self.p_out = self.KP*self.e
        self.d_out = (1-self.ALPHA)*self.KD*self.de + self.ALPHA*self.d_out_1

        self.F = self.p_out + self.d_out # Update control force

        self.e_1 = self.e # Update last error
        self.d_out_1 = self.d_out # Update last derivative term

        self.a = self.F/self.M # Update acceleration
        self.v = self.v + self.a*self.T # Update velocity
        self.x = self.x + self.v*self.T # Update position
        if self.x < 0:
            self.x = np.array([0.03])
        return self.x

# 初始化
def init_func(img):
    pos = detector.get_pos(img)
    # pos = get_pos(img)
    if not pos:
        return None
    pts = get_pts(img, pos)
    rot = get_rotation(pts)
    return np.concatenate([rot])

# 主循环
def loop_func():
    global feature

    feature = np.array([0,0,0,0,0,0,0,0,0,0,0])
    init_rot = init_func(cv2.imread('./files/std_face.jpg'))

    KalmanX = KalmanFilter(68, 1,10)
    KalmanY = KalmanFilter(68, 1,10)
    uu_ = np.zeros((68))

    Kalman_X = KalmanFilterSimple()
    Kalman_Y = KalmanFilterSimple()
    Kalman_yaw = KalmanFilterSimple()
    Kalman_pitch = KalmanFilterSimple()
    Kalman_roll = KalmanFilterSimple()

    SAXY = SlidingAverageXY(5)

    SlidingAverage1 = SlidingAverage(2,5)
    SlidingAverage2 = SlidingAverage(3,5)

    SlidingAverage3 = SlidingAverage(2,5)
    SlidingAverage4 = SlidingAverage(2,2)

    mouth_ctl = DynamicsControl(M=1, ALPHA=0.7, KP=0.04, KD=1)
    l_eye_ctl = DynamicsControl(M=1, ALPHA=0.8, KP=0.04, KD=1)
    r_eye_ctl = DynamicsControl(M=1, ALPHA=0.8, KP=0.04, KD=1)

    l_brow_ctl = DynamicsControl(M=1, ALPHA=0.7, KP=0.04, KD=1)
    r_brow_ctl = DynamicsControl(M=1, ALPHA=0.7, KP=0.04, KD=1)

    cap = cv2.VideoCapture(0)
    logging.warning('FaceTracter Looping......')
    while True:
    # for i in range(1000):
        # t_s = time.time()
        ret, img = cap.read()
        img = cv2.resize(img, (640,480), interpolation=cv2.INTER_CUBIC)
        # img = cv2.flip(img, 1, dst = None)
        # feature = gf(img) - init_gf

        pos = detector.get_pos(img)
        # pos = get_pos(img)
        if pos is not None:
            rlt_pos = get_rlt_pos(img, pos)
            pts = get_pts(img, pos)

            pts_orig = pts
            pts = SAXY.update(pts)

            rot = get_rotation(pts) - init_rot
            face = face_size(pts)
            eye = eye_size2(pts_orig)
            brow = brow_height(pts)
            mouth = mouth_size(pts_orig)

            rlt_pos[0] =  Kalman_X.update(rlt_pos[0])
            rlt_pos[1] =  Kalman_X.update(rlt_pos[1])
            rot[0] = Kalman_yaw.update(rot[0])
            rot[1] = Kalman_pitch.update(rot[1])
            rot[2] = Kalman_roll.update(rot[2])

            rlt_pos = SlidingAverage1.update(rlt_pos)
            rot = SlidingAverage2.update(rot)

            mouth = mouth_ctl.update(mouth)

            if eye[0] < 0.08:
                eye[0] = 0.03
            else:
                eye[0] = l_eye_ctl.update(eye[0])

            if eye[1] < 0.08:
                eye[1] = 0.03
            else:
                eye[1] = r_eye_ctl.update(eye[1])

            # eye = SlidingAverage4.update(eye)

            # eye[0] = l_eye_ctl.update(eye[0])
            # eye[1] = r_eye_ctl.update(eye[1])

            brow[0] = l_brow_ctl.update(brow[0])
            brow[1] = r_brow_ctl.update(brow[1])

            brow = SlidingAverage3.update(brow)

            feature = np.concatenate([rlt_pos, rot, face, eye, brow, mouth])
            # 绘图监测
            # img //= 2
            # img[pos.top():pos.bottom(), pos.left():pos.right()] *= 2 
            # for i, (px, py) in enumerate(pts):
            #     cv2.putText(img, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255))
            #     # cv2.putText(img, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 0, 0))

            # cv2.imshow('', img[:, ::-1])
            # cv2.waitKey(1)

            # 绘图监测
            # img2 = np.ones([512, 512], dtype=np.float32)
            # img2[pos.top():pos.bottom(), pos.left():pos.right()] *= 2 
            # for i, (px, py) in enumerate(pts):
            #     cv2.putText(img2, str(i), (int(px), int(py)), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 0, 0))
    
            # cv2.imshow('', img2[:, ::-1])
            # cv2.waitKey(1)
        # time.sleep(1/30)
        # print(1/(time.time()-t_s))

# from line_profiler import LineProfiler
# lp = LineProfiler()
# lp_wrapper = lp(loop_func)
# lp_wrapper()
# lp.print_stats()

def get_feature():
    return feature

t = threading.Thread(target=loop_func)
t.setDaemon(True)
t.start()
logging.warning('FaceTracter Starting......')

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=400)
if __name__ == '__main__':
    while True:
        time.sleep(0.1)
        # x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
        print(feature)