import time
import logging

import numpy as np
import yaml

import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

import win32api

import facetracter

import threading

# ---------- 矩阵定义 ----------
def scale(x, y, z):
    a = np.eye(4, dtype=np.float32)
    a[0, 0] = x
    a[1, 1] = y
    a[2, 2] = z
    return a


def rotate(r, axis: tuple):
    a = np.eye(4, dtype = np.float32)
    a[axis[0], axis[0]] = np.cos(r)
    a[axis[0], axis[1]] = np.sin(r)
    a[axis[1], axis[0]] = - np.sin(r)
    a[axis[1], axis[1]] = np.cos(r)
    return a


def translate(x, y, z):
    a = np.eye(4, dtype = np.float32)
    a[3, 0] = x
    a[3, 1] = y
    a[3, 2] = z
    return a


def perspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 1 / 1000
    a[3, 2] = -0.0001
    a[2, 3] = 1
    a[3, 3] = 0
    return a

def inperspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 0
    a[3, 2] = 1
    a[2, 3] = -10000
    a[3, 3] = 10
    return a


# ---------- 图层类定义 ----------
class layer:
    def __init__(self, name, bbox, z, npdata, visual):
        self.name = name
        self.npdata = npdata
        self.visual = visual
        self.texture_num, texture_pos = self.get_texture()

        q, w = texture_pos
        a, b, c, d = bbox
        if type(z) in [int, float]:
            depth = np.array([[z, z], [z, z]])
        else:
            depth = np.array(z)
        assert len(depth.shape) == 2
        self.shape = depth.shape

        [[p1, p2],
         [p4, p3]] = np.array([
             [[a, b, 0, 1, 0, 0, 0, 1], [a, d, 0, 1, w, 0, 0, 1]],
             [[c, b, 0, 1, 0, q, 0, 1], [c, d, 0, 1, w, q, 0, 1]],
         ])
        x, y = self.shape
        self.vertex = np.zeros(shape=[x, y, 8])
        for i in range(x):
            for j in range(y):
                self.vertex[i, j] = p1 + (p4-p1)*i/(x-1) + (p2-p1)*j/(y-1)
                self.vertex[i, j, 2] = depth[i, j]

    # 生成纹理
    def get_texture(self):
        w, h = self.npdata.shape[:2]
        d = 2**int(max(np.log2(w), np.log2(h)) + 1)
        texture = np.zeros([d, d, 4], dtype = self.npdata.dtype)
        texture[:, :, :3] = 255
        texture[:w, :h] = self.npdata

        width, height = texture.shape[:2]
        texture_num = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_num)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_num, (w / d, h / d)

    def get_vertex(self):
        return self.vertex.copy()



# ---------- Vitural类定义 ----------
class Virtural:
    def __init__(self, inf_yaml, shape_yaml, others_yaml, test_yaml, size=(1024, 1024), pic_size=(1024, 1024)):
        with open(inf_yaml, encoding='utf8') as f:
            inf = yaml.safe_load(f)

        with open(shape_yaml, encoding='utf8') as f:
            self.change_inf = yaml.safe_load(f)
        
        self.Layers = []
        self.psd_size = pic_size
        self.size = size
        self.feature = np.zeros(8)
        self.test_dx = []
        
        for l in inf:
            a, b, c, d = inf[l]['bbox']
            self.Layers.append(layer(
                name=l,
                z=inf[l]['depth'],
                bbox=(b, a, d, c),
                npdata=np.load(inf[l]['path']),
                visual=inf[l]['visual']
            ))

        def update_num(yaml_inf):
            global test
            global t_c1
            test = yaml_inf['test']
            t_c1 = yaml_inf['t_c1']

        def update_test(yaml_inf):
            self.test_dx = yaml_inf

        def reload(path):
            try:
                with open(path, encoding='utf8') as f:
                    return yaml.safe_load(f)
            except Exception as error:
                    logging.exception(error)

        def reload_thread():
            logging.warning('Reload Thread Looping......')
            while True:
                update_num(reload(others_yaml))
                update_test(reload(test_yaml))
                time.sleep(1)

        t = threading.Thread(target = reload_thread)
        t.setDaemon(True)
        t.start()

    # 缩放
    def add_cut(self, a):
        model_g = \
                scale(2 / self.psd_size[0], 2 / self.psd_size[1], 1) @ \
                translate(-1, -1, 0) @ \
                rotate(- np.pi / 2, axis=(0, 1))
        return a @ model_g

    # 位置
    def add_pos(self, face_size, x, y, a):
        f = 750/(800-face_size)
        extra = translate(x, -y, 0) @ \
                scale(f, f, 1)
        return a @ extra

    # 旋转
    # def add_rot(self, rot, a):
    #     # yaw, pitch, roll
    #     view = \
    #             translate(0, 0, -1) @ \
    #             rotate(rot[0], axis=(0, 2)) @ \
    #             rotate(rot[1], axis=(2, 1)) @ \
    #             rotate(rot[2], axis=(0, 1)) @ \
    #             translate(0, 0, 1)
    #     return a @ view

    def add_rot(self, rot, name, a):

        if name in self.test_dx:
            dx, dy = self.test_dx[name]
            view = perspective() @ translate(-dx, -dy, 0) @ inperspective()
            view = view @ translate(0, 0, -1)
            view = view @ rotate(rot[0], axis=(0, 2)) @ rotate(rot[1], axis=(2, 1)) @ rotate(rot[2], axis=(0, 1))
            view = view @ rotate(rot[2], axis=(0, 1))
            view = view @ translate(0,0,1)
            view = view @ perspective() @ translate(dx, dy, 0) @ inperspective()
        else:
            view = translate(0, 0, -1)
            view = view @ rotate(rot[0], axis=(0, 2)) @ rotate(rot[1], axis=(2, 1)) @ rotate(rot[2], axis=(0, 1))
            view = view @ translate(0,0,1)
            
        return a @ view

    # 叠加变形
    def add_changes(self, Changes, layer_name, a):
        for change_name, intensity in Changes:
            change = self.change_inf[change_name]
            if layer_name in change:
                if 'pos' in change[layer_name]:
                    d = np.array(change[layer_name]['pos'])
                    a[:, :2] += d.reshape(a.shape[0], 2) * intensity
        return a

    # 更新是否可见
    def update_layer_visual(self, layer):
        if layer.name == "Q":
            layer.visual = key_state[0]
        if layer.name == "hand":
            layer.visual = key_state[1]

    def draw(self, layer):
        global p_a
        rlt_x, rlt_y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth = self.feature
        # roll = t_c1[2]*1.5*(time.time() - t0)
        # yaw = 0
        # pitch = 0
        vertex = layer.get_vertex()
        x, y, _ = vertex.shape
        ps = vertex.reshape(x*y, 8)
        a, b = ps[:, :4], ps[:, 4:]
        a = self.add_cut(a)
        a = a @ translate(t_c1[0],t_c1[1],0)
        a = a @ scale(1,1,1)

        z = a[:, 2:3]
        z -= 0.1
        a[:, :2] *= z

        p_a = np.array([t_c1[8],t_c1[9],t_c1[7],1])
        p_z = p_a[2:3]
        p_z -= 0.1
        p_a[:2] *= p_z

        a = self.add_changes([
            ['close_mouth', 1 - mouth],
            ['l_brow', brow_l],
            ['r_brow', brow_r],
            ['l_eye',1 - eye_l],
            ['r_eye',1 - eye_r]
        ], layer.name, a)
        if layer.name == 'body':
            a = self.add_rot(np.array([yaw, pitch, roll/10]),layer.name,a)
        else:
            a = self.add_rot(np.array([yaw, pitch, roll]),layer.name,a)
        if layer.name == 'hand':
            a = a @ translate(0.02, 0.11, 0) @ \
                    rotate(- mouse_theta, axis=(0, 1))@ \
                    translate(-0.05, -0.11, 0)
        # a = a @ scale(2,2,2) \
        #     @ rotate(0.5, axis=(0, 2)) \
        #     @ translate(0.4, 0, 0.2)
        a = a @ perspective()
        a = self.add_pos(face,rlt_x,rlt_y,a)

        p_a = p_a @ perspective()
        p_a = self.add_pos(face,rlt_x,rlt_y,p_a)

        b *= z
        ps[:, :4], ps[:, 4:] = a, b
        ps = ps.reshape([x, y, 8])

        glBegin(GL_QUADS)
        for i in range(x-1):
            for j in range(y-1):
                for p in [ps[i, j], ps[i, j+1], ps[i+1, j+1], ps[i+1, j]]:
                    glTexCoord4f(*p[4:])
                    glVertex4f(*p[:4])
        glEnd()

    def draw_loop(self, window, feature):
        while not glfw.window_should_close(window):
            glfw.poll_events()
            glClearColor(0,0,0,0)
            glClear(GL_COLOR_BUFFER_BIT)
            self.feature = feature()
            for layer in self.Layers:
                self.update_layer_visual(layer)
                if layer.visual == True:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, layer.texture_num)
                    glColor4f(1, 1, 1, 1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    self.draw(layer)

                    # 框图
                    glDisable(GL_TEXTURE_2D)
                    glColor4f(0, 0, 0, 1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    self.draw(layer)

            glPointSize(t_c1[6])
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_POINTS)
            glVertex4f(*p_a)
            glEnd()

            glfw.swap_buffers(window)
            get_mouse_theta()
            time.sleep(1/25)



# ---------- 窗口生成 ----------
monitor_size = None
window_pos = None
def init_window(v_size=(512,512)):
    global monitor_size
    global window_pos
    glfw.init()
    glfw.window_hint(glfw.DECORATED, True)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
    glfw.window_hint(glfw.FLOATING, True)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*v_size, 'V', None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - v_size[0], monitor_size.height - v_size[1])
    glfw.set_window_pos_callback(window, window_pos_callback)
    glfw.set_key_callback(window, key_callback)
    window_pos = np.array([monitor_size.width - v_size[0], monitor_size.height - v_size[1]])
    glViewport(0, 0, *v_size)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    return window


# ---------- 数据生成及处理 ----------
S = np.zeros((8,11))
def SlidingAverage(i):
    global S
    S = np.delete(S,0,axis=0)
    S = np.vstack((S,i))
    return np.average(S,axis=0)

# feature定义
# x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
feature = None
coefficient = np.array([1,1,0.3,0.3,0.4,1,150,150,100,100,80]) # 各个参数的系数
bias = np.array([-0.5,-0.4,0,0,0,0,-0.006,-0.006,-0.03,-0.03,-0.06]) # 各个参数的偏置
def feature_generate():
    global feature
    feature = SlidingAverage((facetracter.get_feature() + bias) * coefficient)
    feature[6:11] = np.clip(feature[6:11],0,1)
    return feature

def window_pos_callback(window, x, y):
    global window_pos
    window_pos = np.array([x,y])

mouse_theta = 0
def get_mouse_theta():
    global mouse_theta
    x, y = win32api.GetCursorPos()
    if (window_pos[0] - x - 1) != 0:
        mouse_theta = np.arctan((window_pos[1] - y - 1)/(window_pos[0] - x - 1))

key_state = np.array([False] * 2)
def key_callback(window, key, scancode, action, mods):
    global key_state
    if key == glfw.KEY_F1 and action == glfw.PRESS:
        key_state[0] = not(key_state[0])
    if key == glfw.KEY_F2 and action == glfw.PRESS:
        key_state[1] = not(key_state[1])

test = np.array([0,0,0,0,0,140,1,1,1,1,1])
def test_feature():
    return test

t0 = time.time()
t_c1 = np.zeros(10)

if __name__ == '__main__':
    window = init_window()
    V = Virtural(inf_yaml='./Pic/init_inf.yaml',\
                 shape_yaml='./Pic/change_inf.yaml',\
                 test_yaml='./test.yaml',\
                 others_yaml='./others.yaml')
    # V.draw_loop(window, feature = test_feature)
    V.draw_loop(window, feature = feature_generate)