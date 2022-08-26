import sys
import time
import math
import logging
import random

import numpy as np
import yaml

import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

import win32api
import matrix
import facetracter

v_size = 256,256

# 图层类
class layer:
    def __init__(self, name, bbox, z, npdata):
        self.name = name
        self.npdata = npdata
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
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        texture = np.zeros([d, d, 4], dtype=self.npdata.dtype)
        texture[:w, :h] = self.npdata

        width, height = texture.shape[:2]
        texture_num = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_num)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_num, (w / d, h / d)
    
    # 生成方块节点
    def get_square(self):
        x, y, _ = self.vertex.shape
        square = []
        for i in range(x-1):
            for j in range(y-1):
                square.append(
                    [[self.vertex[i, j], self.vertex[i, j+1]],
                     [self.vertex[i+1, j], self.vertex[i+1, j+1]]]
                )
        return square


class Virtural:
    def __init__(self, inf_yaml, shape_yaml, size=(1024, 1024), pic_size=(1024, 1024)):
        with open(inf_yaml, encoding='utf8') as f:
            inf = yaml.safe_load(f)

        with open(shape_yaml, encoding='utf8') as f:
            self.change_inf = yaml.safe_load(f)
        
        self.Layers = []
        self.psd_size = pic_size
        self.size = size
        
        for l in inf:
            a, b, c, d = inf[l]['bbox']
            npdata = np.load(inf[l]['path'][0])
            self.Layers.append(layer(
                name=l,
                z=inf[l]['depth'],
                bbox=(b, a, d, c),
                npdata=npdata
            ))

    # 缩放
    def add_cut(self, a):
        # model_g = \
        #         matrix.scale(2 / self.size[0], 2 / self.size[1], 1) @ \
        #         matrix.translate(-1, -1, 0) @ \
        #         matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
        model_g = \
                matrix.scale(2 / self.psd_size[0], 2 / self.psd_size[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
        return a @ model_g

    # 位置
    def add_pos(self, face_size, x, y, a):
        f = 0.01 * face_size
        extra = matrix.translate(x, -y, 0) @ \
                matrix.scale(f, f, 1)
        return a @ extra

    # 旋转
    def add_rot(self, rot, a):
        # yaw, pitch, roll
        view = \
                matrix.translate(0, 0, -1) @ \
                matrix.rotate_ax(rot[0], axis=(0, 2)) @ \
                matrix.rotate_ax(rot[1], axis=(2, 1)) @ \
                matrix.rotate_ax(rot[2], axis=(0, 1)) @ \
                matrix.translate(0, 0, 1)
        return a @ view

    # 叠加变形
    def add_changes(self, Changes, layer_name, a):
        for change_name, intensity in Changes:
            change = self.change_inf[change_name]
            if layer_name not in change:
                return a
            if 'pos' in change[layer_name]:
                d = change[layer_name]['pos']
                d = np.array(d)
                a[:, :2] += d.reshape(a.shape[0], 2) * intensity
        return a

    def draw_loop(self, window, feature):

        def draw(Layers):
            Vertexs = []
            for square in Layers.get_square():
                [[p1, p2],
                 [p4, p3]] = square
                Vertexs += [p1, p2, p3, p4]
            ps = np.array(Vertexs)

            a = ps[:, :4]
            b = ps[:, 4:]
            a = self.add_cut(a)
            z = a[:, 2:3]
            a[:, :2] *= z
            b *= z
            
            a = self.add_changes([
                ['close_mouth', 1 - mouth],
                ['close_eyes', 1 - eye],
            ], Layers.name, a)

            if Layers.name == 'body':
                a = self.add_rot(np.array([yaw, pitch, roll/10]),a)
            else:
                a = self.add_rot(np.array([yaw, pitch, roll]),a)
            a = self.add_pos(face,x,y,a)
            # a = a @ matrix.scale(2,2,2) \
            #     @ matrix.rotate_ax(0.5, axis=(0, 2)) \
            #     @ matrix.translate(0.4, 0, 0.2)
            a = a @ matrix.perspective(999)
            for i in range(len(Vertexs)):
                if i % 4 == 0:
                    glBegin(GL_QUADS)
                glTexCoord4f(*b[i])
                glVertex4f(*a[i])
                if i % 4 == 3:
                    glEnd()

        while not glfw.window_should_close(window):
            glfw.poll_events()
            glClearColor(0,0,0,0)
            glClear(GL_COLOR_BUFFER_BIT)
            x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth = feature()
            eye = (eye_l + eye_r)/2
            for Layers in self.Layers:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, Layers.texture_num)
                glColor4f(1, 1, 1, 1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                draw(Layers)

                # 框图
                glDisable(GL_TEXTURE_2D)
                glColor4f(0, 0, 0, 1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                draw(Layers)
            glfw.swap_buffers(window)
            time.sleep(1/30)
            test_mouse()


# 窗口初始化
monitor_size = None
window_pos = None
def init_window():
    global monitor_size
    global window_pos
    glfw.init()
    glfw.window_hint(glfw.DECORATED, True)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
    #glfw.window_hint(glfw.FLOATING, True)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*v_size, 'V', None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - v_size[0], monitor_size.height - v_size[1])
    glViewport(0, 0, *v_size)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    return window


# 滤波
S = np.zeros((8,11))
def SlidingAverage(i):
    global S
    S = np.delete(S,0,axis=0)
    S = np.vstack((S,i))
    return np.average(S,axis=0)


# feature定义
# x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
feature = None
coefficient = np.array([0.1,0.1,0.3,0.3,0.4,1,1,1,50,50,80]) # 各个参数的系数
bias = np.array([-0.3,-0.5,0,0,0,0,0,0,-0.03,-0.03,-0.05]) # 各个参数的偏置
def feature_generate():
    global feature
    feature = SlidingAverage((facetracter.get_feature() + bias) * coefficient)
    feature[6:11] = np.clip(feature[6:11],0,1)
    return feature


def test_feature():
    # x, y, yaw, pitch, roll, face, eye_l, eye_r, brow_l, brow_r, mouth
    return np.array([0,0,0,0,0,130,1,1,0,0,1])


def test_mouse():
    x, y = win32api.GetCursorPos()
    mouse_pos = np.array([x+1,y+1])/np.array(monitor_size)
    # print(mouse_pos)


window = init_window()
V = Virtural(inf_yaml='test_deep_inf.yaml', shape_yaml='test_inf.yaml')
# V.draw_loop(window, feature = test_feature)
V.draw_loop(window, feature = feature_generate)
