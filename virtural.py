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
import psd_tools

import matrix

class layer:
    def __init__(self, name, bbox, z, npdata):
        self.name = name
        self.npdata = npdata
        self.texture_num, texture_pos = self.get_texture()
        self.变形 = []

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