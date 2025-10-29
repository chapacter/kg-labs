import math, random, sys

import numpy as np
from PIL import Image, ImageOps
# from math import sin, cos, pi


MAX_SIZE = 2000
BG_COLOR = 50
img_mat = np.zeros((MAX_SIZE, MAX_SIZE, 3), dtype = np.uint8)
z_buffer = np.zeros((MAX_SIZE, MAX_SIZE, 1))

# Краски
cY = (180, 255, 10)
cP = (255, 180, 10)
cR = (255,0,0)
cG = (0,255,0)
cB = (0,0,255)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def draw_line6(x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    # y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        derror += dy
        if derror > (x1 - x0):
            derror -= 2 * (x1 - x0)
        y += y_update

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_file(filename, v, f):
    file = open(filename)
    for s in file:
        Sp = s.split()
        if Sp[0] == 'v':
            v.append([float(Sp[1]), float(Sp[2]), float(Sp[3])])
        if Sp[0] == 'f':
            f.append([int(Sp[1].split('/')[0]), int(Sp[2].split('/')[0]), int(Sp[3].split('/')[0])])

def calc_bar(x0, y0, x1, y1, x2, y2, x, y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

def calc_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    a = np.array([x1-x2, y1-y2, z1-z2])
    b = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(a, b)

def cos_angle(n):
    l = [0,0,1]
    n0 = np.dot(n,l) # Скалярное произведение
    n1 = np.sqrt(np.dot(n, n)) # Длина вектора нормы
    return n0 / n1

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    xmin = int(min(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    xmax = int(max(x0, x1, x2))+1 # округление вверх
    ymax = int(max(y0, y1, y2))+1

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > MAX_SIZE: xmax = MAX_SIZE
    if ymax > MAX_SIZE: ymax = MAX_SIZE

    c_angle = cos_angle(calc_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2))
    if c_angle >= 0: return
    color = (c_angle * -75, 0, c_angle * 130)

    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            l = calc_bar(x0, y0, x1, y1, x2, y2, x, y)
            z = l[0]*z0 + l[1]*z1 + l[2]*z2

            if l[0] >= 0 and l[1] >= 0 and l[2] >= 0 and z <= z_buffer[y,x]:
                img_mat[y][x] = color
                # img_mat[y][x] = (c_angle * -75, 0, c_angle * 130)
                z_buffer[y, x] = z

def draw_triangle():
    for k in range(len(f)):
        x0 = 9000 * v[f[k][0] - 1][0] + (MAX_SIZE/2)
        x1 = 9000 * v[f[k][1] - 1][0] + (MAX_SIZE/2)
        x2 = 9000 * v[f[k][2] - 1][0] + (MAX_SIZE/2)

        y0 = 9000 * v[f[k][0] - 1][1] + (MAX_SIZE/2)
        y1 = 9000 * v[f[k][1] - 1][1] + (MAX_SIZE/2)
        y2 = 9000 * v[f[k][2] - 1][1] + (MAX_SIZE/2)

        z0 = 9000 * v[f[k][0] - 1][2] + (MAX_SIZE/2)
        z1 = 9000 * v[f[k][1] - 1][2] + (MAX_SIZE/2)
        z2 = 9000 * v[f[k][2] - 1][2] + (MAX_SIZE/2)



        triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - -

for y in range(MAX_SIZE):
    for x in range(MAX_SIZE):
        z_buffer[y][x] = sys.maxsize

v = []
f = []
read_file('model_1.obj', v, f)
# print(v)

# v = 

for i in range(MAX_SIZE):
    for j in range(MAX_SIZE):
        img_mat[i][j] = 10

draw_triangle()

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
img.show('img.png')
