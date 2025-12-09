import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi
import math
import random
import sys

height = 2000
width = 2000

a_x = 1800
a_y = 1800

img_mat = np.zeros((height, width, 3), dtype=np.uint8)

t_x = -0.00
t_y = -0.03
t_z = 0.12

alfa = 0
beta = 3.14
gama = 0

def barycentric_coordinates (x0,y0,x1,y1,x2,y2,x,y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

def drawing_triangle (x0,y0,z0,x1,y1,z1,x2,y2,z2):
    x0_new = x0 * a_x / z0 + width / 2
    y0_new = y0 * a_y / z0 + height / 2
    x1_new = x1 * a_x / z1 + width / 2
    y1_new = y1 * a_y / z1 + height / 2
    x2_new = x2 * a_x / z2 + width / 2
    y2_new = y2 * a_y / z2 + height / 2

    xmin = int(min(x0_new, x1_new, x2_new))
    xmax = int(max(x0_new, x1_new, x2_new))+ 1
    ymin = int(min(y0_new, y1_new, y2_new))
    ymax = int(max(y0_new, y1_new, y2_new))+ 1
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > 2000): xmax = 2000
    if (ymax > 2000): ymax = 2000

    # color = (random.randrange(0,255), random.randrange(0,255), random.randrange(0,255))

    cos=cos_angle(n(x0, y0, x1, y1, x2, y2, z0, z1, z2))
    if (cos>=0):
        return
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            temporary = barycentric_coordinates(x0_new, y0_new, x1_new, y1_new, x2_new, y2_new, x, y)
            z = temporary[0] * z0 + temporary[1] * z1 + temporary[2] * z2
            if temporary[0] >= 0 and temporary[1] >= 0 and temporary[2] >= 0 and z<z_buffer[y, x]:
                img_mat[y][x] = (cos * 0, cos * -255, cos * 0)
                # img_mat[y][x] = color
                z_buffer[y, x] = z

# Вычисление векторного произведения
def n(x0, y0, x1, y1, x2, y2, z0, z1, z2):
    a = np.array([x1-x2, y1-y2, z1-z2])
    b = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(a, b)


def cos_angle(n):
    l = [0,0,1]
    nFirst = np.dot(n,l) # Скалярное произведение вектора
    nSecond = np.sqrt(np.dot(n, n)) # Вычисление длины вектора
    return nFirst / nSecond

z_buffer = np.zeros((2000,2000,1))
for i in range(2000):
    for j in range(2000):
        z_buffer[i][j] = sys.maxsize

for i in range(2000):
    for j in range(2000):
        img_mat[i, j]= 0

file=open('model_1.obj')
v=[]
f=[]
for s in file:
    sp=s.split()
    if(sp[0]=='v'):
        v.append([float(sp[1]), float(sp[2]),float(sp[3])])
# print(v)
    if(sp[0]=='f'):
        f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])

cos_alfa = math.cos(alfa)
sin_alfa = math.sin(alfa)

cos_beta = math.cos(beta)
sin_beta = math.sin(beta)

cos_gama = math.cos(gama)
sin_gama = math.sin(gama)

Rx = np.array([[1, 0, 0], [0, cos_alfa, sin_alfa], [0, -sin_alfa, cos_alfa]])
Ry = np.array([[cos_beta, 0, sin_beta], [0,1,0], [-sin_beta,0,cos_beta]])
Rz = np.array([[cos_gama,sin_gama,0],[-sin_gama,cos_gama,0],[0,0,1]])
R = Rx @ Ry @ Rz #делаем тут умножение матриц
vr=[]
for i  in range(len(v)):
    # vr.append(np.dot(np.array(v[i]), R)+[t_x, t_y, t_z])
    vr.append(np.array(v[i]) @ R + [t_x, t_y, t_z])

for k in range (len(f)):
    x0=vr[f[k][0]-1][0] #вместо v теперь vr тут массив
    y0=vr[f[k][0]-1][1]
    x1=vr[f[k][1]-1][0]
    y1=vr[f[k][1]-1][1]
    x2=vr[f[k][2]-1][0]
    y2=vr[f[k][2]-1][1]
    z0 = vr[f[k][0] - 1][2]
    z1 = vr[f[k][1] - 1][2]
    z2 = vr[f[k][2] - 1][2]
    drawing_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)

img=Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
img.show('img.png')
