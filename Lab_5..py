import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi
import math
import random
import sys

height = 5000
width = 1200

a_x = 1800
a_y = 1800

img_mat = np.zeros((height, width, 3), dtype=np.uint8)

# смещение
# t_x = -0.00
# t_y = -0.03
# t_z = 0.12



alfa = 0
beta = 3.14
gama = 0

n_arr = []

# Разрешение текстуры
w_t = 1024
h_t = 1024

# Константы для поворота с помощью кватернионов
u = [1, 1, 0] # Единичный вектор
temp_norm = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
u[0] /= temp_norm
u[1] /= temp_norm
u[2] /= temp_norm
tetta = 0

class Quaternion:
    def __init__(self, Re, i,j,k):
        self.Re = Re
        self.i = i
        self.j = j
        self.k = k
    def __add__ (self, number):
        return Quaternion(self.Re + number.Re, self.i + number.i, self.j+ number.j, self.k + number.k)

    def __mul__ (self, number):
        return Quaternion(self.Re * number.Re - self.i * number.i - self.j * number.j - self.k * number.k,
                            self.Re * number.i + self.i * number.Re + self.j * number.k - self.k * number.j,
                            self.Re * number.j - self.i * number.k + self.j * number.Re + self.k * number.i,
                            self.Re * number.k + self.i * number.j - self.j * number.i + self.k * number.Re)
    def sopr (self):
        return Quaternion(self.Re, -self.i, -self.j, -self.k)
    def norma(self):
        return Quaternion(self.Re**2, self.i**2, self.j**2, self.k**2)


def barycentric_coordinates (x0,y0,x1,y1,x2,y2,x,y):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

# Вычисляем направление света в вершине
def I_l(n, l):
    np0 = np.dot(n, l)  # Вычисляем скалярное произведение
    n_l = np.sqrt(np.dot(n, n))  # Вычисляем длину вектора нормы
    temp = np0 / n_l
    return temp


def drawing_triangle (x0, y0, z0, x1, y1, z1, x2, y2, z2, n_temp, k):
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
    if (xmax > width): xmax = width
    if (ymax > height): ymax = height

    # color = (random.randrange(0,255), random.randrange(0,255), random.randrange(0,255))
    cos = cos_angle(x0, y0, x1, y1, x2, y2, z0, z1, z2, n_temp)
    if (cos>=0):
        return
    # Вычисляем направление света в вершинах
    l = [0,0,1]
    I0 = I_l(v_n[f[k][0]-1], l)
    I1 = I_l(v_n[f[k][1]-1], l)
    I2 = I_l(v_n[f[k][2]-1], l)
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            temporary = barycentric_coordinates(x0_new, y0_new, x1_new, y1_new, x2_new, y2_new, x, y)
            z = temporary[0] * z0 + temporary[1] * z1 + temporary[2] * z2
            Im=-(temporary[0]*I0+temporary[1]*I1+temporary[2]*I2) # Вычисляем свет в точке (яркость пикселя)
            if temporary[0] >= 0 and temporary[1] >= 0 and temporary[2] >= 0 and z<z_buffer[y, x]:
                img_mat[y][x] = (Im * 0, Im * 255, Im * 0)

                WT = int(w_t * (temporary[0] * vt[ft[k][0] - 1][0] + temporary[1] * vt[ft[k][1] - 1][0] + temporary[2] * vt[ft[k][2] - 1][0]))
                HT = int(h_t * (temporary[0] * vt[ft[k][0] - 1][1] + temporary[1] * vt[ft[k][1] - 1][1] + temporary[2] * vt[ft[k][2] - 1][1]))
                #                       R                    G                      B
                img_mat[y, x] = [file_img[HT][WT][0] * Im, file_img[HT][WT][1] * Im, file_img[HT][WT][2] *Im]

                # img_mat[y][x] = color
                z_buffer[y, x] = z

# Вычисление векторного произведения (нормали к поверхности треугольника)
def n(x0, y0, x1, y1, x2, y2, z0, z1, z2):
    a = np.array([x1-x2, y1-y2, z1-z2])
    b = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(a, b)

# Вычисляем нормали ко всем вершинам модели
def n_top(v, f, n_arr):
    v_n=np.zeros((len(v), 3), dtype=np.float64) # Заполняем полигоны нулевыми векторами по количеству вершин (=3)
    for k in range(len(f)):
        # Вычисляем нормаль к полигону
        v_n[f[k][0]-1]+= n_arr[k]
        v_n[f[k][1]-1]+= n_arr[k]
        v_n[f[k][2]-1]+= n_arr[k]
    for i in range (len(v_n)):
        v_n[i]/=np.sqrt(np.dot(v_n[i], v_n[i])) # Нормируем полигон, разделяя на длину вектора
    return v_n

def cos_angle (x0, y0, x1, y1, x2, y2, z0, z1, z2, n_temp):
    l = [0,0,1]
    nFirst = np.dot(n_temp,l) # Скалярное произведение вектора
    nSecond = np.sqrt(np.dot(n_temp, n_temp)) # Вычисление длины вектора
    return nFirst / nSecond

# Наш z-буфер
z_buffer = np.zeros((height,width,1))
for i in range(height):
    for j in range(width):
        z_buffer[i][j] = sys.maxsize

for i in range(height):
    for j in range(width):
        img_mat[i, j]= 150

# Поворот по углам Эйлера
def euler_rotation(t_x, t_y, t_z):
    cos_alfa = math.cos(alfa)
    sin_alfa = math.sin(alfa)

    cos_beta = math.cos(beta)
    sin_beta = math.sin(beta)

    cos_gama = math.cos(gama)
    sin_gama = math.sin(gama)

    Rx = np.array([[1, 0, 0], [0, cos_alfa, sin_alfa], [0, -sin_alfa, cos_alfa]])
    Ry = np.array([[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]])
    Rz = np.array([[cos_gama, sin_gama, 0], [-sin_gama, cos_gama, 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz  # делаем тут умножение матриц

    for i in range(len(v)):
        # vr.append(np.dot(np.array(v[i]), R)+[t_x, t_y, t_z])
        vr.append(np.array(v[i]) @ R + [t_x, t_y, t_z])

# Поворот по кватернионам
def Quaternion_rotation(tetta, t_x, t_y, t_z):
    cos_tetta = math.cos(tetta/2)
    sin_tetta = math.sin(tetta / 2)
    q = Quaternion(cos_tetta, u[0]*sin_tetta,u[1]*sin_tetta,u[2]*sin_tetta)
    for i in range(len(v)):
        p = Quaternion(0, v[i][0], v[i][1], v[i][2])
        p_shtrih = q * p * q.sopr()
        vr.append(np.array([p_shtrih.i, p_shtrih.j, p_shtrih.k]) + [t_x, t_y, t_z]) # Аналог np.array
        # ВАЖНО: ВОЗМОЖНА ПРОБЛЕМА С ИНДЕКСОМ




# def parser(name):
#     file=open(name)
#     for s in file:
#         sp=s.split()
#         if sp[0]== 'v':
#             v.append([float(sp[1]), float(sp[2]),float(sp[3])])
#     # print(v)
#         if sp[0]== 'f':
#             f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])
#             ft.append([int(sp[1].split('/')[1]), int(sp[2].split('/')[1]), int(sp[3].split('/')[1])])
#         if sp[0] == 'vt':
#             vt.append([float(sp[1]), float(sp[2])])
#     # print(vt)


def parser1(name):
    file = open(name)
    for s in file:
        sp = s.split()
        # print(len(sp))
        if (len(sp) > 0):
            if (sp[0] == 'v'):
                v.append([float(sp[1]), float(sp[2]), float(sp[3])])
            if (sp[0] == 'f'):
                for itor in range(2, len(sp) - 1):
                    f.append([int(sp[1].split('/')[0]), int(sp[itor].split('/')[0]), int(sp[itor + 1].split('/')[0])])
                    ft.append([int(sp[1].split('/')[1]), int(sp[itor].split('/')[1]), int(sp[itor + 1].split('/')[1])])
            if (sp[0] == 'vt'):
                vt.append([float(sp[1]), float(sp[2])])
        # print(len(v))
        # print(len(f))
        # print(len(vt))

def patriotism_model(t_x, t_y, t_z, tetta, name, rot_type): # иде не нравилось otrisovat_model
    parser1(name)
    if (rot_type == 1):
        Quaternion_rotation(tetta, t_x, t_y, t_z)
    else:
        euler_rotation(t_x, t_y, t_z)

    for k in range (len(f)): #тут 3д
        x0=vr[f[k][0]-1][0]
        y0=vr[f[k][0]-1][1]
        x1=vr[f[k][1]-1][0]
        y1=vr[f[k][1]-1][1]
        x2=vr[f[k][2]-1][0]
        y2=vr[f[k][2]-1][1]

        z0=vr[f[k][0]-1][2]
        z1=vr[f[k][1]-1][2]
        z2=vr[f[k][2]-1][2]

        n_arr.append(n(x0, y0, x1, y1, x2, y2, z0, z1, z2)) # считаем нормаль к полигону

    global v_n
    v_n = n_top(v, f, n_arr)

    for k in range (len(f)):
        x0=vr[f[k][0]-1][0] #вместо v теперь vr тут массив
        y0=vr[f[k][0]-1][1]
        z0 = vr[f[k][0] - 1][2]

        x1=vr[f[k][1]-1][0]
        y1=vr[f[k][1]-1][1]
        z1 = vr[f[k][1] - 1][2]

        x2=vr[f[k][2]-1][0]
        y2=vr[f[k][2]-1][1]
        z2 = vr[f[k][2] - 1][2]

        drawing_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, n_arr[k], k)

def bunny(t_x, t_y, t_z, tetta, name_obj, name_img, rot_type):
    global v, f, ft, vt, vr, file_img
    v=[]
    f=[]
    ft=[]
    vt=[]
    vr=[]
    file_img = np.array(ImageOps.flip(Image.open(name_img)))
    patriotism_model(t_x, t_y, t_z, tetta, name_obj, rot_type) # Не удаляять!!!! Это важный вызов всего, чего мы добились!!


# t_x = -0.00
# t_y = -0.03
# t_z = 0.12

# bunny(0.05, -0.04, 0.22, 0, 'model_1.obj', 'bunny-atlas.jpg', 1)
# bunny(-0.04, -0.04, 0.12, 0, 'model_1.obj', 'img.png', 2)
bunny(-0.04, 0.04, 0.70, 0, '12221_Cat_v1_l3.obj', 'Cat_diffuse.jpg', 1)

img=Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
img.show('img.png')
