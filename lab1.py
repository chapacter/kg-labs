import math
import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def draw_line1(image, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0-x1)**2 + (y0 - y1)**2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line3(image, x0, y0, x1, y1, color):
    for x in range(int(x0), int(x1)):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line3f1(image, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(int(x0), int(x1)):
        t = (x-x0)/(x1-x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line3f2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line3f2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    y0, y1 = y1, y0
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line4(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
        y += y_update

def draw_line5(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 2.0 * (x1 - x0) * 0.5):
            derror -= 2.0 * (x1 - x0) * 1.0
        y += y_update

def draw_line6(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
        y += y_update

def triangle:
    pass



# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_file(filename, v, f):
    file = open(filename)
    for s in file:
        Sp = s.split()
        if (Sp[0] == 'v'):
            v.append([float(Sp[1]), float(Sp[2]), float(Sp[3])])
        if (Sp[0] == 'f'):
            f.append([int(Sp[1].split('/')[0]), int(Sp[2].split('/')[0]), int(Sp[3].split('/')[0])])


def draw_obj(filename, image):
    v = []
    f = []
    read_file(filename, v, f)
    # print(v)

    for k in range (len(f)):
        x0 = int(9000*v[f[k][0]-1][2]+1000)
        y0 = int(9000*v[f[k][0]-1][1]+1000)
        x1 = int(9000*v[f[k][1]-1][2]+1000)
        y1 = int(9000*v[f[k][1]-1][1]+1000)
        x2 = int(9000*v[f[k][2]-1][2]+1000)
        y2 = int(9000*v[f[k][2]-1][1]+1000)
        draw_line6(image,x0,y0,x1,y1, 255)
        draw_line6(image,x1,y1,x2,y2, (180, 255, 10))
        draw_line6(image,x0,y0,x2,y2, (150, 180, 255))

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

# for i in range(200):
#     for j in range(200):
#         if i > j: img_mat[i,j] = (60, 255, 255)
#         else: img_mat[i,j] = (255, 200, 20)

# for k in range(13):
#     x0, y0 = 100, 100
#     x1 = int(100 + 95 * cos(2*pi/13*k))
#     y1 = int(100 + 95 * sin(2*pi/13*k))
#     draw_line1(img_mat, 100, 50, x1, y1, 100,(0, 200, 20))
#     draw_line2(img_mat, 200, 100, x1, y1,(255, 200, 20))
#     draw_line3f1(img_mat, 100, 150, x1, y1, (255, 200, 20))
#     draw_line6(img_mat, x0, y0, x1, y1, (255, 200, 20))

draw_obj('model_1.obj', img_mat)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img.png')
img.show('img.png')
