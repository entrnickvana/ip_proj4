import code
import os
import skimage
from skimage import io
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
#from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte, util
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram
from read_json_mod import *
#from scipy.linalg import svd
from numpy.linalg import svd
from lin_trans import *

#TODO
#
#1) Ask about square matrices issue
#2) Ask about Z dimension
#3) I'm I on the right path in general?
#4) Why is this wrapping?
#5) Dividing by w, where does w come from?


#def solve_svd(A,b):
#    # compute svd of A
#    U,s,Vh = svd(A)
#
#    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
#    c = np.dot(U.T,b)
#    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
#    w = np.dot(np.diag(1/s),c)
#    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
#    x = np.dot(Vh.conj().T,w) to_warp[ii][jj]
#    return x
#

def color2grey(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

def map_xy(x, y, P):
    p11 = P[0, 0]
    p12 = P[0, 1]
    p13 = P[0, 2]
    p21 = P[1, 0]
    p22 = P[1, 1]
    p23 = P[1, 2]
    p31 = P[2, 0]
    p32 = P[2, 1]
    p33 = 1

    xp_float = (p11*x + p12*y + p13)/(p31*x + p32*y + 1)
    yp_float = (p21*x + p22*y + p23)/(p31*x + p32*y + 1)
    #print(f"xp float: {xp_float}")
    #print(f"yp float: {yp_float}")    
    xp = int(xp_float)
    yp = int(yp_float)    
    #print(f"xp int: {xp}")
    #print(f"yp int: {yp}")
    return xp, yp
    


def my_svd(A,b, full_matrices=False):
    # compute svd of A
    U,s,Vh = svd(A)
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    x = np.dot(Vh.conj().T,w)
    return x, s


#The images are ['w0.ppm', 'w1.ppm', 'w2c.ppm', 'w3c.ppm']
#There are 3 sets of correspondences
#x3There are 6 correspdondences between image w3c.ppm and image w0c.ppm
#[[264 369 257 367 152 146]
# [105  84 252 239 141 242]]
#[[134 231 132 231  12  11]
# [ 10   8 128 127  20 110]]
#(6, 2)
#(6, 2)
#There are 13 correspdondences between image w0c.ppm and image w1c.ppm
#[[434 451 540 550 542 563 454 355 402 535 563 565 566]
# [  6 136 125  21 222 270 216 244 295 216 267 282 306]]
#[[ 81 106 210 218 212 243 110   9  63 203 243 246 248]
# [ 72 227 203  69 328 383 325 360 418 322 383 399 429]]
#(13, 2)
#(13, 2)
#There are 12 correspdondences between image w0c.ppm and image w2c.ppm
#[[256 313 256 313 145 218 145 218  30  16   3   2]
# [164 165 196 192 246 246 342 342 221 362 249 340]]
#[[428 489 425 488 302 382 296 376 174 155 140 139]
# [107 109 142 141 195 197 299 300 163 317 192 293]]
#(12, 2)
#(12, 2)
#The output file is mosaic_out.tif

# Start with a warping from images with least correspndences between w3c.ppm and w0c.ppm

#corra, corrb = read_json()

# Read in images
to_warp = color2grey(io.imread('w3c.png'))
target = color2grey(io.imread('w0c.png'))
#to_warp = io.imread('w3c.png')
#target = io.imread('w0c.png')

#plt.subplot(1,2,1)
#plt.imshow(to_warp)
#plt.subplot(1,2,2)
#plt.imshow(target)
#plt.show()

corrs = read_json()

print("Corrs:")
print(corrs)

#for ii in range(len(corra)):
#    for jj in range(len(corra)):
#        print(f"IDX: {ii}  {corra[ii]}  ")

# manually compare w3c.png and w0c.png
points_anchor = corrs[0][1][1] # target of warp
points = corrs[0][0][1] # to warp
x = []
y = []
xp = []
yp = []
for ii in range(len(points)):
    x.append(points[ii][0])
    y.append(points[ii][1])
    xp.append(points_anchor[ii][0])
    yp.append(points_anchor[ii][1])

print("x, y:")
print(x)
print(y)

print("xp, yp:")
print(xp)
print(yp)

#x = np.asarray(x)
#py = np.asarray(y)
#xp = np.asarray(xp)
#yp = np.asarray(yp)

x = np.asarray(x[0:4])
y = np.asarray(y[0:4])
xp = np.asarray(xp[0:4])
yp = np.asarray(yp[0:4])

xxp = x*xp
yxp = y*xp
xyp = x*yp
yyp = y*yp

q1 = np.c_[x, y, np.ones(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]
q1 = -1*q1
q1 = np.c_[q1, xxp, yxp]

q2 = np.c_[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), x, y, np.ones(len(x))]
q2 = -1*q2
q2 = np.c_[q2, xyp, yyp]
A = np.r_[q1, q2]

b = np.r_[xp, yp]
P, w = my_svd(A, b)
P = np.r_[P, 1]
P = P.reshape((3,3))
print('P:')
print(P)

x_new = np.zeros(len(x))
y_new = np.zeros(len(y))

### get four corners

# top left
TL = map_xy(0, 0, P)
# top right
TR = map_xy(to_warp.shape[1], 0, P)
# bottom left
BL = map_xy(0, to_warp.shape[0], P)
# bottom right
BR = map_xy(to_warp.shape[1], to_warp.shape[0], P)

# Find out how big polygon is to make a canvas
x_len = abs(max([TL[0], TR[0], BL[0], BR[0]]) - min([TL[0], TR[0], BL[0], BR[0]]))
y_len = abs(max([TL[1], TR[1], BL[1], BR[1]]) - min([TL[1], TR[1], BL[1], BR[1]]))



canvas = np.zeros((3*y_len, 3*x_len))
poly_canvas = np.array(canvas)
orig_x = x_len
orig_y = y_len

print(f"x_len: {x_len}, canvas x_len: {canvas.shape[1]}, poly orig: orig: {orig_x}")
print(f"y_len: {y_len}, canvas y_len: {canvas.shape[0]}, poly orig: orig: {orig_y}")

#canvas[orig_y + TL[1], orig_x + TL[0]] = to_warp[0,0]
#canvas[orig_y + TR[1], orig_x + TR[0]] = to_warp[0,to_warp.shape[1]-1]
#canvas[orig_y + BL[1], orig_x + BL[0]] = to_warp[to_warp.shape[0]-1, 0]
#canvas[orig_y + BR[1], orig_x + BR[0]] = to_warp[to_warp.shape[0]-1, to_warp.shape[1]-1]

df = 4

poly_canvas[orig_y + TL[1]-df:orig_y + TL[1]+df, orig_x + TL[0]-df: orig_x + TL[0]+df] = 255
poly_canvas[orig_y + TR[1]-df:orig_y + TR[1]+df, orig_x + TR[0]-df:orig_x + TR[0]+df] = 255
poly_canvas[orig_y + BL[1]-df:orig_y + BL[1]+df, orig_x + BL[0]-df: orig_x + BL[0]+df] = 255
poly_canvas[orig_y + BR[1]-df:orig_y + BR[1]+df, orig_x + BR[0]-df:orig_x + BR[0]+df] = 255


tar_len_x = to_warp.shape[1]
tar_len_y = to_warp.shape[0]

for ii in range(tar_len_x):
    for jj in range(tar_len_y):

        #get new transformed coordinate
        new_tmp_cord = map_xy(ii, jj, P)
        canvas[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = to_warp[tar_len_y - jj -1, tar_len_x - ii -1]
        #canvas[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = 255
        if(ii % 100 == 0 and jj % 400 == 0):
            code.interact(local=locals())    


## get whole new mapping
#for ii in range(len(x)):
#     = map_xy(x, y, P)

code.interact(local=locals())    


#warped = linear_transformation(to_warp, P)
#warped = lt_mod(to_warp, P)
#
#warped_norm = warped/w







## example linear system to test with
#
# #x + z = 6
# #z − 3y = 7
# #2x + y + 3z = 15
# 
# #=>
# 
# # 1x + 0y + 1z = 6
# # 0x - 3y + 1z = 7
# # 2x + 1y + 3z = 15
# 
# #  |1   0  1 |   | x1 |  =   6
# #  |0  -3  1 | * | x2 |  =   7
# #  |2   1  3 |   | x3 |  =  15
# 
# A = np.array([1, 0 , 1, 0, -3, 1, 2, 1, 3]).reshape((3,3))
# b = np.array([6, 7, 15]).reshape((3, 1))
# x = solve_svd(A, b)
# print(x)



            




