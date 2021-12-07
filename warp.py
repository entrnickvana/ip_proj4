import code
import os
import skimage
from skimage import io
import scipy
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
from poly_perim import *
import json
#from lin_trans import *

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

def conv_poly(poly):
    new_poly = [[],[],[],[]]
    new_poly[0] = [poly[0][1], poly[0][0]] 
    new_poly[1] = [poly[1][1], poly[1][0]] 
    new_poly[2] = [poly[2][1], poly[2][0]] 
    new_poly[3] = [poly[3][1], poly[3][0]]
    return new_poly


def my_svd(A,b, full_matrices=False):
    # compute svd of A
    U,s,Vh = svd(A)
    c = np.dot(U.T,b)
    w = np.dot(np.diag(1/s),c)
    x = np.dot(Vh.conj().T,w)
    return x

def get_P(A,b):
    u, s, vh = svd(A, full_matrices=False)
    w = np.diag(1/s)
    P = vh.T @ w @ u.T @ b
    return P

### get four corners
def get_poly_dim(to_warp, P):
    
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
    poly = [TL, TR, BL, BR]

    return poly, x_len, y_len

def build_canvas(to_warp, target, P, poly, dbg=0):

    TL = poly[0]
    TR = poly[1]
    BL = poly[2]
    BR = poly[3]
    
    poly, x_len, y_len = get_poly_dim(to_warp, P)

    tar_x_len = target.shape[1]
    tar_y_len = target.shape[0]
    #max_x_len = max([x_len, tar_x_len])
    #max_y_len = max([y_len, tar_y_len])    
    
    canvas = np.zeros((5*tar_y_len, 5*tar_x_len))
    #orig_x = x_len
    #orig_y = y_len
    orig_x = 2*tar_x_len
    orig_y = 2*tar_y_len
    

    if( dbg == 1):
        
        poly_canvas = np.array(canvas)
        df = 4

        #poly_canvas[orig_y: orig_y + target.shape[0], orig_x: orig_x + target.shape[1]] = target
        poly_canvas[orig_y + TL[1]-df:orig_y + TL[1]+df, orig_x + TL[0]-df: orig_x + TL[0]+df] = 255
        poly_canvas[orig_y + TR[1]-df:orig_y + TR[1]+df, orig_x + TR[0]-df:orig_x + TR[0]+df] = 255
        poly_canvas[orig_y + BL[1]-df:orig_y + BL[1]+df, orig_x + BL[0]-df: orig_x + BL[0]+df] = 255
        poly_canvas[orig_y + BR[1]-df:orig_y + BR[1]+df, orig_x + BR[0]-df:orig_x + BR[0]+df] = 255

        return poly_canvas
    return canvas

# get_forward_transform
def get_transform(points, points_p):
    x = []
    y = []
    xp = []
    yp = []
    
    for ii in range(len(points)):
        x.append(points[ii][0])
        y.append(points[ii][1])
        xp.append(points_p[ii][0])
        yp.append(points_p[ii][1])

    #  convert to numpy
    x = np.asarray(x)
    y = np.asarray(y)
    xp = np.asarray(xp)
    yp = np.asarray(yp)

    # create multiplied col vectors
    xxp = x*xp
    yxp = y*xp
    xyp = x*yp
    yyp = y*yp

    # Stack top half of A matrix
    q1 = np.c_[x, y, np.ones(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]
    q1 = -1*q1
    q1 = np.c_[q1, xxp, yxp]

    # Stack bottom half of A matrix    
    q2 = np.c_[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), x, y, np.ones(len(x))]
    q2 = -1*q2
    q2 = np.c_[q2, xyp, yyp]

    # Stack top and bottom half of A
    A = np.r_[q1, q2]

    # Stack to create byy
    b = np.r_[-1*xp, -1*yp]

    # call svd to get optimal solution P
    P = get_P(A, b)

    # Append a 1
    P = np.r_[P, 1]
    P = P.reshape((3,3))

    return P

def get_poly(target, to_warp, P, P_back,  canvas):

    orig_x = 2*target.shape[1]
    orig_y = 2*target.shape[0]

    tar_len_x = target.shape[1]
    tar_len_y = target.shape[0]
    
    #tst_canvas1 = np.array(canvas)
    #tst_canvas2 = np.array(canvas)

    canvas_poly = np.array(canvas)
    
    for ii in range(tar_len_x):
        for jj in range(tar_len_y):
    
            #get new transformed coordinate
            new_tmp_cord = map_xy(ii, jj, P)
            canvas_poly[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = 255        
            #tst_canvas1[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = to_warp[tar_len_y - jj -1, tar_len_x - ii -1]
            #tst_canvas2[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = to_warp[jj, ii]        
    
            
    mask = np.array(canvas_poly)
    mask[mask > 0] = 255
    k_sz = 11
    
    kern = np.full((k_sz, k_sz), 1/(k_sz*k_sz))
    polygon = scipy.ndimage.filters.convolve(mask, kern)
    final_mask = np.array(polygon)
    final_mask[final_mask > 0] = 255

    reverse_canvas = np.array(final_mask)

    for ii in range(canvas.shape[1]-1):
        for jj in range(canvas.shape[0]-1):
            if(final_mask[jj,ii] == 255):
                backward_cord = map_xy(ii-orig_x-1, jj-orig_y-1, P_back)
                if(backward_cord[0] >= 0 and backward_cord[1] >= 0):
                    if(backward_cord[0] < to_warp.shape[1] -1 and backward_cord[1] < to_warp.shape[0]-1):
                        #print(f"Poly idx x: {ii} idx y: {jj}")                    
                        #print(f"Grabbing index y: {backward_cord[1]} x: {backward_cord[0]}")
                        #print(f"\n\n")
                        reverse_canvas[jj, ii] = to_warp[backward_cord[1], backward_cord[0]]
                    else:
                        reverse_canvas[jj, ii] = 0
                else:
                    reverse_canvas[jj, ii] = 0
    
                #final_mask[jj,ii] = 64
    
    return reverse_canvas
    

    


# Reading all json files taken from:
# https://stackoverflow.com/questions/30539679/python-read-several-json-files-from-a-folder
file_names = []
path_to_json = './'

# Find all JSON files in current directory
for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
  file_names.append(file_name)
  #with open(path_to_json + file_name) as json_file:
  #  data = json.load(json_file)
  #  #print(data)
  
num_files = len(file_names)

print(f"Found {num_files} JSON files in current directory:")
for ii in range(len(file_names)):
    print(f"{ii}:\t{file_names[ii]}")

# Iterate over json files in current directory
for json_file in range(len(file_names)):
    
    # Read all correspondence sets for current JSON files
    print(f"\n\nReading json file _____x")
    corrs =  read_json_file(file_names[json_file])

    mosaic = []
    full = []
    
    # Iterate over each set of correspondences
    for ii in range(len(corrs)):

        
        # Parse file name in current directory
        imageNamea = corrs[ii][0][0]
        imageNameb = corrs[ii][1][0]
        imageNamea = imageNamea.split('.')
        imageNameb = imageNameb.split('.')
        
        # Read PNG image files for target image to warp to and image to be warped
        target = io.imread(imageNamea[0]+'.png')
        to_warp = io.imread(imageNameb[0]+'.png')

        if( target.ndim > 2):
            print("Converting target to greyscale...")
            target = color2grey(target)
        if( to_warp.ndim > 2):
            print("Converting to_warp to greyscale...")            
            to_warp = color2grey(to_warp)
        
        print(f"\n\nFound images:\n{imageNameb[0]}.png\n{imageNamea[0]}.png")
        print(f"Warping {imageNameb[0]}.png to fit {imageNamea[0]}.png...\n")
        
        # Assign correspondences to forward and backward sets, ie (x, x_prime)
        points = corrs[ii][1][1] # 
        points_p = corrs[ii][0][1] #

        # Get tranformation matrix P
        P = get_transform(points, points_p)
        
        # Get backward transform matrix
        P_back = get_transform(points_p, points)

        # Get a representation of the polygon of the to_warp image
        poly, x_len, y_len = get_poly_dim(to_warp, P)
        poly_yx = conv_poly(poly)

        
        canvas = build_canvas(to_warp, target, P, poly, dbg = 0)
        canvas_poly = np.array(canvas)

        orig_x = 2*target.shape[1]
        orig_y = 2*target.shape[0]        

        if(len(mosaic) == 0):
            mosaic = get_poly(target, to_warp, P, P_back, canvas)
        else:
            mosaic = get_poly(target, to_warp, P, P_back, canvas)

        
        if(len(full) == 0):
            full = np.array(mosaic)            
        else:
            full = mosaic + full            

        code.interact(local=locals())
        

        
    
code.interact(local=locals())


#tar_len_x = to_warp.shape[1]
#tar_len_y = to_warp.shape[0]

orig_x = x_len
orig_y = y_len

tst_canvas1 = np.array(canvas)
tst_canvas2 = np.array(canvas)


for ii in range(tar_len_x):
    for jj in range(tar_len_y):

        #get new transformed coordinate
        new_tmp_cord = map_xy(ii, jj, P)
        canvas_poly[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = 255        
        #tst_canvas1[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = to_warp[tar_len_y - jj -1, tar_len_x - ii -1]
        #tst_canvas2[orig_y + new_tmp_cord[1], orig_x + new_tmp_cord[0]] = to_warp[jj, ii]        

        
mask = np.array(canvas_poly)
mask[mask > 0] = 255
k_sz = 11

kern = np.full((k_sz, k_sz), 1/(k_sz*k_sz))
polygon = scipy.ndimage.filters.convolve(mask, kern)
final_mask = np.array(polygon)
final_mask[final_mask > 0] = 255

H, edges = np.histogram(polygon, bins = 256)

reverse_canvas = np.array(final_mask)

for ii in range(canvas.shape[1]-1):
    for jj in range(canvas.shape[0]-1):
        if(final_mask[jj,ii] == 255):
            backward_cord = map_xy(ii-orig_x-1, jj-orig_y-1, P_back)
            if(backward_cord[0] >= 0 and backward_cord[1] >= 0):
                if(backward_cord[0] < to_warp.shape[1] -1 and backward_cord[1] < to_warp.shape[0]-1):
                    #print(f"Poly idx x: {ii} idx y: {jj}")                    
                    #print(f"Grabbing index y: {backward_cord[1]} x: {backward_cord[0]}")
                    #print(f"\n\n")
                    reverse_canvas[jj, ii] = to_warp[backward_cord[1], backward_cord[0]]
                else:
                    reverse_canvas[jj, ii] = 0
            else:
                reverse_canvas[jj, ii] = 0

            final_mask[jj,ii] = 64 
            #if(jj % 20 == 0):
            #    code.interact(local=locals())            

code.interact(local=locals())

exit()




            




