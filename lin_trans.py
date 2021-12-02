import code
import os
import skimage
from skimage import io
from scipy import fftpack

# Here is the link to the article I read, also see pwr_exp.py for what is done in the article
# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte, util
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram

def linear_transformation(src, A):
    M, N = src.shape
    points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    new_points = np.linalg.inv(A).dot(points).round().astype(int)
    x, y = new_points.reshape((2, M, N), order='F')
    indices = x + N*y
    return np.take(src, indices, mode='wrap')

def lt_mod(src, A):
    M, N = src.shape
    points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    points = np.vstack([points, np.ones(points.shape[1])])
    new_points = np.linalg.inv(A).dot(points).round().astype(int)
    x, y, z = new_points.reshape((3, M, N), order='F')
    indices = x + N*y
    #return np.take(src, indices, mode='wrap')
    return np.take(src, indices)




mpl.rcParams.update({'image.cmap': 'Accent',
                     'image.interpolation': 'none',
                     'xtick.major.width': 0,
                     'xtick.labelsize': 0,
                     'ytick.major.width': 0,
                     'ytick.labelsize': 0,
                     'axes.linewidth': 0})

aux = np.ones((100, 100), dtype=int)
src = np.vstack([np.c_[aux, 2*aux], np.c_[3*aux, 4*aux]])

A = np.array([[1.5, 0], [0, 1]])
dst = linear_transformation(src, A)


plt.subplot(1,2, 1)
plt.imshow(src)
plt.subplot(1,2, 2)
plt.imshow(dst)
plt.show()

code.interact(local=locals())





