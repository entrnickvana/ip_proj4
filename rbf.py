import code
import os
import skimage
from skimage import io
from scipy import fftpack

# Here is the link to the article I read, also see pwr_exp.py for what is done in the article
# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/



import matplotlib.pyplot as plt
import numpy as np
#from ip_functions import *
from skimage import data, filters, color, morphology, exposure, img_as_float, img_as_ubyte, util
from skimage.util import img_as_ubyte
from skimage.segmentation import flood, flood_fill
from skimage.morphology import extrema
from skimage.exposure import histogram

#from numpy import unravel_index

def rbf(x, y, xi, yi):
    dx = x - xi
    dy = y - yi

    nrm  = np.sqrt(dx*dx + dy*dy)
    nrm_squared = nrm*nrm
    nrm_log = np.log(nrm)
    phi = nrm_squared*nrm_log
    return phi

g = np.zeros((256, 256))
g[0:256:64, ::] = 1
g[::, 0:256:64] = 1
f = np.cos(g)
#plt.subplot(1,2,1)
#plt.imshow(g, cmap='gray')
#plt.subplot(1,2,2)
#plt.imshow(f, cmap='gray')
#plt.show()

M, N = 3, 4

matrix = np.arange(M*N).reshape((M, N))
points = np.mgrid[0:N, 0:M].reshape((2, M*N))
x, y = np.mgrid[0:N, 0:M]
points = np.vstack([x.ravel(), y.ravel()])
A = np.array([[2, 0], [0, 1]])
new_points = np.linalg.inv(A).dot(points).astype(int)
new_points


code.interact(local=locals())
              
    
