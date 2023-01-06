#!/usr/bin/env python3
-- coding: utf-8 --

"""
Created on Wed Sep  7 16:24:00 2022

@author: venkat
"""

#!/usr/bin/env python3
-- coding: utf-8 --

"""
Created on Thu Sep  1 17:51:45 2022

@author: venkat
"""
import numpy as np
import cv2
from skimage import color, data, restoration
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
from skimage.util import img_as_ubyte
from mpl_toolkits import mplot3d
sigma(standard deviation) and muu(mean) are the parameters of gaussian

def g_filter(kernel_size, sigma_x, mx, sigma_y, my):
# Initializing value of x,y as grid of kernel size
# in the range of kernel size




dst= (((x-mx) // sigma_x) ** 2)+ (((y-my) / sigma_y)** 2)

# lower normal part of gaussian
normal = 1/(2*np.pi*sigma_x*sigma_y)

#normal=np.sqrt(normal)

# Calculating Gaussian filter
gauss = normal*np.exp(-dst) 

return gauss/gauss.sum()

kernel_size=3

#x, y = np.meshgrid(np.linspace(0,18, kernel_size),
#    np.linspace(0.1988, 0.0502, kernel_size))

x=np.linspace(0,18, kernel_size)

y=[0.198899999999995,0.2256,0.499499999999998,-0.557999999999993,-1.2495,-3.289,-5.9893,-9.0428,-11.0551,-11.4142,-9.3563,-7.1286,-4.6992,-3.7905,
-2.6393,-2.0398,-1.0990,-0.7714,0.0502]

y=[0.1988, -0.5579, -5.9893, -11.4142, -4.6992, -2.0398, 0.0502]

y=np.array(y)

x,y=np.meshgrid(x,y)

gaussian = g_filter(kernel_size, sigma_x=5.4772, mx=9, sigma_y=3.9021, my=-3.8499)

#gaussian=np.array(gaussian, dtype='float64')

print("gaussian filter of{} X {} :".format(kernel_size,kernel_size))

print(gaussian)
