#!/usr/bin/env python3
-- coding: utf-8 --

"""
Created on Mon Aug  1 10:13:03 2022

@author: venkat
"""
import cv2
import numpy as np
#from collections import deque
#import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
##from skimage import exposure
#import time
#from skimage.util import random_noise
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
read the image

astro = cv2.imread('/home/venkat/Data Set/1_1.jpg')

astro=cv2.cvtColor(astro, cv2.COLOR_BGR2GRAY)
#Genarate Gaussian Noise

gauss=np.random.normal(0,1, astro.shape)

#Add noise to the image

noisy_img=astro+2*gauss

filtered_image = wiener(noisy_img, (5, 5))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(noisy_img)
ax[1].set_title('Noisy Image')

ax[2].imshow(filtered_image, vmin=noisy_img.min(), vmax=noisy_img.max())

ax[2].set_title('Restoration using Wiener Filter')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
top=0.9, bottom=0.05, left=0, right=1)
plt.imshow(astro)
plt.show()
Calculation of performance metrics

MSE = mean_squared_error(astro, filtered_image)
PSNR = peak_signal_noise_ratio(astro, filtered_image, data_range=astro.max() - astro.min())
SSIM=ssim(astro, filtered_image)
print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM:', SSIM )

print(astro.shape)
