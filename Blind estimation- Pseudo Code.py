import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from Affiliated import denoise, deconv, motion_blur, addNoise
from Affiliated import utils, auxiliary
from skimage.metrics import mean_squared_error
from Kernel_Estimation import kernel_estimation
#from medpy.filter.smoothing import anisotropic_diffusion
import cv2

#import pandas as pd

#file_path = glob.glob('/home/venkat/Downloads/Image-deblur-using-image-pairs-master/Data Set') # Read out all the test image in the /image/ file directory
img_dir = "/home/venkat/Data_Set/"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
num_to_cal = 1 # This parameter is used to control the num of the image you want to read out from the ./images datafile
files = files[1:num_to_cal]
num = 0
is_random_kernel = True    # Decide wheather to generate the random kernel
#size_of_kernel = 3
#niter=20
#kappa=1
#gamma=0.25
#voxelspacing=None
#option=1
Decide the size of the kernel generated and estimate

for i,fname in enumerate(files):
savedir = '/home/venkat/Downloads/Image-deblur-using-image-pairs-master' + str(1000+num)+'/'
csvname = savedir + "IQE_%s.csv" % (os.path.basename(fname))   # Used to estimate the quality of the image, Image quality estimation
 
if not os.path.exists(savedir):
    os.mkdir(savedir)    
    
f= open(csvname,mode = 'w') 
    
f_csv = csv.writer(f, delimiter = ',')
    
f_csv.writerow(['i','Method','size_of_kernel','lambda','SSIM','PSNR', 'MSE'])
    
for size_of_kernel in np.arange(3, 5, 2):   
    
    f_csv.writerow(['='])
    
           
   
    for la in np.linspace(1,2,1):    
        
        f_csv.writerow(['='])


        print("--"*10)
        print("Starting the {} round".format(num+1))
    #    savename = 'eval_' + str(2000+num)
        
        
        I = io.imread(fname,as_gray=True)
        I = img_as_float(I)
        
        if is_random_kernel:
            #blur_kernel = utils.kernel_generator(size_of_kernel)
            #B = utils.blur(I,blur_kernel)
            #B=cv2.GaussianBlur(I, (size_of_kernel,size_of_kernel),0)
            B=cv2.blur(I, (size_of_kernel,size_of_kernel))
        else:
            motion_degree = np.random.randint(0,360)    # Generate one specific motion deblur
            B , blur_kernel = motion_blur(I,size_of_kernel,motion_degree)
            
        N = addNoise(I,0,0.01)
        Nd = denoise(N)
        
       #Nd=anisotropic_diffusion(N, niter, kappa, gamma, voxelspacing, option)
        
        K_estimated = kernel_estimation(Nd,B,lens=size_of_kernel,lam=la,method='l1ls')
    #    K_estimated = blur_kernel
        
        auxiliary.kernel_write(K_estimated,"estimated (img=%d,size_of_kernel=%d, lambda=%d)" % (i,size_of_kernel,la),savedir)
        #auxiliary.kernel_write(blur_kernel,"true (img=%d, size_of_kernel=%d,lambda=%d)" % (i,size_of_kernel,la),savedir)
    
        plt.imsave(savedir+"original(img=%d,size_of_kernel=%d, lambda=%d).png" % (i,size_of_kernel,la),I,cmap = 'gray')
        plt.imsave(savedir+"blurred(img=%d,size_of_kernel=%d, lambda=%d).png" % (i,size_of_kernel,la),B,cmap = 'gray')
        plt.imsave(savedir+"denoised(img=%d, size_of_kernel=%d,lambda=%d).png" % (i,size_of_kernel,la),Nd,cmap = 'gray')
    
    
        

        
        deconvmode = ['detailedRL','lucy','resRL','gcRL']
    ## deconvolution, can be 'lucy', 'resRL', 'gcRL' and 'detailedRL'.
        
        for demode in deconvmode:
            # deBlur = deblur(Nd,B,unikernel = True,deconvmode=demode)
            deBlur = deconv(Nd,B,K_estimated,mode=demode)
            
            plt.imsave(savedir+"deblurred_" +demode+'(img=%d, size_of_kernel=%d, lambda=%d)' %(i,size_of_kernel,la) +".png",deBlur,cmap = 'gray')
            
            ssim1 = ssim(I,deBlur)
            psnr1 = peak_signal_noise_ratio(I,deBlur)
            mse1=mean_squared_error(I, deBlur)
           
            result = [i,demode,size_of_kernel,la,ssim1,psnr1, mse1]  
           #result=result.insert(1, demode)
            f= open(csvname,mode = 'a+')
            f_csv = csv.writer(f,delimiter = ',')
            f_csv.writerow(result)
    

            
            
num += 1        
print("Complete the {} round".format(num))
print("__"*10)
print("Complete all cycles")
print("__"*10)

#all_results = np.float32(all_results)
#np.savetxt(savedir + "/IQE(img=%d).csv" % (i), all_results,'%10.5f', header='i,lambda,SSIM,PSNR,MSE', delimiter=',')
