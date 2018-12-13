#-*-coding:utf-8-*-
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p
import cv2
np.set_printoptions(threshold=np.inf)
dist ='/home/victor/software/OpenNI-Linux-x64-2.3/Redist'
mod = SourceModule \
    (
        """
#include<stdio.h>
#define INDEX(a, b) a*320+b
__global__ void rgb2gray(float *dest,float *r_img, float *g_img, float *b_img)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a = idx/320;
  a<240;
  unsigned int b = idx%320;
dest[INDEX(a, b)] = (288.126*r_img[INDEX(a, b)]+0*g_img[INDEX(a, b)]+156.578*b_img[INDEX(a, b)]);
}

__global__ void rgb2gray2(float *dest2,float *r_img, float *g_img, float *b_img)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a = idx/320;
  a<240;
  unsigned int b = idx%320;
dest2[INDEX(a, b)] = (0*r_img[INDEX(a, b)]+288.780*g_img[INDEX(a, b)]+124.968*b_img[INDEX(a, b)]);
}
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  dest[i] = a[i]/b[i];
}
"""
    )
openni2.initialize(dist)
if (openni2.is_initialized()):
    print("openNI2 initialized")
else:
    print("openNI2 not initialized")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
depth_stream.set_mirroring_enabled(False)
depth_stream.start()
def get_depth():
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(240,320)
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1)
    d4d = cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    d4d = 255 - d4d    
    return dmap, d4d
s=0
done = False
while not done:
    key = cv2.waitKey(1)
    key = cv2.waitKey(1) & 255
    if key == 27:
        print("\tESC key detected!")
        done = True
    elif chr(key) =='s':
        print("\ts key detected. Saving image and distance map {}".format(s))
        cv2.imwrite("ex1_"+str(s)+'.png', d4d)
        np.savetxt("ex1dmap_"+str(s)+'.out',dmap)
    dmap,d4d = get_depth()
    #print (dmap)
    zuobiao=[]
    for d in range(dmap.shape[0]):
	for e in range(dmap.shape[1]):
		x=np.array([d,e,1])
		if dmap[d,e]<20:
			pass
		else:
			addfdsfds=[]
			#print ("此处要显示x格式")
			#print(x.dtype)
	        #print ("此处要显示k格式")
		#print(k1.dtype)
		zuobiao.append(x)
    #这个坐标就是cuda里面的放入内核函数里面的坐标。
    zuobiao=np.array(zuobiao).astype(np.float32)
    zuobiao=zuobiao.reshape(25600,3,3)
    r_img =zuobiao[:, :, 0].reshape(76800, order='F')
    g_img =zuobiao[:, :, 1].reshape(76800, order='F')
    b_img =zuobiao[:, :, 2].reshape(76800, order='F')
    dmap_img=dmap.reshape(76800,order='F')
    dest=r_img
    dest2=r_img

    rgb2gray = mod.get_function("rgb2gray")
    rgb2gray(drv.Out(dest), drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(1024, 1, 1), grid=(75, 1, 1))
#dest=np.reshape(dest,(3,3), order='F')
#print dest
    abc=np.array(dest)
    rgb2gray2 = mod.get_function("rgb2gray2")
    rgb2gray2(drv.Out(dest2), drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(1024, 1, 1), grid=(75, 1, 1))
#dest2=np.reshape(dest2,(3,3), order='F')
    #print dest2
    destzuizhong=[]
#print abc
    #####!!!!!!!!////destzuizhong=np.dstack((abc,dest2))
    #print destzuizhong
#print destzuizhong.shape

    multiply_them = mod.get_function("multiply_them")
    destlast_abc = np.zeros_like(abc)
    destlast_dest2=np.zeros_like(dest2)
    multiply_them(
        drv.Out(destlast_abc), drv.In(abc), drv.In(dmap_img),block=(1024, 1, 1), grid=(75, 1, 1))
    multiply_them(drv.Out(destlast_dest2),drv.In(dest2),drv.In(dmap_img),block=(1024,1,1),grid=(75,1,1))
    destzuizhong=np.dstack((destlast_abc,destlast_dest2))
    destzuizhong=destzuizhong.reshape(76800,2)
    ones=np.ones(76800)
    destzuizhong=np.c_[destzuizhong,ones]
    #print ("dest")
    print(destzuizhong.shape)
    print ("以上代码解决了在cuda里面进行深度信息坐标转化问题success!")

    cv2.imshow('depth', d4d)
cv2.destroyAllWindows()
depth_stream.stop()
openni2.unload()
print ("Terminated")

#        R_tuple=([0.999983,0.00264383,-0.0052673],
#            [-0.00264698,0.999996,-0.000589696],
#            [0.00526572,0.000603628,0.999986])

