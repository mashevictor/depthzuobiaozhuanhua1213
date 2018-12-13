#-*-coding:utf-8-*-
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p
import cv2
mod = SourceModule \
    (
        """
#include<stdio.h>
#define INDEX(a, b) a*3+b
__global__ void rgb2gray(float *dest,float *r_img, float *g_img, float *b_img)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a = idx/3;
  unsigned int b = idx%3;
dest[INDEX(a, b)] = (288.126*r_img[INDEX(a, b)]+0*g_img[INDEX(a, b)]+156.578*b_img[INDEX(a, b)]);
}

__global__ void rgb2gray2(float *dest2,float *r_img, float *g_img, float *b_img)
{

unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

  unsigned int a = idx/3;
  unsigned int b = idx%3;
dest2[INDEX(a, b)] = (0*r_img[INDEX(a, b)]+288.780*g_img[INDEX(a, b)]+124.968*b_img[INDEX(a, b)]);
}
__global__ void multiply_them(float *dest, float *a, float *b,float *c)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  dest[i] = a[i] * b[i]/c[i];
}

"""
    )


#a = scm.imread('Lenna.png').astype(np.float32)
#a=np.array([[[ 3,  2,  1],[ 6,  5,  4],[ 9,  8,  7]],[[12, 11, 10],[15, 14, 13],[18, 17, 16]],[[21, 20, 19],[24, 23, 22],[27, 26, 25]]])
#k1=([288.126,0,156.578],[0,288.780,124.968],[0,0,1])

a=cv2.imread('Lenna.png').astype(np.float32)
print a
r_img = a[:, :, 0].reshape(9, order='F')
g_img = a[:, :, 1].reshape(9, order='F')
b_img = a[:, :, 2].reshape(9, order='F')
dest=r_img
dest2=r_img

#print(r_img)
#print(g_img)
#print(b_img)
rgb2gray = mod.get_function("rgb2gray")
rgb2gray(drv.Out(dest), drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(1024, 1, 1), grid=(75, 1, 1))
#dest=np.reshape(dest,(3,3), order='F')
#print dest
abc=np.array(dest)
rgb2gray2 = mod.get_function("rgb2gray2")
rgb2gray2(drv.Out(dest2), drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(1024, 1, 1), grid=(75, 1, 1))
#dest2=np.reshape(dest2,(3,3), order='F')
print dest2
destzuizhong=[]
print abc
destzuizhong=np.dstack((abc,dest2))
print destzuizhong
print destzuizhong.shape
print ("以上代码解决了在cuda里面进行深度信息坐标转化问题success!")
a = np.array([[1,2],[3,4]]).astype(np.float32)
b = np.array([[10,20],[30,40]]).astype(np.float32)
c = np.array([[1,2],[3,4]]).astype(np.float32)
multiply_them = mod.get_function("multiply_them")
destlast = np.zeros_like(a)
multiply_them(
        drv.Out(destlast), drv.In(a), drv.In(b),drv.In(c),
        block=(1024, 1, 1), grid=(75, 1, 1))
print ("dest")
print(destlast)
