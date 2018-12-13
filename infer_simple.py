#!/usr/bin/env python
#-*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from numpy import *
import cv2
from primesense import openni2
from primesense import _openni2 as c_api
from collections import defaultdict
import argparse
import cv2
import glob
import logging
import os
import sys
import time
from caffe2.python import workspace
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
from itertools import chain
c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)
np.set_printoptions(threshold=np.inf)
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
from timeit import default_timer as timer
import pycuda.autoinit

kernel_code_template = """
__global__ void MatrixMulKernel(float *A, float *B, float *C)
{

  const uint wA = %(MATRIX_SIZE)s;
  const uint wB = %(MATRIX_SIZE)s;  
  
  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  const uint aBegin = wA * %(BLOCK_SIZE)s * by;
  // Index of the last sub-matrix of A processed by the block
  const uint aEnd = aBegin + wA - 1;
  // Step size used to iterate through the sub-matrices of A
  const uint aStep = %(BLOCK_SIZE)s;

  // Index of the first sub-matrix of B processed by the block
  const uint bBegin = %(BLOCK_SIZE)s * bx;
  // Step size used to iterate through the sub-matrices of B
  const uint bStep = %(BLOCK_SIZE)s * wB;

  // The element of the block sub-matrix that is computed
  // by the thread
  float Csub = 0;
  // Loop over all the sub-matrices of A and B required to
  // compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) 
    {
      // Shared memory for the sub-matrix of A
      __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
      // Shared memory for the sub-matrix of B
      __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

      // Load the matrices from global memory to shared memory
      // each thread loads one element of each matrix
      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];
      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
      for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
        Csub += As[ty][k] * Bs[k][tx];

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

  // Write the block sub-matrix to global memory;
  // each thread writes one element
  const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
  C[c + wB * ty + tx] = Csub;
}
"""
dist ='/home/victor/software/OpenNI-Linux-x64-2.3/Redist'
openni2.initialize(dist)
if (openni2.is_initialized()):
    print ("openNI2 initialized")
else:
    print ("openNI2 not initialized")
dev = openni2.Device.open_any()
rgb_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
rgb_stream.start()
depth_stream.start()
depth_stream.set_mirroring_enabled(False)
rgb_stream.set_mirroring_enabled(False)

def shandiao(total_data):
	sb=[]
	median = np.median(total_data)
	b = 1.4826
	mad = b * np.median(np.abs(total_data-median))
	lower_limit = median - (3*mad)
	upper_limit = median + (3*mad)
	qudiao=[]
	for i in total_data:
		if lower_limit<i<upper_limit:
			pass
		else:
			qudiao.append(i)
	std = np.std(total_data)
	mean = np.mean(total_data)
	b = 3
	lower_limit = mean-b*std
	upper_limit = mean+b*std
	for i in total_data:
		if lower_limit<i<upper_limit:
			pass
		else:
			qudiao.append(i)
	zuizhong=[]
	zuizhong=set(total_data).difference(set(qudiao))
	list2=list(zuizhong)
	sb=[]	
	sb=np.array(list2)
	return sb

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /home/victor/facebook/infer_simple)',
        default='/home/victor/detectron/detectron-visualizations',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
utils.logging.setup_logging(__name__)
args = parse_args()
def get_rgb():
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb
def get_depth():
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(240,320)
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1)
    d4d = cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    d4d = 255 - d4d
    return dmap, d4d
s=0
done = False
logger = logging.getLogger(__name__)
merge_cfg_from_file(args.cfg)
cfg.NUM_GPUS = 1
args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg()
model = infer_engine.initialize_model_from_cfg(args.weights)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()
while not done:
    key = cv2.waitKey(1) & 255
    if key == 27: 
        print ("\tESC key detected!")
        done = True
    elif chr(key) =='s': #screen capture
        print ("\ts key detected. Saving image {}".format(s))
        cv2.imwrite("ex2_"+str(s)+'.png', rgb)
    im = get_rgb()
    dmap,d4d = get_depth()
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )

    vis_utils.vis_one_image(
        im[:, :, ::-1],  # BGR -> RGB for visualization
        args.output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_coco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
        kp_thresh=2
) 
    im=vis_utils.vis_one_image_opencv(im,cls_boxes, cls_segms,cls_keyps,dataset=dummy_coco_dataset,thresh=0.7,show_class=True)
    classxy=vis_utils.vis_one_image_opencv2(im,cls_boxes, cls_segms,cls_keyps,dataset=dummy_coco_dataset,thresh=0.7,show_class=True)
    
    if type(classxy[0][0])==np.ndarray:
	pass
    else:
    	print(classxy)
    yinying=[]
    yinying=    vis_utils.vis_one_image(
        im,
        args.output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_coco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
)
    if yinying is None:
	pass
    else:
	k1_tuple=([288.126,0,156.578],
	   [0,288.780,124.968],
	   [0,0,1])
	k1=np.array(k1_tuple)
	k2_tuple=([256.204,0,163.978],
	   [0,256.450,118.382],
	   [0,0,1])
	k2=np.array(k2_tuple)
	R_tuple=([0.999983,0.00264383,-0.0052673],
	   [-0.00264698,0.999996,-0.000589696],
	   [0.00526572,0.000603628,0.999986])
	R=np.array(R_tuple)
	T_tuple1=x=np.array([[[1,0,-24.2641],[0,1,-0.439535],[0,0,1]]])


	for p in range(yinying.shape[0]):
		zhuan=tuple(yinying[p])
		f=[]
		woca=[]
		for i in range(len(zhuan)):
			zuobiao=zhuan[i]
			for d in range(dmap.shape[0]):
				for e in range(dmap.shape[1]):
					x=np.array([d,e,1])
					if dmap[d,e]<20:
						pass
					else:
					#print ("此处要显示x格式")
					#print(x.dtype)
				        #print ("此处要显示k格式")
					#print(k1.dtype)
										
						depthworld=np.dot(k1,x)/dmap[d,e] #np.dot(k1,x).shape==(3,)
						rgbworld=[]
						rgbworld=np.dot(R,depthworld)
						rgbworld[2]=1
						rgbworld2=[]
						rgbworld2=np.dot(T_tuple1,rgbworld)
						hello=[]
						hellowei=[]
						k2_mat=mat(k2)
						k2_I=k2_mat.I
						hello=np.dot(rgbworld2,k2_I)*dmap[d,e]
					#print(hello)
						hello=np.array(hello)
						hello_list=list(chain(*hello))
						#print(hello_list)
						yesx=hello_list[0]
						yesy=hello_list[1]
						hellowei=[yesx,yesy]
						if np.isnan(hellowei).any()==True:
							pass	
						if np.isinf(hellowei).any()==True:
							pass
						else:
							hello=[int(yesx),int(yesy)]
							if (hello==zuobiao).all():
							#print("panduanzhihou*******************")
							#print(hello)
								woca.append(dmap[d,e])
	

	
		sumhe=0
		pingjun=0
		if woca==[]:
			pass
		else:
			woca=np.array(woca)
			woca_unique=[]
			wocazuizhong=[]
			woca_unique=np.unique(woca)
			if woca_unique.shape[0]==1:
				wocazuizhong=woca_unique
			else:
				wocazuizhong=shandiao(woca_unique)
			for i in range(len(wocazuizhong)):
				sumhe=sumhe+wocazuizhong[i]
			print ("**********深度距离******************")
			print (sumhe/len(wocazuizhong))


	
    cv2.imshow('rgb', im)
    cv2.imshow('depth', d4d)
  

cv2.destroyAllWindows()
rgb_stream.stop()
openni2.unload()
print ("Terminated")
