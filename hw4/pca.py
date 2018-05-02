import numpy as np
import skimage
from skimage import io
import os
import sys

num_components=4
imagedir = sys.argv[1]
imagelist = [os.path.join(imagedir,p) for p in os.listdir(imagedir)]
target = np.asarray(io.imread(os.path.join(imagedir,sys.argv[2]))).astype(float).reshape(1,-1)
image = []
for i in range(415):
	image.append(io.imread(imagelist[i]))
images = np.asarray(image).astype(float)
image_avg = np.average(images,axis=0)
image_avg_flat = image_avg.reshape(1,-1)
images_flat = images.reshape(415,-1)
images_flat = images_flat-image_avg_flat

eigvec,eigval,V = np.linalg.svd(images_flat.T,full_matrices=False)
eigface = -eigvec.T
weight = np.matmul(target-image_avg_flat,eigface.T)
reconstruct = np.matmul(weight[0,:num_components],eigface[:num_components])+image_avg_flat
reconstruct = reconstruct.reshape(600,600,3)
reconstruct-=np.min(reconstruct)
reconstruct/=np.max(reconstruct)
reconstruct*=255
io.imsave('reconstruction.jpg',reconstruct.astype(np.uint8))
