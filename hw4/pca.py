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
io.imsave('AVG.jpg',image_avg.astype(np.uint8))
image_avg_flat = image_avg.reshape(1,-1)
images_flat = images.reshape(415,-1)
images_flat = images_flat-image_avg_flat

eigvec,eigval,V = np.linalg.svd(images_flat.T,full_matrices=False)
total = np.sum(eigval)
importance = (eigval[:num_components]/total)*100
print(importance)

eigface = -eigvec.T
eigface_graph = np.copy(eigface[:num_components].reshape(num_components,600,600,3))
for i in range(num_components):
	eigface_graph[i]-=np.min(eigface_graph[i])
	eigface_graph[i]/=np.max(eigface_graph[i])
	eigface_graph[i]*=255
	io.imsave('eigface'+str(i)+'.jpg',eigface_graph[i].astype(np.uint8))
'''
weight = np.matmul(image_flat,eigface.T)
for i in range(4):
	reconstruct = np.matmul(weight[i,:num_components],eigface[:num_components])+image_avg_flat
	reconstruct = reconstruct.reshape(600,600,3)
	reconstruct-=np.min(reconstruct)
	reconstruct/=np.max(reconstruct)
	reconstruct*=255
	io.imsave('reconstruct'+str(i)+'.jpg',reconstruct.astype(np.uint8))
'''
weight = np.matmul(target-image_avg_flat,eigface.T)
reconstruct = np.matmul(weight[0,:num_components],eigface[:num_components])+image_avg_flat
reconstruct = reconstruct.reshape(600,600,3)
reconstruct-=np.min(reconstruct)
reconstruct/=np.max(reconstruct)
reconstruct*=255
io.imsave('reconstruct.jpg',reconstruct.astype(np.uint8))
