#! /usr/bin/env python
# coding:utf-8

#########################################
#            Image Compress             #
#########################################
from PIL import Image
from numpy import *
from matplotlib.pyplot import *
import k_means

class ImgCompresser(k_means.K_means):

	def __init__(self,imagefile):
		self.Img=self.loadImageData(imagefile)
		# Make all values in the range 0-1:
		#	self.X=self.Img/255
		#
		# P.S.
		# matplotlib.pyplot already provide
		# the imread() function, it's more helpful.
		im=imread(imagefile)
		self.Im=im
		self.X=im.reshape((im.shape[0]*im.shape[1],3))

	# Return a Nx3 matrix of pixels
	def loadImageData(self,imagefile):
		im=Image.open(imagefile)
		m,n=im.size
		data=im.getdata()
		imgMat=zeros((m*n,3))

		for i in xrange(m*n):
			imgMat[i]=data[i]

		return imgMat

	##################################
	# Image Compression With K-means #
	##################################
	def runCompression(self):
		x=self.X
		K=16
		max_iters=10
		initial_centroids=self.kMeansInitCentroids(x,K)

		centroids,idx=self.runKMeans(x,initial_centroids,max_iters)
		# Recover the image from the indices by mapping pixel to the centroid value
		x_recovered=centroids[idx].reshape(x.shape)

		return x_recovered

	#################
	# Plot Function #
	#################
	def plotImage(self):
		im=self.Im
		im_compressed=self.runCompression()
		im_compressed=im_compressed.reshape(im.shape)

		subplot(121)
		title('Original')
		imshow(im)
		subplot(122)
		title('Compressed, with 16 colors')
		imshow(im_compressed)

		return self

if __name__=='__main__':
	test=ImgCompresser('bird_small.png')
	test.plotImage().show()