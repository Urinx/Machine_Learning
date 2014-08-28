#! /usr/bin/env python
# coding:utf-8

#########################################
#                  PCA                  #
#########################################
from numpy import *
import numpy as np
from random import random
from matplotlib.pyplot import *
from pylab import *
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_cg
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import k_means

class ML():
	def __init__(self,x=[],y=[]):
		self.X=x
		self.Y=y
		self.Theta=[]
		self.Alpha=0.01
		self.Iterations=50
		self.Lambda=1

	def load(self,fname,d=','):
		data=loadtxt(fname,delimiter=d)
		self.X=data[:,:-1]
		self.Y=data[:,-1:]

	def loadMat(self,fname):
		return loadmat(fname)

	def initXY(self,data):
		m=data.shape[0]
		x=hstack((ones((m,1)),data))
		return x,self.Y,m

	# Feature Normalize
	def Normalization(self,data):
		mu=mean(data,0)
		sigma=std(data,0)
		data_Norm=(data-mu)/sigma
		return data_Norm,mu,sigma

	def sigmoid(self,z):
		return 1/(1+exp(-z))

	def sigmoidGradient(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def J(self):
		pass

	def predict(self,x):
		return array([1]+x).dot(self.Theta)

	def evaluate(self):
		pass

	# x,x^2,x^3,....x^p
	def polyFeatures(self,x,p):
		x_poly=zeros((x.shape[0],p))
		for i in xrange(p):
			x_poly[:,i:i+1]=x**(i+1)
		return x_poly

	# x1,x2,x1*x2,...
	def mapFeature(self,data,k):
		x1,x2=data[:,0:1],data[:,1:]
		m=x1.shape[0]
		x=ones((m,1))
		for i in xrange(1,k+1):
			for j in xrange(i+1):
				x=hstack((x,x1**j+x2**(i-j)))
		return x

	def addOne(self,x):
		m=x.shape[0]
		one=ones((m,1))
		return hstack((one,x))

	def plot(self):
		pass

	def show(self):
		show()

# Principal Component Analysis
class PCA(ML):

	def __init__(self,fname):
		self.Lambda=1
		self.Theta=[]

		mat=self.loadMat(fname)
		self.X=mat['X']
		#self.Y=mat['y']
		if 'Xval' in mat:
			self.Xval=mat['Xval']
			self.Yval=mat['yval']
		#self.Xtest=mat['Xtest']

	# Return the eigenvectors U, the eigenvalues in S
	def pca(self,x):
		m,n=x.shape
		U=zeros((n,n))
		S=zeros((n,n))
		sigma=x.T.dot(x)/m
		U,S,V=linalg.svd(sigma)
		return U,S

	# Compute the reduced data representation
	# when projecting only to the top k eigenvectors
	def projectData(self,x,U,k):
		z=zeros((x.shape[0],k))
		U_reduce=U[:,:k]
		z=x.dot(U_reduce)
		return z

	# Recover an approximation of the original data when using the projected data
	def recoverData(self,z,U,k):
		x_rec=zeros((z.shape[0],U.shape[0]))
		U_reduce=U[:,:k]
		x_rec=z.dot(U_reduce.T)
		return x_rec

	#################
	# Plot Function #
	#################
	# Plot 2D Data
	def plotData(self):
		x=self.X
		plot(x[:,0],x[:,1],'ro',markersize=7,linewidth=0)
		return self

	def drawLine(self,p1,p2):
		plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=2)
		return self

	def displayFace(self,x):
		n=x.shape[0]
		width=round(sqrt(n))
		x=x.reshape((width,width)).T
		imshow(x)
		return self

	##########################
	def twoDimensionPCA(self):
		x=self.X
		x_norm,mu,sigma=self.Normalization(x)
		U,S=self.pca(x_norm)

		self.plotData()
		self.drawLine(mu,mu+1.5*S[0]*U[:,0].T)
		self.drawLine(mu,mu+1.5*S[1]*U[:,1].T)
		self.show()

	def dimensionReduction(self):
		x=self.X
		k=1
		x_norm,mu,sigma=self.Normalization(x)
		U,S=self.pca(x_norm)
		
		z=self.projectData(x_norm,U,k)
		x_rec=self.recoverData(z,U,k)
		
		plot(x_norm[:,0],x_norm[:,1],'bo')
		plot(x_rec[:,0],x_rec[:,1],'ro')
		for i in xrange(x.shape[0]):
			self.drawLine(x_norm[i,:],x_rec[i,:])
		self.show()

	def pcaOnImage(self,imagefile):
		x=imread(imagefile)
		m,n=x.shape[:2]
		x.shape=m*n,3

		x_norm,mu,sigma=self.Normalization(x)
		U,S=self.pca(x_norm)
		z=self.projectData(x_norm,U,1)
		x_rec=self.recoverData(z,U,1)
		
		x_back=x_rec*sigma+mu
		x_back.shape=m,n,3

		imshow(x_back)
		self.show()

	def pcaForVisualization(self,imagefile):
		x=imread(imagefile)
		a,b,c=x.shape
		x=x.reshape((a*b,c))
		K=16
		max_iters=10

		kmeans=k_means.K_means('ex7data1.mat')
		initial_centroids=kmeans.kMeansInitCentroids(x,K)
		centroids,idx=kmeans.runKMeans(x,initial_centroids,max_iters)

		ax=gca(projection='3d')
		for i in xrange(K):
			p=where(idx==i)[0]
			ax.scatter(x[p,0],x[p,1],x[p,2],color=np.random.rand(1,3))
		self.show()

		x_norm,mu,sigma=self.Normalization(x)
		U,S=self.pca(x_norm)
		z=self.projectData(x_norm,U,2)
		kmeans.plotDataPoints(z,idx,K)
		self.show()

if __name__=='__main__':
	test=PCA('ex7data1.mat')
	#test.plotData().show()
	#test.twoDimensionPCA()
	#test.dimensionReduction()

	face=PCA('ex7faces.mat')
	#face.displayFace(face.X[0]).show()

	image=PCA('ex7data1.mat')
	#image.pcaOnImage('bird_small.png')
	#image.pcaForVisualization('bird_small.png')