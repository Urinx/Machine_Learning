#! /usr/bin/env python
# coding:utf-8

#########################################
#                K-Means                #
#########################################
from numpy import *
import numpy as np
from random import random
from matplotlib.pyplot import *
from pylab import *
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_cg
from scipy.io import loadmat

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

class K_means(ML):

	def __init__(self,fname,x=[],y=[]):
		self.Lambda=1
		self.Theta=[]

		mat=self.loadMat(fname)
		self.X=mat['X']
		#self.Y=mat['y']
		if 'Xval' in mat:
			self.Xval=mat['Xval']
			self.Yval=mat['yval']
		#self.Xtest=mat['Xtest']

	def findClosestCentroids(self,x,initial_centroids):
		K=initial_centroids.shape[0]
		m=x.shape[0]
		tmp=zeros((m,K))
		for i in xrange(K):
			tmp[:,i:i+1]=sum((x-initial_centroids[i])**2,1).reshape((m,1))
		idx=argmin(tmp,1).reshape((m,1))
		return idx

	def computeCentroids(self,x,idx,K):
		m,n=x.shape
		centroids=zeros((K,n))
		for i in xrange(K):
			Ck=sum(idx==i)
			centroids[i]=sum(x[where(idx==i)[0]],0)/Ck
		return centroids

	def kMeansInitCentroids(self,x,K):
		m,n=x.shape
		centroids=zeros((K,n))

		randidx=[]
		while len(randidx)<K:
			r=np.random.randint(m)
			if r not in randidx: randidx+=[r]
		centroids=x[randidx]

		return centroids

	def runKMeans(self,x,initial_centroids,max_iters,plot_progress=False):
		m,n=x.shape
		centroids=initial_centroids
		previous_centroids=centroids
		K=centroids.shape[0]
		idx=zeros((m,1))

		for i in xrange(max_iters):
			idx=self.findClosestCentroids(x,centroids)
			centroids=self.computeCentroids(x,idx,K)

			if plot_progress:
				self.plotProgresskMeans(centroids,previous_centroids)
				previous_centroids=centroids

		if plot_progress:
			self.plotDataPoints(x,idx,K).show()

		return centroids,idx

	#################
	# Plot Function #
	#################
	# Plot 2D Data
	def plotData(self):
		x=self.X
		plot(x[:,0],x[:,1],'ro',markersize=7,linewidth=0)
		return self

	# Plot data points in x, coloring them so that those with the same
	def plotDataPoints(self,x,idx,K):
		mark=['s','o','^','v','>','<','d','p','h','8','+','*']*2
		for i in xrange(K):
			p=where(idx==i)[0]
			scatter(x[p,0],x[p,1],s=20,marker=mark[i],color=np.random.rand(1,3),label=str(i))
		legend()
		return self

	def drawLine(self,p1,p2):
		plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=2)
		return self

	def plotProgresskMeans(self,centroids,previous_centroids):
		m=centroids.shape[0]

		plot(centroids[:,0],centroids[:,1],'rx',markersize=7)
		for i in xrange(m):
			self.drawLine(centroids[i],previous_centroids[i])

		return self

	##################
	def testFindCentroidsAndMeans(self):
		K=3
		initial_centroids=array([[3,3],[6,2],[8,5]])
		x=self.X
		idx=self.findClosestCentroids(x,initial_centroids)
		print 'Closest centroids for the first 3 examples:'
		print idx[:3].flatten()
		print '(the closest centroids should be 0,2,1 respectively)'

		centroids=self.computeCentroids(x,idx,K)
		print 'Centroids computed after initial finding of closest centroids:'
		print centroids
		print 'the centroids should be:'
		print '[2.428301 3.157924]'
		print '[5.813503 2.633656]'
		print '[7.119387 3.616684]'

	def testKMeans(self):
		K=3
		max_iters=10
		x=self.X
		initial_centroids=array([[3,3],[6,2],[8,5]])
		
		print 'Running K-Means clustering on example dataset'
		print 'Ploting ...'
		centroids,idx=self.runKMeans(x,initial_centroids,max_iters,True)
		print 'Done!'

if __name__=='__main__':
	test2=K_means('ex7data2.mat')
	#test2.testFindCentroidsAndMeans()
	#test2.testKMeans()