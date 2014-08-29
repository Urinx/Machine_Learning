#! /usr/bin/env python
# coding:utf-8

#########################################
#          Anomaly Detection            #
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

class AD(ML):

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

	# Estimate the parameters of a Gaussian distribution
	def estimateGaussian(self,x):
		m,n=x.shape
		mu=mean(x,0).reshape((n,1))
		sigma2=var(x,0).reshape((n,1))
		return mu,sigma2

	# Compute the probability density function of the multivariate gaussian distribution
	def multivariateGaussian(self,x,mu,sigma2):
		x=x-mu.T
		p=e**(-x**2/(2*sigma2.T))/sqrt(2*pi*sigma2.T)
		return p

	# Find the best threshold (epsilon) to use for selecting outliners
	def selectThreshold(self,yval,pval):
		bestEpsilon=0
		bestF1=0
		F1=0
		stepsize=(pval.max()-pval.min())/1000

		for epsilon in arange(pval.min(),pval.max(),stepsize):
			predictions=pval<epsilon
			tp=sum((double(yval==1)+double(predictions==1))==2)
			fp=sum((double(yval==0)+double(predictions==1))==2)
			fn=sum((double(yval==1)+double(predictions==0))==2)
			
			if tp!=0:
				prec=tp*1./(tp+fp)
				rec=tp*1./(tp+fn)
				F1=2*prec*rec/(prec+rec)

			if F1>bestF1:
				bestF1=F1
				bestEpsilon=epsilon

		return bestEpsilon,bestF1

	#################
	# Plot Function #
	#################
	# Plot 2D Data
	def plotData(self):
		x=self.X
		plot(x[:,0],x[:,1],'bo',markersize=2,linewidth=0)
		xlabel('Latency (ms)')
		ylabel('Throughput (mb/s)')
		return self

	def drawLine(self,p1,p2):
		plot([p1[0],p2[0]],[p1[1],p2[1]],linewidth=2)
		return self

	def visualizeFit(self,x,mu,sigma2):
		r=arange(x.min(),x.max(),.5)
		x1,x2=meshgrid(r,r)
		m,n=x1.shape
		x12=hstack((x1.flatten().reshape((m*n,1)),x2.flatten().reshape((m*n,1))))
		
		z=self.multivariateGaussian(x12,mu,sigma2)
		z=(z[:,0]*z[:,1]).reshape((m,n))
		self.plotData()
		contour(x1,x2,z)

		return self

	##########################
	def testAD(self):
		x=self.X
		xval=self.Xval
		yval=self.Yval

		mu,sigma2=self.estimateGaussian(x)
		# Get the density
		p=self.multivariateGaussian(x,mu,sigma2)
		# Visualize the fit
		self.visualizeFit(x,mu,sigma2)

		# Find Outliers
		pval=self.multivariateGaussian(xval,mu,sigma2)
		epsilon,F1=self.selectThreshold(yval,pval)
		outliners=where(p<epsilon)[0]
		plot(x[outliners,0],x[outliners,1],'ro',markersize=5)

		self.show()
		
if __name__=='__main__':
	test=AD('ex8data1.mat')
	#test.plotData().show()
	#test.testAD()