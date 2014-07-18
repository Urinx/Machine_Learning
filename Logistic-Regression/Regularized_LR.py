#! /usr/bin/env python
# coding:utf-8

from numpy import *
from matplotlib.pyplot import *
from pylab import *
from scipy.optimize import fmin_bfgs

class ML():
	def __init__(self,x=[],y=[]):
		self.X=x
		self.Y=y
		self.Theta=[]
		self.Alpha=0.01
		self.Iterations=1500

	def load(self,fname,d=','):
		data=loadtxt(fname,delimiter=d)
		self.X=data[:,:-1]
		self.Y=data[:,-1:]

	def initXY(self,data):
		m=data.shape[0]
		x=hstack((ones((m,1)),data))
		return x,self.Y,m

	def Normalization(self):
		mu=mean(data,0)
		sigma=std(data,0)
		data_Norm=(data-mu)/sigma
		return data_Norm,mu,sigma

	def J(self):
		pass

	def predict(self,x):
		return array([1]+x).dot(self.Theta)

	def evaluate(self):
		pass

	def mapFeature(self,data,k):
		x1,x2=data[:,0:1],data[:,1:]
		m=x1.shape[0]
		x=ones((m,1))
		for i in xrange(1,k+1):
			for j in xrange(i+1):
				x=hstack((x,x1**j+x2**(i-j)))
		return x

	def plot(self):
		pass

	def show(self):
		show()

class LogisticRegression(ML):

	def __init__(self,fname,k,x=[],y=[]):
		self.Lambda=1
		self.load(fname)
		self.feature_X=self.mapFeature(self.X,k)

	def sigmoid(self,z):
		return 1/(1+exp(-z))

	#########################################################
	# CostFunc:
	# sigmoid() has a problem for that when z is too huge the
	# return result is equal to 1. which cause log(h) to nan
	# Solution:
	# normalize the feature first
	#########################################################
	def J(self,theta):
		x,y=self.feature_X,self.Y
		m=x.shape[0]
		theta=theta.reshape(theta.shape[0],1)
		lada=self.Lambda
		h=self.sigmoid(x.dot(theta))
		j=sum((-y.T.dot(log(h))-(1-y).T.dot(log(1-h))))/m+self.Lambda*sum(theta[1:]**2)/(2*m)
		return j

	def gradient(self,theta):
		x,y=self.feature_X,self.Y
		m=x.shape[0]
		theta=theta.reshape(theta.shape[0],1)
		h=self.sigmoid(x.dot(theta))
		grad=((h-y).T.dot(x)).T/m
		grad[1:]=grad[1:]+self.Lambda*theta[1:]/m
		return grad.flatten()

	def minJ(self):
		initial_theta=zeros((self.feature_X.shape[1]))
		self.Theta=fmin_bfgs(self.J,initial_theta,fprime=self.gradient)

	def plot(self):
		pos,neg=where(self.Y==1),where(self.Y==0)
		plot(self.X[pos,0].T,self.X[pos,1].T,'b+')
		plot(self.X[neg,0].T,self.X[neg,1].T,'ro')
		return self

	def plotDecisionBoundary(self):
		n=50
		x_a,x_b=min(self.X[:,0]),max(self.X[:,0])
		y_a,y_b=min(self.X[:,1]),max(self.X[:,1])
		x=linspace(x_a,x_b,n)
		y=linspace(y_a,y_b,n)
		z=zeros((n,n))
		for i in xrange(n):
			for j in xrange(n):
				z[i][j]=self.mapFeature(array([[x[i],y[j]]]),6).dot(self.Theta)
		self.plot()
		contour(x,y,z,(0,0))
		return self

if __name__=='__main__':
	test=LogisticRegression('ex2data2.txt',k=6)
	test.minJ()
	test.plotDecisionBoundary().show()