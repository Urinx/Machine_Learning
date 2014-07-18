#! /usr/bin/env python
# coding:utf-8

from numpy import *
from matplotlib.pyplot import *
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

	def Normalization(self,data):
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

	def plot(self):
		pass

	def show(self):
		show()

class LogisticRegression(ML):

	def __init__(self,x=[],y=[]):
		self.Lambda=1
		self.data_Norm=0
		self.mu=0
		self.sigma=0

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
		x,y,m=self.initXY(self.data_Norm)
		theta=theta.reshape(theta.shape[0],1)
		h=self.sigmoid(x.dot(theta))
		j=(-y.T.dot(log(h))-(1-y).T.dot(log(1-h)))/m
		return sum(j)

	def gradient(self,theta):
		x,y,m=self.initXY(self.data_Norm)
		theta=theta.reshape(theta.shape[0],1)
		h=self.sigmoid(x.dot(theta))
		grad=((h-y).T.dot(x)).T/m
		return grad.flatten()

	def minJ(self):
		initial_theta=zeros((self.X.shape[1]+1))
		self.Theta=fmin_bfgs(self.J,initial_theta,fprime=self.gradient)

	def featureNormalize(self):
		self.data_Norm,self.mu,self.sigma=self.Normalization(self.X)

	def plot(self):
		pos,neg=where(self.Y==1),where(self.Y==0)
		plot(self.X[pos,0].T,self.X[pos,1].T,'b+')
		plot(self.X[neg,0].T,self.X[neg,1].T,'ro')
		return self

	def plotDecisionLine(self):
		theta=self.Theta
		x=self.data_Norm
		plot_x=array([min(x[:,0]),max(x[:,0])])
		plot_y=(-1/theta[2])*(theta[1]*plot_x+theta[0])

		pos,neg=where(self.Y==1),where(self.Y==0)
		plot(x[pos,0].T,x[pos,1].T,'b+')
		plot(x[neg,0].T,x[neg,1].T,'ro')

		plot(plot_x,plot_y)
		return self

if __name__=='__main__':
	test=LogisticRegression()
	test.load('ex2data1.txt')
	test.featureNormalize()
	test.minJ()
	test.plotDecisionLine().show()