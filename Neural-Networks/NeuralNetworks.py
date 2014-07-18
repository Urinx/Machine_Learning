#! /usr/bin/env python
# coding:utf-8

#######################################
# Handwritten Digit (0-9) Recognizion #
#######################################
from numpy import *
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
		self.Iterations=1500
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

	def Normalization(self):
		mu=mean(data,0)
		sigma=std(data,0)
		data_Norm=(data-mu)/sigma
		return data_Norm,mu,sigma

	def sigmoid(self,z):
		return 1/(1+exp(-z))

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

class NeuralNetworks(ML):

	def __init__(self,fname,weightsfile,x=[],y=[]):
		self.Lambda=1
		self.All_Theta=[]
		mat=self.loadMat(fname)
		self.X=mat['X']
		self.Y=mat['y']
		weight=self.loadMat(weightsfile)
		self.Theta1=weight['Theta1']
		self.Theta2=weight['Theta2']

	def addOne(self,x):
		m=x.shape[0]
		one=ones((m,1))
		return hstack((one,x))

	def predict(self):
		theta1=self.Theta1
		theta2=self.Theta2
		x=self.X
		m=x.shape[0]

		a1=self.addOne(x)
		z2=a1.dot(theta1.T)
		a2=self.sigmoid(z2)
		a2=self.addOne(a2)
		z3=a2.dot(theta2.T)

		p=argmax(z3,axis=1).reshape(m,1)+1

		accuracy=array(p==self.Y,dtype=int).sum()*1./m
		print 'Training Set Accuracy:',accuracy

	def terminalPlot(self):
		x=self.X
		for n in xrange(0,x.shape[0],500):
			num=x[n,:].reshape((20,20)).T
			for line in num:
				s=''
				for i in line:
					if abs(i)<0.1: s+='0'
					else: s+='1'
				print s

if __name__=='__main__':
	test=NeuralNetworks('ex3data1.mat','ex3weights.mat')
	#test.terminalPlot()
	test.predict()