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

class Multi_class_LR(ML):

	def __init__(self,fname,x=[],y=[]):
		self.Lambda=1
		self.All_Theta=[]
		mat=self.loadMat(fname)
		self.X=mat['X']
		self.Y=mat['y']
		self.X1=hstack((ones((self.X.shape[0],1)),self.X))

	def J(self,theta,x,y):
		m=x.shape[0]
		theta=theta.reshape(theta.shape[0],1)
		lada=self.Lambda

		h=self.sigmoid(x.dot(theta))
		j=sum((-y.T.dot(log(h))-(1-y).T.dot(log(1-h))))/m+self.Lambda*sum(theta[1:]**2)/(2*m)
		return j

	def gradient(self,theta,x,y):
		m=x.shape[0]
		theta=theta.reshape(theta.shape[0],1)

		h=self.sigmoid(x.dot(theta))
		grad=((h-y).T.dot(x)).T/m
		grad[1:]=grad[1:]+self.Lambda*theta[1:]/m
		return grad.flatten()

	def minJ(self):
		initial_theta=zeros((self.X1.shape[1]))

		j=lambda theta:self.J(theta,self.X1,self.Y)
		grad=lambda theta:self.gradient(theta,self.X1,self.Y)

		# 拟Newton法       very slow
		#self.Theta=fmin_bfgs(j,initial_theta,fprime=grad)
		# 非线性共轭梯度法   much fast
		self.Theta=fmin_cg(j,initial_theta,fprime=grad)

	def oneVsAll(self,num_lables):
		initial_theta=zeros((self.X1.shape[1]))
		temp=zeros((self.X1.shape[1]))

		for k in xrange(1,num_lables+1):
			y=array(self.Y==k,dtype=int)

			j=lambda theta:self.J(theta,self.X1,y)
			grad=lambda theta:self.gradient(theta,self.X1,y)

			theta_k=fmin_cg(j,initial_theta,fprime=grad)
			temp=vstack((temp,theta_k))
		self.All_Theta=temp[1:]

	def predict(self,x):
		all_theta=self.All_Theta
		x=hstack((1,x)).reshape(x.shape[0],1)
		p=all_theta.dot(x)
		print argmax(p)+1

	def predictOneVsAll(self):
		all_theta=self.All_Theta
		x=self.X1
		m=x.shape[0]
		p=zeros((m,1))

		for i in xrange(m):
			p[i]=argmax(all_theta.dot(x[i].T))+1

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
	test=Multi_class_LR('ex3data1.mat')
	test.terminalPlot()
	test.oneVsAll(10)
	test.predictOneVsAll()