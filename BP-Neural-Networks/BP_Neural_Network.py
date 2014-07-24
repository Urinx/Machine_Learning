#! /usr/bin/env python
# coding:utf-8

#######################################
# Handwritten Digit (0-9) Recognizion #
#######################################
from numpy import *
import numpy as np
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

	def Normalization(self):
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

class BP_Neural_Network(ML):

	def __init__(self,fname,x=[],y=[]):
		self.Lambda=1
		self.Input_Layer_Size=400
		self.Hidden_Layer_Size=25
		self.Output_Layer_Size=10

		mat=self.loadMat(fname)
		self.X=mat['X']
		self.Y=mat['y']
		# Because the number 0 in y is labled as 10
		self.Y[self.Y==10]=0

		i,h,o=self.layerSize()
		self.Y_k=eye(o)[self.Y][:,0]

		total=(i+1)*h+(h+1)*o
		self.Theta_Grad=0.2*ones((1,total)).flatten()

	def layerSize(self):
		return self.Input_Layer_Size,self.Hidden_Layer_Size,self.Output_Layer_Size

	def feedForward(self,theta,x):
		m=x.shape[0]
		i,h,o=self.layerSize()

		theta1=theta[:h*(1+i)].reshape(h,1+i)
		theta2=theta[h*(1+i):].reshape(o,1+h)

		a1=self.addOne(x)
		z2=a1.dot(theta1.T)
		a2=self.addOne(self.sigmoid(z2))
		z3=a2.dot(theta2.T)
		a3=self.sigmoid(z3)
		return m,i,h,o,theta1,theta2,a1,a2,a3,z2,z3

	def J(self,theta,x,y,cal_grad=0):
		lada=self.Lambda
		m,i,h,o,theta1,theta2,a1,a2,a3,z2,z3=self.feedForward(theta,x)

		j=sum(-y*log(a3)-(1-y)*log(1-a3))/m
		j+=lada*(sum(theta1[:,1:]**2)+sum(theta2[:,1:]**2))/(2*m)

		# Calculate The Gradient
		if cal_grad:
			delta3=a3-y
			delta2=delta3.dot(theta2[:,1:])*self.sigmoidGradient(z2)
			
			Delta2=delta3.T.dot(a2)
			Delta1=delta2.T.dot(a1)
			
			r1=lada*theta1/m
			r2=lada*theta2/m
			r1[:,0]=0
			r2[:,0]=0
			theta1_grad=(Delta1/m+r1).flatten()
			theta2_grad=(Delta2/m+r2).flatten()
			self.Theta_Grad=hstack((theta1_grad,theta2_grad))
		# End

		return j

	def gradient(self,theta):
		return self.Theta_Grad

	def randInitialWeights(self,L_in,L_out):
		epsilon_init=sqrt(6)/sqrt(L_in+L_out)
		W=np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init
		return W

	def minJ(self):
		i,h,o=self.layerSize()

		init_theta1=self.randInitialWeights(i,h)
		init_theta2=self.randInitialWeights(h,o)
		initial_theta=hstack((init_theta1.flatten(),init_theta2.flatten()))
		
		args=(self.X,self.Y_k,1)
		j=lambda theta: self.J(theta,*args)
		self.Theta=fmin_cg(j,initial_theta,fprime=self.gradient,maxiter=50)

	def predict(self):
		m,i,h,o,theta1,theta2,a1,a2,a3,z2,z3=self.feedForward(self.Theta,self.X)

		p=argmax(z3,axis=1).reshape(m,1)
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

	def checkNNGradients(self):
		'''
		In the function checkNNGradients, our code creates a small
		random model and dataset which is used with computeNumericalGradient
		for gradient checking. Furthermore, after you are confident
		that your gradient computations are correct, you should turn
		off gradient checking before running your learning algorithm.
		'''
		pass

	def computeNumbericalGradient(self):
		epsilon=1e-4
		x=self.X
		y=self.Y_k
		theta=self.Theta
		m=len(theta)
		f=[]

		for i in xrange(m):
			tmp=zeros(m)
			tmp[i]=epsilon
			f.append((self.J(theta+tmp,x,y)-self.J(theta-tmp,x,y))/(2*epsilon))
		return f

if __name__=='__main__':
	test=BP_Neural_Network('ex4data1.mat')
	#test.terminalPlot()
	test.minJ()
	test.predict()