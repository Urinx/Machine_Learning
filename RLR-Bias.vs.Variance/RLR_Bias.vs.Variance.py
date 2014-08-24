#! /usr/bin/env python
# coding:utf-8

#########################################
# Water Flowing Out of a Dam Prediction #
#########################################
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

class Bias_vs_Variance(ML):

	def __init__(self,fname,x=[],y=[]):
		self.Lambda=1
		self.Theta=[]

		mat=self.loadMat(fname)
		self.X=mat['X']
		self.Y=mat['y']
		self.Xval=mat['Xval']
		self.Yval=mat['yval']
		self.Xtest=mat['Xtest']

	# Regularized linear regression cost function
	# x: addOne(X)
	def J(self,x,y,theta,lada):
		m,n=x.shape
		theta=theta.reshape((n,1))
		h=x.dot(theta)
		j=sum((h-y)**2)/(2*m)
		j+=lada*(sum(theta**2))/(2*m)
		return j

	# Regularized linear regression gradient
	# x: addOne(X)
	def gradient(self,x,y,theta,lada):
		m,n=x.shape
		theta=theta.reshape((n,1))
		h=x.dot(theta)
		grad=x.T.dot(h-y)/m
		tmp=lada*theta/m
		tmp[1]=0
		grad+=tmp
		return grad.flatten()

	# x: addOne(x)
	def minJ(self,x,y,lada):
		initial_theta=zeros((x.shape[1],1))
		
		j=lambda theta: self.J(x,y,theta,lada)
		g=lambda theta: self.gradient(x,y,theta,lada)
		theta=fmin_cg(j,initial_theta,fprime=g,maxiter=500)
		return theta

	def learningCurve(self,x,y,xval,yval,lada):
		m=x.shape[0]
		error_train=zeros((m,1))
		error_val=zeros((m,1))

		for i in xrange(m):
			theta=self.minJ(x[:i],y[:i],lada)
			j_train=self.J(x[:i],y[:i],theta,0)
			error_train[i]=j_train
			j_val=self.J(xval,yval,theta,0)
			error_val[i]=j_val

		return error_train,error_val

	def validationCurve(self,x,y,xval,yval):
		# Selected values of lambda
		lambda_vec=array([[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]]).T

		m=lambda_vec.shape[0]
		error_train=zeros((m,1))
		error_val=zeros((m,1))

		for i in xrange(m):
			theta=self.minJ(x,y,lambda_vec[i])
			error_train[i]=self.J(x,y,theta,0)
			error_val[i]=self.J(xval,yval,theta,0)

		return lambda_vec,error_train,error_val

	def polynomialRegression(self):
		x,xval,xtest=self.X,self.Xval,self.Xtest
		y,yval=self.Y,self.Yval

		# Feature Mapping for Polynomial Regression
		p=8
		x_poly=self.polyFeatures(x,p)
		x_poly,mu,sigma=self.Normalization(x_poly)
		x_poly=self.addOne(x_poly)

		x_poly_val=self.polyFeatures(xval,p)
		x_poly_val=(x_poly_val-mu)/sigma
		x_poly_val=self.addOne(x_poly_val)

		x_poly_test=self.polyFeatures(xtest,p)
		x_poly_test=(x_poly_test-mu)/sigma
		x_poly_test=self.addOne(x_poly_test)

		# Polynomial Regression Fit Curve
		lada=0
		theta=self.minJ(x_poly,y,lada)
		self.plotFit(x.min(),x.max(),mu,sigma,theta,p).show()

		# Polynomial Regression Learning Curve
		clf()
		title('Learning curve for polynomial regression')
		self.plotLearningCurve(x_poly,y,x_poly_val,yval,lada).show()


	#################
	# Plot Function #
	#################
	def plotData(self):
		plot(self.X,self.Y,'bx',markersize=8,linewidth=10)
		xlabel('Change in water level (x)')
		ylabel('Water flowing out of the dam (y)')
		return self

	# x,xval: addOne()
	def plotLearningCurve(self,x,y,xval,yval,lada):
		m=x.shape[0]
		error_train,error_val=self.learningCurve(x,y,xval,yval,lada)
		plot(range(m),error_train,range(m),error_val)
		legend(('Train','Cross Validation'))
		xlabel('Number of training examples')
		ylabel('Error')

		return self

	# Linear Regression Learning Curve
	def plotLRLC(self):
		x,y=self.addOne(self.X),self.Y
		xval,yval=self.addOne(self.Xval),self.Yval
		lada=0

		self.plotLearningCurve(x,y,xval,yval,lada)
		title('Learning curve for linear regression')
		return self

	def plotFit(self,min_x,max_x,mu,sigma,theta,p):
		self.plotData()
		title('Polynomial Regression Fit (lambda = 0)')

		x=arange(min_x-15,max_x+15,0.05)
		x=x.reshape((x.shape[0],1))
		x_poly=self.polyFeatures(x,p)
		x_poly=(x_poly-mu)/sigma
		x_poly=self.addOne(x_poly)

		theta=theta.reshape((theta.shape[0],1))

		plot(x,x_poly.dot(theta),'y--')
		return self

	def plotValidationCurve(self):
		x,xval=self.X,self.Xval
		y,yval=self.Y,self.Yval

		p=8
		x_poly=self.polyFeatures(x,p)
		x_poly,mu,sigma=self.Normalization(x_poly)
		x_poly=self.addOne(x_poly)

		x_poly_val=self.polyFeatures(xval,p)
		x_poly_val=(x_poly_val-mu)/sigma
		x_poly_val=self.addOne(x_poly_val)

		lambda_vec,error_train,error_val=self.validationCurve(x_poly,y,x_poly_val,yval)
		plot(lambda_vec,error_train,lambda_vec,error_val)
		legend(('Train','Cross Validation'))
		xlabel('lambda')
		ylabel('Error')
		return self

	def predict(self):
		m,i,h,o,theta1,theta2,a1,a2,a3,z2,z3=self.feedForward(self.Theta,self.X)
		p=argmax(z3,axis=1).reshape(m,1)
		accuracy=array(p==self.Y,dtype=int).sum()*1./m
		print 'Training Set Accuracy:',accuracy

if __name__=='__main__':
	test=Bias_vs_Variance('ex5data1.mat')
	#test.plotData().show()
	#print test.J(test.addOne(test.X),test.Y,array([[1],[1]]),1)
	#print test.gradient(test.addOne(test.X),test.Y,array([[1],[1]]),1)
	#test.plotLRLC().show()
	#test.polynomialRegression()
	test.plotValidationCurve().show()