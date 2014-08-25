#! /usr/bin/env python
# coding:utf-8

#########################################
#            Spam Classifier            #
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

class SVM(ML):

	def __init__(self,fname,x=[],y=[]):
		self.Lambda=1
		self.Theta=[]

		mat=self.loadMat(fname)
		self.X=mat['X']
		self.Y=mat['y']
		if 'Xval' in mat:
			self.Xval=mat['Xval']
			self.Yval=mat['yval']
		#self.Xtest=mat['Xtest']

	# x1,x2: column vectors
	def linearKernel(self,x1,x2):
		sim=x1.T.dot(x2)
		return sim

	# To find non-linear decision boundaries
	def gaussianKernel(self,x1,x2,sigma):
		sim=e**(-sum((x1-x2)**2)/(2.0*sigma**2))
		return sim

	def svmTrain(self,X,Y,C,kernelFunction,tol=1e-3,max_passes=5):
		# An SVM classifier using a simplified version of the SMO algorithm
		# X: matrix of training examples
		# Y: column matrix containing 1 for positive examples and 0 for negative ones
		# C: standard SVM regularization parameter
		# tol: tolerance value used for determining equality of floating point number
		# max_passes: control the number of iterations over the dataset
		
		m,n=X.shape
		# Map 0 to -1
		# The data in Octave is alright but loaded in python
		# cause wrong calculate result, and it cost my some 
		# time to find this problem.
		# Because Y.dtype is unit8, it's overfloat when I set -1
		# So change type to int64
		Y=Y.astype('int64')
		Y[Y==0]=-1
		# Variables
		alphas=zeros((m,1))
		b=0
		E=zeros((m,1))
		passes=0
		eta=0
		L=0
		H=0

		fcn=kernelFunction.func_name
		if fcn=='linearKernel':
			K=X.dot(X.T)
		elif fcn=='gaussianKernel':
			X2=sum(X**2,1).reshape((m,1))
			K=X2+X2.T-2*(X.dot(X.T))
			K=kernelFunction(1,0)**K
		else:
			K=zeros((m,m))
			for i in xrange(m):
				for j in xrange(j,m):
					K[i,j]=kernelFunction(X[i,:].T,X[j,:].T)
					K[j,i]=K[i,j]

		# Train
		while passes<max_passes:
			num_change_alphas=0
			for i in xrange(m):
				E[i]=b+sum(alphas*Y*K[:,i:i+1])-Y[i]

				if (Y[i]*E[i]<-tol and alphas[i]<C) or (Y[i]*E[i]>tol and alphas[i]>0):

					# randomly select j 
					j=int(floor(m*random()))
					while j==i:
						j=int(floor(m*random()))

					E[j]=b+sum(alphas*Y*K[:,j:j+1])-Y[j]

					# Save old alphas
					# Here is a uneasy find problem if your code like this:
					#	alpha_i_old=alphas[i]
					# The alpha_i_old will change if alphas[i] has changed
					# So fixed as following:
					alpha_i_old=float(alphas[i])
					alpha_j_old=float(alphas[j])

					# Compute L and H
					if Y[i]==Y[j]:
						L=max(0,float(alphas[j]+alphas[i]-C))
						H=min(C,float(alphas[j]+alphas[i]))
					else:
						L=max(0,float(alphas[j]-alphas[i]))
						H=min(C,float(C+alphas[j]-alphas[i]))

					if L==H:
						continue

					# Compute eta
					eta=2*K[i,j]-K[i,i]-K[j,j]
					if eta>=0:
						continue

					# Compute and clip new value for alpha[j]
					alphas[j]=alphas[j]-(Y[j]*(E[i]-E[j]))/eta
					alphas[j]=min(H,float(alphas[j]))
					alphas[j]=max(L,float(alphas[j]))

					# Check if change in alpha is significant
					if abs(alphas[j]-alpha_j_old)<tol:
						alphas[j]=alpha_j_old
						continue

					# Determine value for alpha[i]
					alphas[i]=alphas[i]+Y[i]*Y[j]*(alpha_j_old-alphas[j])

					# Compute b1 and b2
					b1=b-E[i]-Y[i]*(alphas[i]-alpha_i_old)*K[i,j]-Y[j]*(alphas[j]-alpha_j_old)*K[i,j]
					b2=b-E[j]-Y[i]*(alphas[i]-alpha_i_old)*K[i,j]-Y[j]*(alphas[j]-alpha_j_old)*K[j,j]

					# Compute b
					if 0<alphas[i] and alphas[i]<C :
						b=b1
					elif 0<alphas[j] and alphas[j]<C:
						b=b2
					else:
						b=(b1+b2)/2

					num_change_alphas+=1

			if num_change_alphas==0:
				passes+=1
			else:
				passes=0

		class Model():
			def __init__(self):
				self.X=array([])
				self.Y=array([])
				self.kernelFunction=0
				self.b=0
				self.alphas=0
				self.w=0

		# Save the model
		model=Model()
		#idx=alphas>0
		#model.X=X[idx,:]
		#model.Y=Y[idx,:]
		idx=where(alphas>0)[0]
		model.X=X[idx]
		model.Y=Y[idx]
		model.kernelFunction=kernelFunction
		model.b=b
		model.alphas=alphas[idx]
		model.w=((alphas*Y).T.dot(X)).T

		return model

	# x: m x n matrix, each example is a row
	# pred: m x 1 column of prediction of {0,1} values
	def svmPredict(self,model,x):
		m=x.shape[0]
		M=model.X.shape[0]
		p=zeros((m,1))
		pred=zeros((m,1))

		if model.kernelFunction.func_name=='linearKernel':
			p=x.dot(model.w)+model.b
		elif model.kernelFunction.func_name=='gaussianKernel':
			x1=sum(x**2,1).reshape((m,1))
			x2=sum(model.X**2,1).reshape((M,1)).T
			K=x1+(x2-2*x.dot(model.X.T))
			K=model.kernelFunction(1,0)**K
			K=model.Y.T*K
			K=model.alphas.T*K
			p=sum(K,1)
		else:
			for i in xrange(m):
				prediction=0
				for j in xrange(M):
					prediction+=model.alphas[j]*model.Y[j]*model.kernelFunction(x[i:i+1,:].T,model.X[j:j+1,:].T)
				p[i]=prediction+model.b

		pred[p>=0]=1
		pred[p<0]=0
		return pred

	# Determine the best C and sigma parameter to use
	def selectParams(self,x,y,xval,yval):
		C=1
		sigma=0.3
		m=1
		param=[0.01,0.03,0.1,0.3,1,3,10,30]

		for s in param:

			def gaussianKernel(x1,x2):
				return self.gaussianKernel(x1,x2,s)

			for c in param:
				model=self.svmTrain(x,y,c,gaussianKernel)
				prediction=self.svmPredict(model,xval)
				tmp=mean(double(prediction!=yval))

				if tmp<m:
					m=tmp
					C=c
					sigma=s

		return C,sigma

	#################
	# Plot Function #
	#################
	def plotData(self):
		pos,neg=where(self.Y==1),where(self.Y==0)
		plot(self.X[pos,0],self.X[pos,1],'k+',markersize=7,linewidth=1)
		plot(self.X[neg,0],self.X[neg,1],'ro',markersize=7,linewidth=1)
		return self

	# plot a linear decision boundary learned by the SVM
	def visualizeBoundaryLinear(self,x,y,model):
		w=model.w
		b=model.b
		xp=array([min(x[:,0]),max(x[:,0])])
		yp=-(w[0]*xp+b)/w[1]
		self.plotData()
		plot(xp,yp,'g-')
		return self

	def visualizeBoundary(self,x,y,model):
		self.plotData()
		x1plot=linspace(min(x[:,0]),max(x[:,0]),100)
		x2plot=linspace(min(x[:,1]),max(x[:,1]),100)
		x1,x2=meshgrid(x1plot,x2plot)
		vals=zeros(x1.shape)
		for i in xrange(x1.shape[1]):
			this_x=hstack((x1[:,i:i+1],x2[:,i:i+1]))
			vals[:,i:i+1]=self.svmPredict(model,this_x)
		contour(x1,x2,vals,(0,0))
		return self

	##################
	def trainLinearSVM(self):
		x,y=self.X,self.Y
		C=1
		kernel=self.linearKernel
		model=self.svmTrain(x,y,C,kernel,1e-3,20)
		self.visualizeBoundaryLinear(x,y,model).show()

	def trainNonLinearSVM(self):
		x,y=self.X,self.Y
		C=1
		sigma=0.1

		def gaussianKernel(x1,x2):
			return self.gaussianKernel(x1,x2,sigma)

		kernel=gaussianKernel
		model=self.svmTrain(x,y,C,kernel)
		self.visualizeBoundary(x,y,model).show()

	def findBestParams(self):
		x,y=self.X,self.Y
		xval,yval=self.Xval,self.Yval
		C,sigma=self.selectParams(x,y,xval,yval)

		def gaussianKernel(x1,x2):
			return self.gaussianKernel(x1,x2,sigma)

		model=self.svmTrain(x,y,C,gaussianKernel)
		self.visualizeBoundary(x,y,model).show()

if __name__=='__main__':
	test=SVM('ex6data1.mat')
	#test.plotData().show()
	#test.trainLinearSVM()

	test2=SVM('ex6data2.mat')
	#test2.plotData().show()
	#test2.trainNonLinearSVM()

	test3=SVM('ex6data3.mat')
	#test3.plotData().show()
	#test3.findBestParams()