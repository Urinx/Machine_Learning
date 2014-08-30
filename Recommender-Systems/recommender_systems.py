#! /usr/bin/env python
# coding:utf-8

#########################################
#         Recommender Systems           #
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
import re

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

class CollaborativeFiltering(ML):

	def __init__(self,fname):
		self.Lambda=1
		self.Theta=[]

		mat=self.loadMat(fname)
		self.R=mat['R']
		self.Y=mat['Y']

	def normalizeRatings(self,y,R):
		m,n=y.shape
		y_mean=zeros((m,1))
		y_norm=zeros((m,n))

		for i in xrange(m):
			idx=where(R[i]==1)[0]
			y_mean[i]=mean(y[i,idx])
			y_norm[i,idx]-=y_mean[i]

		return y_norm,y_mean

	# Compute the collaborative filtering cost
	def cofiCost(self,params,y,R,num_users,num_movies,num_features,lada):
		x=params[:num_movies*num_features].reshape((num_movies,num_features))
		theta=params[num_movies*num_features:].reshape((num_users,num_features))

		Delta=R*(x.dot(theta.T)-y)
		j=sum(Delta**2)/2+lada*(sum(theta**2)+sum(x**2))/2
		
		return j

	def cofiGradient(self,params,y,R,num_users,num_movies,num_features,lada):
		x=params[:num_movies*num_features].reshape((num_movies,num_features))
		theta=params[num_movies*num_features:].reshape((num_users,num_features))

		Delta=R*(x.dot(theta.T)-y)
		x_grad=Delta.dot(theta)+lada*x
		theta_grad=Delta.T.dot(x)+lada*theta

		grad=hstack((x_grad.flatten(),theta_grad.flatten()))
		return grad

	def minJ(self,y,R,num_users,num_movies,num_features,lada):
		# Set Initial parameters (theta,x)
		x=np.random.randn(num_movies,num_features)
		theta=np.random.randn(num_users,num_features)
		initial_parameters=hstack((x.flatten(),theta.flatten()))
		
		j=lambda theta: self.cofiCost(theta,y,R,num_users,num_movies,num_features,lada)
		g=lambda theta: self.cofiGradient(theta,y,R,num_users,num_movies,num_features,lada)
		params=fmin_cg(j,initial_parameters,fprime=g,maxiter=100)
		
		x=params[:num_movies*num_features].reshape((num_movies,num_features))
		theta=params[num_movies*num_features:].reshape((num_users,num_features))
		return x,theta

class movieRecommenderSystem(CollaborativeFiltering):
	def __init__(self,ratingFile,paramFile):
		rMat=self.loadMat(ratingFile)
		self.R=rMat['R']
		self.Y=rMat['Y']

		pMat=self.loadMat(paramFile)
		self.X=pMat['X']
		self.Theta=pMat['Theta']
		self.Num_features=pMat['num_features']
		self.Num_users=pMat['num_users']
		self.Num_movies=pMat['num_movies']

	def loadMovieList(self,movieListFile):
		movieList=[]
		with open(movieListFile,'r') as f:
			for line in f.readlines():
				movieList+=[re.sub(r'^[0-9]+\s','',line.replace('\n',''))]
			f.close()
		return movieList

	def testCost(self):
		num_users=4
		num_movies=5
		num_features=3
		x=self.X[:num_movies,:num_features]
		theta=self.Theta[:num_users,:num_features]
		y=self.Y[:num_movies,:num_users]
		R=self.R[:num_movies,:num_users]

		params=hstack((x.flatten(),theta.flatten()))
		j=self.cofiCost(params,y,R,num_users,num_movies,num_features,1.5)
		print j

	def testRating(self):
		movieList=self.loadMovieList('movie_ids.txt')
		my_ratings=zeros((1682,1))
		# Suppose did not enjoy Silence of the Lamds (1991)
		my_ratings[97]=2
		my_ratings[6]=3
		my_ratings[11]=5
		my_ratings[53]=4
		my_ratings[63]=5
		my_ratings[65]=3
		my_ratings[68]=5
		my_ratings[182]=4
		my_ratings[225]=5
		my_ratings[354]=5

		print 'New user ratings:'
		for i in xrange(1682):
			if my_ratings[i]>0:
				print 'Rated',double(my_ratings[i]),'for',movieList[i]

		# Learning Movie Ratings
		# Y is a1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
		# R is a 1682x943 matrix, where R[i,j]=1 if and only if user j gave a rating to movie i
		y=self.Y
		R=self.R

		# Add our own ratings to the data matrix
		y=hstack((my_ratings,y))
		R=hstack((double(my_ratings!=0),R))

		# Normalize Ratings
		y_norm,y_mean=self.normalizeRatings(y,R)

		# Useful Values
		num_users=y.shape[1]
		num_movies=y.shape[0]
		num_features=10
		lada=10

		print '='*20
		print 'Training collaborative filtering...'
		x,theta=self.minJ(y,R,num_users,num_movies,num_features,lada)
		print 'Recommender system learning completed.'
		print '='*20

		# Recommendation for you
		p=x.dot(theta.T)
		my_predictions=p[:,0:1]+y_mean
		idx=argsort(my_predictions,0)[::-1]
		print 'Top 10 recommendations for you'
		for i in xrange(10):
			j=idx[i]
			print 'Predicting rating %.1f for %s' % (double(my_predictions[j]),movieList[j])

if __name__=='__main__':
	test=movieRecommenderSystem('ex8_movies.mat','ex8_movieParams')
	#test.testCost()
	test.testRating()