#! /usr/bin/env python
# coding:utf-8

from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import inv

class normalEqn():
	
	def __init__(self,x=[],y=[]):
		self.X=x
		self.Y=y
		self.theta=[]

	def load(self,fname,d=','):
		data=loadtxt(fname,delimiter=d)
		self.X=data[:,:-1]
		self.Y=data[:,-1:]

	def Theta(self):
		y=self.Y
		m=y.shape[0]
		x=hstack(( ones((m,1)) ,self.X))
		self.theta=inv(x.T.dot(x)).dot(x.T).dot(y)

	def predict(self,x):
		if len(self.theta)==0: self.Theta()
		return array([1]+x).dot(self.theta)

	def plot(self):
		if len(self.theta)==0: self.Theta()
		m,n=self.X.min(),self.X.max()
		axis([m,n,self.Y.min(),self.Y.max()])
		plot(self.X.T,self.Y.T,'rx')

		t1=array([m,n])
		t2=array([[1,m],[1,n]]).dot(self.theta).T[0,:]
		plot(t1,t2)
		show()

if __name__=='__main__':
	test=normalEqn()
	test.load('ex1data1.txt')
	print test.predict([8.4084])
	print test.theta
	test.plot()