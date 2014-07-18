#! /usr/bin/env python
# coding:utf-8
'''
梯度逼近算法(bgd)
'''

from numpy import *
from matplotlib.pyplot import *

class gradientDescent():

	def __init__(self,x=[],y=[]):
		self.X=x
		self.Y=y
		self.Theta=[]
		self.Alpha=0.01
		self.iterations=1500

	def load(self,fname,d=','):
		data=loadtxt(fname,delimiter=d)
		self.X=data[:,:-1]
		self.Y=data[:,-1:]

	def J(self):
		m=self.X.shape[0]
		x=hstack(( ones((m,1)) ,self.X))
		t=x.dot(self.Theta)-self.Y
		j=t.T.dot(t)/(2.*m)
		return j

	def gradientDesc(self):
		iters=self.iterations
		m=self.X.shape[0]
		x=hstack(( ones((m,1)) ,self.X))
		self.Theta=zeros((x.shape[1],1))
		J_history=zeros((iters,1))
		for i in xrange(iters):
			t=x.dot(self.Theta)-self.Y
			self.Theta-=self.Alpha/m*(x.T.dot(t))
			J_history[i]=self.J()

	def predict(self,x):
		if len(self.Theta)==0: self.gradientDesc()
		return array([1]+x).dot(self.Theta)

	def plot(self):
		if len(self.Theta)==0: self.gradientDesc()
		m,n=self.X.min(),self.X.max()
		axis([m,n,self.Y.min(),self.Y.max()])
		plot(self.X.T,self.Y.T,'rx')

		t1=array([m,n])
		t2=array([[1,m],[1,n]]).dot(self.Theta).T[0,:]
		plot(t1,t2)
		show()

if __name__=='__main__':
	test=gradientDescent()
	test.load('ex1data1.txt')
	print test.predict([8.4084])
	print test.Theta
	test.plot()