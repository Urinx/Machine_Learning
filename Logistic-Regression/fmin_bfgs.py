#! /usr/bin/env python

from numpy import *
from scipy.optimize import fmin_bfgs

def sigmoid(z):
    return 1/(1+exp(-z));

def cost(theta):
	theta=theta.reshape(n+1,1)
	h=sigmoid(x.dot(theta))
	t1=log(h)
	t2=log(1-h)
	t=y*t1+(1-y)*t2
	j=-sum(t)/m
	return j

def grad(theta):
	theta=theta.reshape(n+1,1)
	h=sigmoid(x.dot(theta))
	g=x.T.dot(h-y)/m
	return g.flatten()

data=loadtxt('ex2data1.txt',delimiter=',')
x=data[:,:-1]
y=data[:,-1:]
m,n=x.shape
x=hstack(( ones((m,1)) ,x))

initial_theta=array([-25,0,0])
print fmin_bfgs(cost,initial_theta,grad)
#print cost(initial_theta)
#print grad(initial_theta)