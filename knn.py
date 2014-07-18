#! /usr/bin/env python
# coding:utf-8

'''
k-近邻算法(knn)
'''
from numpy import *

def autoNorm(dataSet):
	mins=dataSet.min(axis=0)
	maxs=dataSet.max(axis=0)
	ranges=maxs-mins
	m=dataSet.shape[0]
	normDataSet=(dataSet-tile(mins,(m,1)))*1./tile(ranges,(m,1))
	return normDataSet,mins,ranges

def knn(inputX,dataSet,labels,k):
	m=dataSet.shape[0]
	diffM=tile(inputX,(m,1))-dataSet
	distances=(diffM**2).sum(axis=1)**0.5
	sortedIndex=distances.argsort(axis=0)
	result={}
	for i in range(k):
		label=labels[sortedIndex[i]]
		result[label]=result.get(label,0)+1
	return sorted(result.iteritems(),key=lambda x:x[1],reverse=True)[0][0]

def createDataSet():
	group=array([[9,400],[200,5],[100,77],[40,300]])
	labels=['A','B','C','A']
	return group,labels

def simpleTest():
	group,labels=createDataSet()
	print knn(array([199,4]),group,labels,1)

def dataClassTest():
	Ratio=0.1
	error=0.0
	dataM,dataL=file2matrix('data.txt')
	normMat,mins,ranges=autoNorm(dataM)

	testNum=int(normData.shape[0]*Ratio)
	for i in xrange(testNum):
		r=knn(normData[i],normData[testNum:],dataL[testNum:],3)
		if r!=dataL[i]: error+=1
	print 'Error Rate is',error/testNum

if __name__=='__main__':
	simpleTest()