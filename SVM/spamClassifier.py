#! /usr/bin/env python
# coding:utf-8

#########################################
#            Spam Classifier            #
#########################################
import svm
import re
import poterStemmer
from numpy import *

# Use SVMs to build spam filter
class SpamClassifier(svm.SVM):
	def __init__(self,data_file,vacab_file):
		svm.SVM.__init__(self,data_file)
		self.VocabList=self.getVocabList(vacab_file)

	def getVocabList(self,vacab_file):
		vocabList=[]
		with open(vacab_file,'r') as f:
			vocabList=[line.split('\t')[1].replace('\n','') for line in f.readlines()]
		return vocabList

	# Preprocesses the body of an email and returns a list of word_indices
	def processEmail(self,email_contents):
		vocabList=self.VocabList
		n=len(vocabList)
		word_indices=[]

		# =========Preprocess Email=========
		# Lower case
		email_contents=email_contents.lower()
		# Strip all HTML
		email_contents=re.sub(r'<[^<>]+>','',email_contents)
		# Handle Numbers
		email_contents=re.sub(r'[0-9]+','number',email_contents)
		# Handle URLs
		email_contents=re.sub(r'http[s]?://[^\s]*','httpaddr',email_contents)
		# Handle Email Addresses
		email_contents=re.sub(r'[^\s]+@[^\s]+','emailaddr',email_contents)
		# Handle $ sign
		email_contents=re.sub(r'[$]+','dollar',email_contents)
		# Remove any non alphanumeric characters
		email_contents=re.sub(r'[^a-zA-Z0-9]+',' ',email_contents)
		# Strip space
		email_contents=email_contents.strip()

		# Stem the word
		# Apply the Porter Stemming algorithm
		# Original code provided at:
		# http://tartarus.org/~martin/PorterStemmer/python.txt
		p=poterStemmer.PorterStemmer()
		output=[]
		for word in email_contents.split(' '):
			output+=[p.stem(word,0,len(word)-1)]
		email_contents=' '.join(output)

		# Look up the word in the dictionary and add to word_indices if found
		for word in email_contents.split(' '):
			if word in vocabList:
				word_indices+=[vocabList.index(word)]

		return word_indices,email_contents

	# Takes in a word_indices array and produces a feature vector from the word indidces
	def emailFeatures(self,word_indices):
		n=len(self.VocabList)
		x=zeros((n,1))
		x[word_indices]=1
		return x

	###################
	def example(self,sample_file):
		print '\nPreprocessing sample email ('+sample_file+')\n'
		
		print '===== Origin Email ======\n'
		email_contents=open(sample_file,'r').read()
		print email_contents

		word_indices,processed_email=self.processEmail(email_contents)
		print '==== Processed Email ====\n'
		print processed_email
		print
		print '====== Word Indices =====\n'
		print word_indices
		print
		print '======= Spam or Not ======\n'
		if 'Model' in dir(self):
			model=self.Model
			x=self.emailFeatures(word_indices)
			pred=self.svmPredict(model,x)
			print 'Prediction:',pred
		else:
			print 'The spam classifier has not trained yet!'
		print
		print 'P.S. 1 indicates spam, 0 indicates not spam'

	# Train Linear SVM for Spam Classification
	def trainSpamClassifier(self):
		x,y=self.X,self.Y
		C=0.1
		kernel=self.linearKernel
		model=self.svmTrain(x,y,C,kernel)
		self.Model=model
		pred=self.svmPredict(model,x)
		print 'Training Accuracy:',mean(double(pred==y))*100,'%'

	# Test Spam Classification
	def testSpamClassifier(self,test_set):
		mat=self.loadMat(test_set)
		self.Xtest=mat['Xtest']
		pred=self.svmPredict(self.Model,self.Xtest)
		print 'Test Accuracy:',mean(double(pred==y))*100,'%'

if __name__=='__main__':
	test=SpamClassifier('spamTrain.mat','vocab.txt')
	#test.example('emailSample1.txt')
	#test.example('emailSample2.txt')
	#test.trainSpamClassifier()
	#est.testSpamClassifier('spamTest.mat')
	#test.example('spamSample1.txt')
	#test.example('spamSample2.txt')
