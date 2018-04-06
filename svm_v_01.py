#########################################################################
##
##    Name: svm.py
##    Description : This program implements support vector machine
##                  to classify unkown vectors based on learned machine
##                  algorithm with training feature vectors class.
##
##    Written By  : Mili Biswas (MSc Computer Science)
##    Organization: University of Fribourg
##    Date : 30.03.2018
##
##########################################################################

#==========================
#   Modules Import
#==========================

import numpy as np
import cvxopt as cvx
import math
import operator
import csv
import matplotlib.pyplot as plt

'''

    Algorithm for Mclass SVM
       Input Data : 
        1. n dimensional feature vectors from Training Set
        2. n dimensional vectors as test.
        3. each training data vector will be provided with y (+1 or -1) based on class
       Output:
        1. Linear Discriminant Function parameters
        2. Classification of Test Data Vector
        3. Graphical Representation of Test Data Vector classification
    Step1:
    
      Calculate Kernel for the processing
      
    Step2:
      Using the training data vector and Kernel from Step1 solve the 
      quadratic optimization problem for Langrange Multipliers and subsequently
      calculate the weight vector W for optimal solution and b for bias.
      
      Here use cvxopt solver.
      
    Step3:
    
    ........ To be determined
    

'''

class svm:
        def __init__(self,kernel='linear',C=None,sigma=0.05,threshold=0.1,degree=1.):
                '''
                     This is constructor which will get values of different varaibales---
                     ---- kernel    : Type of Kernel
                     ---- C         : Slack Variable
                     ---- sigma     : Standard Deviation
                     ---- threshold : Threshold value to choose lamda (langrange multipliers from solver) where lamdas > threshold
		     ---- degree    : Needed for calculating Kernel
                '''
                self.kernel = kernel
                self.sigma = sigma
                self.C = C
                self.threshold = threshold
                self.degree=degree
                
        def __kernel__(self,l_x):
                '''
                   This function will build the kernel and gives output kernel
                --- Linear Kernel
                --- Polynomial
                --- RBF   
                '''
                self.k = (1. + 1./self.sigma*np.dot(l_x,l_x.T))**self.degree
               # self.k = np.dot(l_x,l_x.T)
                
        def learn_module(self,l_x,l_target):
                '''
                   This function will train the claissifier based on Training Data.
                   Quadratic solver will be called in this function.
                   
                   parameters
                   
                      l_x      : Training Data without class identification
                      l_target : +1 / -1 (based on class pair) 
                '''
                self.n = np.shape(l_x)[0]
                self.__kernel__(l_x)
                P =np.dot(l_target,l_target.T)*self.k
                q = -np.ones((self.n,1))
                G = -np.eye(self.n)
                h = np.zeros((self.n,1))
                b = 0.0
                A = l_target.reshape(1,self.n)
                
                #  Quadratic Solver
                
                sol = cvx.solvers.qp(cvx.matrix(P),cvx.matrix(q),cvx.matrix(G),cvx.matrix(h), cvx.matrix(A), cvx.matrix(b))
                                
                l_lamda = np.array(sol['x'])   # This holds lamda value
                
                '''
                find support vector
                '''
                
                self.sv = np.where(l_lamda>self.threshold)[0]
                
                self.nsupport = np.shape(self.sv)[0]
                self.l_x = l_x[self.sv,:]
                self.l_target = l_target[self.sv,:]
                self.l_lamda = l_lamda[self.sv,:]
                
                #calculate bias b (or w0)
                self.b = np.sum(self.l_target)
                
                for i in range(self.nsupport):
                        self.b -= np.sum(self.l_lamda*self.l_target*np.reshape(self.k[self.sv[i],self.sv],(self.nsupport,1)))
                self.b = self.b/self.nsupport
                
                
                #print(self.b)
		
	# building classifier -- This is neede when recalculate the class for test data with only support vector
        
        def classifier(Y,soft=False):
                k = (1. + 1./self.sigma*np.dot(Y,self.l_x.T))**self.degree
                self.y = np.zeros((np.shape(Y)[0],1))
                for j in range(np.shape(Y)[0]):
                        for i in range(self.nsupport):
                                self.y[j] += self.lambdas[i]*self.l_target[i]*k[j,i]
                        self.y[j] += self.b
                if soft:
                        return self.y
                else:
                        return np.sign(self.y)      
        
def read_train_data():	
        '''
        this function is for reading data from input file
        '''
        with open("train.csv",'r') as l_csvfile:
                l_train = csv.reader(l_csvfile)
                l_train_data = np.array(list(l_train),dtype = int)[0:5000,0:]
        return l_train_data
	
def read_test_data():
	with open("test.csv",'r') as l_csvfile:
		l_test = csv.reader(l_csvfile)
		l_test_data = np.array(list(l_test),dtype = int)[0:5000,0:]
	return l_test_data


def prepare_target(l_target,l_data,l_class):
	l = []
	l_no_class = np.where(l_data[:,0]==l_class)
	l_total = np.shape(l_no_class)[1]
	if l_total != 0:
		l_target[0:l_total,0] = 1
		
		#l.append('Y')
		
	return l_target	
	
if __name__ == "__main__":
	l_data = read_train_data()
	l_target = -np.ones((5000,1))
	obj_svm0 = svm()
	obj_svm1 = svm()
	obj_svm2 = svm()
	obj_svm3 = svm()
	obj_svm4 = svm()
	obj_svm5 = svm()
	obj_svm6 = svm()
	obj_svm7 = svm()
	obj_svm8 = svm()
	obj_svm9 = svm()
	
	# learn svm0 classifier
	l = prepare_target(l_target,l_data,0)
	#if l[0] == 'Y':
	obj_svm0.learn_module(l_data[:,1:],l)
	
	# learn svm1 classifier
	l = prepare_target(l_target,l_data,1)
	#if l[0] == 'Y':
	obj_svm1.learn_module(l_data[:,1:],l)
	
	# learn svm2 classifier
	l = prepare_target(l_target,l_data,2)
	#if l[0] == 'Y':
	obj_svm2.learn_module(l_data[:,1:],l)
	
	# learn svm3 classifier
	l = prepare_target(l_target,l_data,3)
	#if l[0] == 'Y':
	obj_svm3.learn_module(l_data[:,1:],l)
	
	# learn svm4 classifier
	l = prepare_target(l_target,l_data,4)
	#if l[0] == 'Y':
	obj_svm4.learn_module(l_data[:,1:],l)
	
	# learn svm5 classifier
	l = prepare_target(l_target,l_data,5)
	#if l[0] == 'Y':
	obj_svm5.learn_module(l_data[:,1:],l)
	
	# learn svm6 classifier
	l = prepare_target(l_target,l_data,6)
	#if l[0] == 'Y':
	obj_svm6.learn_module(l_data[:,1:],l)
	
	# learn svm7 classifier
	l = prepare_target(l_target,l_data,7)
	#if l[0] == 'Y':
	obj_svm7.learn_module(l_data[:,1:],l)
	
	# learn svm8 classifier
	l = prepare_target(l_target,l_data,8)
	#if l[0] == 'Y':
	obj_svm8.learn_module(l_data[:,1:],l)
	
	# learn svm9 classifier
	l = prepare_target(l_target,l_data,9)
	#if l[0] == 'Y':
	obj_svm9.learn_module(l_data[:,1:],l)	
	
	l_test_data = read_test_data()
	output0 = svm0.classifier(l_test_data,soft=False)
	output1 = svm1.classifier(l_test_data,soft=False)
	output2 = svm2.classifier(l_test_data,soft=False)
	output3 = svm3.classifier(l_test_data,soft=False)
	output4 = svm4.classifier(l_test_data,soft=False)
	output5 = svm5.classifier(l_test_data,soft=False)
	output6 = svm6.classifier(l_test_data,soft=False)
	output7 = svm7.classifier(l_test_data,soft=False)
	output8 = svm8.classifier(l_test_data,soft=False)
	output9 = svm9.classifier(l_test_data,soft=False)

	