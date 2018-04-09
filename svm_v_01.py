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
        def __init__(self,kernel='linear',C=None,sigma=1.0,threshold=0.0,degree=1.0):
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
                self.k = np.dot(l_x,l_x.T).astype(float)
                if self.kernel=='linear':
                        self.k = (1. + 1./self.sigma*self.k)**self.degree
               # 
                elif self.kernel=='rbf':
                        self.xsquared = (np.diag(self.k)*np.ones((1,self.n))).T
                        b = np.ones((self.n,1))
                        self.k -= 0.5*(np.dot(self.xsquared,b.T) + np.dot(b,self.xsquared.T))
                        self.k = np.exp(self.k/(2.*self.sigma**2))	       
                
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
                
                cvx.solvers.options['show_progress'] = False
                sol = cvx.solvers.qp(cvx.matrix(P),cvx.matrix(q),cvx.matrix(G),cvx.matrix(h), cvx.matrix(A), cvx.matrix(b))
                                
                l_lamda = np.array(sol['x'])   # This holds lamda value
                
                '''
                find support vector
                '''

                self.sv = np.where(l_lamda>self.threshold)[0]
                #print(self.sv)
                #print(l_lamda)
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
                #print(self.nsupport)
		
	# building classifier -- This is neede when recalculate the class for test data with only support vector
        
        def classifier(self,Y,soft=False):
                k=np.dot(Y,self.l_x.T)
                if self.kernel=='linear':
                        k = (1. + 1./self.sigma*k)**self.degree
                        self.y = np.zeros((np.shape(Y)[0],1))
                        for j in range(np.shape(Y)[0]):
                                for i in range(self.nsupport):
                                        self.y[j] += self.l_lamda[i]*self.l_target[i]*k[j,i]
                                self.y[j] += self.b
                elif self.kernel=='rbf':
                        c = (1./self.sigma * np.sum(Y**2,axis=1)*np.ones((1,np.shape(Y)[0]))).T
                        c = np.dot(c,np.ones((1,np.shape(k)[1])))
                        aa = np.dot(self.xsquared[self.sv],np.ones((1,np.shape(k)[0]))).T
                        k = k - 0.5*c - 0.5*aa
                        k = np.exp(k/(2.*self.sigma**2))

                        self.y = np.zeros((np.shape(Y)[0],1))
                        for j in range(np.shape(Y)[0]):
                                for i in range(self.nsupport):
                                        self.y[j] += self.l_lamda[i]*self.l_target[i]*k[j,i]
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
                l_train_data = np.array(list(l_train),dtype = int)[:,0:]
        return l_train_data
	
def read_test_data():
	with open("test.csv",'r') as l_csvfile:
		l_test = csv.reader(l_csvfile)
		l_test_data = np.array(list(l_test),dtype = int)[:,0:]
	return l_test_data


def prepare_target(l_target,l_data,l_class):
	l_target[np.where(l_data[:,0]==l_class)]=1
	return l_target
	
if __name__ == "__main__":
	
	l_data = read_train_data()
	l_test_data = read_test_data()
	
	l_target = -np.ones((len(l_data),1))   # setting target array elements to -1
	output  = np.zeros((np.shape(l_test_data)[0],10)) # setting output array to zero.
	
	
	#print(l_data)
	#print(l_test_data)
	#print(l_target)
	#print(output)
	
	
	for l_class in range(10):
		obj_svm = svm(kernel='linear',degree=1.0,C=0.1)
		l = prepare_target(l_target,l_data,l_class)
		obj_svm.learn_module(l_data[:,1:],l)
		#print(obj_svm.classifier(l_test_data))
		output[:,l_class] = obj_svm.classifier(l_test_data[:,1:],soft=True).T
		
	# Make a decision about which class
	# Pick the one with the largest margin
	bestclass = np.argmax(output,axis=1)
	#print(bestclass)
	#print(output)
	
	err = np.where(bestclass!=l_test_data[:,0])[0]
	print ("% of accuracy (linear,C=0.1) : ",100.0-(len(err)/np.shape(l_test_data[:,0])[0])*100)
	'''
	for l_class in range(10):
		obj_svm = svm(kernel='rbf',degree=1.0,C=0.1)
		l = prepare_target(l_target,l_data,l_class)
		obj_svm.learn_module(l_data[:,1:],l)
		#print(obj_svm.classifier(l_test_data))
		output[:,l_class] = obj_svm.classifier(l_test_data[:,1:],soft=True).T
	# Make a decision about which class
	# Pick the one with the largest margin
	bestclass = np.argmax(output,axis=1)
	#print(bestclass)
	#print(output)
	
	err = np.where(bestclass!=l_test_data[:,0])[0]
	print ("% of accuracy (rbf,C=0.1) : ",100.0-(len(err)/np.shape(l_test_data[:,0])[0])*100)
	'''