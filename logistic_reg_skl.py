#Jay Rawal
#Machine Learning Assignment 1

import math
import idx2numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression

#const
colors = ['b','g','r','c','m','y','k']
color_c = 0
roc_list = list()

def takeInput():
	train_X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
	train_Y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

	test_X = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
	test_Y = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
	
	l = [train_X,train_Y,test_X,test_Y]
	
	return l

def logistic_reg(penlty,train_X,train_Y,test_X,test_Y):
	lr_l = LogisticRegression(penalty=penlty)
	lr_l.fit(train_X,train_Y)
	prob_test = lr_l.predict_proba(test_X)

	roc_x, roc_y, ts = roc_curve(test_Y, prob_test[:,1])	
	roc_list.append([roc_x,roc_y])

	return lr_l.score(train_X,train_Y),lr_l.score(test_X,test_Y)

def plot_roc():
	# print(roc_list)
	for i in range(len(roc_list)):
		lbl= "Roc for " + str(i) 
		plt.plot(roc_list[i][0],roc_list[i][1],colors[i%7],label=lbl)
		# plt.show()
	plt.legend()
	plt.show()

if __name__ == "__main__":
	l = takeInput()
	train_X,train_Y,test_X,test_Y = l[0],l[1],l[2],l[3]
	
	train_X = train_X.reshape(60000,784)
	test_X = test_X.reshape(10000,784)

	# print(np.shape(train_X),np.shape(test_X))
	# print(train_Y,test_Y)

	for i in range(10):
		print("\n Working on",i)
		Y_cur_test = (i==test_Y).astype(int)
		Y_cur_train = (i==train_Y).astype(int)
		
		# print(Y_cur_train,Y_cur_test)

		score_l1_train,score_l1_test = logistic_reg("l1",train_X,Y_cur_train,test_X,Y_cur_test)
		print("Result for l1 on ",i)
		print("\t \t Training score : ",score_l1_train)
		print("\t \t Testing score : ",score_l1_test)
		
		score_l2_train,score_l2_test = logistic_reg("l2",train_X,Y_cur_train,test_X,Y_cur_test)
		print("Result for l2 on ",i)
		print("\t \t Training score : ",score_l2_train)
		print("\t \t Testing score : ",score_l2_test)

	plot_roc()