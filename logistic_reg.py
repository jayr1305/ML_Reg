# Jay Rawal
# ML Assignment 1 

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model 
from sklearn.preprocessing import LabelEncoder
import pandas as pd

#constants
numb_of_iter = 2000
alpha = 0.3
kfolds = 5

def inputData(filename):
	print("Taking Input "+filename)
	data_raw = pd.read_csv(filename)
	
	# encoder inspired from
	# https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
	###################################################
	le = LabelEncoder()
	categorical_feature_mask = data_raw.dtypes==object
	categorical_cols = data_raw.columns[categorical_feature_mask].tolist()

	data_raw[categorical_cols] = data_raw[categorical_cols].apply(lambda colm: le.fit_transform(colm))
	data = data_raw.to_numpy()
	####################################################

	m = len(data)

	features = data[:,:-1]
	# print(features)
	outcome = data[:,-1].reshape(m,1)
	# print(outcome)
	return features,outcome
	#Func ends-----

def sigmoidy(x):

	return 1 / (1 + np.exp(-x))

def getAcc(train_predicted_output,train_outcome):
	bool_predict = train_predicted_output > 0.5
	prediction = np.where(bool_predict==False,0,bool_predict)

	bool_count = (prediction - train_outcome) == 0
	accuracy = np.where(bool_count==False,0,bool_count)

	return accuracy.sum()

def gradient_des(train_features,train_outcome,test_features,test_outcome,v_f,v_o):
	print("gradient_des without regularisation")
	
	m = len(train_features)
	thetas = np.zeros((len(train_features[0]),1),dtype=np.float)
	
	valid_acc = list()
	test_acc = list()
	acc = list()
	test_error=list()

	for i in range(numb_of_iter):
		
		train_predicted_output = sigmoidy(np.matmul(train_features,thetas)) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)								#m*1

		acc.append(getAcc(train_predicted_output,train_outcome))
		print(i)

		#valid
		v_predicted_output = sigmoidy(np.matmul(v_f,thetas))

		valid_acc.append(getAcc(v_predicted_output,v_o))
	
		#TEST
		test_predicted_output = sigmoidy(np.matmul(test_features,thetas))
		test_acc.append(getAcc(test_predicted_output,test_outcome))

		#Error
		cur_error = np.sum(np.multiply(test_outcome,np.log(test_predicted_output)) + np.multiply((1-test_outcome),np.log(1 - test_predicted_output)))
		# print(cur_error)
		test_error.append((-1)*cur_error/m)

		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1

		thetas = thetas - train_subt_theta*((alpha)/m)									#features*1

	return 	acc,thetas,test_acc,valid_acc,test_error
	#Func ends-----

def gradient_des_L1(train_features,train_outcome,lambda_reg,test_features,test_outcome,v_f,v_o):

	print("gradient_des with L1 regularisation")
	m = len(train_features)
	thetas = np.random.rand(len(train_features[0]),1)
	
	acc = list()
	test_acc = list()
	valid_acc = list()
	test_error = list()

	for i in range(numb_of_iter):

		#To compute next thethas	
		train_predicted_output = sigmoidy(np.matmul(train_features,thetas)) 						#m*1
		train_diff = np.subtract((train_predicted_output),train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1
		print(i)

		acc.append(getAcc(train_predicted_output,train_outcome))
		
		#valid
		v_predicted_output = sigmoidy(np.matmul(v_f,thetas))
		valid_acc.append(getAcc(v_predicted_output,v_o))
	
		#TEST
		test_predicted_output = sigmoidy(np.matmul(test_features,thetas))
		test_acc.append(getAcc(test_predicted_output,test_outcome))

		#Error
		cur_error = np.sum(np.multiply(test_outcome,np.log(test_predicted_output)) + np.multiply((1-test_outcome),np.log(1 - test_predicted_output)))
		test_error.append((-1)*cur_error/m)

		cond = (thetas > 0)
		plus_minus = np.where(cond==False,-1,cond)*(lambda_L1/2)

		train_subt_theta = train_subt_theta+plus_minus									#change for L1

		thetas = thetas - train_subt_theta*((alpha)/m)						#features*1

	return acc,thetas,test_acc,valid_acc,test_error
	#func ends here

def gradient_des_L2(train_features,train_outcome,lambda_reg,test_features,test_outcome,v_f,v_o):
	print("gradient_des with L2 regularisation")

	m = len(train_features)
	# print(m)

	thetas = np.random.rand(len(train_features[0]),1)
	
	acc=list()
	test_acc=list()
	valid_acc=list()
	test_error=list()

	for i in range(numb_of_iter):
		print(i)
		
		#To compute next thethas	
		train_predicted_output = sigmoidy(np.matmul(train_features,thetas)) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1				

		acc.append(getAcc(train_predicted_output,train_outcome))
		
		#valid
		v_predicted_output = sigmoidy(np.matmul(v_f,thetas))
		valid_acc.append(getAcc(v_predicted_output,v_o))
		
		#TEST
		test_predicted_output = sigmoidy(np.matmul(test_features,thetas))
		test_acc.append(getAcc(test_predicted_output,test_outcome))
	
		#Error
		cur_error = np.sum(np.multiply(test_outcome,np.log(test_predicted_output)) + np.multiply((1-test_outcome),np.log(1 - test_predicted_output)))
		test_error.append((-1)*cur_error/m)
	
		thetas = thetas*(1- (alpha/m)*lambda_reg) - train_subt_theta*((alpha)/m)						#features*1

	return acc,thetas,test_acc,valid_acc,test_error
	#Func ends-----

def normalize(features):

	meanVal = features.mean(axis=0)
	sd = np.std(features,axis = 0)
	features = features - meanVal
	features = features/sd
	return features

def plot(m,acc,s,k):
	plt.plot(range(1,numb_of_iter+1),acc,'b+')
	plt.title(k+" "+s)
	plt.xlabel('No of Iterations')

	if(k=="test error"):
		plt.ylabel("ERROR")
	else:
		plt.ylabel('Correct predicted %')

	if(s=="L2"):
		plt.show()
	else:
		plt.figure()


if __name__ == "__main__":
	filename = "train.csv"
	features,outcome = inputData(filename)
	features = normalize(features)

	features=np.insert(features,0,1,axis=1)

	# print(features)
	m = len(features)

	val_features = features[(3*m)//4:,:]
	val_outcomes = outcome[(3*m)//4:,:]

	filename = "test.csv"
	test_features,test_outcome = inputData(filename)
	test_features = normalize(test_features)

	test_features=np.insert(test_features,0,1,axis=1)

	#L1
	reg1 = linear_model.LassoCV(cv=kfolds)
	reg1.fit(features,outcome.ravel())
	lambda_L1 = reg1.alpha_										#lambda1
	#L2
	reg2 = linear_model.RidgeCV(cv=kfolds)
	reg2.fit(features,outcome)
	lambda_L2 = reg2.alpha_										#lambda2
	
	# print(lambda_L1,lambda_L2)

	acc_L,thetas_L,test_L,v_acc,t_er_0 = gradient_des(features,outcome,test_features,test_outcome,val_features,val_outcomes)
	acc_L1,thetas_L1,test_L1,v_L1,t_er_1 = gradient_des_L1(features,outcome,lambda_L1,test_features,test_outcome,val_features,val_outcomes)
	acc_L2,thetas_L2,test_L2,v_L2,t_er_2 = gradient_des_L2(features,outcome,lambda_L1,test_features,test_outcome,val_features,val_outcomes)
	
	acc_L = ((np.array(acc_L)/m)*100).tolist()	
	acc_L1 = ((np.array(acc_L1)/m)*100).tolist()
	acc_L2 = ((np.array(acc_L2)/m)*100).tolist()
	
	plot(m,acc_L,'','Train accuracy')
	plot(m,acc_L1,'L1','Train accuracy')
	plot(m,acc_L2,'L2','Train accuracy')

	m=len(val_features)

	v_acc = ((np.array(v_acc)/m)*100).tolist()
	v_L1 = ((np.array(v_L1)/m)*100).tolist()
	v_L2 = ((np.array(v_L2)/m)*100).tolist()

	plot(m,v_acc,'','Val accuracy')
	plot(m,v_L1,'L1','Val accuracy')
	plot(m,v_L2,'L2','Val accuracy')

	m=len(test_features)

	test_L = ((np.array(test_L)/m)*100).tolist()
	test_L1 = ((np.array(test_L1)/m)*100).tolist()
	test_L2 = ((np.array(test_L2)/m)*100).tolist()

	plot(m,test_L,'','test accuracy')
	plot(m,test_L1,'L1','test accuracy')
	plot(m,test_L2,'L2','test accuracy')

	plot(m,t_er_0,'','test error')
	plot(m,t_er_1,'L1','test error')
	plot(m,t_er_2,'L2','test error')