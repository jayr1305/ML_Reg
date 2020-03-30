# Jay Rawal
# ML Assignment 1 

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model 

#constants
filename = "data.csv"
numb_of_iter = 1000
alpha = 0.11
kfolds = 5

def inputData():
	data = np.genfromtxt(filename, delimiter=',', dtype=np.float)
	m = (len(data))
	ones = np.ones((m,1),dtype=np.float)

	data = np.append(ones,data,axis=1)

	features = data[1:,:-1]
	outcome = data[1:,-1].reshape(m-1,1)
	return features,outcome
	#Func ends-----

def normalize(features):

	meanVal = features[:,1].mean(axis=0)
	sd = np.std(features[:,1],axis = 0)

	features[:,1] = features[:,1] - meanVal
	features[:,1] = features[:,1]/sd

	return features

def gradient_des(train_features,train_outcome):

	m = len(train_features)
	thetas = np.zeros((len(train_features[0]),1),dtype=np.float)
	# print(thetas,train_features,train_outcome)

	for i in range(numb_of_iter):
		
		train_predicted_output = np.matmul(train_features,thetas) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1

		thetas = thetas - train_subt_theta*((alpha)/m)						#features*1

	return 	np.matmul(train_features,thetas),thetas
	#Func ends-----

def gradient_des_L1(train_features,train_outcome,lambda_reg):

	m = len(train_features)
	# print(m)
	thetas = np.random.rand(len(train_features[0]),1)
	
	for i in range(numb_of_iter):
		
		#To compute next thethas	
		train_predicted_output = np.matmul(train_features,thetas) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1

		cond = (thetas > 0)
		plus_minus = np.where(cond==False,-1,cond)*(lambda_reg/2)

		train_subt_theta = train_subt_theta+plus_minus									#change for L1

		thetas = thetas - train_subt_theta*((alpha)/m)						#features*1

	return np.matmul(train_features,thetas),thetas
	#func ends here

def gradient_des_L2(train_features,train_outcome,lambda_reg):

	m = len(train_features)
	# print(m)

	thetas = np.random.rand(len(train_features[0]),1)
	
	for i in range(numb_of_iter):
		
		#To compute next thethas	
		train_predicted_output = np.matmul(train_features,thetas) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1				

		thetas = thetas*(1- (alpha/m)*lambda_reg) - train_subt_theta*((alpha)/m)						#features*1

	return np.matmul(train_features,thetas),thetas
	#Func ends-----

def plot(features,outcome,predicted,s):
	plt.plot(features[:,1],outcome,'b+')
	plt.title('Training '+s)
	plt.xlabel('normalized_Brain_weight')
	plt.ylabel('body_weight')

	plt.plot(features[:,1],predicted, 'g')
	plt.show()
	#Func ends-----

def plot_all(features,outcome,predicted,predicted_L1,predicted_L2):
	plt.plot(features[:,1],outcome,'y+')
	plt.plot(features[:,1],predicted, 'r')
	plt.plot(features[:,1],predicted_L1, 'g')
	plt.plot(features[:,1],predicted_L2, 'b')
	plt.show()

def norml_eq(features,outcome):
	
	return np.matmul(np.matmul(np.linalg.inv(np.matmul(features.transpose(),features)),features.transpose()),outcome)

def getRmse(predicted,outcome):
	diff = np.subtract(predicted,outcome)
	return np.linalg.norm(diff,2)/np.sqrt(len(outcome))


if __name__ == "__main__":
	features,outcome = inputData()
	features = normalize(features)
	
	#L1
	reg1 = linear_model.LassoCV(cv=kfolds)
	reg1.fit(features,outcome.ravel())
	lambda_L1 = reg1.alpha_										#lambda1
	
	#L2
	reg2 = linear_model.RidgeCV(cv=kfolds)
	reg2.fit(features,outcome)
	lambda_L2 = reg2.alpha_										#lambda2
	
	
	# Without regularisation
	predicted_L,thetas_L = gradient_des(features,outcome)
	plot(features,outcome,predicted_L,"Grad Desc")
	# print(predicted_L[:10,:],outcome[:10,:])

	#L1 ================
	predicted_L1,thetas_L1 = gradient_des_L1(features,outcome,lambda_L1)
	plot(features,outcome,predicted_L1,"L1")

	#L2 ================
	predicted_L2,thetas_L2 = gradient_des_L2(features,outcome,lambda_L2)
	plot(features,outcome,predicted_L2,"L2")

	#"""============ Extra
	print("lambda1 =",lambda_L1,"\t lambda2=",lambda_L2)
	print("thetas_L : ",thetas_L)
	print("thetas_L1 : ",thetas_L1)
	print("thetas_L2 : ",thetas_L2)
	print("\n")
	print("L rmse : ",getRmse(predicted_L,outcome))
	print("L1 rmse : ",getRmse(predicted_L1,outcome))
	print("L2 rmse : ",getRmse(predicted_L2,outcome))
	plot_all(features,outcome,predicted_L,predicted_L1,predicted_L2)
	#"""
	
################################# 