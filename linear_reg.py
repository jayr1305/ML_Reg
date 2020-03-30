# Jay Rawal
# ML Assignment 1 

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model 

#constants
filename = "Dataset.data"
numb_of_iter = 1000
alpha = 0.2	
kfolds = 5

def inputData():
	data = np.genfromtxt(filename, delimiter=' ', dtype='unicode')
	m = (len(data))
	ones = np.ones((m,1),dtype='unicode')

	data = np.append(ones,data,axis=1)

	for i in range(len(data)):
		if(data[i][1] == "M"):
			data[i][1]="0"
		elif(data[i][1] == "F"):
			data[i][1]="1"
		else:
			data[i][1]="2"
			# print(data[i])

	data = data.astype(np.float)

	features = data[:,:-1]
	outcome = data[:,-1].reshape(m,1)
	return features,outcome
	#Func ends-----

def gradient_des(train_features,train_outcome,valid_features,valid_outcome):

	m = len(train_features)
	# print(m)

	thetas = np.random.rand(len(train_features[0]),1)

	train_rmse = list()
	valid_rmse = list()
	
	for i in range(numb_of_iter):
		
		#To compute next thethas	
		train_predicted_output = np.matmul(train_features,thetas) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1

		# Validation set rmse with this theta for ==>> plotting later
		#To plot RMSE == 2-norm(diff)/root(m) for plotting
		train_rmse.append(np.linalg.norm(train_diff,2)/np.sqrt(m))					
		valid_rmse.append(getRmse(thetas,valid_features,valid_outcome))

		thetas = thetas - (np.multiply(train_subt_theta,(alpha)/m))						#features*1

	return train_rmse,valid_rmse
	#Func ends-----

def grd_dsc_kfolds(features,outcome):
	M =len(features)
	print("Gradient Descent")
	jump = M//kfolds
	start=0

	train_rmse_matrix = list()
	valid_rmse_matrix = list()

	for i in range(kfolds):	
		print("	", (i+1) , " Fold Iteration")
		excd = range(start,start+jump)
		
		train_features = np.delete(features, excd ,axis=0)
		train_outcome = np.delete(outcome, excd ,axis=0)
		
		valid_features = features[excd,:]
		valid_outcomes = outcome[excd,:]

		rmse_train,rmse_valid = gradient_des(train_features,train_outcome,valid_features,valid_outcomes)
		
		train_rmse_matrix.append(rmse_train)		
		valid_rmse_matrix.append(rmse_valid)

		start = start+jump

	train_rmse_matrix = np.array(train_rmse_matrix)
	valid_rmse_matrix = np.array(valid_rmse_matrix)

	return train_rmse_matrix,valid_rmse_matrix
	#Func ends-----

def takeColMean(train_rmse_matrix,valid_rmse_matrix):
	#taking mean accross 5 folds
	train_mean_rmse = train_rmse_matrix.mean(axis=0)
	valid_mean_rmse = valid_rmse_matrix.mean(axis=0)

	return train_mean_rmse,valid_mean_rmse
	#Func ends-----

def plot(rmse_Train,rmse_Valid ):
	plt.plot(range(1,numb_of_iter+1),rmse_Train,'r')
	plt.title('Training')
	plt.xlabel('Iterations')
	plt.ylabel('mean RMSE')

	plt.figure()
	plt.plot(range(1,numb_of_iter+1),rmse_Valid, 'g')
	plt.title('Testing')	
	plt.xlabel('Iterations')
	plt.ylabel('mean RMSE')

	plt.show()
	#Func ends-----

def norml_eq(features,outcome):
	
	return np.matmul(np.matmul(np.linalg.inv(np.matmul(features.transpose(),features)),features.transpose()),outcome)

def getRmse(thetas,features,outcome):
	predicted_output = np.matmul(features,thetas) 
	diff = np.subtract(predicted_output,outcome)
	return np.linalg.norm(diff,2)/np.sqrt(len(features))

def getRMSE_kfolds_opthetas(op_thetas,features,outcome):
	M =len(features)

	jump = M//kfolds
	start=0

	RMSE = list()

	for i in range(kfolds):	

		excd = range(start,start+jump)
		
		train_features = np.delete(features, excd ,axis=0)
		train_outcome = np.delete(outcome, excd ,axis=0)
		
		valid_features = features[excd,:]
		valid_outcome = outcome[excd,:]

		rmse_opthetha_train = getRmse(op_thetas,train_features,train_outcome)
		rmse_opthetha_valid = getRmse(op_thetas,valid_features,valid_outcome)

		print("Optimal Thetas Fold ",(i+1))
		print("			 	Training Rmse with optimal thetas: ", rmse_opthetha_train)
		print("			 	Validation Rmse with optimal thetas: ",rmse_opthetha_valid)

		RMSE.append([rmse_opthetha_train,rmse_opthetha_valid])
		
		start+=jump

	return RMSE

def gradient_des_L1(train_features,train_outcome,lambda_reg):

	m = len(train_features)
	# print(m)
	thetas = np.random.rand(len(train_features[0]),1)

	train_rmse = list()
	
	for i in range(numb_of_iter):
		
		#To compute next thethas	
		train_predicted_output = np.matmul(train_features,thetas) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1

		# Validation set rmse with this theta for ==>> plotting later
		#To plot RMSE == 2-norm(diff)/root(m) for plotting
		train_rmse.append(np.linalg.norm(train_diff,2)/np.sqrt(m))	

		cond = (thetas > 0)
		plus_minus = np.where(cond==False,-1,cond)*(lambda_reg/2)

		train_subt_theta = train_subt_theta+plus_minus									#change for L1

		thetas = thetas - (np.multiply(train_subt_theta,(alpha)/m))						#features*1

	return train_rmse,thetas
	#func ends here

def gradient_des_L2(train_features,train_outcome,lambda_reg):

	m = len(train_features)
	# print(m)

	thetas = np.random.rand(len(train_features[0]),1)

	train_rmse = list()
	
	for i in range(numb_of_iter):
		
		#To compute next thethas	
		train_predicted_output = np.matmul(train_features,thetas) 						#m*1
		train_diff = np.subtract(train_predicted_output,train_outcome)					#m*1
		train_subt_theta = np.matmul(train_features.transpose(),train_diff)				#features*1

		# Validation set rmse with this theta for ==>> plotting later
		#To plot RMSE == 2-norm(diff)/root(m) for plotting
		train_rmse.append(np.linalg.norm(train_diff,2)/np.sqrt(m))					

		thetas = thetas*(1- (alpha/m)*lambda_reg) - (np.multiply(train_subt_theta,(alpha)/m))						#features*1

	return train_rmse,thetas
	#Func ends-----
	
if __name__ == "__main__":
	features,outcome = inputData()

	train_rmse_matrix,valid_rmse_matrix = grd_dsc_kfolds(features,outcome)				#matrix of 5 folds
	rmse_Train,rmse_Valid = takeColMean(train_rmse_matrix,valid_rmse_matrix)			#mean of all 5 folds, col wise mean

	op_thetas = norml_eq(features,outcome)								#Normal Equation optimal thetas
	print(" \n Normal Equation Optimal Thetas : ", op_thetas)
	

	rmse_normalEq = getRmse(op_thetas,features,outcome)					#root mean square error over whole data set
	print("Root mean sq. error with optimal thetas for whole data set ", rmse_normalEq)
	
	Rmse = getRMSE_kfolds_opthetas(op_thetas,features,outcome)			#get rmse with optimal thetas over validation set over 5 folds

	mean_normalEq = np.array(Rmse)[:,1].mean(axis=0)					# mean Rmse with optimal thetas(from normal-eq)
	mean_gradDes = rmse_Valid[-1]										# mean Rmse with gradient descent converging theta

	print("Gradient Descent mean rmse", mean_gradDes)
	print("Normal Equation mean rmse ", mean_normalEq)
	# print(Rmse)

	plot(rmse_Train,rmse_Valid)


													#Regularisation Part B
	print("\n			Question 2 Regularisation \n")
	minValidationErrorRange = np.argmin(valid_rmse_matrix[:,-1])
	print("Minimum validation error fold with gradient descent ",minValidationErrorRange+1)
	
	jump = len(features)//kfolds
	start= jump
	excd = range(start,start+jump)								#range to be used as test

	train_reg_features = np.delete(features, excd ,axis=0)
	train_reg_outcome = np.delete(outcome, excd ,axis=0)
	
	test_features = features[excd,:]
	test_outcomes = outcome[excd,:]
	# print(len(train_reg_features),len(train_reg_outcome),len(train_reg_features[0]))

	reg1 = linear_model.LassoCV(cv=kfolds)
	reg1.fit(train_reg_features,train_reg_outcome.ravel())
	lambda_L1 = reg1.alpha_										#lambda1
	print("Lambda L1 " ,lambda_L1)

	reg2 = linear_model.RidgeCV(cv=kfolds)
	reg2.fit(train_reg_features,train_reg_outcome)	
	lambda_L2 = reg2.alpha_										#lambda2
	print("Lambda L2 " ,lambda_L2)

	L1_gradientDesc,L1_thetas = gradient_des_L1(train_reg_features,train_reg_outcome,lambda_L1)
	print("RMSE Value of L1 gradient descent for Test ", getRmse(L1_thetas,test_features,test_outcomes))

	plt.plot(range(1,numb_of_iter+1),L1_gradientDesc, 'g')
	plt.title('L1 Gradient Descent')	
	plt.xlabel('Iterations')
	plt.ylabel('RMSE')
	plt.show()

	L2_gradientDesc,L2_thetas = gradient_des_L2(train_reg_features,train_reg_outcome,lambda_L2)
	print("RMSE Value of L2 gradient descent for Test ", getRmse(L2_thetas,test_features,test_outcomes))

	plt.plot(range(1,numb_of_iter+1),L2_gradientDesc, 'r')
	plt.title('L2 Gradient Descent')	
	plt.xlabel('Iterations')
	plt.ylabel('RMSE')
	plt.show()

################################# 