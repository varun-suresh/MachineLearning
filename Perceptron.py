# This program is the implementation of a simple perceptron algorithm.
# The algorithm is as follows :
# 1) During training, an update will be performed on an example (x,y) if y(w'x + b) < 0
# where : x - feature vector of length n
# y - label - 0 or 1
# w : weight vector of length n
# b : Bias 
# 2) The update equation is i) w(new) = w(old) + ryx ii) b(new) = b(old) + ry
# where r - Learning rate
############################################################################################################################################
import csv
import sys
import numpy as np
import scipy
import random
"""
Run the perceptron algorithm
perceptron.py <train file> <test file> <output file> <epochs> <no of runs>

"""
# Read the input from the file

# First, find the length of the input feature vector :

# The input file containing the feature vector is expected to be of the following form :

# 1) The first column contains the label
# 2) The feature vector is the entire row except the first column. 

# In the case of missing values, I fill up those columns with zeros. There should be a better way, I'm ignoring it for now.

# Training
training_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
f = open(output_file,'w')
f.write("Number of updates Learning Rate Training Set Errors Test Set Errors \n")
no_of_epochs = int(sys.argv[4])
no_of_runs = int(sys.argv[5])
# For no_of_epochs number of passes :
for rate in np.arange(0,1.1,0.1):
	errors_train_avg = 0
	errors_test_avg = 0
	for runs in range(0,no_of_runs):
		reader=csv.reader(open(training_file,"rb"),delimiter=',')
#		reader = csv.reader(open("sampleTable.txt"),delimiter = ' ')
		x=list(reader)
		result=np.array(x).astype('int')
		nColumns = result.shape[1] - 1
		bias = random.random()
		updates = 0
		weightVector = np.random.random([1,nColumns])
		for k in range(0,no_of_epochs):
			np.random.shuffle(result)
	#		print result.shape
			featureMat = scipy.delete(result,0,1)
			labels = result[:,0]
			for i in range(0,len(labels)):
		#		print np.dot(weightVector,featureMat[i]) 
			#	print labels
				if labels[i]*(np.dot(weightVector,featureMat[i]) + bias) <= 0 :
					updates += 1
					weightVector = weightVector + rate * labels[i] * featureMat[i]
					bias += rate * labels[i]
#			print weightVector
		#	print updates



		#######################################################################################################################################
		# Testing 
		# Read the test data file
		reader = csv.reader(open(training_file,"rb"),delimiter = ',')
#		reader = csv.reader(open("sampleTable.txt"),delimiter = ' ')
		x = list(reader)
		result = np.array(x).astype('int')
	
		# Check no of training errors
		# Store the labels in a list
		labels = result[:,0]
		featureMat = scipy.delete(result,0,1)
		#print featureMat.shape
		#print weightVector.T
		result_Trainingbias = np.dot(featureMat,weightVector.T) + bias * np.ones(shape = (len(labels),1))
		labels_predicted = []
		for i in range(0,result_Trainingbias.shape[0]) :
			if result_Trainingbias[i] < 0 :
				labels_predicted.append(-1)
			else :
				labels_predicted.append(1)
		#print labels_predicted
		errors_train = sum(abs(labels - labels_predicted)/2)
	#	print "Number of Training Errors for the learning rate %f are %d" %(rate,errors_train)
		# Check no of testing errors
		reader = csv.reader(open(test_file,"rb"),delimiter = ',')
#		reader = csv.reader(open("sampleTable.txt"),delimiter = ' ')
		x = list(reader)
		result = np.array(x).astype('int')
		# Store the labels in a list
		labels = result[:,0]
		featureMat = scipy.delete(result,0,1)
		#print featureMat.shape
		#print weightVector.T
		result_Testbias = np.dot(featureMat,weightVector.T) + bias * np.ones(shape = (len(labels),1))
		labels_predicted = []
		for i in range(0,result_Testbias.shape[0]) :
			if result_Testbias[i] < 0 :
				labels_predicted.append(-1)
			else :
				labels_predicted.append(1)
		#print labels_predicted
		errors_test = sum(abs(labels - labels_predicted)/2)
#		print errors_test
	#	print "Number of Testing Errors for the learning rate %f are %d" %(rate,errors_test)
		errors_train_avg += errors_train
		errors_test_avg += errors_test
#		print errors_test_avg
	errors_train_avg = float(errors_train_avg)/no_of_runs
	errors_test_avg = float(errors_test_avg)/no_of_runs
	f.write("%d \t %f \t %d \t %d \n " %(updates,rate,errors_train_avg,errors_test_avg))	

#######################################################################################################################################
	
#print labels_predicted.shape[0]
