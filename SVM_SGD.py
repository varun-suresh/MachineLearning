# This is a program to implement SVM using Stochastic Gradient Descent
import pandas as pd
import numpy as np
from numpy import size
from __builtin__ import int
# This function takes a training data file and trains a weight vector on the SVM objective
# If Cross-validation needs to be done, the value of nCrossValidation must be specified. 
# It does a nCrossValidation-fold cross validation. If the weights have to be trained without cross-validation, specify it as 1.
# Inputs : Training Data File : A CSV file with delimiter = ','. The data must be converted to a sparse format from the libSVM format.
#          nEpochs : No of epochs that must be used for Stochastic Gradient Descent
#          r_0 : Initial Training Rate
#          C - Hyper Parameter for regularization
           
def SVM_SGD(trainingDataFile,nEpochs,r_0,C,nCrossValidation):
    # The first column contains the labels and the other columns are data
    labels = trainingDataFile[0]
    trainingData = trainingDataFile.ix[:,1:size(trainingDataFile,1)]
    # Initialize Average Error
    error_avg = 0
    # Divide the training set into k=nCrossValidation parts :
    divisionSize = len(trainingData)/nCrossValidation
    for i in range(0,nCrossValidation):
        
        if(nCrossValidation > 1) :
            crossValidate_testList = range(i*divisionSize,(i+1)*divisionSize)
            crossValidate_test = trainingDataFile.ix[crossValidate_testList,:]
            crossValidate_test = crossValidate_test.as_matrix()
            crossValidate_testData = crossValidate_test[:,1:size(trainingDataFile,1)]
            labels_test = crossValidate_test[:,0]
            crossValidate_trainList = list(set(range(0,len(trainingData)))^set(crossValidate_testList))
        else :
            crossValidate_trainList = range(0,len(trainingData))
        crossValidate_train = trainingDataFile.ix[crossValidate_trainList,:]
        crossValidate_train = crossValidate_train.as_matrix()
        timeIndex = 0
        # Initialize the average weight vector to be a zero vector :
        w_avg =  np.zeros(size(trainingData,1))
        # Train the data for the required number of times : nEpoch number of times
        for j in range(0,nEpochs):
            # Shuffle the data
            np.random.shuffle(crossValidate_train)
            labels_CrossValidate = crossValidate_train[:,0]
            crossValidate_trainData = crossValidate_train[:,1:size(trainingDataFile,1)]
#             print size(crossValidate_trainData)
            # Initilaize the weight vector
            w = np.zeros(size(trainingData,1))
            # Stochastic Gradient descent :
                
            for m in range(0,len(crossValidate_trainData)) :
                learningRate = r_0/(1+r_0*timeIndex/C)
        #       print np.dot(w,crossValidate_trainData[m,:])
                if ((labels_CrossValidate[m] * np.dot(w,crossValidate_trainData[m,:])) <= 1) :
                    grad_E = w - C*labels_CrossValidate[m]*crossValidate_trainData[m,:]
                else :
                    grad_E = w
                w = w - learningRate * grad_E
                timeIndex += 1
        #       print m
            w_avg += w
        w_avg = w_avg/nEpochs
    #     print "Average weight Vec is",w_avg
        # After nEpochs, check the error :
        if nCrossValidation > 1 :
            result_Testbias = np.dot(crossValidate_testData,w_avg.T)
            labels_predicted = []
            for n in range(0,result_Testbias.shape[0]) :
                if result_Testbias[n] < 0 :
                    labels_predicted.append(-1)
                else :
                    labels_predicted.append(1)
        #     print labels_predicted
        #     print labels_test
            errors_test = sum(abs(labels_test - labels_predicted))/(2.0 * len(labels_predicted))
        #     print errors_test
            error_avg += errors_test
        error_avg = error_avg/nCrossValidation
#     print error_avg
    return error_avg,w_avg

# Read the training file
trainingDataFile = pd.read_csv("badges-train-features.csv",delimiter = ',',header = None)
# Set the number of epochs
nEpochs = 10
# Set the initial learning rate :
initialLearningRate = [0.001,0.01,0.1,1]
# r_0 = 0.001
# Hyper-parameter to control regularization
# C = 0.1
f = open("SVM_SGD_Parameters.txt",'w')
f.write("r_0 \t C \t Error \n")
nCrossValidation = 10
regularizationParameter = [0.1,1,10,100,1000]
# To get the best parameters :
error_best = 1
for r_0 in initialLearningRate :
    for C in regularizationParameter :
        (error_avg,w_avg) = SVM_SGD(trainingDataFile, nEpochs, r_0, C, nCrossValidation)
        f.write("%f \t %f \t %f \n"%(r_0,C,error_avg))
        if error_avg < error_best :
            error_best = error_avg
            r_0_best = r_0
            C_best = C
        
print "The best parameters are r_0 = %f and C = %f"%(r_0_best,C_best)

# Use the best parameters to train on the entire dataset :
nEpochs = 30
nCrossValidation = 1
# The weight vector returned here is used to test on the test file 
(error_avg,w_avg) = SVM_SGD(trainingDataFile, nEpochs, r_0_best, C_best, nCrossValidation)

# Read the test file :
testDataFile = pd.read_csv("badges-test-features.csv",delimiter = ',',header = None)
# Get the labels and the data :
labels = testDataFile[0]
testData = testDataFile.ix[:,1:size(testDataFile,1)]
testData = testData.as_matrix()
result_Testbias = np.dot(testData,w_avg.T)
labels_predicted = []
for n in range(0,result_Testbias.shape[0]) :
    if result_Testbias[n] < 0 :
        labels_predicted.append(-1)
    else :
        labels_predicted.append(1)
#     print labels_predicted
#     print labels_test
errors_test = sum(abs(labels - labels_predicted))/(2.0 * len(labels_predicted))
print "Error on the test set is ",errors_test


        
