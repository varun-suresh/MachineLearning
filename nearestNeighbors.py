# This program is for classifying data using the K-nearest neighbor
# This program finds K = {1,2..5} neighbors.
# The features are assumed to be categorical and the distance measure used is Hamming distance.
import csv
def findHammingDistance(trainList,row):
	hammingDistance = []
	for i in range(0,len(trainList)):
		hd = 0
		for j in range(0,len(row)):
			if trainList[i][j] != row[j] :
				hd += 1
		hammingDistance.append(hd)
	return hammingDistance

def findNnearestNeighbors(hammingDistance):
# Input : An array which contains the Hamming distance of the test point from all the training points.
	minDistance = min(hammingDistance)
	distance = list(set(hammingDistance))
	N = 2
#	print distance
#	print minDistance
	indices = []
	for j in range(0,len(distance)):
		for i in range(0,len(hammingDistance)):
			if(distance[j] == hammingDistance[i]):
				indices.append(i)

#	print len(indices)
	return indices[0:5]

def majorityLabel(indices,N):
#	print labels_list
#	print indices
	labels = []
	for i in range(0,N):
		labels.append(trainListLabels[indices[i]])
#	print labels
	plus_count = labels.count('+')
	minus_count = labels.count('-')	
	if plus_count >= minus_count :
		return '+'
	else :
		return '-'
#	print plus_count,minus_count
#	print labels
	
# Open the file and find out the number of rows and columns.
datafilename = 'badges-trainFeatures.csv'
d = ',' # Delimiter
trainFile=open(datafilename,'r')
trainReader=csv.reader(trainFile,delimiter=d)

testDataFile = 'badges-testfeatures.csv'
testFile = open(testDataFile,'r')
testReader = csv.reader(testFile,delimiter=d)
trainList = []
testList = []
trainListLabels = []
testListLabels = []
indices = []
for row in trainReader:
	trainList.append([row[0],row[1],row[2],row[3]])
	trainListLabels.append(row[4])
#print len(trainList)

for row in testReader:
	testList.append([row[0],row[1],row[2],row[3]])
	testListLabels.append(row[4])
#print len(testList)

# Get the index of the 5-nearest neighbors

for i in range(0,len(testList)): 
	hammingDistance = findHammingDistance(trainList,testList[i])
	indices.append(findNnearestNeighbors(hammingDistance))

# For K = 1
labels1 = []
error = 0
for i in range(0,len(testList)):
	labels1.append(trainListLabels[indices[i][0]])
	if(labels1[i] != testListLabels[i]):
		error += 1
#print labels1
errorPercentage = float(error)/len(testList)
print "For k=1, Error is",errorPercentage

# For K = 2
labels2 = []
error = 0
for i in range(0,len(testList)):

	labels2.append(majorityLabel(indices[i][0:2],2))
	if(labels2[i] != testListLabels[i]):
		error += 1
errorPercentage = float(error)/len(testList)
print "For k=2, Error is",errorPercentage

# For K = 3
labels3 = []
error = 0
for i in range(0,len(testList)):

	labels3.append(majorityLabel(indices[i][0:3],3))
	if(labels3[i] != testListLabels[i]):
		error += 1
errorPercentage = float(error)/len(testList)
print "For k=3, Error is",errorPercentage

# For K = 4
labels4 = []
error = 0
for i in range(0,len(testList)):

	labels4.append(majorityLabel(indices[i][0:4],4))
	if(labels4[i] != testListLabels[i]):
		error += 1
errorPercentage = float(error)/len(testList)
print "For k=4, Error is",errorPercentage

# For K = 5
labels5 = []
error = 0
for i in range(0,len(testList)):

	labels5.append(majorityLabel(indices[i][0:5],5))
	if(labels5[i] != testListLabels[i]):
		error += 1
errorPercentage = float(error)/len(testList)
print "For k=5, Error is",errorPercentage


