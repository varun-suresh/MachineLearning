# Homework -1
# Machine Learning - CS 6350

############################################################################################################################################
import csv
import decisionTreeFunctions as dtf

# Main Program :

# Open the file and find out the number of rows and columns.
datafilename = 'badges-trainFeatures.csv'
d = ',' # Delimiter
f=open(datafilename,'r')
reader=csv.reader(f,delimiter=d)
col_count=len(next(reader)) # Read first line and count columns
f.seek(0)        
#print col_count
row_count = sum(1 for row in f)  
#print row_count
f.seek(0)

# Convert the table into a list of lists.

listOfTables = []
attribute_table = []
for i in range(0,col_count):
#	print i
#	print col_number
	column = []
	for row in reader:
		column.append(row[i])
	f.seek(0)
	attribute_table.append(column)

listOfTables.append(attribute_table)
tableIndex = 0
isEndOfTable = False
splitColumnNo = []
parentsChildren = []
attributesOrLabel = []
# Recursively build the tree by updating the list of tables :
while(isEndOfTable == False):
#	splitColumnNo.append(dtf.informationGain_majorityError(listOfTables[tableIndex]))
	splitColumnNo.append(dtf.informationGain_entropy(listOfTables[tableIndex]))
#	print splitColumnNo
#	print len(listOfTables),tableIndex
#	parentsChildren.append(tableIndex)
	listOfTables,temp,att = dtf.getRemainder(listOfTables,tableIndex,splitColumnNo[tableIndex])
	parentsChildren.append(temp)
	attributesOrLabel.append(att)
	isEndOfTable = dtf.isEndOfTables(listOfTables,tableIndex)
	tableIndex += 1
print "The decision tree based on Entropy measure is"
print parentsChildren
print "The corresponding labels are :"
print attributesOrLabel
	
#splitColumnNo.append(dtf.informationGain(listOfTables[tableIndex]))
#listOfTables = dtf.getRemainder(listOfTables,tableIndex,splitColumnNo[tableIndex])
#print len(listOfTables)
#print listOfTables
#print splitColumnNo

# To test the decision tree :
# Steps
# 1)Read a table one row at a time
# 2)Look at the splitColumnNo list,parentsChildren and attributesOrLabel to decide what next.
# 3)Do this recursively till a Label is reached.

datafilename = 'badges-testfeatures.csv'
d = ',' # Delimiter
f=open(datafilename,'r')
reader=csv.reader(f,delimiter=d)
testLabels = []
trueLabels = []
for row in reader :
	#print row
	trueLabels.append(row[-1])
	i = 0
	while len(parentsChildren[i]) != 1 :
		attribute = row[splitColumnNo[i]]
	#	print attribute
	#	print attributesOrLabel[i][0].index(attribute)
		k = attributesOrLabel[i][0].index(attribute) + 1
		i = parentsChildren[i][k]
#	print attributesOrLabel[i][0]
	testLabels.append(attributesOrLabel[i][0])
#print len(testLabels)
#print len(trueLabels)
 
# Check the error rate :
errorCount = 0
for i in range(0,len(trueLabels)):
	if trueLabels[i] != testLabels[i] :
		errorCount += 1
errorPercentage = float(errorCount)/len(trueLabels)
print "Error Percentage is",errorPercentage
