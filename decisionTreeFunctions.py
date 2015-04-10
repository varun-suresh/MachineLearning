# This file contains the functions necessary to build a decision tree.
import numpy as np
import csv
import re
# This is a function that takes a table as its input and returns the 
# Input : Table containing the features as well as the labels. The last column of the table is assumed to be the labels.
# Output : The column number on which the feature decision tree must be split. This is the column which shows the maximum information gain
def informationGain_entropy(table):
#	print table
	no_cols = len(table)
	no_rows = len(table[0])
	# The last column is always the one which contains the labels
	# Use the last column to find the Entropy
	labels = table[no_cols-1] # The last column
	labels_list = list(set(labels))
	no_labels = len(labels_list)
#	print no_cols,no_rows,no_labels

	# Calculate the entropy of the set - i.e the entropy of the labels
	entropy_labels = 0
	for i in range(0,no_labels):
		p_label = float(labels.count(labels_list[i]))/len(labels)
		entropy_labels += -p_label * np.log2(p_label)
#	print entropy_labels  
	
	# Calculate the information gain for all the features and store it in a list
	informationGain_entropy = []
	for feature in range(0,no_cols-1):
		informationGain_feature = 0
		attribute = table[feature]
#		print attribute
		attributes_list = list(set(attribute))
#		print attributes_list
		no_attributes = len(attributes_list)
#		print no_attributes
	  	for k in range(0,no_attributes): # To find the information gain of each attribute in a feature
			entropy_attribute = 0
			for j in range(0,no_labels): # This loop is to find the entropy of each type of label.
				count = 0
				for i in range(0,len(attribute)):
					if (attribute[i] == attributes_list[k] and labels[i] == labels_list[j]) :
						count += 1
	#			print count
				p_attribute = float(count)/attribute.count(attributes_list[k])
				if (p_attribute > 0) :
					entropy_attribute += -p_attribute * np.log2(p_attribute)
	#	  	print entropy_attribute
		  	informationGain_feature += attribute.count(attributes_list[k])/float(len(attribute))*entropy_attribute
	  	informationGain_feature = entropy_labels - informationGain_feature
		informationGain_entropy.append(informationGain_feature)
#	print informationGain
#	print informationGain.index(max(informationGain))
	if max(informationGain_entropy) > 0 :
		return informationGain_entropy.index(max(informationGain_entropy))
	else :
		return None


#############################################################################################################################################

# A function that creates a subset of a table based on certain conditions
def getRemainder(listOfTables,tableIndex,splitColNo):
	table = listOfTables[tableIndex]
	parentsChildren = []
	parentsChildren.append(tableIndex)
	attributesOrLabel = []
#	print isTableLeaf(table)
	if (isTableLeaf(table) == False) :
		# Count the number of attributes the feature can take in the column to be split.
		attributes = list(set(table[splitColNo]))
#		print attributes
		attributesOrLabel.append(attributes)
		no_cols = len(table)
		no_rows = len(table[0])
		no_attributes = len(attributes)
#		print no_attributes
		for no_of_tables in range(0,len(attributes)):
			new_table = []
			for cols in range(0,no_cols):
				new_col = []
				for rows in range(0,no_rows):
					new_element = []
					if(table[splitColNo][rows] == attributes[no_of_tables]):
						new_element = table[cols][rows]					
					if new_element :
						new_col.append(new_element)					
				if new_col :
					new_table.append(new_col)
#			print new_table
			listOfTables.append(new_table)
#			print len(listOfTables)
			parentsChildren.append(len(listOfTables)-1)
#			print parentsChildren
	else :
		labels = list(table[-1])
		labelsItems = list(set(labels))	
		if len(labelsItems) > 1:
			for i in range(0,len(labelsItems)):
				maxCount = 0
				temp = labels.count(labelsItems[i])
				if (temp > maxCount):
					maxCount = temp
					maxCountIndex = i
			finalLabel = labelsItems[maxCountIndex]
		else :
			finalLabel = labelsItems[0]
#		print finalLabel
		attributesOrLabel.append(finalLabel)
#	print splitColNo
#	print len(listOfTables)
	return listOfTables,parentsChildren,attributesOrLabel
	
	
#############################################################################################################################################
# To check if a table is a leaf node
# Input : Table containing the features
# Conditions to be checked :
# 1) If all the labels are the same.
# 2) If all the attributes have been checked.
def isTableLeaf(table):
	no_cols = len(table)
	labels_list = list(set(table[no_cols-1]))
	featureLength = []
	for i in range(0,no_cols-1):
		featureLength.append(len(list(set(table[i]))))
	if len(labels_list) == 1 :
		return True
	elif max(featureLength) == 1 :
		return True
	elif informationGain_entropy(table) == None :
		return True
	elif informationGain_majorityError(table) == None :
		return True
	else :
		return False
	
#############################################################################################################################################	
# To check if we have gone through all the tables
def isEndOfTables(listOfTables,tableIndex):
	if len(listOfTables)-1 == tableIndex :
		return True
	else :
		return False
#############################################################################################################################################
# This is a function that takes a table as its input and returns the 
# Input : Table containing the features as well as the labels. The last column of the table is assumed to be the labels.
# Output : The column number on which the feature decision tree must be split. This is the column which shows the maximum information gain
def informationGain_majorityError(table):
#	print table
	no_cols = len(table)
	no_rows = len(table[0])
	# The last column is always the one which contains the labels
	# Use the last column to find the Entropy
	labels = table[no_cols-1] # The last column
	labels_list = list(set(labels))
	no_labels = len(labels_list)
#	print no_cols,no_rows,no_labels

	# Calculate the majority error of the set - i.e the majority error of the labels
	
	p_label = float(labels.count(labels_list[0]))/len(labels)
	majorityError_labels = 1 - max(p_label,1-p_label)
#	print entropy_labels  
	
	# Calculate the information gain for all the features and store it in a list
	informationGain_majorityError = []
	for feature in range(0,no_cols-1):
		informationGain_feature = 0
		attribute = table[feature]
#		print attribute
		attributes_list = list(set(attribute))
#		print attributes_list
		no_attributes = len(attributes_list)
#		print no_attributes
	  	for k in range(0,no_attributes): # To find the information gain of each attribute in a feature
#			entropy_attribute = 0
			for j in range(0,no_labels): # This loop is to find the entropy of each type of label.
				count = 0
				for i in range(0,len(attribute)):
					if (attribute[i] == attributes_list[k] and labels[i] == labels_list[j]) :
						count += 1
	#			print count
				p_attribute = float(count)/attribute.count(attributes_list[k])
#				print p_attribute
				majorityError_attribute = 1 - max(p_attribute,1-p_attribute)
#				if (p_attribute > 0) :
#					entropy_attribute += -p_attribute * np.log2(p_attribute)
	#	  	print entropy_attribute
		  	informationGain_feature += attribute.count(attributes_list[k])/float(len(attribute)) * majorityError_attribute
	  	informationGain_feature = majorityError_labels - informationGain_feature
		informationGain_majorityError.append(informationGain_feature)
#	print informationGain_majorityError
#	print informationGain.index(max(informationGain))
	if max(informationGain_majorityError) > 0 :
		return informationGain_majorityError.index(max(informationGain_majorityError))
	else :
		return None
