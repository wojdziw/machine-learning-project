from parser import *
from Story import *
from sklearn import svm
import numpy as np
from SVM import *

print ("1")

stories, dictionary = parseFile('train.txt')
allPoints = np.zeros([1,len(dictionary)])
allLabels = np.zeros(1)

print ("2")

for story in stories:
	points = story.constructPoints(dictionary)
	allPoints = np.concatenate((allPoints, points), axis = 0)
	labels = story.constructLabels(dictionary)
	allLabels = np.concatenate((allLabels, labels), axis = 0)

allPoints = allPoints[1:]
allLabels = allLabels[1:]

print ("3")

testStories, testDictionary = parseFile('test.txt')
testData = np.zeros([1,len(dictionary)])

testData = testData[1:]

print ("4")

for story in testStories:
	points = story.constructPoints(dictionary)
	testData = np.concatenate((testData, points), axis = 0)

print ("5")

# model = svmTrain(allPoints, allLabels)

# print ("8")

# pred = model.predict(allPoints)

# print ("9")

# # translator(pred)

# print ("10")

# for prediction in pred:
# 		wordIndexes = reverseUniqueMapping(int(prediction))
# 		for wordIndex in wordIndexes:
# 			if wordIndex == -1:
# 				print ("nothing")
# 			else:
# 				print (dictionary[wordIndex])
# 		print ("---")

