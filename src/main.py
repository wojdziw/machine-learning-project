from parser import *
from Story import *
from sklearn import svm
import numpy as np
from sklearn import svm

from SVM import *

print ("1")

stories, dictionary = parseFile('train.txt')
trainData = np.zeros([1,len(dictionary)])
trainLabels = np.zeros(1)

print ("2")

for story in stories:
	points = story.constructPoints(dictionary)
	trainData = np.concatenate((trainData, points), axis = 0)
	labels = story.constructLabels(dictionary)
	trainLabels = np.concatenate((trainLabels, labels), axis = 0)

trainData = trainData[1:]
trainLabels = trainLabels[1:]

print ("3")

testStories, testDictionary = parseFile('test.txt', hasAnswers=False)
testData = np.zeros([1,len(dictionary)])

testData = testData[1:]

print ("4")

for story in testStories:
	points = story.constructPoints(dictionary)
	testData = np.concatenate((testData, points), axis = 0)

print ("5")

model = svmTrain(trainData, trainLabels)
# model = svm.SVR(C=1.0, epsilon=0.2)

print ("6")

pred = model.predict(testData)

print ("7")

for prediction in pred:
	if int(prediction) == -1:
		print "nothing"
	else:
		print dictionary[int(prediction)]

translator(pred, testStories, dictionary)

# for prediction in pred:
# 		wordIndexes = reverseUniqueMapping(int(prediction))
# 		for wordIndex in wordIndexes:
# 			if wordIndex == -1:
# 				print ("nothing")
# 			else:
# 				print (dictionary[wordIndex])
# 		print ("---")

