from parser import *
from Story import *
from sklearn import svm
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from SVM import *

print ("Parsing the train data")

stories, dictionary = parseFile('train.txt')
trainData = np.zeros([1,len(dictionary)])
trainLabels = np.zeros(1)

print ("Producing the train data points and labels")

for story in stories:
	points = story.constructPoints(dictionary)
	trainData = np.concatenate((trainData, points), axis = 0)
	labels = story.constructLabels(dictionary)
	trainLabels = np.concatenate((trainLabels, labels), axis = 0)

trainData = trainData[1:]
trainLabels = trainLabels[1:]

print ("Parsing the test data")

testStories, testDictionary = parseFile('test.txt', hasAnswers=False)
testData = np.zeros([1,len(dictionary)])

testData = testData[1:]

print ("Producing the test data points and labels")

for story in testStories:
	points = story.constructPoints(dictionary)
	testData = np.concatenate((testData, points), axis = 0)

print ("Initialising the SVMs")

# SVMs
modelR = svm.SVR(C=1.0, epsilon=0.2)
modelC = svm.SVC()

# print ("Fitting the regression SVM")
print ("Fitting the classification SVM")
modelC.fit(trainData, trainLabels)

print ("Predicting the labels")
predC = modelC.predict(testData)

# print "SVM Binary accuracy is:" + str(binaryAccuracy(predC,trainLabels))

# Neural net
'''
net = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
net.fit(trainData, trainLabels)
predNet = net.predict(trainData)
predProb = net.predict_proba(trainData)
print "Neural Net Binary accuracy is:" + str(binaryAccuracy(predNet,trainLabels))
'''

# Random forest
'''
rf = RandomForestClassifier(n_estimators=10)
rf.fit(trainData, trainLabels)
predRf = rf.predict(trainData)
print "Neural Net Binary accuracy is:" + str(binaryAccuracy(predRf,trainLabels))
'''

'''
print ("Printing the results")
for prediction in predC:
	prediction -= 10
	if int(prediction) == -1:
		print "nothing"
	elif int(prediction) > 32:
		inds = reverseUniqueMapping(prediction+10)
		words = ''
		for ind in inds:
			words = words + dictionary[ind-10] + ', '
		print words
	else:
		print dictionary[int(prediction)]
'''


print ("Saving this into a csv")
translator(predC, testStories, dictionary)
