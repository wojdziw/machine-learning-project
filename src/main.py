from parser import *
from Story import *
from sklearn import svm
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from SVM import *
import sys

if len(sys.argv) < 3:
	print("Correct usage: python main.py <path_to_train_data> <path_to_test_data>")
	sys.exit(1)

trainFile = sys.argv[1]
testFile = sys.argv[2]

# print ("Parsing the train data")
#
# stories, dictionary = parseFile(trainFile)
# trainData = np.zeros([1,len(dictionary)])
# trainLabels = np.zeros(1)
#
# print ("Producing the train data points and labels")
#
# for story in stories:
# 	points = story.constructPoints(dictionary)
# 	trainData = np.concatenate((trainData, points), axis = 0)
# 	labels = story.constructLabels(dictionary)
# 	trainLabels = np.concatenate((trainLabels, labels), axis = 0)
#
# trainData = trainData[1:]
# trainLabels = trainLabels[1:]

# print ("Parsing the test data")
#
# testStories, testDictionary = parseFile(testFile, hasAnswers=False)
# testData = np.zeros([1,len(dictionary)])
#
# testData = testData[1:]
#
# print ("Producing the test data points and labels")
#
# for story in testStories:
# 	points = story.constructPoints(dictionary)
# 	testData = np.concatenate((testData, points), axis = 0)

print ("Parsing the train data")

stories, dictionary = parseFile(trainFile)
trainData = np.zeros([1,len(dictionary)])
trainLabels = np.zeros([1, len(dictionary)])

print ("Producing the train data points and labels")

for story in stories:
	points = story.constructPoints(dictionary)
	trainData = np.concatenate((trainData, points), axis = 0)
	labels = story.constructBinaryLabels(dictionary)
	trainLabels = np.concatenate((trainLabels, labels), axis = 0)

trainData = trainData[1:]
trainLabels = trainLabels[1:]

print ("Parsing the test data")

testStories, testDictionary = parseFile(testFile, hasAnswers=True)
testData = np.zeros([1,len(dictionary)])
testLabels = np.zeros([1, len(dictionary)])

print ("Producing the test data points and labels")

for story in testStories:
	points = story.constructPoints(dictionary)
	labels = story.constructBinaryLabels(dictionary)
	testData = np.concatenate((testData, points), axis = 0)
	testLabels = np.concatenate((testLabels, labels), axis = 0)

testData = testData[1:]
testLabels = testLabels[1:]
print(testData.shape)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

mor = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(trainData, trainLabels)
output = mor.predict(testData)

# np.save("output", output)
# np.save("testLabels", testLabels)

outLabels = np.zeros(output.shape)

'''
	Give our final classifications depending on our real-valued data
	If we've predicted 'nothing' to be most probable, we output 'nothing'
	Else, we ouput the highest scoring label and everything that is within
	98% of it
'''
for i in range(output.shape[0]):
	p = output[i]
	maxCoef = np.max(p)
	for j in range(output.shape[1]):
		coef = p[j]
		if dictionary[j] == 'nothing' and coef == maxCoef:
			outLabels[i, j] = 1
			break
		if maxCoef - coef < 0.02 * maxCoef: # The real value here is important
			outLabels[i, j] = 1

# for i in range(20):
# 	print("Question " + str(i + 1))
# 	print("--------")
# 	print(list(zip(dictionary, (list(zip(output[i], outLabels[i], testLabels[i]))))))
# 	print("==========================")


# for i in range(5):
# 	print(list(zip(dictionary, (list(zip(outLabels[i], testLabels[i]))))))
# 	print("-------------------")

acc = mltBinaryAccuracy(outLabels, testLabels)
print(acc)

#
# print ("Initialising the SVMs")
#
# # SVMs
# modelC = svm.SVC()
#
# # print ("Fitting the regression SVM")
# print ("Fitting the classification SVM")
# modelC.fit(trainData, trainLabels)
#
# print ("Predicting the labels")
# predC = modelC.predict(testData)
#
# print("SVM Binary accuracy is:" + str(binaryAccuracy(predC, testLabels)))
#
# # Neural net
# '''
# net = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
# net.fit(trainData, trainLabels)
# predNet = net.predict(trainData)
# predProb = net.predict_proba(trainData)
# print "Neural Net Binary accuracy is:" + str(binaryAccuracy(predNet,trainLabels))
# '''

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


# print ("Saving this into a csv")
# translator(predC, testStories, dictionary)
