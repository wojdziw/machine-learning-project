from parser import *
from sklearn import svm
import numpy as np

print "1"

stories, dictionary = parseFile('train.txt')
allPoints = np.zeros([1,len(dictionary)])
allLabels = np.zeros(1)

print "2"

for story in stories:
	points = story.constructPoints(dictionary)
	allPoints = np.concatenate((allPoints, points), axis = 0)
	labels = story.constructLabels(dictionary)
	allLabels = np.concatenate((allLabels, labels), axis = 0)

allPoints = allPoints[1:]
allLabels = allLabels[1:]

print "3"

testStories, testDictionary = parseFile('train.txt')
testData = np.zeros([1,len(dictionary)])
testLabels = np.zeros(1)

testData = testData[1:]
testLabels = testLabels[1:]

print "4"

for story in testStories:
	points = story.constructPoints(dictionary)
	testData = np.concatenate((testData, points), axis = 0)
	labels = story.constructLabels(dictionary)
	testLabels = np.concatenate((testLabels, labels), axis = 0)

print "5"

# HERE RETURN THE THINGS IN THE REQUIRED KAGGLE FORMAT
def SVMPredict(data, label, model):
	pred = model.predict(data)
	E = 0.00
	N = len(pred)
	for i in range(N):
		k = abs((label[i]-pred[i]))/(2.0)
		E += k
	return (1-1.00/N*E)
	return 0

print "6"


def svmTrain(dataTrain, labelTrain, *args):
	C = 1.0
	kernel = 'rbf'
	degree = 3
	gamma = 'auto'
	coef0 = 0.0

	if args:
		C = args[0]
		kernel = args[1]
		degree = args[2]
		gamma = args[3]
		coef0 = args[4]

	# Uncomment if regression
	# svc = svm.SVR(C, epsilon=0.2)
	svc = svm.SVC(C, kernel, degree, gamma, coef0)
	svc.fit(dataTrain, labelTrain)
	return svc

print "7"

model = svmTrain(allPoints, allLabels)

print "8"

pred = model.predict(testData)

for prediction in pred:
		if prediction == -1:
			word = "nothing"
		else:
			word = dictionary[int(prediction)]
		print word
		print prediction