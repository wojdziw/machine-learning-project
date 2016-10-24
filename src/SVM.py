from parser import *
from Story import *
from sklearn import svm
import numpy as np

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

testStories, testDictionary = parseFile('train.txt')
testData = np.zeros([1,len(dictionary)])
testLabels = np.zeros(1)

testData = testData[1:]
testLabels = testLabels[1:]

print ("4")

for story in testStories:
	points = story.constructPoints(dictionary)
	testData = np.concatenate((testData, points), axis = 0)
	labels = story.constructLabels(dictionary)
	testLabels = np.concatenate((testLabels, labels), axis = 0)

print ("5")

def translator(pred):
	wordIndices = []
	for prediction in pred:
		wordIndices += reverseUniqueMapping(int(prediction))
	mainString = ''
	counter = 0
	for i in range(len(stories)):
		print (stories[i].wordToNumHash)
		for j in range(len(stories[i].questionIndices)):
			string = str(i) + '_' + str(j) + ','
			answerNum = wordIndices[counter]
			answerWord = dictionary[answerNum]
			if answerNum == -1:
				answerWord = 'nothing'
			print(answerWord)
			if answerWord not in stories[i].wordToNumHash.keys():
				string += '0' + '\n'
			else:
				string += str(stories[i].wordToNumHash[answerWord]) + '\n'
			counter += 1
			mainString += string
	f = open('Answers.txt', 'w')
	f.write(mainString)
	f.close()

def SVMPredict(data, label, model):
	pred = model.predict(data)
	E = 0.00
	N = len(pred)
	for i in range(N):
		k = abs((label[i]-pred[i]))/(2.0)
		E += k
	return (1-1.00/N*E)
	return 0

print ("6")


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
	# svc = svm.SVR(C, kernel, degree, gamma, coef0)
	svc = svm.SVC(C, kernel, degree, gamma, coef0)
	svc.fit(dataTrain, labelTrain)
	return svc

print ("7")

model = svmTrain(allPoints, allLabels)

print ("8")

pred = model.predict(testData)

print ("9")

translator(pred)


'''
for prediction in pred:
		wordIndexes = reverseUniqueMapping(int(prediction))
		for wordIndex in wordIndexes:
			if wordIndex == -1:
				print ("nothing")
			else:
				print (dictionary[wordIndex])

		print ("---")

'''