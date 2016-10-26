from parser import *
from Story import *
from sklearn import svm
import numpy as np
from sklearn.metrics import f1_score

def translator(pred, stories, dictionary):
	wordIndices = []
	for prediction in pred:
		if int(prediction) > 32:
			multWords = reverseUniqueMapping(prediction)
			wordIndices.append(multWords)
		else:
			wordIndices.append([int(prediction)])
	mainString = 'textID,sortedAnswerList\n'
	counter = 0
	for i in range(len(stories)):
		for j in range(len(stories[i].questionIndices)):
			string = str(i+1) + '_' + str(j+1) + ','
			currentWordIndices = wordIndices[counter]
			if currentWordIndices[0] == -1:
				string += '-1'
			else:
				currentStoryWordIndices = []
				for wordIndex in currentWordIndices:
					word = dictionary[wordIndex]
					currentStoryWordIndices.append(stories[i].wordToNumHash[word])
				currentStoryWordIndices.sort()
				for currentStoryWordIndex in currentStoryWordIndices:
					string += str(currentStoryWordIndex) + ' '

			if string[len(string)-1] == " ":
				string = string[0:len(string)-1]

			string += '\n'
			counter += 1
			mainString += string
	f = open('Answers.csv', 'w')
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

def svmTrain(data, labels, *args):
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
	# svc = svm.SVR(C=1.0, epsilon=0.2)
	svc = svm.SVC(C, kernel, degree, gamma, coef0)
	svc.fit(data, labels)
	return svc

def binaryAccuracy(labels, predictions):
	correct = 0.0
	for i in range(len(labels)):
		if labels[i] == predictions[i]:
			correct += 1
	return correct/len(labels)

def mltBinaryAccuracy(labels, predictions):
	'''
		Gives us overall acc as well as acc on different types of quesitons.
		labels[0] and predictions[0] correspond to 'nothing'
	'''
	correct = 0.0
	nothings = 0.0
	nothings_c = 0.0
	multilabel = 0.0
	multilabel_c = 0.0
	singlelabel = 0.0
	singlelabel_c = 0.0
	for i in range(labels.shape[0]):

		isCorrect =  np.all(labels[i] == predictions[i])
		if isCorrect:
			correct += 1
		if labels[i][0] > 0:
			nothings += 1
			if isCorrect:
				nothings_c += 1
		elif np.sum(labels[i]) > 1.5:
			multilabel += 1
			if isCorrect:
				multilabel_c += 1
		else: # single
			singlelabel += 1
			if isCorrect:
				singlelabel_c += 1
	nothing_acc = nothings_c / nothings
	multilabel_acc = multilabel_c / multilabel
	singlelabel_acc = singlelabel_c / singlelabel
	if nothings + multilabel + singlelabel != labels.shape[0]:
		print("WTFFFF: ", str(labels.shape[0] - nothings - multilabel - singlelabel))
	return correct / len(labels), nothing_acc, multilabel_acc, singlelabel_acc

def f1Accuracy(labels, predictions):
	return f1_score(labels, predictions, average='macro')
