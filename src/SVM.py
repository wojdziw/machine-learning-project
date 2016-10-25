from parser import *
from Story import *
from sklearn import svm
import numpy as np

def translator(pred, stories, dictionary):
	wordIndices = []
	for prediction in pred:
		wordIndices += [int(prediction)]
		# wordIndices += reverseUniqueMapping(int(prediction))
	mainString = ''
	counter = 0
	for i in range(len(stories)):
		print (stories[i].wordToNumHash)
		for j in range(len(stories[i].questionIndices)):
			string = str(i+1) + '_' + str(j+1) + ','
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
