from parser import *
from sklearn import svm


stories, dictionary = parseFile('train.txt')
allPoints = np.zeros([1,len(dictonary)])
allLabels = np.zeros(1)


for story in stories:
	points = stories.constructPoints(dictionary)
	allPoints.concatenate((points,allPoints), axis = 0)
	labels = stories.constructLabels(dictionary)
	allPoints.concatenate((points,allPoints), axis = 0)


def SVMPredict(data, label, model):
	pred = model.predict(data)
	E = 0.00
	N = len(pred)
	for i in range(N):
		k = abs((label[i]-pred[i]))/(2.0)
		E += k
	return (1-1.00/N*E)


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

	svc = svm.SVC(C, kernel, degree, gamma, coef0)
	svc.fit(dataTrain, labelTrain)
	return svc

model = svmTrain(allPoints, allLabels)
#pred = model.predict(testData)