from parser import *
from Story import *
from sklearn import svm
import numpy as np
from constants import *
from utils import *
from MLPC import *
from TwoStagePredictor import *
from GBC import *

# cl = TwoStagePredictor()
cl = MLPC()


print("Loading data from .npy files...")
trainData_um = np.load(TRAIN_DATA_UM_FILE)
trainData_bow = np.load(TRAIN_DATA_BOW_FILE)
trainLabels = np.load(TRAIN_LABELS_FILE)
trainStories = np.load(TRAIN_STORIES_FILE).tolist()
dictionary = np.load(DICTIONARY_FILE).tolist()

testData_um = np.load(TEST_DATA_UM_FILE)
testData_bow = np.load(TEST_DATA_BOW_FILE)
testStories = np.load(TEST_STORIES_FILE).tolist()

print("Training classifier...")
cl.fit(trainData_bow, trainLabels)
# cl.fit(trainData_um, trainLabels)

print("Predicting...")
output = cl.predict(testData_bow)
# output = cl.predict(testData_um)

in_sample = cl.predict(trainData_bow)
# in_sample = cl.predict(trainData_um)

print("Accuracies in: [margin] [overall] [nothings] [multilabel] [singlelabel]")
print(mltBinaryAccuracy(trainLabels, in_sample))

# for i in range(20):
# 	print(i)
# 	print(list(zip(dictionary,output[i].tolist())))
#
# print("Writing answers to csv...")
# writeToCsv(output, testStories, dictionary, ANSWERS_CSV_FILE)
