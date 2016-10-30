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
clName = "mlpc_bgr_hlz12_а10е-4" # used to save the classifier


print("Loading data from .npy files...")
trainData_um = np.load(TRAIN_DATA_UM_FILE)
trainData_bow = np.load(TRAIN_DATA_BOW_FILE)
trainData_bgr = np.load(TRAIN_DATA_BGR_FILE)
trainLabels = np.load(TRAIN_LABELS_FILE)
trainStories = np.load(TRAIN_STORIES_FILE).tolist()
dictionary = np.load(DICTIONARY_FILE).tolist()

testData_um = np.load(TEST_DATA_UM_FILE)
testData_bow = np.load(TEST_DATA_BOW_FILE)
testData_bgr = np.load(TEST_DATA_BGR_FILE)
testStories = np.load(TEST_STORIES_FILE).tolist()

print("Training classifier...")
# cl.fit(trainData_bow, trainLabels)
# cl.fit(trainData_um, trainLabels)
cl.fit(trainData_bgr, trainLabels)

print("Predicting...")
# output = cl.predict(testData_bow)
# output = cl.predict(testData_um)
output = cl.predict(testData_bgr)

# in_sample = cl.predict(trainData_bow)
# in_sample = cl.predict(trainData_um)
in_sample = cl.predict(trainData_bgr)

print("Accuracies in: [margin] [overall] [nothings] [multilabel] [singlelabel]")
print(mltBinaryAccuracy(trainLabels, in_sample))
#
# for i in range(20):
# 	print(i)
# 	print(list(zip(dictionary,output[i].tolist())))

# save model
print("Saving model...")
np.save('../models/main_' + clName + "_" + timestamp_str, np.array([cl]))

print("Writing answers to csv...")
writeToCsv(output, testStories, dictionary, ANSWERS_CSV_FILE)
