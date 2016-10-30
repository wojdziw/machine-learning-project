import numpy as np
import matplotlib.pyplot as plt
from SVM import *
from sklearn import svm
from constants import *
from utils import *
from MLPC import *
from TwoStagePredictor import *
from GBC import *
from MKNN import *
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

###############
# Change the lines in this block to tweak your test

testStoriesCount = 600 # how many examples to leave for testing
cl = MLPC()
# cl = MKNN()
# cl = MultiOutputClassifier(svm.SVC())
# cl = TwoStagePredictor(nulls_policy='return_nothing')
# cl = TwoStagePredictor()
# cl = GBC()1

# Make sure you change the name below to sth meaningful, so we have a record
# of the models we have tried
clName = "mlpc_scale_100_a10e-2_bgr" # used to save the classifier
##############

print("Loading data from .npy files...")
data_um = np.load(TRAIN_DATA_UM_FILE)
data_bow = np.load(TRAIN_DATA_BOW_FILE)
data_bgr = np.load(TRAIN_DATA_BGR_FILE)
labels = np.load(TRAIN_LABELS_FILE)
stories = np.load(TRAIN_STORIES_FILE).tolist()
dictionary = np.load(DICTIONARY_FILE).tolist()

# Split the the data and labels into train and test
print("Splitting data...")
testStories = stories[:testStoriesCount]
testQuestionCount = 0 # Count the number of questions in the test stories
for s in testStories:
    testQuestionCount += len(s.questionIndices)
trainData_um = data_um[testQuestionCount:, :]
trainData_bow = data_bow[testQuestionCount:, :]
trainData_bgr = data_bgr[testQuestionCount:, :]
testData_um = data_um[:testQuestionCount, :]
testData_bow = data_bow[:testQuestionCount, :]
testData_bgr = data_bgr[:testQuestionCount, :]
trainLabels = labels[testQuestionCount:, :]
testLabels = labels[:testQuestionCount, :]


# Train a classifier
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


for i in range(output.shape[0]):
    assert np.sum(output[i]) > 0, "Nothing is predicted for question " + str(i+1)

# save model
print("Saving model...")
np.save('../models/test_' + clName + "_" + timestamp_str, np.array([cl]))

print("Accuracies in: [overall] [nothings] [multilabel] [singlelabel]")
print(mltBinaryAccuracy(trainLabels, in_sample))
print("Accuracies out: [overall] [nothings] [multilabel] [singlelabel]")
print(mltBinaryAccuracy(testLabels, output))
# print("F1 score: " + str(f1_score(testLabels, output, average='macro')))
