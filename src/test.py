import numpy as np
import matplotlib.pyplot as plt
from SVM import *
from constants import *
from utils import *
from MLPC import *
from TwoStagePredictor import *
from GBC import *

###############
# Change the lines in this block to tweak your test

testStoriesCount = 600 # how many examples to leave for testing
cl = MLPC()
# cl = TwoStagePredictor()
# cl = GBC()1

##############

print("Loading data from .npy files...")
data_um = np.load(TRAIN_DATA_UM_FILE)
data_bow = np.load(TRAIN_DATA_BOW_FILE)
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
testData_um = data_um[:testQuestionCount, :]
testData_bow = data_bow[:testQuestionCount, :]
trainLabels = labels[testQuestionCount:, :]
testLabels = labels[:testQuestionCount, :]


# Train a classifier
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
print("Accuracies out: [margin] [overall] [nothings] [multilabel] [singlelabel]")
print(mltBinaryAccuracy(testLabels, output))
