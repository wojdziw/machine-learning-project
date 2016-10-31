from parser import *
from Story import *
import numpy as np
from constants import *
from utils import *
from MLPC import *
from validation import validation
###############
# Change the lines in this block to tweak your test

param_grid = {
    'activation': ['relu'],
    'alpha': [5.0, 6.0],
    'hidden_layer_sizes': [(10,)]}#, (20,), (30,), (40,), (50,), (60,)]}

cl = MLPC()

# Make sure you change the name below to sth meaningful, so we have a record
# of the models we have tried
clName = "mlpc_validated" # used to save the classifier

##############

print("Loading data from .npy files...")
trainData_trgr_f = np.load(TRAIN_DATA_TRGR_FOUND_FILE)
trainLabelsAns = np.load(TRAIN_LABELS_ANS_FILE)
answer_dictionary = np.load(ANSWER_DICTIONARY_FILE).tolist()

testData_trgr_f = np.load(TEST_DATA_TRGR_FOUND_FILE)
testStories = np.load(TEST_STORIES_FILE).tolist()

print("Training classifier, running cross-validation...")
cl = validation(cl, trainData_trgr_f, trainLabelsAns, param_grid)

print("Predicting...")
output = cl.predict(testData_trgr_f)

# save model
print("Saving model...")
np.save('../models/main_' + clName + "_" + timestamp_str, np.array([cl]))

print("Writing answers to csv...")
writeToCsv(output, testStories, answer_dictionary, ANSWERS_CSV_FILE)
