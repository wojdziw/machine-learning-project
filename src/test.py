import numpy as np
import matplotlib.pyplot as plt
# from SVM import *
from sklearn import svm
from constants import *
from utils import *
from MLPC import *
from TwoStagePredictor import *
from GBC import *
from MKNN import *
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from validation import validation


###############
# Change the lines in this block to tweak your test
cl = MLPC()

param_grid = {
'activation': ['relu'],
'alpha': [0.0001, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
# 'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,)]}
'hidden_layer_sizes': [(60,), (80,), (100,), (20, 20), (30, 30)]}

# Make sure you change the name below to sth meaningful, so we have a record
# of the models we have tried
clName = "mlpc_validated" # used to save the classifier
##############

print("Loading data from .npy files...")
data_um = np.load(TRAIN_DATA_UM_FILE)
data_bow = np.load(TRAIN_DATA_BOW_FILE)
data_bgr = np.load(TRAIN_DATA_BGR_FILE)
data_bgr_f = np.load(TRAIN_DATA_BGR_FOUND_FILE)
data_trgr_f = np.load(TRAIN_DATA_TRGR_FOUND_FILE)
labels = np.load(TRAIN_LABELS_FILE)
labelsAns = np.load(TRAIN_LABELS_ANS_FILE)
stories = np.load(TRAIN_STORIES_FILE).tolist()
dictionary = np.load(DICTIONARY_FILE).tolist()

# Train a classifier
print("Training classifier and running cross validation...")
# cl.fit(trainData_bow, trainLabels)
# cl.fit(trainData_um, trainLabels)
# cl.fit(trainData_bgr, trainLabels)
# cl.fit(trainData_bgr, trainLabelsAns)
# cl.fit(trainData_bgr_f, trainLabelsAns)
# cl.fit(trainData_trgr_f, trainLabelsAns)
cl = validation(cl, data_trgr_f, labelsAns, param_grid)
