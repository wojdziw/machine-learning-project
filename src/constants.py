'''
    Lists constants to be used throughout our source code
'''
import time
import datetime

timestamp_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M')

###############
# Filenames
###############
KAGGLE_TRAIN_TXT_FILE = '../data/kaggle/train_kaggle.txt'
KAGGLE_TEST_TXT_FILE = '../data/kaggle/test_kaggle.txt'

DICTIONARY_FILE = '../data/dictionary.npy'
BIGRAMS_FILE = '../data/bigrams.npy'

TRAIN_DATA_UM_FILE = '../data/train_um.npy'
TRAIN_DATA_BOW_FILE = '../data/train_bow.npy'
TRAIN_DATA_BGR_FILE = '../data/train_bgr.npy'
TRAIN_DATA_BGR_ORDER_FILE = '../data/train_bgr.npy'
TRAIN_STORIES_FILE = '../data/train_stories.npy'

TRAIN_LABELS_FILE = '../data/train_labels.npy'

TEST_DATA_UM_FILE = '../data/test_um.npy'
TEST_DATA_BOW_FILE = '../data/test_bow.npy'
TEST_DATA_BGR_FILE = '../data/test_bgr.npy'
TEST_DATA_BGR_ORDER_FILE = '../data/test_bgr.npy'
TEST_STORIES_FILE = '../data/test_stories.npy'

ANSWERS_CSV_FILE = '../out/Answers_' + timestamp_str + '.csv'

################
# Other
################
NOTHING = 'nothing'
