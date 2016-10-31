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
BIGRAMS_FOUND_FILE = '../data/bigramsFound.npy'
TRIGRAMS_FOUND_FILE = '../data/trigramsFound.npy'
ANSWER_DICTIONARY_FILE = '../data/answer_dictionary.npy'

TRAIN_DATA_UM_FILE = '../data/train_um.npy'
TRAIN_DATA_BOW_FILE = '../data/train_bow.npy'
TRAIN_DATA_BGR_FILE = '../data/train_bgr.npy'
TRAIN_DATA_BGR_FOUND_FILE = '../data/train_bgr_found.npy'
TRAIN_DATA_TRGR_FOUND_FILE = '../data/train_trgr_found.npy'
TRAIN_STORIES_FILE = '../data/train_stories.npy'

TRAIN_LABELS_FILE = '../data/train_labels.npy'
TRAIN_LABELS_ANS_FILE = '../data/train_labels_ans.npy'

TEST_DATA_UM_FILE = '../data/test_um.npy'
TEST_DATA_BOW_FILE = '../data/test_bow.npy'
TEST_DATA_BGR_FILE = '../data/test_bgr.npy'
TEST_DATA_BGR_FOUND_FILE = '../data/test_bgr_found.npy'
TEST_DATA_TRGR_FOUND_FILE = '../data/test_trgr_found.npy'
TEST_STORIES_FILE = '../data/test_stories.npy'

ANSWERS_CSV_FILE = '../out/Answers_' + timestamp_str + '.csv'

################
# Other
################
NOTHING = 'nothing'
