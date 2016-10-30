import Story
from constants import *
import numpy as np
from utils import generateAllPossibleBigrams

def parseFile(filename, hasAnswers = True):
    '''
        filename - a name of a file that is formatted as per spec
        hasAnswers - Are there answers to the questions in the file
        returns:
            stories - a list of Story objects - one for each set of statements
            dictionary - alphabetically sorted list of all the words in the text
    '''
    dictionary = set()
    labels = set()
    stories = []
    currentPoint = Story.Story()
    with open(filename, "r") as ins:
        for line in ins:
            # split on whitespace, convert to lowercase
            lst = list(map(lambda x: x.lower(), line.strip().split()))

            # When we see an index of 1, we push the current point and reset it
            if int(lst[0]) == 1:
                stories.append(currentPoint)
                currentPoint = Story.Story()

            # check if the sentence is a statement
            if lst[-1][-1] == '.':
                lst[-1] = lst[-1][:-1]  # remove punct. mark
                currentPoint.addStatement(lst[1:])
            else:
                answers = []
                if hasAnswers:
                    answers = lst[-1].split(',')
                    lst = lst[:-1]
                lst[-1] = lst[-1][:-1]  # remove question mark
                for a in answers:
                    labels.add(a)
                currentPoint.addQuestion(lst[1:], answers)

            # add words to dictionary
            for w in lst[1:]:
                dictionary.add(w)

        stories.append(currentPoint)
        dictionary = list(dictionary)
        dictionary.sort()
        labels = list(labels)
        labels.sort()
        # Put 'nothing at posisiton 0'
        labels.remove(NOTHING)
        labels = [NOTHING] + labels

        bigrams = generateAllPossibleBigrams(dictionary)
        bigrams.sort()
        dictionary = ["nothing"] + dictionary
        return stories[1:], dictionary, bigrams, labels

def parseAll():
    print ("Parsing the train data...")

    stories, dictionary, bigrams, ansLabels = parseFile(KAGGLE_TRAIN_TXT_FILE)
    trainData_um = np.zeros([1,len(dictionary) - 1]) # with uniqueMapping
    trainData_bow = np.zeros([1,len(dictionary) - 1]) # as bag-of-words
    trainData_bgr = np.zeros([1,len(bigrams)]) # as bag-of-words
    trainLabels = np.zeros([1, len(dictionary)])
    trainLabelsAns = np.zeros([1, len(ansLabels)])

    print ("Producing the train data points and labels...")

    # Make points and labels for each
    for story in stories:
        points_um = story.constructPoints(dictionary[1:], bagOfWords=False)
        points_bow = story.constructPoints(dictionary[1:], bagOfWords=True)
        points_bgr = story.constructBigramPoints(bigrams)
        trainData_um = np.concatenate((trainData_um, points_um), axis = 0)
        trainData_bow = np.concatenate((trainData_bow, points_bow), axis = 0)
        trainData_bgr = np.concatenate((trainData_bgr, points_bgr), axis = 0)
        labels = story.constructBinaryLabels(dictionary)
        labelsAns = story.constructBinaryLabels(ansLabels)
        trainLabels = np.concatenate((trainLabels, labels), axis = 0)
        trainLabelsAns = np.concatenate((trainLabelsAns, labelsAns), axis = 0)

    trainData_um = trainData_um[1:]
    trainData_bow = trainData_bow[1:]
    trainData_bgr = trainData_bgr[1:]
    trainLabels = trainLabels[1:]
    trainLabelsAns = trainLabelsAns[1:]

    print ("Parsing the test data...")

    testStories, _, _, _ = parseFile(KAGGLE_TEST_TXT_FILE, hasAnswers=False)
    testData_um = np.zeros([1,len(dictionary) - 1])
    testData_bow = np.zeros([1,len(dictionary) - 1])
    testData_bgr = np.zeros([1,len(bigrams)])

    print ("Producing the test data points and labels...")

    for story in testStories:
        points_um = story.constructPoints(dictionary[1:], bagOfWords=False)
        points_bow = story.constructPoints(dictionary[1:], bagOfWords=True)
        points_bgr = story.constructBigramPoints(bigrams)
        testData_um = np.concatenate((testData_um, points_um), axis = 0)
        testData_bow = np.concatenate((testData_bow, points_bow), axis = 0)
        testData_bgr = np.concatenate((testData_bgr, points_bgr), axis = 0)

    testData_um = testData_um[1:]
    testData_bow = testData_bow[1:]
    testData_bgr = testData_bgr[1:]


    # Save everything to files
    print("Saving parsed data to .npy files")
    np.save(DICTIONARY_FILE, dictionary)
    np.save(ANSWER_DICTIONARY_FILE, ansLabels)
    np.save(BIGRAMS_FILE, bigrams)

    np.save(TRAIN_DATA_UM_FILE, trainData_um)
    np.save(TRAIN_DATA_BOW_FILE, trainData_bow)
    np.save(TRAIN_DATA_BGR_FILE, trainData_bgr)
    np.save(TRAIN_LABELS_FILE, trainLabels)
    np.save(TRAIN_LABELS_ANS_FILE, trainLabelsAns)
    np.save(TRAIN_STORIES_FILE, np.array(stories))

    np.save(TEST_DATA_UM_FILE, testData_um)
    np.save(TEST_DATA_BOW_FILE, testData_bow)
    np.save(TEST_DATA_BGR_FILE, testData_bgr)
    np.save(TEST_STORIES_FILE, np.array(testStories))

# parseAll()
