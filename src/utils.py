from Story import *

def uniqueMapping(indexes):

    if len(indexes) > 5:
        indexes = indexes[0:5]

    if len(indexes) == 0:
        a = 0
        b = 0
    elif indexes[0] == -1:
        return -1
    elif len(indexes) == 1:
        return indexes[0]
        a = indexes[0]
        b = 0
    elif len(indexes) == 2:
        a = indexes[0]
        b = indexes[1]
    else:
        a = indexes[0]
        b = uniqueMapping(indexes[1:])

    output = cantorMapping(a,b)

    return output

def reverseUniqueMapping(mapping):

    if mapping == -1:
        return [-1]

    a, b = reverseCantorMapping(mapping)
    indexes = [int(a)]

    # threshold - basically the number of elements in the array
    threshold = 32

    if b > threshold:
        indexes += reverseUniqueMapping(b)
    else:
        indexes.append(int(b))

    # if a == 0:
    #     return [int(b)]

    return indexes

def cantorMapping(a,b):
    output = (a+b)*(a+b+1)/2+b

    return output

def reverseCantorMapping(mapping):
    w = math.floor((math.sqrt(8*mapping+1)-1)/2)
    t = w*(w+1)/2

    b = mapping-t
    a = w-b

    returnList = [a,b]
    returnList.sort()

    return returnList


def mltBinaryAccuracy(labels, predictions):
    '''
        Gives us overall acc as well as acc on different types of quesitons.
        labels[0] and predictions[0] correspond to 'nothing'
    '''
    correct = 0.0
    nothings = 0.0
    nothings_c = 0.0
    multilabel = 0.0
    multilabel_c = 0.0
    singlelabel = 0.0
    singlelabel_c = 0.0
    for i in range(labels.shape[0]):
        isCorrect =  np.all(labels[i] == predictions[i])
        if isCorrect:
            correct += 1
        if labels[i][0] > 0:
            nothings += 1
            if isCorrect:
                nothings_c += 1
        elif np.sum(labels[i]) > 1.5:
            multilabel += 1
            if isCorrect:
                multilabel_c += 1
        else: # single
            singlelabel += 1
            if isCorrect:
                singlelabel_c += 1
    nothing_acc = nothings_c / nothings
    multilabel_acc = multilabel_c / multilabel
    singlelabel_acc = singlelabel_c / singlelabel
    assert nothings + multilabel + singlelabel == labels.shape[0], "Sth's fishy"
    assert nothings_c + multilabel_c + singlelabel_c == correct, "Sth's fishy"
    if nothings + multilabel + singlelabel != labels.shape[0]:
        print("WTFFFF: ", str(labels.shape[0] - nothings - multilabel - singlelabel))
    return correct / len(labels), nothing_acc, multilabel_acc, singlelabel_acc


def writeToCsv(outLabels, testStories, dictionary, outFile):
    '''
    Translate the labels to answer format and write them to a csv
    '''
    mainString = 'textID,sortedAnswerList'
    out_index = 0
    missing_predicted = 0
    for i, st in enumerate(testStories):
        for qi in range(1, len(st.questionIndices) + 1):
            line = str(i + 1) + "_" + str(qi) + "," # a single line in our csv
            out_point = outLabels[out_index] # a single line with the length of the dict, contatining 0s and 1s
            writtenSth = False # have we predicted any word that is in the story - HACKY AF
            writtenN = False # have we written "-1"
            predictions = [] # 1-based story indices of the predicted words
            for j, res in enumerate(out_point):
                if res == 1: # we predict dictionary[j]
                    predicted = dictionary[j]
                    # make sure we're not predicting a word that's not in the story
                    # assert predicted in st.wordToNumHash, "WORD THAT IS NOT IN STORY PREDICTED: " + predicted + " is not in story " + str(i) + " with question index " + str(qi)
                    if predicted in st.wordToNumHash:
                        predictions.append(st.wordToNumHash[dictionary[j]])
                        writtenSth = True
                        assert st.wordToNumHash[dictionary[j]] == -1 or (not writtenN), "Writing Nothing and sth else!" # make sure we're not wrinig 'nothing' and sth else
                        if st.wordToNumHash[dictionary[j]] == -1:
                            writtenN = True
                    else:
                        missing_predicted += 1
            # assert writtenSth, "No answers written for question " + str(out_index) # make sure we've written std
            if not writtenSth:
                predictions = [-1]
            assert len(predictions) > 0, "Empty predictions for " + str(i) + " " + str(qi)
            predictions.sort() # we need to output sorted indices
            for p in predictions:
                line += str(p) + " "
            mainString += "\n" + line.strip() # remove trailing whitespace
            out_index += 1
    f = open(outFile, "w")
    f.write(mainString)
    f.close()
    print("Written to " + outFile + " - " + str(missing_predicted) + " predictions of words that were not in the story are replaced with -1")

def generateAllPossibleBigrams(words):
    bigrams = []
    for i in range(1, len(words)):
        bigrams += list(zip(words, words[i:] + words[:i]))
    assert len(bigrams) == (len(words) * (len(words) - 1)),  "Bigram list length is not correct: " + str(len(bigrams))
    return bigrams

def generateBigramsFromSent(sentence):
    bigrams = list(zip(sentence, sentence[1:]))
    assert len(bigrams) == len(sentence) - 1 or len(sentence) == 0, "Wrong bigram count"
    return bigrams

def generateTrigramsFromSent(sentence):
    trigrams = list(zip(sentence, sentence[1:], sentence[2:]))
    assert len(trigrams) == len(sentence) - 2 or len(sentence) == 0, "Wrong bigram count"
    return trigrams
