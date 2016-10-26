from itertools import *
import numpy as np
import copy
import math

def uniqueMapping(indexes):

    if len(indexes) > 5:
        indexes = indexes[0:5]

    if len(indexes) == 0:
        a = 0
        b = 0
    elif indexes[0] == -1:
        return -1
    elif len(indexes) == 1:
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

    if a == 0:
        return [int(b)]

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

class Story:
    def __init__(self):
        self.wordToNumHash = dict()   # one-based
        self.wordToNumHash['nothing'] = -1 # as per spec
        self.statementWordIndices = dict()     # Keeps a list of indices of occurences
                                      # for each word (0-based)
        self.wordIndicesBeforeQuestion = []     # at index i, the state of statementWordIndices
                                                # before question i, updated with the
                                                # indices in the question
        self.wordCount = 0
        self.statementIndices = []    # zero based
        self.questionIndices = []     # zero based
        self.answers = []             # answers[i] is a list of answers to
                                      # sentences[questionIndices[i]]
        self.sentences = []           # holds questions and statements

    def addStatement(self, statement):
        '''
            Takes a list of word
            Ex.: ['John', 'takes', 'the', 'apple']
        '''
        for word in statement:
            self.wordCount += 1
            if not word in self.wordToNumHash:
                self.wordToNumHash[word] = self.wordCount
                self.statementWordIndices[word] = [self.wordCount - 1] # 0-based
            else:
                self.statementWordIndices[word] += [self.wordCount - 1] # 0-based
        self.statementIndices.append(len(self.sentences))
        self.sentences.append(statement)

    def addQuestion(self, question, answers):
        '''
            question is a list of words
            Ex. ['What', 'is', 'John', 'holding']
            answers is a (possibly empty) list of single-word answers
            Ex.: ['apple', 'milk']
        '''
        # copy the current state of statementWordIndices, add the word indices
        # from the question in the loop below and record it
        currentWordIndices = copy.deepcopy(self.statementWordIndices)
        for word in question:
            self.wordCount += 1
            if not word in self.wordToNumHash:
                self.wordToNumHash[word] = self.wordCount
            if not word in currentWordIndices:
                currentWordIndices[word] = [self.wordCount - 1] # 0-based
            else:
                currentWordIndices[word] += [self.wordCount - 1] # 0-based


        self.wordIndicesBeforeQuestion.append(currentWordIndices)
        self.questionIndices.append(len(self.sentences))
        self.sentences.append(question)
        self.answers.append(answers)


    def wordCountsBefore(self, questionNumber, words):
        '''
            Returns a list containing the number of occurences of individual
            words in *words* in statemens before question no. *questionNumber* (0-based)
            and in said question
        '''
        wordCounts = dict(zip(words, repeat(0)))
        for w in self.sentences[self.questionIndices[questionNumber]]:
            if w in wordCounts:
                wordCounts[w] += 1
        for i in self.statementIndices:
            if i > self.questionIndices[questionNumber]:
                break
            else:
                for word in self.sentences[i]:
                    if word in wordCounts:
                        wordCounts[word] += 1
        return [wordCounts[w] for w in words]

    def makeFeatureValuesBefore(self, questionNumber, words):
        '''
            Returns a 1xN numpy array containig the values for each word in the
            dictionary based on the indices of occurence of that word in the
            statements before question no. *questionNumber* as well as in that
            question
        '''
        featureVector = np.empty(len(words))
        for i, w in enumerate(words):
            # Get a list of indices for the word and feed it to the generator function
            if w in self.wordIndicesBeforeQuestion[questionNumber]:
                featureVector[i] = uniqueMapping(
                    self.wordIndicesBeforeQuestion[questionNumber][w])
            else:
                featureVector[i] = uniqueMapping([-1])
        return featureVector

    def constructPoints(self, dictionary):
        '''
            dictionary - python list of words
            returns:
                points - NxM np.array, containing the data points, corresponding
                to the questions in the Story, where N is the number of questions
                and M is the length of the dictionary
        '''
        n = len(self.questionIndices)
        M = len(dictionary)
        points = np.empty([n, M])
        for i in range(n):
            points[i] = self.makeFeatureValuesBefore(i, dictionary)
        return points

    def constructLabels(self, dictionary):
        '''
            dictionary - python list of words
            returns:
                labels - 1xN np.array, containing the label for each question
                where N is the number of questions in this Story
        '''
        n = len(self.answers)
        labels = np.empty(n)
        for i, ans in enumerate(self.answers):
            # Map the words in each answer to their index in the dictionary
            ansNumbers = []
            for w in ans:
                if w == 'nothing':
                    ansNumbers.append(-1)
                else:
                    ansNumbers.append(dictionary.index(w))
            # Make a unique mapping from the indices to an integer
            # labels[i] = uniqueMapping(ansNumbers)
            ansNumbers[:] = [x+10 for x in ansNumbers]
            if len(ansNumbers) > 1:
                labels[i] = uniqueMapping(ansNumbers)
            else:
                labels[i] = ansNumbers[0]

        return labels


    def __str__(self):
        ret = ""
        q = s = 0
        while(q + s < len(self.sentences)):
            if(q < len(self.questionIndices) and
                (s == len(self.statementIndices) or
                self.questionIndices[q] < self.statementIndices[s])):
                ret += str(q + s + 1) + " " + " ".join(self.sentences[q + s]) + "?\t"
                ret += ",".join(self.answers[q])
                q += 1
            else:
                ret += str(q + s + 1) + " " + " ".join(self.sentences[q + s]) + "."
                s += 1
            ret += "\n"
        return ret
