from itertools import *
import nltk

class DataPoint:
    def __init__(self):
        self.wordToNumHash = dict()   # one-based
        self.wordToNumHash['nothing'] = -1 # as per spec
        self.wordCount = 0
        self.statementIndices = []    # zero based
        self.questionIndices = []     # zero based
        self.answers = []             # answers[i] is a list of answers to
                                        # sentences[questionIndices[i]]
        self.sentences = []             # holds questions and statements

    def addStatement(self, statement):
        '''
            Takes a list of word
            Ex.: ['John', 'takes', 'the', 'apple']
        '''
        for word in statement:
            self.wordCount += 1
            if not word in self.wordToNumHash:
                self.wordToNumHash[word] = self.wordCount
        self.statementIndices.append(len(self.sentences))
        self.sentences.append(statement)

    def addQuestion(self, question, answers):
        '''
            question is a list of words
            Ex. ['What', 'is', 'John', 'holding']
            answers is a (possibly empty) list of single-word answers
            Ex.: ['apple', 'milk']
        '''
        for word in question:
            self.wordCount += 1
            if not word in self.wordToNumHash:
                self.wordToNumHash[word] = self.wordCount
        self.questionIndices.append(len(self.sentences))
        self.sentences.append(question)
        self.answers.append(answers)

    def wordCountsBefore(self, constaints, questionNumber, words):
        '''
            Returns a map containing the number of occurences of individual
            words in *words* in sentences containing all the words in
            *constaints* before question no. *questionNumber* (0-based)
        '''
        wordCounts = dict(zip(words, repeat(0)))
        for i in self.statementIndices:
            if i > self.questionIndices[questionNumber]:
                break
            else:
                if all([(w in self.sentences[i]) for w in constaints]):
                    for word in self.sentences[i]:
                        if word in wordCounts:
                            wordCounts[word] += 1
        return wordCounts

    def constructFeatures(self, nouns, verbs):
        pts = []
        for i, qi in enumerate(self.questionIndices):
            name = ""
            for w in self.sentences[qi]:
                if nltk.pos_tag([w])[0][1] == 'NNP':
                    name = w
                    break
            point = []
            for n in nouns:
                verbCounts = self.wordCountsBefore([name, n], i, verbs)
                for v, c in verbCounts.items():
                    point.append((n + ' ' + v, c))
            pts.append(point)
        return pts


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
