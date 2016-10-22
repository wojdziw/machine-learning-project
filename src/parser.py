import DataPoint
import nltk

def parseFile(filename, hasAnswers = True):
    '''
        filename - a name of a file that is formatted as per spec
        hasAnswers - Are there answers to the questions in the file
        returns:
            dataPoints - a list of DataPoint objects - one for each set of statements
            verbs - alphabetically sorted list of all the verbs in the text
            nouns - alphabetically sorted list of all the nouns in the text
    '''
    verbs = set()
    nouns = set()
    dataPoints = []
    currentPoint = DataPoint.DataPoint()
    with open(filename, "r") as ins:
        for line in ins:
            # split on whitespace
            lst = line.strip().split()

            # When we see an index of 1, we push the current point and reset it
            if int(lst[0]) == 1:
                dataPoints.append(currentPoint)
                currentPoint = DataPoint.DataPoint()

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
                currentPoint.addQuestion(lst[1:], answers)

            # get verbs and nouns
            for (w, tag) in nltk.pos_tag(lst[1:]):
                if tag[0] == 'V':
                    verbs.add(w)
                elif (tag == 'NN' or tag == 'NNS') and (not w == 'journeyed'):
                    nouns.add(w)

        dataPoints.append(currentPoint)
        verbs = list(verbs)
        nouns = list(nouns)
        verbs.sort()
        nouns.sort()
        return dataPoints[1:], verbs, nouns
