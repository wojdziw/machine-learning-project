import Story

def parseFile(filename, hasAnswers = True):
    '''
        filename - a name of a file that is formatted as per spec
        hasAnswers - Are there answers to the questions in the file
        returns:
            stories - a list of Story objects - one for each set of statements
            dictionary - alphabetically sorted list of all the words in the text
    '''
    dictionary = set()
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
                currentPoint.addQuestion(lst[1:], answers)

            # add words to dictionary
            for w in lst[1:]:
                dictionary.add(w)

        stories.append(currentPoint)
        dictionary = list(dictionary)
        dictionary.sort()

        # adding a dummy element to the ditionary to make it base 1
        dictionary = ["nothing"] + dictionary
        return stories[1:], dictionary
