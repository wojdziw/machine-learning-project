Log:

The current version of the code gives us accuracy of 51.101%.

(Woj) The method for computing the labels is a little bit convoluted, so I will explain it here. I will do it through explaining the problem first, then proposing a solution, that will give a better understanding of my train of thought. Choo choo!

Problem: We want to output a single number as a label, from which we can uniquely identify which words it representes. That is, if the word indices are 11 and 13, then uniqueMapping([11,13])==313 then reverseUniqueMapping(313)=[11,13].

Solution: I implemented this function and ran an SVM on those labels.

Problem: These labels didn't prove to be very effective.

Solution: Instead, as a first iteration I just decided to output the first label and ditch the uniqueMapping altogether. That is, if the supposed label is [11,13], I will only output 11.

Problem: Clearly, this doesn't get us closer to the ground truth.

Solution: If the label is only one word, output its index normally. If it's more than two words, compute the uniqueMapping(words) and output that. Hopefully, the classifier would then learn the uniqueMapping and use it if there was more than one word in the label.

Problem: No problem! Working perfectly (sort of...)


REGARDING uniqueMapping - bear in mind it's not only used to generate the labels. It is also heavily used to generate the datapoints, since the datapoint value for a word is uniqueMapping(occurences_in_the_statement).


Okay, so now:
1) We change the method substantially to improve the score, or...
2) Introduce tweaks to the current algorithm to make it more robust.

Ad #2, possible issues are:
1) uniqueMapping produces large numbers. VERY LARGE, and they overflow. Hence, if a word appears in the story more than five times, its uniqueMapping(occurences) is likely to just be the max_int, because len(occurences)>5. There is little we can do to fix this - after all, we cannot represent arbitrarily large numbers. Then again, it might not be that much of an issue, because any substantial words are not gonna appear more than five times.
1.5) Please bear in mind uniqueMapping([largeNumber1, largeNumber2]) doesn't produce any big number by itself. It's big lists that do it.
2) The classification is imperfect - we are only producing labels that appeared in the training set, which is clearly not very robust. If (football, apple, milk) didn't appear in TRAIN then we're out of luck - even if it's a correct answer in TEST then the classifier is not going to catch it.
3) We are overfitting heavily - we get over 95% accuracy on the training set. It's surprising that it generalizes to a 50% accuracy on the testing given how similar the two are. Instead, though, we could try using a different classifier that wouldn't overfit and generalize better, of course at the expense of Ein.
4) Maybe there is some silly mistake in the code that doesn't give us the right answer. Literally, most of the examples I checked by hand we get right.
5) Maybe it's all wrong, life is a lie and we just live in a computer simulation.

TL;DR:
Try neural nets or something to generalize better...