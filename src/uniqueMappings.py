import math

def uniqueMapping(indexes):

	if len(indexes) == 0:
		a = 0
		b = 0
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
	a, b = reverseCantorMapping(mapping)
	indexes = [int(a)]

	# threshold - basically the number of elements in the array
	threshold = 32

	if b > threshold:
		indexes += reverseUniqueMapping(b)
	else:
		indexes.append(int(b))

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