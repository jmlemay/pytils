#######################################################################################################################
# Miscellaneous functions.
########################################################################################################################

import datetime, numpy, math
from typing import Union

def now(units:str="ms") -> numpy.datetime64:
	"""
	Gets the current time.
	@param ms	{string}		Units (default="ms"). Note that all digits smaller than microseconds will be zero,
								regardless of units requested.
	@return		{datetime64}	Now.
	"""
	return numpy.datetime64(datetime.datetime.now(), units)

def today() -> numpy.datetime64:
	"""
	Gets the current date.
	@return	{datetime64}	Today.
	"""
	return numpy.datetime64("today")

def timeDelta(Y:int=0, M:int=0, W:int=0, D:int=0, h:int=0, m:int=0, s:int=0, ms:int=0, us:int=0, ps:int=0, fs:int=0, a:int=0) -> numpy.timedelta64:
	"""
	Creates a timedelta using a combination of units. Note that this can cause an overflow if mixing big units with
	smaller ones (as combining expands to the smallest unit). Some types are also incompatible (e.g. 'Y' and 'm').
	@param Y	{int}		Years.
	@param M	{int}		Months.
	@param W	{int}		Weeks.
	@param D	{int}		Days.
	@param h	{int}		Hours.
	@param m	{int}		Minutes.
	@param s	{int}		Seconds.
	@param ms	{int}		Milliseconds.
	@param us	{int}		Microseconds.
	@param ps	{int}		Picoseconds.
	@param fs	{int}		Femtoseconds.
	@param a	{int}		Attoseconds (note: breaks pattern because 'as' is a keyword).
	@return		{timedelta}	Time delta.
	"""
	res = numpy.timedelta64()
	if Y != 0: res += numpy.timedelta64(Y, "Y")
	if M != 0: res += numpy.timedelta64(M, "M")
	if W != 0: res += numpy.timedelta64(W, "W")
	if D != 0: res += numpy.timedelta64(D, "D")
	if h != 0: res += numpy.timedelta64(h, "h")
	if m != 0: res += numpy.timedelta64(m, "m")
	if s != 0: res += numpy.timedelta64(s, "s")
	if ms != 0: res += numpy.timedelta64(ms, "ms")
	if us != 0: res += numpy.timedelta64(us, "us")
	if ps != 0: res += numpy.timedelta64(ps, "ps")
	if fs != 0: res += numpy.timedelta64(fs, "fs")
	if a != 0: res += numpy.timedelta64(a, "as")
	return res

def ps(n:Union[int,float]) -> str:
	"""
	Pluralizes -s words.
	@param n	{number}	Number of things.
	@return		{string}	"s" if the word should be pluralized, "" otherwise.
	"""
	return _p("", "s", n)

def pes(n:Union[int,float]) -> str:
	"""
	Pluralizes -es words.
	@param n	{number}	Number of things.
	@return		{string}	"es" if the word should be pluralized, "" otherwise.
	"""
	return _p("", "es", n)

def pies(n:Union[int,float]) -> str:
	"""
	Pluralizes -ies words.
	@param n	{number}	Number of things.
	@return		{string}	"ies" if the word should be pluralized, "y" otherwise.
	"""
	return _p("y", "ies", n)

def _p(s:str, p:str, n:Union[int,float]) -> str:
	"""
	Pluralizes all words.
	@param s	{string}	Suffix if singular.
	@param p	{string}	Suffix if plural.
	@param n	{number}	Number of things.
	@return		{string}	'p' if the word should be pluralized, 's' otherwise.
	"""
	return [s, p][n != 1]

def distinct(x:Union[list,tuple,numpy.ndarray]) -> Union[list,tuple,numpy.ndarray]:
	"""
	Returns the distinct elements of a list-like object. Note that order is not preserved; result is in ascending order.
	@param x	{list|tuple|array}	List/tuple/array.
	@return		{list|tuple|array}	Distinct elements.
	"""
	typ = type(x)
	res = numpy.unique(x) # Lean on numpy
	if typ == list: res = list(res) # Preserve type
	elif typ == tuple: res = tuple(res) # Ditto
	return res

def getVal(d:dict, k, f):
	"""
	Gets key value from a dictionary if it exists, or a default value if it doesn't.
	@param d	{dict}	Dictionary to pull from.
	@param k	{any}	Key to get.
	@param f	{any}	Default (fill) value to provide if the key doesn't exist.
	@return		{any}	The value of the key or the default value.
	"""
	if k in d.keys():
		return d[k]
	else:
		return f

def mkList(obj) -> list:
	"""
	Makes into a list if it isn't already. Note: where we say list, we mean strictly list; even it's a tuple or numpy
	array or other, we're checking only lists.
	Note: Retains a reference to the original object, if applicable.
	@param obj	{any}	Any object.
	@return		{list}	The object untouched, if it was already a list, or the object enlisted if it wasn't.
	"""
	return obj if type(obj) == list else [obj]

def asof(x:list, y) -> int:
	"""
	Does an "asof" search in a list sorted in ascending order (essentially `bin` in q).
	@param x	{list}	Sorted list.
	@param y	{any}	Value to search.
	@return		{int}	Returns the smallest index `i` such that `x[i] =< y < 0W^x[i+1]`.
						Returns `-1` if `y` is smaller that `x[0]`.
	"""
	l = len(x)
	if (l == 0) or (y < x[0]): return -1 # Trivial or smaller than our smallest
	if l == 1: return 0 # Not much of a search
	return _asof(x, y, 0, len(x)-1)

def _asof(x:list, y, i:int, j:int) -> int:
	"""
	Private version of "asof" search. Calls itself recursively narrowing down its search.
	@param x	{list}	Sorted list.
	@param y	{any}	Value to search.
	@param i	{int}	Left index.
	@param j	{int}	Right index. Note: assume `j > i`.
	@return		{int}	Returns the smallest index `k` such that `x[k] =< y < 0W^x[k+1]`.
	"""
	# Base case: i and j are right next to each other. So the answer is one of them.
	d = j - i
	if d == 1: return j if x[j] <= y else i

	mid = i + math.floor(d / 2) # Mid-point between left and right indices
	return _asof(x, y, mid, j) if y > x[mid] else _asof(x, y, i, mid) # Go left or right

def where(b:list) -> list:
	"""
	Where.
	@param b	{bool[]}	List of booleans.
	@return		{int[]}		Where booleans are true.
	"""
	res = []

	for i in range(len(b)):
		if b[i]: res.append(i)

	return res

def remove(a:list, b:Union[any,list]) -> list:
	"""
	Removes elements in "b" from "a".
	@param a	{list}		List.
	@param b	{atom|list}	Elements to remove from "a".
	@return		{list}		a - b.
	"""
	return list(set(a).difference(set(mkList(b))))

def first(x:[list,tuple,numpy.ndarray]):
	"""
	Returns the first element of the list.
	@param x	{list|tuple|array}	List.
	@return		{any}				First element of the list.
	"""
	return x[0]

def last(x:[list,tuple,numpy.ndarray]):
	"""
	Returns the last element of the list.
	@param x	{list|tuple|array}	List.
	@return		{any}				Last element of the list.
	"""
	return x[-1]

def find(x:[list,tuple,numpy.ndarray], y) -> int:
	"""
	Finds the index of the first occurrence of an element in a list, or -1 if none.
	@param x	{list|tuple|array}	List to search.
	@param y	{any}				Item to search.
	@return		{int}				Index of the first of occurrence of y in x.
	"""
	for i in range(len(x)):
		if x[i] == y: return i

	return -1

def within(x, y: list) -> [bool,tuple,numpy.ndarray]:
	"""
	Determines if x is within y.
	@param x	{atom|vector}	Item(s) to test.
	@param y	{list[2]}		Two-element list.
	@return		{bool|vector}	True if within, false otherwise. Shape matches x.
	"""
	if type(x) in [list, tuple, numpy.ndarray]:
		res = [_within(i, y) for i in x]
		if type(x) == tuple: return tuple(res)
		if type(x) == numpy.ndarray: return numpy.array(res)
		return res

	return _within(x, y)

def _within(x, y:list) -> bool:
	"""
	Determines if x is within y.
	@param x	{atom}		Item to test.
	@param y	{list[2]}	Two-element list.
	@return		{bool}		True if x is within y.
	"""
	return y[0] <= x and x <= y[1]

def vin(x:Union[list,numpy.ndarray], y:Union[list, numpy.ndarray]) -> numpy.ndarray:
	return numpy.array([x[i] in y for i in range(len(x))])
