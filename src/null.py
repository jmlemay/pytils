########################################################################################################################
# Module to handle nulls, which numpy doesn't always do correctly.
########################################################################################################################

import numpy

# Null constants.
INT = -999999999999 ##~ Might be weird with actual large-magnitude negatives
FLOAT = numpy.nan
DATETIME = numpy.datetime64("")
TIMEDELTA = numpy.timedelta64("")
STRING = ""
OBJECT = None
BOOL = False

def getNull(typ:str):
	"""
	Gets the null of the appropriate type.
	@param typ	{string}	Type of null we want.
	@return		{any}		Null of the appropriate type.
	"""
	if typ == "int64": return INT
	if typ == "float64": return FLOAT
	if typ == "object": return OBJECT
	if typ == "bool": return BOOL
	if typ.startswith("datetime64"): return DATETIME
	if typ.startswith("timedelta64"): return TIMEDELTA
	if typ.startswith("str"): return STRING

	raise Exception("No null defined for type={}".format(typ))

def isNull(val) -> bool:
	"""
	Determines if a value is null. Note that this is really only handy for single values.
	If doing this on a column, consider using `numpy.isnan`.
	@param val	{any}	A value.
	@return		{bool}	True if the value is null.
	"""
	if val is None: return True

	typ = type(val)
	styp = str(typ)

	if typ == object: return False
	if typ in [str, numpy.str_]: return val == ""
	if typ in [bool, numpy.bool_]: return not val ##~ Imperfect, should create a nullable boolean
	if styp.find("int") != -1: return val == INT
	if styp.find("float") != -1: return numpy.isnan(val)
	if styp.find("datetime64") != -1: return numpy.isnan(val)
	if styp.find("timedelta64") != -1: return numpy.isnan(val)

	raise Exception("Unrecognized type for null check, type={} val={}".format(styp, val))