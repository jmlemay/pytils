########################################################################################################################
# Mocking module. The versions that exist are too complicated for what we're trying to do, and they seem more geared
# towards mocking classes than functions in modules. Hence, we have a simplified version here that's more appropriate to
# our use case.
########################################################################################################################

from types import ModuleType, FunctionType

# Global dictionaries
_mocks = {} 	# Mocked values
_sentries = {}	# Sentries

def mock(module:ModuleType, attr:str, mockAttr):
	"""
	Mocks a attribute.
	@param module	{module}	Module in which attribute resides.
	@param attr		{string}	Name of the attribute.
	@param mockAttr	{any}		New value we want to assign to attribute.
	"""
	if not _isAttrStored(module, attr):
		_backupAttr(module, attr) # Back up so we can restore after

	setattr(module, attr, mockAttr) # Set new mocked attr

def resetMocks():
	"""
	Resets all mocks to their original values.
	"""
	global _mocks # Because we are changing the value

	for key, value in _mocks.items(): # key = (module, attrName), value = ogDef
		setattr(key[0], key[1], value) # Reset mocked attrs one by one

	_mocks = {} # Forget mocks

def deleteMock(module:ModuleType, attr:str):
	"""
	Deletes an attribute (no restore, it's gone).
	@param module	{module}	Module in which attribute resides.
	@param attr		{string}	Name of the attribute.
	"""
	delattr(module, attr) # Maybe doesn't need to be a function, but for the sake of a consistent interface

def sentry(name:str) -> FunctionType:
	"""
	Creates a sentry -- a function that records how many times it was called and the last arguements it was called with,
	but otherwise does nothing.
	@param name	{string}	Name of the sentry.
	@return		{fn}		Sentry.
	"""
	global _sentries
	_sentries[name] = {"count": 0, "args": None} # Initialize a new sentry

	def f(*x):
		_sentries[name]["count"] += 1 # Increment count
		_sentries[name]["args"] = x # Capture (last) args

	return f

def sentryCalled(name:str) -> int:
	"""
	Returns how many times the sentry function was called.
	@param name	{string}	Name of the sentry.
	@return		{int}		Number of times the sentry was called.
	"""
	return _sentries[name]["count"]

def sentryArgs(name:str):
	"""
	Returns the last args that a sentry was called with.
	@param name	{string}	Name of the sentry.
	@return		{*any|None}	Last called-with args, or 'None' if it wasn't called.
	"""
	res = _sentries[name]["args"]

	if res is None: # i.e. wasn't called
		return None
	elif len(res) == 1:
		return res[0]
	else:
		return res

def sentryReset(name:str):
	"""
	Resets a sentry.
	@param name	{string}	Name of the sentry.
	"""
	_sentries[name]["args"] = None
	_sentries[name]["count"] = 0

def _isAttrStored(module:ModuleType, attr:str) -> bool:
	"""
	Determines if an attribute has already be backed up (useful to ensure we don't overwrite).
	@param module	{module}	Module in which attribute resides.
	@param attr		{string}	Name of the attribute.
	@return			{bool}		True if the attr is already backed up, false otherwise.
	"""
	return (module, attr) in _mocks.keys()

def _backupAttr(module:ModuleType, attr:str):
	"""
	Backs up an attribute (so that it can be recovered). If the attribute doesn't exist (e.g. we're mocking a value that
	doesn't exist in the module to begin with), this does nothing..
	@param module	{module}	Module in which attribute resides.
	@param attr		{string}	Name of the attribute.
	"""
	if attr in dir(module):
		_mocks[(module, attr)] = getattr(module, attr) # Store in dictionary