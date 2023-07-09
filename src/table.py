########################################################################################################################
# Table class. This is meant to replace the awful pandas.DataFrame
# (the source of too many headaches) with a more q-like interface.
# To be updated as more features are required!
########################################################################################################################

import numpy, os
from typing import Union
from src import null, misc

class Table:
	_dispWidth = 100 # Default display width
	_dispHeight = 30 # Default display height

	_types = {
		"b": bool,
		"d": numpy.datetime64,
		"s": str,
		"i": numpy.int64,
		"f": numpy.float64,
		"n": numpy.timedelta64,
		"*": object
	}

	def __init__(self, d:dict={}):
		"""
		Init.
		@param d	{dict}	Dictionary representing the table. Must be one of the following types:
								- empty:					Creates an empty table -- no columns, no rows.
								- string --> {string|type}	Creates a table with empty typed columns.
								- string --> {list|array}	Creates a table with columns. Note that if a column is given
															as a list, it is transformed to a numpy array, you're taking
															your chances with the type.	
		"""
		if type(d) != dict: raise TableException("init param not dict")

		self._keys = [] # Always init table unkeyed
		self._isKeyed = False

		# Empty case.
		if len(d) == 0:
			self._dict = {}
			return

		# Type cases.
		self._dict = {}
		first = list(d.values())[0] # Grab first value

		if type(first) == str:
			for k in d.keys(): self._dict[k] = numpy.array([]).astype(Table._getType(d[k]))
			return

		if type(first) == type:
			for k in d.keys(): self._dict[k] = numpy.array([]).astype(d[k])
			return

		# Otherwise, populate the table. ##~ Support for scalar extension
		length = len(list(d.values())[0]) # All columns should have the same length

		for k in d.keys():
			if len(d[k]) != length: raise TableException("length")
			self._dict[k] = numpy.array(d[k]) # Create a numpy array for each column


	#-------------------------------------------------------------------------------------------------------------------
	# Class functions.
	#-------------------------------------------------------------------------------------------------------------------

	def setc(h:float):
		"""
		Sets the vertical console size to display tables.
		@param h	{float}	Vertical console height.
		"""
		# Determine console size and how much vertical space we want to use.
		# Note: "stty" may not exist, in which case we can't do anything.
		size = os.popen('stty size', 'r').read()

		if size == "":
			print("stty undefined, unable to set table height fraction")
			return

		y, x = size.split()
		x = int(x)
		y = int(y)
		Table.setDispSize(x, int(h*y))

	def getDispSize() -> int:
		"""
		Gets the current vertical console size to display tables.
		@return	{(float, float)}	Display width and height, respectively.
		"""
		return Table._dispWidth, Table._dispHeight

	def setDispSize(x:int=None, y:int=None):
		"""
		Sets the display size.
		@param x	{int|None}	Display width, where 'None' means no change.
		@param x	{int|None}	Display height, where 'None' means no change.
		"""
		if not x is None: Table._dispWidth = x
		if not y is None: Table._dispHeight = y

	def fromCSV(file:str, types:Union[list,str], delimiter:str=",") -> "Table":
		"""
		Writes down the table as a csv.
		@param file			{string}		File location.
		@param types		{type[]|string}	Types of the columns. `None` is used to indicate to ignore the column.
		@param delimiter	{string}		Delimmiter -- optional, default is ",".
		@return				{Table}			Table parsed from CSV
		"""
		string = open(file, "r").read()
		return Table.fromCsvString(string, types, delimiter)

	def fromCsvString(string:str, types:Union[list,str], delimiter:str=",") -> "Table":
		"""
		Writes down the table as a csv.
		@param string		{string}		CSV in string format (e.g. "x,y\\n1,2").
		@param types		{type[]|string}	Types of the columns. `None` is used to indicate to ignore the column.
		@param delimiter	{string}		Delimmiter -- optional, default is ",".
		@return				{Table}			Table parsed from CSV
		"""
		rows = string.split("\n")
		if rows[-1] == "": rows = rows[:-1] # Sometimes it imagines a trailing \n

		cols = numpy.array([c.strip() for c in rows[0].split(delimiter)]) # Columns
		if type(types) == str: types = [ None if t == " " else Table._getType(t) for t in types] # Typify strings
		atypes = numpy.array(types + [None]*(len(cols) - len(types))) # Types as array, trailing cols are to be ignored
		gc = atypes != None # Good columns (those we actually care about)
		cols = cols[gc] # Keep only the good columns
		atypes = atypes[gc] # May as well forget the Nones as well
		rng = range(len(cols)) # Range of columns (iterate through this every row)

		# Another example of why python sucks. We can't use the [x]*y shortcut, because lists are by reference.
		res = []

		for i in rng:
			res.append([])

		for row in rows[1:]:
			r = numpy.array([r.strip() for r in row.split(delimiter)])[gc] # Cells of the rows we care about

			for j in rng:
				res[j].append(r[j]) # Append cell contents

		res = {cols[j]: numpy.array(res[j]).astype(atypes[j]) for j in rng} # Build dict of column name --> values
		return Table(res) # Make a table

	def raze(tables:list) -> "Table":
		"""
		Razes a list of tables. Note that the columns of the resulting table are those of the first table in the list;
		any extra columns in the subsequent tables are lost. If a column is missing from one of the subsequent tables,
		the function throws an error.
		@param tables	{Table[]}	List of tables.
		@return			{Table}		Razed tables.
		"""
		res = tables[0].copy()

		for i in range(1, len(tables)):
			res.append(tables[i])

		return res

	def load(loc:str) -> "Table":
		"""
		Loads a table from disk.
		@param loc	{string}	On-disk location of table (root).
		@return		{Table}		On-disk table, now in memory.
		"""
		# Start by getting the column order.
		try:
			dotD = open(loc + "/.d", "r") # Open the .d file
			cols = dotD.read().split("\n") # Read the column names
			if cols[-1] == "": cols = cols[:-1] # Handle trailing \n that always pops up for whatever reason
		except Exception as ex:
			raise TableException("Unable to read .d file: {}".format(ex))

		# Read the columns.
		res = Table() # Start with an empty table

		for c in cols:
			try:
				res[c] = numpy.load(loc + "/" + c + ".npy", allow_pickle=True)
			except Exception as ex:
				raise TableException("Unable to read column '{}': {}".format(c, ex))

		return res


	#-------------------------------------------------------------------------------------------------------------------
	# Public instance functions. Most, but not all, functions have the `inPlace` parameter. If false,
	# the function returns a new table instance. If true (true), the function modifies the current table.
	#-------------------------------------------------------------------------------------------------------------------

	def isKeyed(self) -> bool:
		"""
		Indicates if the table is keyed.
		@return	{bool}	True if keyed, false otherwise.
		"""
		return self._isKeyed

	def keyCols(self) -> list:
		"""
		Gets the key columns of the table.
		@return	{list}	Key columns, or empty if none.
		"""
		return self._keys
		
	def key(self, cols:Union[str,list], inPlace:bool=True):
		"""
		Keys the table. Replaces any existing key columns.
		@param cols		{string|string[]}	Column(s) to be the key columns.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Keyed table.
		"""

		if not inPlace:
			c = self.copy()
			c.key(cols)
			return c

		cols = misc.mkList(cols) # Enlist

		# Handle trivial case.
		if len(cols) == 0:
			self.unkey() # This is probably what is meant
			return

		# Make sure the columns actually exist.
		for c in cols:
			if not c in self._dict.keys():
				raise TableException("Column '{}' not in table".format(c))

		# Set appropriate variables.
		self._isKeyed = True
		self._keys = cols

	def unkey(self, inPlace:bool=True) -> "Table":
		"""
		Unkeys the table.
		@param inPlace	{bool}	Do it in place or not.
		@return			{Table}	Unkeyed table.
		"""
		t = self if inPlace else self.copy()
		t._isKeyed = False
		t._keys = []
		if not inPlace: return t

	def copy(self) -> "Table":
		"""
		Creates a copy of this table.
		@return	{Table}	Copy of this table (new instance).
		"""
		t = Table(self._dict)
		t._isKeyed = self._isKeyed
		t._keys = self._keys
		return t

	def cols(self) -> list:
		"""
		Gets the columns of the table.
		@return	{string[]}	Columns.
		"""
		return list(self._dict.keys())

	def count(self) -> int:
		"""
		Gets the table count (can also use `len`).
		@return	{int}	Length of table.
		"""
		return len(self)

	def getCol(self, col:Union[str,list]) -> Union[numpy.ndarray,list]:
		"""
		Gets one or more column(s) from the table. Note, we return copies of the columns.
		@param col	{string|string[]}	Column(s) to get.
		@return		{array|array[]}		Array or list of arrays.
		"""
		if type(col) in [list, numpy.ndarray]: # Multiple columns
			return [self._dict[c].copy() for c in col]
		else: # Single column, presumably
			return self._dict[col].copy()

	def getRow(self, row:Union[int,list,slice], col:Union[str,list]=[]) -> Union[dict,"Table"]:
		"""
		Gets one or multiple rows from the table.
		@param row	{int|int[]|slice|bool[]}	Row(s) to get.
		@param col	{string|string[]}			Column(s) to get -- optional, default is all.
		@return		{dict|Table}				Single row (as a dictionary) if requesting one row, or subset of table
												if multiple.
		"""
		if len(self.cols()) == 0:
			raise TableException("Can't get rows from an empty table")

		res = {}
		keys = self._dict.keys() if col == [] else [col] if type(col) == str else col

		for k in keys:
			res[k] = self._dict[k][row] # Grab appropriate row(s) from each column

		return res if type(row) in [int, numpy.int64] else Table(res)

	def getKey(self, key, col:Union[str,list]=[]) -> Union[dict,"Table"]:
		"""
		Gets one or multiple key row(s) from the table. Note that null rows are inserted where key lookup fails, and
		that includes the key columns.
		@param key	{various}			Rows of the table to retrieve. The input can be one of the 5 following types:
											1)	A list of lists/arrays.
												The (i, j)-th entry corresponds to key column i and key row j.
												Note: we don't support arrays of lists/arrays since numpy may modify the
												type in the mixed case, which breaks the lookup.
											2)	A list of tuples -- only valid for multi-key tables.
												The (i, j)-th entry corresponds to key row i and key column j.
												Note: as in case 1, we don't support arrays of tuples.
											3)	A single array/list -- only valid if there is a single key column.
											4)	A single atom -- only valid if there is a single key column.
											5)	A single tuple -- this is essentially case 2, but for a single element.
										Note in all cases, the types of the inputs must match the table key column types.
										Note also that, because of this flexibility, one cannot key a table by a single
										tuple or list/array column. Make multiple columns.
		@param col	{string|string[]}	Column(s) to get -- optional, default is all.
		@return		{dict|Table}		Single row  (as a dictionary) if requesting one row, or a table if multiple or
										none.
		"""
		idx, get = self._getKeyRow(key) # Look up keys

		if type(idx) == int:
			if idx == -1:
				res = self.mkNullRow(col) # Create a null row

				for c in res.keys():
					if c in self.keyCols():
						res[c] = get(key, 0, self.keyCols().index(c)) # Fill in (missed) key value

				return res
			else:
				return self.getRow(idx, col)
		else: # list of indices
			b = [i == -1 for i in idx] # Mask of misses

			if False in b: # If we got at least one hit
				res = self.getRow(idx, col) # Get rows (misses resolve to last row)

				if True in b:
					null = res.mkNullRow() # Null row
					misses =[] # Fill in misses with nulls or key values that missed
					w = misc.where(b) # Rows where we missed

					for c in null.keys():
						if c in self.keyCols():
							j = self.keyCols().index(c) # Index of key col
							misses.append([get(key, i, j) for i in w])
						else:
							misses.append(null[c])

					res.setRow(b, misses) # Fill in misses with null or key

				return res
			else: # All misses
				d = self.mkNullRow(col)
				n = len(idx)
				r = range(n)

				for k in d.keys(): 
					if k in self.keyCols():
						j = self.keyCols().index(k)
						d[k] = [get(key, i, j) for i in r]
					else:
						d[k] = [d[k]]*n # Scalar extend

				return Table(d)

	def setCol(self, col: Union[str,list], val: Union[list,numpy.ndarray], inPlace:bool=True) -> "Table":
		"""
		Sets a/many column(s) in the table (overwrites an existing column or defines a new one).
		@param col		{string|string[]}			Column(s) to set.
		@param val		{list|list[]|array|array[]}	Column value(s). Note that value length(s) must match table length.
		@param inPlace	{bool}						Do it in place or not.
		@return			{Table}						Table with column(s) redefined.
		"""
		if not inPlace:
			t = self.copy()
			t.setCol(col, val)
			return t

		# Standardize input.
		tc = type(col)

		if tc in [str, numpy.str_]:
			c = [col]
			v = [val]
		elif tc in [list, numpy.ndarray]:
			c = col
			v = val # Should be a list, though we don't check since it'll fail anyway
		else:
			raise TableException("Unrecognized column type: {}".format(tc))

		if len(c) == 0: return # Nothing to do

		# Validate.
		if len(c) != len(v): raise TableException("Set col/val mismatch")

		l = len(self) if len(self._dict) > 0 else len(v[0])
		r = range(len(c))

		for i in r:
			if type(v[i]) in [list, numpy.ndarray]: # If it's already a list/array
				if len(v[i]) != l: raise TableException("Set col length") # Ensure length
			else:
				v[i] = [v[i]]*l # Otherwise scalar extend

		# Should be safe to set.
		for i in r:
			self._dict[c[i]] = numpy.array(v[i]) # Set

	def setRow(self, row:Union[int,list,numpy.ndarray,slice], val, col:Union[str,list]=[], inPlace:bool=True) -> "Table":
		"""
		Sets row value(s) in the table. 
		@param row		{int|int[]|array|slice|bool[]}	Row(s) to set.
		@param val		{various}						Value(s) to set. Behaviour depends on type
															- tuple|tuple[]:
																Sets rows row-wise. List parallel to `row` and each
																tuple parallel to `col`.
															- atom|list|array|list[]|array[]:
																Sets rows column-wise. Outer list parallel to `col` and
																each inner list/array parallel to `row`.
														Note: we support scalar extension of an atom over multiple rows
														(single column), but nothing else.
		@param col		{string|string[]}				Columns to set -- optional, default is all.
		@param inPlace	{bool}							Do it in place or not. Note that setting strings, datetimes and
														timedeltas forces the whole column to be rewritten if the new
														values don't fit in the previous size of the array.
		@return			{Table}							Table with row(s) redefined.
		"""
		if not inPlace:
			t = self.copy()
			t.setRow(row, val, col)
			return t

		if col == []:
			c = self.cols()
		else:
			c = misc.mkList(col)
			allCols = self.cols()

			for x in c: 
				if not x in allCols: raise TableException("Set row unknown col: '{}'".format(x))

		c = list(self.cols()) if col == [] else misc.mkList(col)
		nc = len(c)
		tv = type(val)

		# Standardize `val` to a list of atoms, lists, arrays, or any mix thereof. i.e. `v[i][j] = i-th column, j-th row`.
		if tv == tuple:
			v = val
		elif tv == list:
			if len(val) == 0: return
			first = type(val[0])

			if first == tuple:
				r = range(len(val))

				for i in r:
					if len(val[i]) != nc: raise TableException("Set val/col mismatch") # Ensure tuples aren't blobby

				v = [[val[i][j] for i in r] for j in range(nc)] # Essentially flip
			elif first in [list, numpy.ndarray]:
				v = val # Should be in the correct shape already
			else:
				v = [val] # Setting a single column
		else:
			v = [val] # Anything else is assumed to be an atomic set

		# Set rows. In case anything goes wrong, backup previous values so we can restore.
		bkup = {}

		for i in range(nc):
			bkup[c[i]] = self.getRow(row, c[i])[c[i]] # Back up rows of the column we're about to change

			try:
				self._setRowOne(row, c[i], v[i])
			except Exception as ex:
				for key in bkup.keys(): self._dict[key][row] = bkup[key] # Restore backups
				raise TableException("Set row error: " + str(ex))

	def setKey(self, key, val, col:Union[str,list]=[], inPlace:bool=True) -> "Table":
		"""
		Sets the values of the given keys.
		@param key		{various}			Rows of the table to set. The input can be one of the 5 following types:
												1)	A list of lists/arrays.
													The (i, j)-th entry corresponds to key column i and key row j.
												2)	A list/array of tuples.
													The (i, j)-th entry corresponds to key row i and key column j.
												3)	A single array/list -- only valid if there is a single key column.
												4)	A single atom -- only valid if there is a single key column.
												5)	A single tuple -- this is essentially case 2, but for a single
													element.
											Note in all cases, the types of the inputs must match the table key column
											types.
											Note also that, because of this flexibility, one cannot key a table by a
											single tuple or list/array column. Make multiple columns.
		@param val		{any}				Values to set. See `setRow`, which this function ultimately calls.
		@param col		{string|string[]}	Column(s) to s -- optional, default is all non-key columns.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Table with key value(s) redefined..
		"""
		if not inPlace:
			t = self.copy()
			t.setKey(key, val, col)
			return t

		ogLen = len(self) # Snap our current length
		idx, get = self._getKeyRow(key) # Lookup keys

		# Handle misses. We append them to the end of the table. Note that appending causes us to fiddle with the memory
		# BEFORE calling `setRow`, so if there's an exception there, we have to revert.
		if type(idx) == int:
			if idx == -1:
				idx = len(self)
				newRow = self.mkNullRow() # Null row
				for i in range(len(self.keyCols())): newRow[self.keyCols()[i]] = get(key, 0, i) # Fill in key values
				self.append(newRow)
		else:
			if -1 in idx:
				n = 0 # Number of misses
				w = [] # Indices of misses

				for i in range(len(idx)):
					if idx[i] == -1:
						w.append(i) # Store index (basically doing 'where')
						idx[i] = ogLen + n # We'll append to end of table
						n += 1 # Bump

				d = self.mkNullRow() # Make a null row

				for k in d.keys():
					if k in self.keyCols():
						j = self.keyCols().index(k)
						d[k] = [get(key, i, j) for i in w] # If it's a key col, fill in keys we have
					else:
						d[k] = [d[k]]*n # Or put nulls

				self.append(Table(d)) # Append

		# By default, we only set non-key cols.
		if col == []:
			cc = []

			for c in self.cols():
				if not c in self._keys:
					cc.append(c)
		else:
			cc = col

		# Attempt to set the rows. If we can't, revert.
		try:
			self.setRow(idx, val, cc) # Set them
		except:
			self.deleteRow(range(ogLen, len(self))) # Revert if something goes wrong
			raise

	def deleteCol(self, col:Union[str,list], inPlace:bool=True) -> "Table":
		"""
		Deletes a/many column(s) from the table.
		@param col		{string|string[]}	Column(s) to delete.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Table with column(s) deleted.
		"""
		if not inPlace:
			t = self.copy()
			t.deleteCol(col)
			return t

		for c in misc.mkList(col): # For each column...
			if c in self.cols(): # If it's in the table...
				self._dict.pop(c) # Pop

	def takeCol(self, col=Union[str,list], inPlace:bool=True) -> "Table":
		"""
		Takes a/many column(s) from the table.
		@param col		{string|string[]}	Column(s) to take.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Table with subset of columns.
		"""
		cc = misc.mkList(col)
		self._chkCols(cc) # Ensure columns exist

		if inPlace:
			toDel = self.cols() # Grab full list of columnns
			for c in cc: toDel.remove(c) # Remove things we want to keep
			self.deleteCol(toDel) # Delete unwanted columns
			self.xcol(col) # Order accordingly
		else:
			return Table({c: self._dict[c] for c in cc})

	def deleteRow(self, row:Union[int,list,slice], inPlace:bool=True) -> Union[dict,"Table"]:
		"""
		Deletes one or multiple rows from the table.
		@param row	{int|int[]|slice|bool[]}	Row(s) to get.
		@return		{dict|Table}				Single row (as a dictionary) if requesting one row, or subset of table
												if multiple.
		"""
		if not inPlace:
			t = self.copy()
			t.deleteRow(row)
			return t

		# Build list of indices to keep.
		idx = list(range(len(self))) # Indices 0, 1, 2, ..., n-1 as a list
		toDel = numpy.array(idx)[row] # Indices to delete
		if type(toDel) == numpy.int64: toDel = [toDel] # Enlist
		toKeep = [] # List of indices to keep, build as we go

		for i in idx:
			if not i in toDel:
				toKeep.append(i)

		# Keep only good indices in each column.
		for col in self._dict.keys():
			self._dict[col] = self._dict[col][toKeep]

	def deleteKey(self, key, inPlace:bool=True) -> "Table":
		"""
		Sets the values of the given keys.
		@param key		{various}	Rows of the table to set. The input can be one of the 5 following types:
										1)	A list/array of lists/arrays.
											The (i, j)-th entry corresponds to key column i and key row j.
										2)	A list/array of tuples.
											The (i, j)-th entry corresponds to key row i and key column j.
										3)	A single array/list -- only valid if there is a single key column.
										4)	A single atom -- only valid if there is a single key column.
										5)	A single tuple -- this is essentially case 2, but for a single element.
									Note in all cases, the types of the inputs must match the table key column types.
									Note also that, because of this flexibility, one cannot key a table by a single
									tuple or list/array column. Make multiple columns.
		@param inPlace	{bool}		Do it in place or not.
		@return			{Table}		Table with key value(s) redefined..
		"""
		if not inPlace:
			t = self.copy()
			t.deleteKey(key)
			return t

		idx, dummy = self._getKeyRow(key) # Lookup keys

		if -1 in misc.mkList(idx):
			raise TableException("Invalid key to delete")

		self.deleteRow(idx)
 
	def type(self, col:Union[str,list]=[]) -> Union[str,list]:
		"""
		Gets column type.
		@param col	{string|string[]}	Column name(s) -- empty list for all.
		@return		{string|string[]}	Type(s).
		"""
		c = self.cols() if col == [] else col
		t = self.getCol(c)
		return [t[i].dtype.name for i in range(len(c))] if type(c) == list else t.dtype.name

	def typeCode(self, col:Union[str,list]=[]) -> Union[str,list]:
		"""
		Gets column type.
		@param col	{string|string[]}	Column name(s) -- empty list for all.
		@return		{string|string[]}	Type(s).
		"""
		t = self.type(col)
		return [Table._typeCodeOne(x) for x in t] if type(col) == list else Table._typeCodeOne(t)

	def toCSV(self, delimiter:str=",", file:str=None) -> Union[str,None]:
		"""
		Produces a CSV from the table.
		@param delimiter	{string}		Delimiter -- optional, default is ",".
		@param file			{string}		Location to save file -- optional, returns csv in string form if not
											specified.
		@return				{string[]|None}	The csv in string form, if no file location is specified, or nothing,
											otherwise.
		"""
		res = [str.join(delimiter, self.cols())] # Column names
		res += [str.join(delimiter, [str(x) for x in self.getRow(i).values()]) for i in range(len(self))] # Their values

		if file is None:
			return res
		else:
			res = str.join("\n", res)
			f = open(file, "w") # Open file, overwriting anything that's already there
			f.write(res)
			f.close()

	def xcol(self, order=Union[str,list], inPlace:bool=True) -> "Table":
		"""
		Sets the column order. Note that since we cannot change the order of the keys in a dictionary, this creates a
		copy of the table, hence it's potentially hard on memory.
		@param order	{string|string[]}	Column order. Any columns in the table that are not included go at the end
											in the order they were originally.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Table with columns reordered.
		"""
		if not inPlace:
			t = self.copy()
			t.xcol(order)
			return t

		lo = misc.mkList(order)
		res = {c: self._dict[c] for c in lo} # Order we want

		cols = list(self._dict.keys()) # All columns
		for c in lo: cols.remove(c) # Remove ones we've already ordered
		for c in cols: res[c] = self._dict[c] # Append remaining columns

		self._dict = res # Set new table

	def append(self, toAdd:Union["Table",dict,tuple], inPlace:bool=True) -> "Table":
		"""
		Appends rows to the table. Note, the columns of the table to append must have all the columns of this table. It
		may contain extras, but they are dropped. Note also that if the table is keyed, any appended rows are added
		without taking into account key upserts; user beware. Also note that we cannot append or append to nested
		columns.
		@param toAdd	{Table|dict|tuple}	Table or row(s) to add.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Table with rows appended.
		"""
		##~ Consider adding support for list[] to append vertically.
		##~ Consider adding support for lists of the above types. The tricky part with this is to append to each column
		##~ only once; hence there's some scanning through each row that needs to be done first. A bit messy and not
		##~ worth it unless a use-case pops up.
		if not inPlace:
			t = self.copy()
			t.append(toAdd)
			return t

		for c in self.cols():
			if len(numpy.shape(self.getCol(c))) > 1: raise TableException("Cannot append to table with nested columns")

		typ = type(toAdd)
		isEmpty = len(self.cols()) == 0 # Is the table empty? We can trivially copy 'toAdd'

		if typ == Table:
			if isEmpty:
				self._dict = toAdd._dict.copy() # Simple copy will do
				return

			# Before we start diddling with the memory, ensure all columns are present.
			for c in self.cols():
				if not c in toAdd.cols(): raise TableException("Append missing column: " + c)
				if len(numpy.shape(toAdd.getCol(c))) > 1: raise TableException("Cannot append nested columns")

			# Once we know all columns are there, we can add safely.
			for c in self.cols():
				self._dict[c] = numpy.append(self._dict[c], toAdd.getCol(c))

		elif typ == dict:
			# If table is empty, do a copy. We assume we're creating a one row table.
			if isEmpty:
				for c in toAdd.keys():
					self._dict[c] = numpy.array([toAdd[c]])

				return

			# Basically the same case as above.
			for c in self.cols():
				if c in toAdd.keys():
					if type(toAdd[c]) in [list, tuple]:
						raise TableException("Cannot append a nested dictionary")
				else:
					raise TableException("Append missing key: " + c)

			for c in self.cols():
				self._dict[c] = numpy.append(self._dict[c], toAdd[c])

		elif typ == tuple:
			lt = len(toAdd)
			c = self.cols()
			lc = len(c)

			if lt != lc:
				raise TableException(
					"Append: tuple of length {} but {} column{} in table".format(lt, lc, "" if lc == 1 else "s"))

			for i in range(lt):
				if type(toAdd[i]) in [list, tuple]:
					raise TableException("Cannot append a nested tuple")

			for i in range(lt):
				self._dict[c[i]] = numpy.append(self._dict[c[i]], toAdd[i])

		else:
			raise TableException("Table unkown append type: {}".format(typ))

	def save(self, loc:str):
		"""
		Saves table to disk.
		@param loc	{string}	Location to save table.
		"""
		# Save all tables splayed:
		# table/
		#	.d		--> column names and order
		#	col1	--> numpy file
		#	col2	--> numpy file
		#	...
		if not os.path.exists(loc): os.mkdir(loc) # Make the directory if it doesn't exist already
		dotD = open(loc + "/.d", "w+") # .d file for column order
		dotD.write(str.join("\n", self.cols())) # Write it as a plain text file (probably fine)
		dotD.close() # Close the file as we're done with it

		for c in self.cols():
			numpy.save(loc + "/" + c, self.getCol(c)) # Save each column as a numpy array

	def sort(self, cols:Union[str,list], desc:bool=False, inPlace:bool=True) -> "Table":
		"""
		Sorts the table in either ascending or descending order.
		@param cols		{string|string[]}	Column(s) to sort on (sorts cols_n within ... within cols_2 within cols_1).
		@param desc		{bool}				Do descending order (default is False).
		@param inPlace	{bool}				Do it in place.
		@return			{Table}				Sorted table.
		"""
		if not inPlace:
			t = self.copy()
			t.sort(cols, desc)
			return t

		# Do the sort (numpy sorts multiple cols in the opposite order you'd expect).
		cc = misc.mkList(cols) # Enlist
		self._dict = self[numpy.lexsort([self[c] for c in cc[::-1]])]._dict # Reverse the column order
		if desc: self._dict = self[::-1]._dict # Reverse the order if we wanted it desending

	def mkNullRow(self, col:Union[str,list]=[]) -> dict:
		"""
		Makes a null row from the table's schema. Use this to do the getKey function better (put null where no key match).
		@param col	{string|string[]}	Column(s) to grab. Empty for all.
		@return		{dict}				Row with null values.
		"""
		if col == []:
			return dict(zip(self.cols(), [null.getNull(t) for t in self.type()]))
		else:
			c = misc.mkList(col)
			return dict(zip(c, [null.getNull(t) for t in self.type(c)]))

	def aj(self, table:"Table", cols:Union[str,list], inPlace:bool=True) -> "Table":
		"""
		As-of join.
		@param table	{Table}		Table to join onto self. Must be sorted on c_n in ascending order within c_1, ..., c_n-1.
		@param cols		{string[]}	List of columns (c_1, ..., c_n) where c_1, ..., c_n-1 are the columns to join on
									and c_n is the as-of column (e.g. time).
		@param inPlace	{bool}		True if should be done in place.
		@return			{Table}		Self as-of joined with `table`.
		"""
		if not inPlace:
			t = self.copy()
			t.aj(table, cols)
			return t

		# Some error handling.
		for c in cols:
			if not c in self.cols():
				raise TableException("aj error: column '{}' not in left".format(c))

		for c in cols[:-1]:
			if not c in table.cols():
				raise TableException("aj error: column '{}' not in right".format(c))

		# Proceed as follows on each row of self:
		#	1) Match all but the last col in table. Do this once at the outset since we have an inbuilt function that
		#	   does this (`_getKeyMask`).
		#	2) Binary search over the last col to find the as-of point.
		#	3) Join all cols of table onto self.
		kc = cols[:-1] # Columns that act as key columns
		zkc = len(kc) == 0 # Are there any key cols?

		if len(kc) == 0:
			mask = [slice(None)]*len(self)
		else:
			mask, dummy, dummy1 = table._getKeyMask(self[kc], kc) # Mask of matching keys in left table

		aoc = cols[-1] # Last column to be used as the as-of column (e.g. time)
		l = self.getCol(aoc) # Left table as-of column
		r = table.getCol(aoc) # Right table as-of column -- assume sorted in ascending order within key
		t = table.deleteCol(cols, inPlace=False) # Subset of columns we now care about
		res = t[:0] # Table we'll join to left
		null = t.mkNullRow() # In case we need it ##~ Just use values?

		for i in range(len(self)):
			sr = r[mask[i]] # Subset of right table for this key
			j = misc.asof(sr, l[i]) # As-of index into (subset of) right table

			if j == -1:
				res.append(null) # Null if no such index (mask empty or as-of point predates first)
			else:
				if zkc:
					res.append(t[j]) ##~ Probably a better idea to fix 'where'
				else:
					res.append(t.getRow(misc.where(mask[i])[j]))

		self.setCol(res.cols(), list(res._dict.values()))

	def lj(self, right:"Table", key:Union[str,list]=[], col:Union[str,list]=[], inPlace:bool=True) -> "Table":
		"""
		Left joins a table. Note that in case of repeat keys in the right table, we use the first occurrence.
		@param right	{Table}				Table to join.
		@param key		{string|string[]}	Column(s) to use as keys in the right table. If not explicitly given, we use
											the key columns of the right table. If not explicitly given and the right
											table is not keyed, then we raise an exception.
		@param col		{string|string[]}	Column(s) to join from the right table. If not explicitly given, we use all
											the columns from the right table except those in 'key'.
		@param inPlace	{bool}				Do it in place or not.
		@return			{Table}				Self left joined with the right table.
		"""
		if not inPlace:
			t = self.copy()
			t.lj(right, key, col)
			return t

		# Resolve keys. Either explicitly given, or the key columns of the right table.
		if key == []:
			if right.keyCols() == []:
				raise TableException("Left join: no keys")

			k = right.keyCols()
			r = right
		else:
			k = misc.mkList(key)
			r = right.key(key, inPlace=False)

		toJoin = r.getKey(self.getCol(k), col) # Lookup left keys in right table
		self.join(toJoin) # Join

	def distinct(self) -> "Table":
		"""
		Returns the distinct elements of the table.
		@return	{Table}	Distinct rows of 'self'.
		"""
		res = self.getRow([]) # Start fresh

		for row in self:
			if not row in res:
				res.append(row)

		return res

	def by(self, byClause:Union[str,list,dict], aggClause:Union[str,list,dict], raggedCols:Union[str,list]=[]) -> "Table":
		"""
		Does a 'select {aggClause} by {byClause} from self'.
		@param byClause		{string|list|dictionary}	Column or list of columns to group by. If a dicitonary, keys are
														target column names, values are source column names or lists of
														the form (function, param_1, ..., param_n), possibly nested.
														Nested target columns not supported, but also not enforced --
														user beware.
		@param aggClause	{string|list|dictionary}	Column or list of columns to aggregate - aggregating in this way
														enlists the columns. If a dictionary, format is the same as
														'byClause'.
		@param raggedCols	{string|string[]}			Target column(s) which is/are ragged, i.e. a nested column with
														non-constant shape. By default (empty), we assume that only
														columns whose 'aggClause' is of the form 'targetCol: sourceCol'.
		@return				{Table}						Result of 'group by'. Note that the table is keyed by the target
														columns of 'byClause'.
		"""
		b = Table._stdClause(byClause) # Standardize by clause to dictionary
		a = Table._stdClause(aggClause) # Ditto for agg clause
		ks = Table(self._resolveClause(b)) # Construct key columns
		res = ks.distinct() # Distinct keys
		v = {k: [] for k in a.keys()} # Values -- build as we go

		for row in res:
			w = misc.where([row == k for k in ks]) # Where this key appears
			subTbl = self.getRow(w) # Sub-table corresponding to this key
			rowVals = subTbl._resolveClause(a) # Resolve clause on sub-table
			for k in v.keys(): v[k].append(rowVals[k]) # Append across row

		# Handle raggedness. If a column is ragged, numpy doesn't like it without specifying 'dtype=object'. However,
		# it's not always easy for us to know what columns are ragged. So we do one of the following:
		#	1) If none specified, we assume the only ragged columns are those where the clause is 'targetCol: sourceCol'.
		#	2) Put the onus on the caller; they can specify which columns need the ragged treatment.
		if raggedCols == []: # Option 1
			for k in a.keys():
				if type(a[k]) == str:
					v[k] = numpy.array(v[k], dtype=object)
		else: # Option 2
			for k in misc.mkList(raggedCols):
				v[k] = numpy.array(v[k], dtype=object)

		res.join(Table(v)) # Join
		res.key(ks.cols()) # Set key cols
		return res

	def join(self, right:"Table", inPlace:bool=True) -> "Table":
		"""
		Joins an equal length table to self (equivalent of doing ,' in q). Note that, like in q, common columns in the
		right table will overwrite existing columns in self.
		@param right	{Table}	Table to join.
		@param inPlace	{bool}	Do it in place or not.
		@return			{Table}	Self joined with right.
		"""
		if not inPlace:
			t = self.copy()
			t.join(right)
			return t

		# Table lengths must match. This error would be caught when trying to define columns, but may as well do it here.
		if len(self) != len(right):
			raise TableException("Join length mismatch, self={} right={}".format(len(self), len(right)))

		c = right.cols() # Columns we're adding (or overwriting)
		self.setCol(c, right.getCol(c)) # Join

	#-------------------------------------------------------------------------------------------------------------------
	# Private functions.
	#-------------------------------------------------------------------------------------------------------------------

	def _stdClause(clause:Union[str,list,dict]) -> dict:
		"""
		Standardizes by or agg clauses to dictionary format.
		@param clause	{string|list|dict}	Column, list of columns or dictionary.
		@return			{dict}				Transforms single column or list of columns to dictionary format.
		"""
		t = type(clause)

		if t == dict:
			res = clause
		elif t == str:
			return dict(zip([clause], [clause]))
		elif t == list:
			res = dict(zip(clause, clause))
		else:
			raise TableException("Incorrect clause type: {}".format(t))

		for k in res.keys():
			if type(k) != str:
				raise TableException("Incorrect clause key type: {}".format(type(k)))

		return res

	def _resolveClause(self, clause:dict) -> dict:
		"""
		Resolves an agg/by clause.
		@param clause	{dict}	Dictionary of target column name to possibly nested elements of the form
								'(function, param_1, param_2, ...)' or 'param'. In the former, the params are parameters
								to the function, in the latter, the param is used as a literal. Parameters are:
									- string:		column name.
									- atom:			used literally in the function.
									- string[1]:	string literal.
		@return			{dict}	Result of each sub-clause.
		"""
		return {k: self._resolveClauseOne(clause[k]) for k in clause.keys()}

	def _resolveClauseOne(self, clause:list):
		"""
		Resolves a single values of a full clause.
		@param clause	{list}		See '_resolveClause'.
		@return			{list|atom}	Column value, where an atom is scalar extended.
		"""
		typ = type(clause)

		if typ == str: # Column name
			return self.getCol(clause)
		elif typ == list:
			if len(clause) == 1 and type(clause[0]) == str: # String literal
				return clause[0]
			elif len(clause) > 0: # (fn, param_1, param_2, ...)
				return clause[0](*[self._resolveClauseOne(c) for c in clause[1:]]) # Apply function with nesting

		return clause # Anything else is assumed to be a literal

	def _getIdxType(idx) -> "type":
		"""
		Gets the type of indices supplied to `__{g|s}etitem__`.
		@param idx	{many}	See `__getitem__`.
		@return		{type}	Effective type.
		"""
		typ = type(idx)

		if typ in [list, numpy.ndarray]:
			return type(idx[0]) if len(idx) > 0 else int # Treat an empty list as you would an int list
		else:
			return type(idx)

	def _setRowOne(self, row:Union[int,list,numpy.ndarray,slice], col:str, val):
		"""
		Sets the row values for a single column.
		@param row	{int|int[]|array|slice|bool[]}	Row(s) to set.
		@param col	{string}						Column to set.
		@param val	{any}							Value(s) to set.
		"""
		# If we're inserting into a string column, expand if required.
		typ = str(self.getCol(col).dtype)

		if typ.startswith("<U"):
			maxWidth = int(typ[2:]) # Max number of characters array can handle
			tv = type(val)

			if tv == list:
				width = max([len(str(x)) for x in val])
			elif tv == numpy.ndarray:
				val = val.astype(str) # Cast now
				width = str(val.dtype)[2:]
			else: # Scalar
				val = str(val)
				width = len(val)

			if width > maxWidth: # One or more values are too wide ==> expand column
				newCol = list(self.getCol(col)) # Unarray-ify
				tr = type(row)

				if tr in [int, numpy.int64, slice]:
					newCol[row] = val # Simple index will do
				elif tr in [list, numpy.ndarray]:
					tf = type(row[0]) # Representative

					if tf in [int, numpy.int64]:
						for i in range(len(row)): newCol[row[i]] = val[i]
					elif tf in [bool, numpy.bool_]:
						j = 0 # Counter through val

						for i in range(len(row)):
							if row[i]:
								newCol[i] = val[j]
								j += 1
				else:
					raise TableException("Unknown row type") # Shouldn't really happy but just to cover our bases

				self.setCol(col, numpy.array(newCol).astype(str)) # Re-array-ify
				return

		self._dict[col][row] = val

	def _getKeyRow(self, key) -> tuple:
		"""
		Gets the row numbers of given key(s).
		@param key	{various}			Key(s) to lookup. See `getKey` or `setKey` for specs.
		@return		{int|int[],fn(3)}	Row number(s) for given key(s), where -1 indicates the key doesn't exist in the
										table, and a "getter" takes (k, i, j) and retuns the j-th component of the i-th
										key in the key "vector" k.
		"""
		if not self._isKeyed: raise TableException("Not keyed") # Get the easy case out of the way
		masks, isSingle, get = self._getKeyMask(key, self._keys)
		idx = [list(mask).index(True) if True in mask else -1 for mask in masks]
		if isSingle: return idx[0], get
		else: return idx, get

	def _getKeyMask(self, key, keyCols=list) -> tuple:
		"""
		Gets a mask of values matching a key or keys.
		@param key		{scalar|list|tuple|array|list[]|tuple[]|array[]}	Key(s) to lookup (see below for description).
		@param keyCols	{string[]}											Functional key columns.
		@return			{bool[][],bool,fn(3)}								List of masks (1 per key), a boolean
																			indicating if the key is a single scalar
																			value, and a dyadic function that takes
																			(k, i, j) and returns the j-th component of
																			i-th key in the key "vector" k.
		"""
		# We proceed based on the shape of `key`. There are 3 cases:
		#	a) ()		Single element lookup.
		#	b) (x)		Either a list/array or a tuple. If it's the former, it's multiple keys
		#				within one column, if it's the later, it's a single key across multiple columns.
		#	c) (x, y)	Either a list of lists or a list of tuples. If it's the former, it's a
		#				then the keys are oriented vertically, if it's the latter, then the keys
		#				are oriented horizontally.
		# We figure out what case we're dealing with and attempt to standardize the input
		# accordingly. That is, for all entries, we create a matrix and define a lambda for
		# accessing the i-th row and j-th column of the matrix. Moreover, we keep track of
		# whether we are requesting one or multiple elements, so that we can get the return
		# type right.
		# We define the following variables in this if-else block:
		#	- isSingle	{bool}		Specifies if requesting one key or multiple.
		#	- get		{lambda}	Function to access the rows and columns from the keys.
		#	- n			{int}		The number of keys to look up.
		shape = numpy.shape(key) # Get shape
		length = len(shape) # Dimension of shape
		nKeys = len(keyCols) # Columns considered keys in the table
		dimErr = "Key dimension mismatch" # Don't want to have to use it, but we will if we have to
		typErr = "Unrecognized key type" # Ditto
		getTupleList = lambda tupleList, row, col: tupleList[row][col] # For case 2
		getArrayList = lambda arrayList, row, col: arrayList[col][row] # For case 1

		if length == 0: # Case a
			if nKeys != 1:
				raise TableException(dimErr)

			isSingle = True
			get = lambda k, row, col: k
			n = 1

		elif length == 1: # Case b
			typ = type(key) # Need to determine list/array or tuple

			if typ in [list, numpy.ndarray]: # Multi-row lookup from a one-keyed table, presumably
				if nKeys != 1:
					raise TableException(dimErr)

				isSingle = False
				get = lambda k, row, col: k[row]
				n = shape[0]

			elif typ == tuple: # Single row lookup from a multi-keyed table
				if nKeys != shape[0]:
					raise TableException(dimErr)

				isSingle = True
				get = lambda k, row, col: k[col]
				n = 1

			else:
				raise TableException(typErr) # Not actually sure how to bring this on...

		elif length == 2: # Case c
			isSingle = False # Necessarily
			first = key[0] # Grab the first element
			typ = type(first) # Get its type, which determines what case we're in

			if typ in [list, numpy.ndarray]: # Column-wise
				if shape[0] != nKeys:
					raise TableException(dimErr)

				get = lambda k, row, col: k[col][row]
				n = shape[1]

			elif typ == tuple: # Row-wise
				if shape[1] != nKeys:
					raise TableException(dimErr)

				get = lambda k, row, col: k[row][col]
				n = shape[0]

			else:
				raise TableException(typErr) # As above, not sure sure how to trigger this...

		else:
			raise TableException("Unrecognized key shape")

		# Do the key look up. We proceed by going left to right, then up to down through the requested keys.
		# Whenever a key column matches, we store the information in case we need to look up the same value again.
		rows = range(n)
		cols = range(nKeys)
		mem = {c: {} for c in cols} # Used to store lookups we've already done (map of column (index) --> value --> boolean mask of matches)
		keyColVals = self.getCol(keyCols) # Key cols [key1, key2, ...] (i.e. list of arrays)
		length = len(self) # Length of the table
		masks = [] # Store matrix of masks

		for i in rows:
			mask = numpy.array([True]*length) # Mask of rows that match this key

			for j in cols:
				v = get(key, i, j) # Value to look up

				if not v in mem[j].keys(): # If we haven't seen this before
					mem[j][v] = keyColVals[j] == v # Remember it from here on out

				mask *= mem[j][v] # Whittle down the mask

			masks.append(mask)

		return masks, isSingle, get

	def _getType(s:str) -> "type":
		"""
		Gets type from character (see _types).
		@param s	{char}	Character type.
		@return		{type}	Actual type.
		"""
		if not s in Table._types.keys():
			raise TableException("Unknown column type: " + s)

		return Table._types[s]

	def _chkCols(self, col:list):
		"""
		Checks for existence of columns in table.
		@param col	{string[]}	Columns to check.
		@throws					If any column doesn't exist.
		"""
		for c in col:
			if not c in self.cols():
				raise TableException("Unknown column: {}".format(c))

	def _typeCodeOne(typ:str) -> str:
		if typ == "timedelta64*": return "n"
		if typ == "object": return "*"
		return typ[0]

	#----------------------------------------------------------------------------------------------------------
	# Python plumbing.
	#----------------------------------------------------------------------------------------------------------

	def __str__(self) -> str:
		"""
		Formats a table to print to console. This is the function that gets invoked when `print` is called.
		@return	{string}	A q-like representation of the table.
		"""
		return self.toString()

	def __repr__(self) -> str:
		"""
		Formats a table to print to console. This is the function that gets invoked when displaying in the command line.
		@return	{string}	A q-like representation of the table.
		"""
		return self.toString()

	def toString(self) -> str:
		"""
		Formats a table to print to console.
		@return	{string}	A q-like representation of the table.
		"""
		m, n = Table.getDispSize()

		# Build table as a matrix of strings.
		table = [] # Init
		cols = self.cols() # Table columns
		isLastKeyCol = [False]*len(cols) # Marker for last key column

		if self._isKeyed:
			for c in self._keys: # Extract keys
				cols.remove(c)

			cols = self._keys + cols # Put them in front
			isLastKeyCol[len(self._keys)-1] = True # Mark last key column

		table.append(cols) # First row is the columns
		tooLong = len(self) > n - 2 # Is the table too long to fully fit?
		
		for i in range(n-3 if tooLong else len(self)): # If too long, reserve last row for trailing ".."
			row = self.getRow(i) # Grab the raw row
			table.append([str(row[c]) for c in cols]) # Cast to string

		# Determine width of each column. We want at least 1 space between each. If the table
		# is too wide, we drop columns. However, we'll always display the first column even
		# if it stretches too far and has to wrap. Create a more sensible table next time!
		width = [] # Witdths reserved for each column
		lastCol = len(table[0]) - 1

		for j in range(len(table[0])):
			width.append((j<lastCol) + isLastKeyCol[j] + 
				max([len(table[i][j]) for i in range(len(table))])) # Max width + " " (+ "|" for last key col)

		if sum(width) > m: # If too wide, we'll have to drop some columns
			nCols = 1 # Always display at least one
			cumWidth = width[0] # Cumulative width

			for i in range(1, len(table[0])):
				cumWidth += width[i]

				if cumWidth > n - 2:
					break # Can't fit anymore

			for i in range(len(table)):
				table[i] = table[i][:nCols] # Trim columns we can't display
				table[i].append("..") # Indicate there are more columns

			width = width[:nCols] # Trim widths as well
			width.append(2) # New column we just added

		# Pad the cells.
		for i in range(len(table)):
			for j in range(len(table[i])):
				if isLastKeyCol[j]:
					table[i][j] = table[i][j].ljust(width[j]-2, " ") + "| "
				else:
					table[i][j] = table[i][j].ljust(width[j], " ") # Pad with trailing spaces

		# Display the table.
		res = ""

		for j in range(len(table[0])):
			res += table[0][j] # First row

		res += "\n" + ("-"*max(1,sum(width))) # Line of dashes

		for i in range(1, len(table)):
			res += "\n"

			for j in range(len(table[i])):
				res += table[i][j]

		if tooLong:
			res += "\n.." # Indicate the table continues

		return res

	def __getitem__(self, idx) -> Union[dict,"Table"]:
		"""
		Indexes into the table.
		@param idx	{many}	Index. Behaviour depends on whether or not the table is keyed.
								- keyed:	`idx` corresponds to row(s). If multiple keys,
											`idx` is a (list of) tuples. If single key, then
											`idx` is a (list of) atoms. In all cases, the
											type(s) must match the type(s) of the key(s).
								- unkeyed:	`idx` corresponds to either row(s), if integers,
											or column(s), if strings.
		@return		{many}	Dictionary, if requesting 1 row, table if requesting rows,
							an array if requesting 1 column, and a list of arrays if requesting
							multiple columns.
		"""	
		if self._isKeyed:
			return self.getKey(idx)

		typ = Table._getIdxType(idx) # Type we're dealing with

		if typ in [int, slice, bool, numpy.bool_, numpy.int64]:
			return self.getRow(idx)
		elif typ in [str, numpy.str_]:
			return self.getCol(idx)
		else:
			raise TableException("Unimplemented getitem type: {}".format(typ))

	def __len__(self) -> int:
		"""
		Gets the length (i.e. number of rows) of a table. This is the function that gets invoked by `len`.
		@return	{int}	Length of table.
		"""
		c = self.cols()
		return len(self._dict[c[0]]) if len(c) else 0

	def __setitem__(self, idx, item):
		"""
		Sets item(s) in the table.
		@param idx	{many}	Index/indices to set (see `__getitem__`).
		@param item	{many}	Item(s) to set index/indices to.
		"""
		if self._isKeyed:
			return self.setKey(idx, item)

		typ = Table._getIdxType(idx) # Type we're dealing with
		
		if typ in [int, slice, bool, numpy.bool_, numpy.int64]:
			return self.setRow(idx, item)
		if typ in [str, numpy.str_]:
			return self.setCol(idx, item)
		else:
			raise TableException("Unimplemented setitem type: {}".format(typ))

	def __eq__(self, other:"Table") -> bool:
		"""
		Checks equality of tables. We have equality if:
			0) Both are tables.
			1) Columns are the same and in the same order.
			2) Key columns are the same.
			3) Types of each column are the same (allow string cols to expand or contract).
			4) Shape of each column (as a numpy array) is the same.
			5) Each element of each column matches.
		@param other	{any}	Object to compare (though typically a Table).
		@return			{bool}	True if they're equal.
		"""
		if not isinstance(other, Table): return False
		if self.cols() != other.cols(): return False
		if self.keyCols() != other.keyCols(): return False

		for c in self.cols():
			sc = self.getCol(c)
			oc = other.getCol(c)
			if sc.shape != oc.shape: return False

			# Compare type. In general, this is easy, except for strings when the width of the strings is significant.
			if sc.dtype != oc.dtype:
				if not sc.dtype.str.startswith("<U") and oc.dtype.str.startswith("<U"): # If both strings, look past it
					return False

			vals = sc == oc
			if type(vals) != numpy.ndarray: raise TableException("Cannot determine equality (nested column)?") ##~
			if vals.all(): continue # Easy case, but because of "nan", inequality isn't necessarily good enough

			# In cases where we have an inequality, we also check for null-ness, since "nan == nan" returns false.
			# We'll return false on the first instance where both entries are not null.
			for i in range(len(sc)):
				if not vals[i]:
					if not (null.isNull(sc[i]) and null.isNull(oc[i])): return False

		return True # If we made it all the way through, they're equal

	def __iter__(self) -> "Table":
		"""
		Sets up iteration.
		@return	{Table}	Self.
		"""
		self._index = 0
		return self

	def __next__(self) -> dict:
		"""
		Gets next row.
		@return	{dict}	Next row.
		"""
		if self._index == len(self):
			raise StopIteration

		i = self._index
		self._index += 1
		return self.getRow(i)


class TableException(Exception):
	pass