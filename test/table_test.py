######################################################################
# Table tests.
######################################################################

import numpy, os, glob
from unittest import TestCase
from src.table import Table, TableException
from src import null, mock

class TableTest(TestCase):
	RESOURCES = "test/resources/"

	def tearDown(self):
		mock.resetMocks()

	def test_init(self):
		# Create an empty table.
		t = Table()
		self.assertEqual(t._dict, {})

		# Create a non-empty table.
		t = Table({
			"x": [1, 2],
			"col2": [numpy.datetime64("2020-04-19T09:00:00"), numpy.datetime64("2020-04-19T10:30:00")],
			"z": numpy.array([True, False])})
		exp = {
			"x": numpy.array([1, 2]),
			"col2": numpy.array([numpy.datetime64("2020-04-19T09:00:00"), numpy.datetime64("2020-04-19T10:30:00")]),
			"z": numpy.array([True, False])}
		self.assertEqual(list(t._dict.keys()), ["x", "col2", "z"])
		self.assertTrue((t["x"] == exp["x"]).all())
		self.assertTrue((t["col2"] == exp["col2"]).all())
		self.assertTrue((t["z"] == exp["z"]).all())

		# Ensure we init by value and not by reference.
		d = {"x": numpy.array([1, 2])}
		t = Table(d)
		self.assertTrue((t["x"] == d["x"]).all()) # Copy values
		self.assertFalse(t["x"] is d["x"]) # But not reference

		# Try to init with bad type.
		self.assertRaisesRegex(TableException, "init param not dict", Table, 1)	

		# Length error.
		self.assertRaisesRegex(TableException, "length", Table, {"short": [1, 2], "long": [1, 2, 3]})

		# Empty typed (string).
		c = ["b1", "s", "d", "i1", "i2", "f", "obj", "b2"]
		s = "bsdiif*bn"
		t = Table(dict(zip(c, s)))
		exp = dict(zip(c, [numpy.array([]).astype(x) for x in
			[bool, str, numpy.datetime64, int, int, float, object, bool, numpy.timedelta64]]))
		self.assertEqual(list(t._dict.keys()), c)
		for col in c:
			self.assertTrue((t[col] == exp[col]).all())

		# Empty typed (type).
		s = [bool, str, numpy.datetime64, int, numpy.int64, float, object, bool, numpy.timedelta64]
		t = Table(dict(zip(c, s)))
		self.assertEqual(list(t._dict.keys()), c)
		for col in c:
			self.assertTrue((t[col] == exp[col]).all())

	def test_size(self):
		mock.mock(Table, "_dispWidth", 1000)
		mock.mock(Table, "_dispHeight", 1000)

		# When popen works.
		mock.mock(os._wrap_close, "read", lambda x: "90 160")
		Table.setc(0.5)
		self.assertEqual(Table.getDispSize(), (160, 45))

		# When popen doesn't work.
		mock.mock(os._wrap_close, "read", lambda x: "")
		Table._dispWidth = 1000
		Table._dispHeight = 1000
		Table.setc(0.6666)
		self.assertEqual(Table.getDispSize(), (1000, 1000))

		# Set size manually.
		Table.setDispSize()
		self.assertEqual(Table.getDispSize(), (1000, 1000))

		Table.setDispSize(2)
		self.assertEqual(Table.getDispSize(), (2, 1000))

		Table.setDispSize(y=10)
		self.assertEqual(Table.getDispSize(), (2, 10))

		Table.setDispSize(16, 9)
		self.assertEqual(Table.getDispSize(), (16, 9))

	def test_fromCSV(self):
		# Bad type.
		self.assertRaisesRegex(TableException, "Unknown column type: y",
			Table.fromCSV, TableTest.RESOURCES + "csv_test_1.csv", "yif")

		# Comma-delimited with whitespaces.
		act = Table.fromCSV(TableTest.RESOURCES + "csv_test_1.csv", "isfb")
		exp = Table({"x": [1, 2], "col2": ["blah", "yada"], "z": [1.2, 3.4], "w": [True, False]})
		self.assertEqual(act, exp)

		# Same as above but ignore columns/change types.
		act = Table.fromCSV(TableTest.RESOURCES + "csv_test_1.csv", "f *")
		exp.deleteCol(["col2", "w"])
		exp["x"] = exp["x"].astype(numpy.float64)
		exp["z"] = exp["z"].astype(str).astype(object)
		self.assertEqual(act, exp)

		# Use actual types instead of chars.
		act = Table.fromCSV(TableTest.RESOURCES + "csv_test_1.csv", [float, None, object])
		self.assertEqual(act, exp)

		# Pipe-delimited.
		act = Table.fromCSV(TableTest.RESOURCES + "csv_test_2.csv", "d d", "|")
		exp = Table({
				"date":
					numpy.array(["2021-10-22", "2021-10-23", "2021-10-23"]).astype(numpy.datetime64),
				"time":
					numpy.array(["2021-10-22 18:49", "2021-10-23 09:00", "2021-10-23 12:20"]).astype(numpy.datetime64)})
		self.assertEqual(act, exp)

		# Empty.
		act = Table.fromCSV(TableTest.RESOURCES + "csv_test_empty.csv", "si")
		exp = Table({"x": "s", "y": "i"})
		self.assertEqual(act, exp)

	# Basically the same as above, so just test that the string input works.
	def test_fromCsvString(self):
		string = "col1,col2,z\n1,1.2,abc\n2,3.4,de"
		act = Table.fromCsvString(string, "ifs")
		exp = Table({"col1": [1, 2], "col2": [1.2, 3.4], "z": ["abc", "de"]})
		self.assertEqual(act, exp)

	# Light since this function just calls 'append' under the covers.
	def test_raze(self):
		# Basic case.
		x = Table({"x": [1, 2, 3], "y": ["abc", "de", "f"],
			"z": numpy.array(["2021-10-23", "2021-10-24", "2021-10-25"]).astype(numpy.datetime64)})
		y = Table({"x": [10, 20], "y": ["xy", "z"],
			"z": numpy.array(["2021-10-26 09:30", "2021-10-26 09:35"]).astype(numpy.datetime64)})
		act = Table.raze([x, y])
		exp = Table({"x": [1, 2, 3, 10, 20], "y": ["abc", "de", "f", "xy", "z"],
			"z": numpy.array(["2021-10-23", "2021-10-24", "2021-10-25", "2021-10-26 09:30", "2021-10-26 09:35"]).
			astype(numpy.datetime64)})
		self.assertEqual(act, exp)

		# Resilient to column order.
		z = Table({"z": [numpy.datetime64("2021-10-31")], "x": [100], "y": ["hello"]})
		act = Table.raze([x, z, y])
		exp = Table({"x": [1, 2, 3, 100, 10, 20], "y": ["abc", "de", "f", "hello", "xy", "z"],
			"z": numpy.array(["2021-10-23", "2021-10-24", "2021-10-25", "2021-10-31", "2021-10-26 09:30", 
			"2021-10-26 09:35"]).astype(numpy.datetime64)})
		self.assertEqual(act, exp)

		# Fail if column mismatch.
		x = Table({"x": [1, 2]})
		y = Table({"y": [3, 4, 5]})
		self.assertRaisesRegex(TableException, "Append missing column: x", Table.raze, [x, y])

		# Doesn't care about type.
		x = Table({"x": [1, 2]})
		y = Table({"x": [True, False]})
		act = Table.raze([x,y])
		exp = Table({"x": [1, 2, 1, 0]})
		self.assertEqual(act, exp)

		# Trims excess columns.
		x = Table({"x": [1, 2], "y": [True, False]})
		y = Table({"x": [10, 20, 30], "extra": [1, 2, 3], "y": [True, True, False]})
		z = Table({"y":[True], "x": [100], "blah": ["abc"]})
		act = Table.raze([x, y, z])
		exp = Table({"x": [1, 2, 10, 20, 30, 100], "y": [True, False, True, True, False, True]})
		self.assertEqual(act, exp)

	def rmTbl(path):
		if "test_table_save_load" in os.listdir(TableTest.RESOURCES):
			for f in glob.glob(path + "/*"): os.remove(f)
			if os.path.isfile(path + "/.d"): os.remove(path + "/.d")
			os.rmdir(path)

	def test_save_load(self):
		testFile = TableTest.RESOURCES + "test_table_save_load"
		TableTest.rmTbl(testFile) # In case it's here from previous run

		# Try to load but it's not there.
		self.assertRaisesRegex(TableException, "Unable to read .d file: *", Table.load, testFile)

		# Try to load but '.d' missing.
		os.mkdir(testFile)
		self.assertRaisesRegex(TableException, "Unable to read .d file: *", Table.load, testFile)

		# Save and reload table.
		exp = Table({"date": numpy.array(["2021-10-23", "2021-10-24"]).astype(numpy.datetime64), "price": [3.45, 9.01]})
		exp.save(testFile)
		act = Table.load(testFile)
		self.assertEqual(act, exp)

		# Error reading column.
		os.remove(testFile + "/price.npy")
		self.assertRaisesRegex(TableException, "Unable to read column 'price':*", Table.load, testFile)

		TableTest.rmTbl(testFile) # Clean up

	def test_key_unkey(self):
		# Unkeyed.
		t = Table({"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [True, False, True]})
		self.assertFalse(t.isKeyed())
		act = t.keyCols()
		exp = []
		self.assertEqual(act, exp)

		# Key it.
		c = t.key("x", inPlace=False)
		self.assertFalse(t.isKeyed()) # Original should be unchanged
		self.assertTrue(c.isKeyed())
		act = c.keyCols()
		exp = ["x"]
		self.assertEqual(act, exp)

		self.assertNotEqual(t, c)
		t.key("x")
		self.assertEqual(t, c)

		# Change keys
		c = t.key(["y", "x"], inPlace=False)
		act = t.keyCols()
		exp = ["x"]
		self.assertEqual(act, exp)
		act = c.keyCols()
		exp = ["y", "x"]
		self.assertTrue(c.isKeyed())
		self.assertEqual(act, exp)

		self.assertNotEqual(t, c)
		t.key(["y", "x"])
		self.assertEqual(t, c)

		# Key empty (removes keys)
		c = t.key([], inPlace=False)
		act = t.keyCols()
		exp = ["y", "x"]
		self.assertEqual(act, exp)
		self.assertFalse(c.isKeyed())
		act = c.keyCols()
		exp = []
		self.assertEqual(act, exp)

		self.assertNotEqual(t, c)
		t.key([])
		self.assertEqual(t, c)

		# Unkey.
		t.key(["x", "y"])
		self.assertTrue(t.isKeyed())
		c = t.unkey(False)
		self.assertTrue(t.isKeyed())
		self.assertFalse(c.isKeyed())
		act = c.keyCols()
		exp = []
		self.assertEqual(act, exp)

		self.assertNotEqual(t, c)
		t.unkey()
		self.assertEqual(t, c)

		# Unkey again (does nothing).
		c = t.unkey(False)
		self.assertEqual(t, c)
		t.unkey()
		self.assertEqual(t, c)

		# Try to key unknown columns error.
		self.assertRaisesRegex(TableException, "Column 'blah' not in table", t.key, ["x", "blah"])
		self.assertEqual(t, c)

		# Multi-character key.
		t = Table({"myKeyCol": [1, 2, 3], "myValCol": [10, 20, 30]})
		t.key("myKeyCol")
		self.assertEqual(t.keyCols(), ["myKeyCol"])

	def test_copy(self):
		t = Table({"col1": numpy.array(["2021-10-27", "2021-10-28"]).astype(numpy.datetime64), "col2": [1, 2]})
		c = t.copy()
		self.assertEqual(t, c) # Should be equivalent
		self.assertFalse(t is c) # But not the same
		self.assertFalse(t["col1"] is c["col1"]) # Same should hold for columns
		self.assertFalse(t["col2"] is c["col2"])

	def test_cols(self):
		t = Table({"col1": [1], "col2": ["a"], "col3": [1.22]})
		act = t.cols()
		exp = ["col1", "col2", "col3"]
		self.assertEqual(act, exp)

	def test_count(self):
		# Empty.
		t = Table()
		act = t.count()
		exp = 0
		self.assertEqual(act, exp)

		# Length 1.
		t = Table({"col1": [1], "col2": ["a"], "col3": [1.22]})
		act = t.count()
		exp = 1
		self.assertEqual(act, exp)

		# Length > 1.
		t = Table({"x": [1, 2, 3, 4, 5]})
		act = t.count()
		exp = 5
		self.assertEqual(act, exp)

	def test_getCol(self):
		# Return a single column.
		t = Table({"col1": [1, 2, 3], "col2": ["abc", "def", "gh"], "col3": [1.2, 3.4, 5.6]})
		act = t.getCol("col1")
		exp = numpy.array([1, 2, 3])
		self.assertTrue((act == exp).all())

		# Return single enlisted.
		act = t.getCol(["col1"])
		exp = [exp]
		self.assertEqual(len(act), 1)
		self.assertTrue((act[0] == exp[0]).all())

		# Return multiple.
		act = t.getCol(["col1", "col3"])
		exp = [numpy.array([1, 2, 3]), numpy.array([1.2, 3.4, 5.6])]
		res = True
		for i in range(len(exp)): res &= (act[i] == exp[i]).all()
		self.assertTrue(res)

		# Column that doesn't exist.
		self.assertRaisesRegex(Exception, "'unknownCol'",  t.getCol, "unknownCol")


	# Slice is explicitly not tested here since it's easier to test as an actual indexing.
	def test_getRow(self):
		# Empty table.
		t = Table()
		self.assertRaisesRegex(TableException, "Can't get rows from an empty table", t.getRow, 0)

		# Zero-row table.
		t = Table({"x": "d", "y": "i", "z": "f"})
		self.assertEqual(t.getRow([]), t)
		self.assertRaises(IndexError, t.getRow, 0)
		act = t.getRow([], ["z", "x"])
		exp = Table({"z": "f", "x": "d"})
		self.assertEqual(act, exp)

		# Single row.
		t = Table({"x1": [1, 2, 3, 4, 5], "x2": ["a", "b", "c", "d", "e"], "x3": [1.1, 1.2, 1.3, 1.4, 1.5]})
		act = t.getRow(1)
		exp = {"x1": 2, "x2": "b", "x3": 1.2}
		self.assertEqual(act, exp)

		act = t.getRow(-1, "x2")
		exp = {"x2": "e"}
		self.assertEqual(act, exp)

		# Multiple row.
		act = t.getRow([3, 1, -1])
		exp = Table({"x1": [4, 2, 5], "x2": ["d", "b", "e"], "x3": [1.4, 1.2, 1.5]})
		self.assertEqual(act, exp)

		act = t.getRow([0, 2], ["x1", "x2"])
		exp = Table({"x1": [1, 3], "x2": ["a", "c"]})
		self.assertEqual(act, exp)

		# Mask.
		act = t.getRow([True, True, False, True, False])
		exp = Table({"x1": [1, 2, 4], "x2": ["a", "b", "d"], "x3": [1.1, 1.2, 1.4]})
		self.assertEqual(act, exp)

		act = t.getRow([True, False, False, False, False], "x2")
		exp = Table({"x2": ["a"]})
		self.assertEqual(act, exp)

		# Unknown col.
		self.assertRaisesRegex(KeyError, "y2", t.getRow, [1, 2], ["x1", "y2", "x3"])

		# Index out of table.
		self.assertRaises(IndexError, t.getRow, 5)
		self.assertRaises(IndexError, t.getRow, [1, 2, 10])
		self.assertRaises(IndexError, t.getRow, [True, False])
		self.assertRaises(IndexError, t.getRow, [True]*6)

		# Empty list.
		self.assertEqual(len(t.getRow([])), 0)

	def test_getKey(self):
		single = Table({"x": ["k1", "k2", "k3"], "y": [1, 2, 3], "z": [True, False, True]})
		single.key("x")
		double = Table({"kc1": ["ab", "cd", "ab", "cd"], "kc2": [1, 1, 2, 2],
			"vc1": [1.1, 1.1, 2.2, 2.2], "vc2": ["a", "c", "a", "c"], "vc3": [True, False, True, False]})
		double.key(["kc1", "kc2"])

		# Case 1: Lists of lists/arrays.
		act = single.getKey([["k1", "k3"]])
		exp = Table({"x": ["k1", "k3"], "y": [1, 3], "z": [True, True]})
		self.assertEqual(act, exp)

		act = single.getKey([numpy.array(["k1", "k3"])])
		self.assertEqual(act, exp)

		l1 = ["cd", "ab", "ab", "ab"]
		l2 = [2, 1, 1, 2]
		act = double.getKey([l1, l2])
		exp = Table({"kc1": l1, "kc2": l2, "vc1": [2.2, 1.1, 1.1, 2.2],
			"vc2": ["c", "a", "a", "a"], "vc3": [False, True, True, True]})
		self.assertEqual(act, exp)

		act = double.getKey([numpy.array(l1), l2])
		self.assertEqual(act, exp)

		act = double.getKey([numpy.array(l1), numpy.array(l2)])
		self.assertEqual(act, exp)

		# Case 1 with null row(s).
		act = single.getKey(["k2", "blah", "k2"])
		exp = Table({"x": ["k2", "blah", "k2"], "y": [2, null.INT, 2], "z": [False]*3})
		self.assertEqual(act, exp)

		act = double.getKey([["ab", "ab", "blah", "blah"], [1, 3, 1, 10]])
		exp = Table({"kc1": ["ab", "ab", "blah", "blah"], "kc2": [1, 3, 1, 10],
			"vc1": [1.1, null.FLOAT, null.FLOAT, null.FLOAT], "vc2": ["a", null.STRING, null.STRING, null.STRING],
			"vc3": [True, null.BOOL, null.BOOL, null.BOOL]})
		self.assertEqual(act, exp)

		# Case 1 with wrong number of keys.
		self.assertRaisesRegex(TableException, "Key dimension mismatch", single.getKey, [["k1", "k2"], ["k1", "k2"]])
		self.assertRaisesRegex(TableException, "Key dimension mismatch", double.getKey, [1, 2, 3])

		# Case 2: List of tuples.
		act = double.getKey([("ab", 1), ("cd", 2)])
		exp = Table({"kc1": ["ab", "cd"], "kc2": [1, 2], "vc1": [1.1, 2.2], "vc2": ["a", "c"], "vc3": [True, False]})
		self.assertEqual(act, exp)

		# Case 2 with null row(s).
		act = double.getKey([("ab", 1), ("cd", 10), ("cd", 2), ("blah", 10)])
		exp = Table({"kc1": ["ab", "cd", "cd", "blah"], "kc2": [1, 10, 2, 10],
			"vc1": [1.1, null.FLOAT, 2.2, null.FLOAT], "vc2": ["a", null.STRING, "c", null.STRING],
			"vc3": [True, null.BOOL, False, null.BOOL]})
		self.assertEqual(act, exp)

		# Case 2 with wrong number of keys.
		self.assertRaisesRegex(TableException, "Key dimension mismatch", single.getKey, 
			[("ab", 1), ("cd", 10), ("cd", 2), ("blah", 10)])
		self.assertRaisesRegex(TableException, "Key dimension mismatch", double.getKey,
			[("ab", 1, 1), ("cd", 10, 1), ("cd", 2, 1), ("blah", 10, 1)])

		# Case 3: Single list.
		act = single.getKey(["k2", "k1"])
		exp = Table({"x": ["k2", "k1"], "y": [2, 1], "z": [False, True]})
		self.assertEqual(act, exp)

		# Case 3 with null rows.
		act = single.getKey(["k2", "k1", "blah"])
		exp = Table({"x": ["k2", "k1", "blah"], "y": [2, 1, null.INT], "z": [False, True, null.BOOL]})
		self.assertEqual(act, exp)

		# Case 4: Single atom.
		act = single.getKey("k1")
		exp = {"x": "k1", "y": 1, "z": True}
		self.assertEqual(act, exp)

		# Case 4 with null rows.
		act = single.getKey("notAnActualKey")
		exp = {"x": "notAnActualKey", "y": null.INT, "z": null.BOOL}
		self.assertEqual(act, exp)

		# Case 4 with wrong number of keys.
		self.assertRaisesRegex(TableException, "Key dimension mismatch", double.getKey, 1)

		# Case 5: Single tuple.
		act = double.getKey(("ab", 2))
		exp = {"kc1": "ab", "kc2": 2, "vc1": 2.2, "vc2": "a", "vc3": True}
		self.assertEqual(act, exp)

		# Case 5 with null rows.
		act = double.getKey(("a", 1))
		exp = {"kc1": "a", "kc2": 1, "vc1": null.FLOAT, "vc2": null.STRING, "vc3": null.BOOL}
		self.assertEqual(act, exp)

		# Case 5 with wrong number of keys.
		self.assertRaisesRegex(TableException, "Key dimension mismatch", single.getKey, ("a", 1))

		# Select only a subset of columns.
		act = single.getKey("kc2", "z")
		exp = {"z": False}
		self.assertEqual(act, exp)

		act = double.getKey([("ab", 1), ("ab", 2), ("blah", 1)], ["kc1", "vc2"])
		exp = Table({"kc1": ["ab", "ab", "blah"], "vc2": ["a", "a", null.STRING]})
		self.assertEqual(act, exp)

		# Key lookup matches multiple rows (should return first match).
		repeat = Table({"x": [1, 1, 2, 2, 2], "y": [1, 2, 1, 2, 1], "z": [10, 20, 30 , 40, 50]})
		repeat.key(["x", "y"])
		act = repeat.getKey([[1, 2], [1, 1]])
		exp = Table({"x": [1, 2], "y": [1, 1], "z": [10, 30]})
		self.assertEqual(act, exp)

		# Table not keyed.
		single.unkey()
		self.assertRaisesRegex(TableException, "Not keyed", single.getKey, [1, 2])

		# Single row miss, subset of colums.
		t = Table({"x": [1, 2], "y": [1.2, 3.4], "z": ["a", "b"]})
		t.key("x")
		act = t.getKey(3, ["z", "x"])
		exp = {"z": null.STRING, "x": 3}
		self.assertEqual(act, exp)

		# Empty table, full single row.
		t = Table({"c1": "i", "c2": "f", "c3": "s"})
		t.key("c1")
		act = t.getKey(1)
		exp = {"c1": 1, "c2": null.FLOAT, "c3": null.STRING}
		self.assertEqual(act, exp)

		# Empty table, subset of single row.
		act = t.getKey(2, "c2")
		exp = {"c2": null.FLOAT}
		self.assertEqual(act, exp)

		# Empty table, full multi row.
		act = t.getKey([3, 4])
		exp = Table({"c1": [3, 4], "c2": [null.FLOAT]*2, "c3": [null.STRING]*2})
		self.assertEqual(act, exp)

		# Empty table, subset of multi row.
		act = t.getKey([5, 6, 7], ["c1", "c3"])
		exp = Table({"c1": [5, 6, 7], "c3": [null.STRING]*3})
		self.assertEqual(act, exp)

	def test_setCol(self):
		og = Table({"x": [1, 2, 3], "y": ["a", "b", "c"]}) # Save original
		t = og.copy() # Working copy

		# Length/mismatch errors
		self.assertRaisesRegex(TableException, "Set col length", t.setCol, "z", [1, 2])
		self.assertRaisesRegex(TableException, "Set col length",
			t.setCol, ["x", "z"], [[10, 20, 30], ["a", "b", "c", "d"]])
		self.assertRaisesRegex(TableException, "Set col/val mismatch",
			t.setCol, ["x", "z"], [[10, 20], [True, False], [-1, -2]])
		self.assertEqual(t, og)

		# Single column list.
		t.setCol("x", [10, 20, 30])
		exp = Table({"x": [10, 20, 30], "y": ["a", "b", "c"]})
		self.assertEqual(t, exp)

		# Single column array.
		t.setCol("y", numpy.array(["abc", "de", "fgh"]))
		exp = Table({"x": [10, 20, 30], "y": ["abc", "de", "fgh"]})
		self.assertEqual(t, exp)

		# New column.
		z = numpy.array(["2021-11-10", "2021-11-11", "2021-11-12"]).astype(numpy.datetime64)
		t.setCol("z", z)
		exp = Table({"x": [10, 20, 30], "y": ["abc", "de", "fgh"], "z": z})
		self.assertEqual(t, exp)

		# Multi column list, no new cols.
		t.setCol(["y", "x"], [["A", "B", "C"], [1.1, 1.2, 1.3]])
		exp = Table({"x": [1.1, 1.2, 1.3], "y": ["A", "B", "C"], "z": z})
		self.assertEqual(t, exp)

		# Multi column array, new cols.
		z = numpy.array(["2022-11-10", "2022-11-11", "2022-11-12"]).astype(numpy.datetime64)
		t.setCol(["z", "col1", "col2"], [z, [True]*3, [1, 0, 1]])
		exp = Table({"x": [1.1, 1.2, 1.3], "y": ["A", "B", "C"], "z": z, "col1": [True]*3, "col2": [1, 0, 1]})
		self.assertEqual(t, exp)

		# Not in place.
		t = og.copy()
		tt = t.setCol("z", z, inPlace=False)
		self.assertEqual(t, og)

		t.setCol("z", z) # Since we've already tested this kind of thing works
		self.assertEqual(tt, t)

		# Bad val shape.
		t = og.copy()
		self.assertRaises(Exception, "Set unknown val shape", t.setCol, "x", 1)
		self.assertEqual(t, og)

		# Do nothing.
		t.setCol([], 1)
		self.assertEqual(t, og)

		# From nothing.
		t = Table()
		self.assertRaisesRegex(TableException, "Set col length", t.setCol, ["x", "y"], [[1,2], [1,2,3]])
		self.assertEqual(t, Table())

		d = {"x": [1, 2], "y": [3, 4]}
		t.setCol(list(d.keys()), list(d.values()))
		exp = Table(d)
		self.assertEqual(t, exp)

		# Scalar extend.
		t = og.copy()
		t.setCol("z", 10)
		t.setCol("str", "abc")
		exp = Table({"x": [1, 2, 3], "y": ["a", "b", "c"], "z":[10]*3, "str": ["abc"]*3})
		self.assertEqual(t, exp)

		# Bad column type.
		t = og.copy()
		self.assertRaisesRegex(TableException, "Unrecognized column type: <class 'int'>", t.setCol, 123, 10)
		self.assertEqual(t, og)
	
	# Note: Some tests here assume setCol is properly tested.
	def test_setRow(self):
		og = Table({"x": [1, 2, 3, 4], "y": ["a" , "b", "c", "d"],
			"z": numpy.array(["2021-11-10", "2021-11-11", "2021-11-12", "2021-11-13"]).astype(numpy.datetime64)})
		t = og.copy()

		# Single row, single col.
		t.setRow(1, 10, "x")
		exp = Table({"x": [1, 10, 3, 4], "y": og["y"], "z": og["z"]})
		self.assertEqual(t, exp)

		# Single row, multi col.
		t.setRow(3, (numpy.datetime64("2021-11-30"), "X"), col=["z", "y"])
		exp = Table({"x": exp["x"], "y": ["a", "b", "c", "X"],
			"z": numpy.array(["2021-11-10", "2021-11-11", "2021-11-12", "2021-11-30"]).astype(numpy.datetime64)})
		self.assertEqual(t, exp)

		# Single row, all cols.
		t.setRow(-1, (-1, "blah", numpy.datetime64("1983-01-01 09:30"))) # Should expand string, but not date
		exp = Table({"x": [1, 10, 3, -1], "y": ["a", "b", "c", "blah"],
			"z": numpy.array(["2021-11-10", "2021-11-11", "2021-11-12", "1983-01-01"]).astype(numpy.datetime64)})
		self.assertEqual(t, exp)

		# Multi row (list), single col, val list.
		t.setRow([2, 1], ["X", "Y"], "y")
		exp.setCol("y", ["a", "Y", "X", "blah"])
		self.assertEqual(t, exp)

		# Multi row (array), single col, val scalar.
		t.setRow(numpy.array([0, 1, 3]), 100, col="x")
		exp.setCol("x", [100, 100, 3, 100])
		self.assertEqual(t, exp)

		# Multi row (mask), single col, val array.
		t.setRow([True, False, False, True], numpy.array(["1977-01-01", "1999-12-06"]).astype(numpy.datetime64), col="z")
		exp.setCol("z", numpy.array(["1977-01-01", "2021-11-11", "2021-11-12", "1999-12-06"]).astype(numpy.datetime64))
		self.assertEqual(t, exp)

		# Multi row (list), multi col, val mixed list.
		t.setRow([1, 3], [numpy.array(["1983-10-31", "2000-11-01"]).astype(numpy.datetime64),
			["hoochie", "mama"]], col=["z", "y"])
		exp.setCol(["y", "z"], [["a", "hoochie", "X", "mama"],
			numpy.array(["1977-01-01", "1983-10-31", "2021-11-12", "2000-11-01"]).astype(numpy.datetime64)])
		self.assertEqual(t, exp)

		# Multi row (array), multi col, val tuple list.
		t.setRow(numpy.array([1, 0, 2]), [(10, "s"), (20, "p"), (30, "i")], col=["x", "y"])
		exp.setCol(["x", "y"], [[20, 10, 30, 100], ["p", "s", "i", "mama"]])
		self.assertEqual(t, exp)

		# Multi row (mask), multi col, val list list.
		t.setRow([True, True, False, True], [[-1, -2, -3], ["a", "b", "c"]], col=["x", "y"])
		exp.setCol(["x", "y"], [[-1, -2, 30, -3], ["a", "b", "i", "c"]])
		self.assertEqual(t, exp)

		# Unknown col + shouldn't overwrite earlier columns.
		self.assertRaisesRegex(TableException, "Set row unknown col: 'not a col'",
			t.setRow, 0, (10, "don't write me", 0), col=["x", "y", "not a col"])
		self.assertEqual(t, exp)

		# Index out of table.
		self.assertRaises(IndexError, t.setRow, [0, 4], ["OK", "index"], "y")
		self.assertEqual(t, exp)

		# Mask of wrong length.
		self.assertRaises(IndexError, t.setRow, [True, False], [10, 20], "x")
		self.assertEqual(t, exp)

		# Length mismatch between values and cols.
		self.assertRaisesRegex(TableException, "Set val/col mismatch", t.setRow, 0, [(1, "a"), (2, "b", 0)], ["x"])
		self.assertEqual(t, exp)

		self.assertRaisesRegex(TableException, "Set row error: *", t.setRow, [0, 1], [[10, 20]], ["x", "y"])
		self.assertEqual(t, exp)

		# Length mismatch between indices and cols.
		self.assertRaisesRegex(TableException, "Set row error: *", t.setRow, [0, 1], [1, 2, 3], "x")
		self.assertEqual(t, exp)

		# Not in place.
		act = t.setRow([0, 1, 2], ["not", "in", "place"], col="y", inPlace=False)
		self.assertEqual(t, exp)
		exp.setCol("y", ["not", "in", "place", "c"])
		self.assertEqual(act, exp)

		# Tests around setting string column. Expand the column every time.
		t = Table({"str": ["a", "b", "c"]})

		t.setRow(0, "ab") # Int
		exp = Table({"str": ["ab", "b", "c"]})
		self.assertEqual(t, exp)

		t.setRow([0, 1], ["a", "111"]) # List
		exp = Table({"str": ["a", "111", "c"]})
		self.assertEqual(t, exp)

		t.setRow(numpy.array([True, False, True]), ["aa", "1234"]) # Array
		exp = Table({"str": ["aa", "111", "1234"]})
		self.assertEqual(t, exp)
		t.setRow(slice(1, 3, 1), ["12345"]*2) # Slice
		exp = Table({"str": ["aa", "12345", "12345"]})
		self.assertEqual(t, exp)

		# Empty row list.
		act = Table({"x": [1, 2]})
		exp = act.copy()
		act.setRow([], [])
		self.assertEqual(act, exp)

	# Note: Assumes that `setRow` is tested.
	def test_setKey(self):
		# Not keyed.
		t = Table({"x": [1, 2], "y": [3, 4]})
		exp = t.copy()
		self.assertRaisesRegex(TableException, "Not keyed", t.setKey, [1, 2], [10, 20])
		self.assertEqual(t, exp)

		# Vertically oriented lists.

		# No misses.
		t = Table({"ks": ["a", "a", "b", "b", "b"], "kn": [1, 2, 1, 2, 3],
			"v1": [10, 20, 30, 40, 50], "v2": [True]*5})
		t.key(["ks", "kn"])
		act = t.copy()
		exp = t.copy()
		act.setKey([["a", "a", "b"], [1, 2 ,2]], [[-1, -2, -3], False])
		exp.setRow([0, 1, 3], [[-1, -2, -3], False], ["v1", "v2"])
		self.assertEqual(act, exp)

		# Some misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([["c", "a", "b"], [1, 2 , 4]], [[-1, -2, -3], False])
		exp.setRow(1, (-2, False), col=["v1", "v2"])
		exp.append(Table({"ks": ["c", "b"], "kn": [1, 4], "v1": [-1, -3], "v2": [False, False]}))
		self.assertEqual(act, exp)

		# All misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([["a", "b"], [10, 20]], [100, 200], col="v1")
		exp.append(Table({"ks": ["a", "b"], "kn": [10, 20], "v1": [100, 200], "v2": [null.BOOL]*2}))
		self.assertEqual(act, exp)

		# Vertically oriented arrays.

		# No misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([numpy.array(["b", "a"]), numpy.array([1, 2])], numpy.array([-1, -2]), col="v1")
		exp.setRow([2, 1], [-1, -2], "v1")
		self.assertEqual(act, exp)

		# Some misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([numpy.array(["b", "a", "c"]), numpy.array([1, 3, 1])], numpy.array([-1, -2, -3]), col="v1")
		exp.append(Table({"ks": ["a", "c"], "kn": [3, 1], "v1": [-2, -3], "v2": [null.BOOL]*2}))
		exp.setRow(2, -1, "v1")
		self.assertEqual(act, exp)

		# All misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([numpy.array(["a", "c"]), numpy.array([3, 1])], numpy.array([-1, -2]), col="v1")
		exp.append(Table({"ks": ["a", "c"], "kn": [3, 1], "v1": [-1, -2], "v2": [null.BOOL]*2}))
		self.assertEqual(act, exp)

		# Vertically oriented mixed.

		# No misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([numpy.array(["a", "b"]), [2, 3]], [[False, False], [-1, -2]], col=["v2", "v1"])
		exp.setRow([1, 4], [[-1, -2], False], col=["v1", "v2"])
		self.assertEqual(act, exp)

		# Horizontally oriented list of tuples.

		# No misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([("a", 2), ("a", 1), ("b", 1)], False, "v2")
		exp.setRow([1, 0, 2], False, "v2")
		self.assertEqual(act, exp)

		# Some misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([("x", 10), ("a", 1), ("b", 1)], False, "v2")
		exp.append(Table({"ks": ["x"], "kn": [10], "v1": [null.INT], "v2": [null.BOOL]}))
		exp.setRow([0, 2], False, "v2")
		self.assertEqual(act, exp)

		# All misses.
		act = t.copy()
		exp = t.copy()
		act.setKey([("x", 1), ("y", 2)], [(10, True), (20, True)])
		exp.append(Table({"ks": ["x", "y"], "kn": [1, 2], "v1": [10, 20], "v2": [True]*2}))
		self.assertEqual(act, exp)

		# Single list (error on a 2 key table).
		exp = t.copy()
		self.assertRaisesRegex(TableException, "Key dimension mismatch", t.setKey, [1, 2], [-1, -2], "v1")
		self.assertEqual(t, exp)

		# Single list.

		# No misses.
		tt = Table({"key": ["a", "b", "c"], "v1": [10, 20, 30], "v2": [True]*3})
		tt.key("key")
		act = tt.copy()
		exp = tt.copy()
		act.setKey(["c", "a"], [numpy.array([-1, -2]), False])
		exp.setRow([2, 0], [[-1, -2], False], col=["v1", "v2"])
		self.assertEqual(act, exp)

		# Some misses.
		act = tt.copy()
		exp = tt.copy()
		act.setKey(["a", "xy", "b", "er"], [[-1, -2, -3, -4], False])
		exp.setRow([0, 1], [[-1, -3], False], col=["v1", "v2"])
		exp.append(Table({"key": ["xy", "er"], "v1": [-2, -4], "v2": [False, False]}))
		self.assertEqual(act, exp)	

		# All misses.
		act = tt.copy()
		exp = tt.copy()
		act.setKey(["xy", "er"], [[-1, -2], False])
		exp.append(Table({"key": ["xy", "er"], "v1": [-1, -2], "v2": [False, False]}))
		self.assertEqual(act, exp)

		# Single array.

		# No missses.
		act = tt.copy()
		exp = tt.copy()
		act.setKey(numpy.array(["c", "a"]), [numpy.array([-1, -2]), False])
		exp.setRow([2, 0], [[-1, -2], False], col=["v1", "v2"])
		self.assertEqual(act, exp)

		# Some misses.
		act = tt.copy()
		exp = tt.copy()
		act.setKey(["a", "b", "x"], [[-1, -2, -3], [False, False, True]])
		exp.append(Table({"key": ["x"], "v1": [-3], "v2": [True]}))
		exp.setRow([0, 1], [[-1, -2], False], ["v1", "v2"])
		self.assertEqual(act, exp)

		# All misses.
		act = tt.copy()
		exp = tt.copy()
		act.setKey(["x", "y", "zed"], [100, 200, 300], "v1")
		exp.append(Table({"key": ["x", "y", "zed"], "v1": [100, 200, 300], "v2": [null.BOOL]*3}))
		self.assertEqual(act, exp)

		# Atom.

		# No miss.
		act = tt.copy()
		exp = tt.copy()
		act.setKey("b", -1, col="v1")
		exp.setRow(1, -1, "v1")
		self.assertEqual(act, exp)

		# Miss.
		act = tt.copy()
		exp = tt.copy()
		act.setKey("newKey", (6, False))
		exp.append(Table({"key": ["newKey"], "v1": [6], "v2": [False]}))
		self.assertEqual(act, exp)

		# Atom in two-key table (error).
		exp = t.copy()
		self.assertRaisesRegex(TableException, "Key dimension mismatch", t.setKey, "a", 10, "v1")
		self.assertEqual(t, exp)

		# Single tuple.

		# No miss.
		act = t.copy()
		exp = t.copy()
		act.setKey(("b", 3), (-1, False))
		exp.setRow(4, (-1, False), col=["v1", "v2"])
		self.assertEqual(act, exp)

		# Miss.
		act = t.copy()
		exp = t.copy()
		act.setKey(("newKey", 0), -1, "v1")
		exp.append(Table({"ks": ["newKey"], "kn": [0], "v1": [-1], "v2": [null.BOOL]}))
		self.assertEqual(act, exp)

		# Change a key column.

		# No miss.
		act = tt.copy()
		exp = tt.copy()
		act.setKey("a", ("newKey", 10), col=["key", "v1"])
		exp.setRow(0, ("newKey", 10), col=["key", "v1"])
		self.assertEqual(act, exp)

		# Miss (silly though it may be).
		act = tt.copy()
		exp = tt.copy()
		act.setKey("x", ("newKey", -1), ["key", "v1"])
		exp.append(Table({"key": ["newKey"], "v1": [-1], "v2": [null.BOOL]}))
		self.assertEqual(act, exp)

		# Not in place.
		cp = tt.copy()
		exp = tt.copy()
		act = cp.setKey("a", -1, col="v1", inPlace=False)
		exp.setRow(0, -1, col="v1")
		self.assertEqual(act, exp)
		self.assertEqual(tt, cp)

		# Incorrect key dimension.
		dimErr = "Key dimension mismatch"

		act = tt.copy() # Multi-key into a single key table (list)
		exp = tt.copy()
		self.assertRaisesRegex(TableException, dimErr, act.setKey, [["a", "b"], [1, 3]], -1, col="v1")
		self.assertEqual(act, exp)

		act = t.copy() # Single key into a multi-key table
		exp = t.copy()
		self.assertRaisesRegex(TableException, dimErr, act.setKey, "a", 1, col="v1")
		self.assertEqual(act, exp)

		act = t.copy() # Wrong number of keys into a multi-key table (tuples)
		exp = t.copy()
		self.assertRaisesRegex(TableException, dimErr, act.setKey, [("a", 1, 2), ("b", 3, 4)], [(-1, False), (-2, False)])
		self.assertEqual(act, exp)

		# Incorrect key shape.
		act = tt.copy()
		exp = tt.copy()
		self.assertRaisesRegex(TableException, "Unrecognized key shap", act.setKey, [[["a"]]], [-1], col="v1")
		self.assertEqual(act, exp)

		# Error in 'setRow', so we revert.
		mock.mock(Table, "setRow", TableTest.mockSetRow)
		act = tt.copy()
		exp = tt.copy()
		self.assertRaisesRegex(TableException, "Some error", act.setKey, "x", 100, "v1")
		self.assertEqual(act, exp) # Shouldn't add a new row

	def mockSetRow(s, x, y, z):
		raise TableException("Some error")

	def test_deleteCol(self):
		# Delete single col.
		t = Table({"c1": ["a", "b", "c"], "c2": [1, 2, 3], "c3": [True, False, True]})
		act = t.copy()
		act.deleteCol("c2")
		exp = Table({"c1": ["a", "b", "c"], "c3": [True, False, True]})
		self.assertEqual(act, exp)

		# Delete multiple columns
		act = t.copy()
		act.deleteCol(["c1", "c2"])
		exp = Table({"c3": [True, False, True]})
		self.assertEqual(act, exp)

		# Not in place.
		cp = t.copy()
		act = t.deleteCol("c1", inPlace=False)
		exp = Table({"c2": [1, 2, 3], "c3": [True, False, True]})
		self.assertEqual(act, exp)
		self.assertEqual(t, cp)

	def test_takeCol(self):
		# Take a single col.
		t = Table({"c1": ["a", "b", "c"], "c2": [1, 2, 3], "c3": [True, False, True]})
		act = t.copy()
		act.takeCol("c1")
		exp = Table({"c1": ["a", "b", "c"]})
		self.assertEqual(act, exp)

		# Take multiple columns (ordered correctly).
		act = t.copy()
		act.takeCol(["c2", "c1"])
		exp = Table({"c2": [1, 2, 3], "c1": ["a", "b", "c"]})
		self.assertEqual(act, exp)

		# Unknown column.
		act = t.copy()
		exp = t.copy()
		self.assertRaisesRegex(TableException, "Unknown column: unknownCol", act.takeCol, ["c1", "c2", "unknownCol"])

		# Not in place.
		cp = t.copy()
		act = t.takeCol("c1", inPlace=False)
		exp = Table({"c1": ["a", "b", "c"]})
		self.assertEqual(act, exp)
		self.assertEqual(t, cp) # Shouldn't affect original
		self.assertFalse(act["c1"] is t["c1"]) # Shouldn't copy by reference

	def test_deleteRow(self):
		# Delete single row.
		t = Table({"x": [1, 2, 3, 4, 5], "y":[10, 20, 30, 40, 50]})
		act = t.copy()
		act.deleteRow(3)
		exp = Table({"x": [1, 2, 3, 5], "y": [10, 20, 30, 50]})
		self.assertEqual(act, exp)

		# Delete multiple rows (list) + wrap.
		act = t.copy()
		act.deleteRow([4, -2, 0])
		exp = Table({"x": [2, 3], "y": [20, 30]})
		self.assertEqual(act, exp)

		# Delete multiple rows array.
		act = t.copy()
		act.deleteRow(numpy.array([4, -2, 0]))
		self.assertEqual(act, exp)

		# Mask.
		act = t.copy()
		act.deleteRow([False, False, True, True, False])
		exp = Table({"x": [1, 2, 5], "y": [10, 20, 50]})
		self.assertEqual(act, exp)

		# Slice.
		act = t.copy()
		act.deleteRow(slice(0, 10, 2))
		exp = Table({"x": [2, 4], "y": [20, 40]})
		self.assertEqual(act, exp)

		# Not in place.
		cp = t.copy()
		act = t.deleteRow([0, 1, 4], inPlace=False)
		exp = Table({"x": [3, 4], "y": [30, 40]})
		self.assertEqual(act, exp)
		self.assertEqual(t, cp)

		# Index error.
		act = t.copy()
		self.assertRaises(IndexError, act.deleteRow, [0, 4, 5])
		self.assertEqual(act, t)

		# Mask length error.
		self.assertRaises(IndexError, act.deleteRow, [True, True])
		self.assertEqual(act, t)

	# Note: relies on deleteRow working.
	def test_deleteKey(self):
		single = Table({"key": [10, 20, 30, 40, 50], "val": ["a", "b", "c", "d", "e"]})
		single.key("key")
		multi = Table({"k1": ["a", "b", "a", "b"], "k2": [1, 1, 2, 2], "x": [10, 20, 30, 40]})
		multi.key(["k1", "k2"])

		# Single key in single-keyed table.
		act = single.copy()
		act.deleteKey(20)
		exp = single.deleteRow(1, inPlace=False)
		self.assertEqual(act, exp)

		# Single key in multi-keyed table.
		act = multi.copy()
		act.deleteKey(("b", 1))
		exp = multi.deleteRow(1, inPlace=False)
		self.assertEqual(act, exp)

		# Multiple keys in multi-keyed table -- tuples
		act = multi.copy()
		act.deleteKey([("a", 1), ("b", 1), ("b", 2)])
		exp = multi.deleteRow([0, 1, 3], inPlace=False)
		self.assertEqual(act, exp)

		# Multiple keys in multi-keyed table -- lists/arrays.
		act = multi.copy()
		act.deleteKey([["a", "a"], numpy.array([1, 2])])
		exp = multi.deleteRow([0, 2], inPlace=False)
		self.assertEqual(act, exp)

		# Missing keys.
		act = multi.copy()
		self.assertRaisesRegex(TableException, "Invalid key to delete", act.deleteKey, [["a", "a"], [1, 3]])
		self.assertEqual(act, multi)

		# Bad shape.
		act = single.copy()
		self.assertRaisesRegex(TableException, "Unrecognized key shape", act.deleteKey, [[[1]]])
		self.assertEqual(act, single)

		# Key/input mismatch
		act = single.copy()
		self.assertRaisesRegex(TableException, "Key dimension mismatch", act.deleteKey, [[10], [20]])
		self.assertEqual(act, single)

		# Multi-list dim mismatch.
		act = multi.copy()
		self.assertRaisesRegex(TableException, "Key dimension mismatch", act.deleteKey, [["a", "b"]])
		self.assertEqual(act, multi)

		# Not keyed.
		act = single.copy().unkey(inPlace=False)
		self.assertRaisesRegex(TableException, "Not keyed", act.deleteKey, 10)
		self.assertEqual(act, single.unkey(inPlace=False))

		# Not in place.
		cp = single.copy()
		act = single.deleteKey([10, 30, 40], inPlace=False)
		exp = single.deleteRow([0, 2, 3], inPlace=False)
		self.assertEqual(act, exp)
		self.assertEqual(single, cp)

	def test_type(self):
		# All.
		t = Table({"date": numpy.array(["2021-12-29"]).astype(numpy.datetime64), "int": [1], "bool": [True]})
		act = t.type()
		exp = ["datetime64[D]", "int64", "bool"]
		self.assertEqual(act, exp)

		# Single.
		act = t.type("int")
		exp = "int64"
		self.assertEqual(act, exp)

		# Multiple.
		act = t.type(["bool", "date"])
		exp = ["bool", "datetime64[D]"]
		self.assertEqual(act, exp)

		# Unknown column.
		self.assertRaises(KeyError, t.type, "unknownCol")

	def test_typeCode(self):
		t = Table({"x": [1, 2], "y": ["abc", "de"], "z": [True, False], "date": [numpy.datetime64("now")]*2})

		# Single.
		act = t.typeCode("y")
		exp = "s"
		self.assertEqual(act, exp)

		# Multi.
		act = t.typeCode(["x", "date"])
		exp = ["i", "d"]
		self.assertEqual(act, exp)

		# All.
		act = t.typeCode([])
		exp = ["i", "s", "b", "d"]
		self.assertEqual(act, exp)

	# Assumes `fromCSV` works.
	def test_toCSV(self):
		file = TableTest.RESOURCES + "test_toCSV.csv"
		if file in os.listdir(TableTest.RESOURCES): os.remove(file) # Clear anything from a previous test run

		# To string with default delimiter.
		t = Table({"x": [1,2], "y": ["abc", "de"]})
		act = t.toCSV()
		exp = ["x,y", "1,abc", "2,de"]
		self.assertEqual(act, exp)

		# To string with custom delimiter.
		act = t.toCSV(delimiter="&!&")
		exp = ["x&!&y", "1&!&abc", "2&!&de"]
		self.assertEqual(act, exp)

		# To file with default delimiter.
		t.toCSV(file=file)
		act = Table.fromCSV(file, "is")
		exp = t
		self.assertEqual(act, exp)

		# To file with custom delimiter (should also overwrite anything previously there).
		t.toCSV(delimiter="|", file=file)
		act = Table.fromCSV(file, "is", delimiter="|")
		exp = t
		self.assertEqual(act, exp)
		os.remove(file)

	def test_xcol(self):
		# Single column.
		t = Table({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
		act = t.copy()
		act.xcol("z")
		exp = Table({"z": [5, 6], "x": [1, 2], "y": [3, 4]})
		self.assertEqual(act, exp)

		# Multi-column incomplete.
		act = t.copy()
		act.xcol(["y", "x"])
		exp = Table({"y": [3, 4], "x": [1, 2], "z": [5, 6]})
		self.assertEqual(act, exp)

		# Multi-column complete.
		act = t.copy()
		act.xcol(["z", "y", "x"])
		exp = Table({"z": [5, 6], "y": [3, 4], "x": [1, 2]})
		self.assertEqual(act, exp)

		# Not in place. Ensure columns are not copied by reference.
		cp = t.copy()
		act = t.xcol("z", inPlace=False)
		exp = Table({"z": [5, 6], "x": [1, 2], "y": [3, 4]})
		self.assertEqual(act, exp)
		self.assertEqual(t, cp)

		for c in t.cols():
			self.assertFalse(act[c] is t[c])

		# Unknown column.
		act = t.copy()
		self.assertRaises(KeyError, act.xcol, "awefawe")

	def test_append(self):
		t = Table({"x": [1, 2], "y": ["a", "b"]})

		# Append Table with missing columns -- error.
		act = t.copy()
		exp = t.copy()
		self.assertRaisesRegex(TableException, "Append missing column: y", act.append, Table({"x": [10, 20, 30]}))
		self.assertEqual(act, exp)

		# Append Table with exact columns.
		ap = Table({"y": [1, 22, 333], "x": [10, 20, 30]})
		act.append(ap)
		exp = Table({"x":[1, 2, 10, 20, 30], "y": ["a", "b", "1", "22", "333"]})
		self.assertEqual(act, exp)

		# Append Table with extra columns.
		act = t.copy()
		ap = Table({"x": [1.2, 3.4], "z": [1, 2], "y": ["c", "d"]})
		act.append(ap)
		exp = Table({"x": [1, 2, 1.2, 3.4], "y": ["a", "b", "c", "d"]})
		self.assertEqual(act, exp)

		# Append dict with missing keys -- error.
		act = t.copy()
		exp = t.copy()
		ap = {"y": "c", "z": 1}
		self.assertRaisesRegex(TableException, "Append missing key: x", act.append, ap)
		self.assertEqual(act, exp)

		# Append dict with exact keys.
		ap = {"x": 3, "y": "c"}
		act.append(ap)
		exp = Table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
		self.assertEqual(act, exp)

		# Append dict with extra keys.
		act = t.copy()
		ap = {"x": 3, "y": "c", "z": 1.2, "w": 10}
		act.append(ap)
		exp = Table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
		self.assertEqual(act, exp)

		# Append dict with nested values -- error.
		act = t.copy()
		exp = t.copy()
		ap = {"x": [3, 4], "y": "c"}
		self.assertRaisesRegex(TableException, "Cannot append a nested dictionary", act.append, ap)
		self.assertEqual(act, exp)

		# Append tuple.
		act = t.copy()
		ap = (10, "c")
		act.append(ap)
		exp = Table({"x": [1, 2, 10], "y": ["a", "b", "c"]})
		self.assertEqual(act, exp)

		# Append tuple with incorrect length -- error.
		act = t.copy()
		exp = t.copy()
		ap = (10, "c", 1.2)
		self.assertRaisesRegex(TableException, "Append: tuple of length 3 but 2 columns in table", act.append, ap)
		self.assertEqual(act, exp)

		# Append tuple with nested values -- error.
		ap = (3, ("c", "d"))
		self.assertRaisesRegex(TableException, "Cannot append a nested tuple", act.append, ap)
		self.assertEqual(act, exp)

		# Not in place.
		cp = t.copy()
		ap = {"x": 3, "y": "c"}
		act = t.append(ap, inPlace=False)
		exp = Table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
		self.assertEqual(act, exp)
		self.assertEqual(t, cp)

		# Append to empty -- table.
		t = Table()
		exp = Table({"sym": ["VOD.L", "MSFT.N"], "price": [1.21, 3.14]})
		t.append(exp)
		self.assertEqual(t, exp)

		# Append to empty -- dict.
		t = Table()
		toAdd = {"x": 1, "y": "abc", "z": 3}
		t.append(toAdd)
		exp = Table({"x": [1], "y": ["abc"], "z": [3]})
		self.assertEqual(t, exp)

		# Append to empty -- tuples (error).
		t = Table()
		toAdd = (1, 2, 3, 4)
		self.assertRaisesRegex(TableException, "Append: tuple of length 4 but 0 columns in table", t.append, toAdd)

		# Append nested table -- error.
		t = Table({"x": [1, 2]})
		s = Table({"x": [[10, 20, 30], [40, 50, 60]]})
		self.assertRaisesRegex(TableException, "Cannot append nested columns", t.append, s)
		self.assertEqual(t, Table({"x": [1, 2]}))

		# Append to a nested table.
		t = Table({"x": [1, 2], "y": [[1, 2], [3, 4]]})
		s = Table({"x": [1], "y": [3]})
		self. assertRaisesRegex(TableException, "Cannot append to table with nested columns", t.append, s)
		self.assertEqual(t, Table({"x": [1, 2], "y": [[1, 2], [3, 4]]}))

	def test_sort(self):
		t = Table({
			"x": [3		,1		,1		,2		,5		,2],
			"y": ["a"	,"z"	,"a"	,"b"	,"c"	,"b"]})

		# Sort single column ascending.
		act = t.copy()
		act.sort("x")
		exp = t.getRow([1, 2, 3, 5, 0, 4])
		self.assertEqual(act, exp)

		# Sort single column descending.
		act = t.copy()
		act.sort("y", desc=True)
		exp = t.getRow([1, 4, 3, 5, 2, 0])
		self.assertEqual(act, exp)

		# Sort multiple columns ascending.
		act = t.copy()
		act.sort(["x", "y"])
		exp = t.getRow([2, 1, 3, 5, 0, 4])
		self.assertEqual(act, exp)

		# Sort multiple columns descending.
		act = t.copy()
		act.sort(["x", "y"], desc=True)
		exp = t.getRow([4, 0, 5, 3, 1, 2])
		self.assertEqual(act, exp)

		# Not in place.
		cp = t.copy()
		act = t.sort("x", desc=True, inPlace=False)
		exp = t.getRow([4, 0, 5, 3, 2, 1])
		self.assertEqual(act, exp)
		self.assertEqual(t, cp)

	def test_mkNullRow(self):
		# All rows.
		t = Table({"i": [1], "f": [1.2], "d": [numpy.datetime64("today")],
			"p": [numpy.datetime64("now", "ns")], "b": [True], "n": [numpy.timedelta64(0)]})
		act = t.mkNullRow()
		exp = {"i": null.INT, "f": null.FLOAT, "d": null.DATETIME,
			"p": null.DATETIME, "b": null.BOOL, "n": null.TIMEDELTA}
		self.assertEqual(act, exp)

		# Single row.
		act = t.mkNullRow("f")
		exp = {"f": null.FLOAT}
		self.assertEqual(act, exp)

		# Multiple rows.
		act = t.mkNullRow(["d", "i"])
		exp = {"d": null.DATETIME, "i": null.INT}
		self.assertEqual(act, exp)

	def test_aj(self):
		trade = Table({
			"sym": ["msft", "ibm", "ge", "msft", "miss"],
			"ts": numpy.array(["2021-12-30 10:01:01", "2021-12-30 10:01:03",
				"2021-12-30 10:01:04", "2021-12-30 10:03:00", "2021-12-30 10:04:00"]).astype(numpy.datetime64),
			"qty": [100, 200, 150, 130, 120]})

		quote = Table({
			"sym": ["ibm", "msft", "msft", "ibm", "msft", "miss"],
			"ts": numpy.array(["2021-12-30 10:01:00", "2021-12-30 10:01:00", "2021-12-30 10:01:00",
				"2021-12-30 10:01:02", "2021-12-30 10:03:00", "2021-12-30 10:04:01"]).astype(numpy.datetime64),
			"px": [100, 99, 101, 98, 102, 104]})

		# Missed col left.
		act = trade.copy()
		self.assertRaisesRegex(TableException, "aj error: column 'px' not in left", act.aj, quote, ["sym", "px", "ts"])
		self.assertEqual(act, trade)

		# Missed col right.
		act = trade.copy()
		self.assertRaisesRegex(TableException, "aj error: column 'qty' not in right", act.aj, quote, ["qty", "ts"])
		self.assertEqual(act, trade)

		# Join on sym + time.
		act = trade.copy()
		act.aj(quote, ["sym", "ts"])
		exp = trade.copy()
		exp["px"] = [101, 98, null.INT, 102, null.INT]
		self.assertEqual(act, exp)

		# Not in place.
		cp = trade.copy()
		act = trade.aj(quote, ["sym", "ts"], inPlace=False)
		self.assertEqual(act, exp)
		self.assertEqual(trade, cp)

		# Join on sym + other + time.
		act = trade.copy()
		act["other"] = 0
		q = quote.copy()
		q["other"] = [0, 0, 1, 0, 0, 0] # Should cause us to ignore the second "msft"
		act.aj(q, ["sym", "other", "ts"])
		exp = trade.copy()
		exp.setCol(["other", "px"], [0, [99, 98, null.INT, 102, null.INT]])
		self.assertEqual(act, exp)

		# Join on time only.
		left = Table({"x": [0, 1, 2, 3], "time": [1, 5, 7, 9]})
		right = Table({"y": [0, 10], "time": [0, 6]})
		act = left.aj(right, ["time"], inPlace=False)
		exp = Table({"x": [0, 1, 2, 3], "time": [1, 5, 7, 9], "y": [0, 0, 10, 10]})
		self.assertEqual(act, exp)

	def test_getitem(self):
		# Row.
		t = Table({"c1": [1, 2, 3], "c2": [1.1, 2.2, 3.3]})
		act = t[1]
		exp = {"c1": 2, "c2": 2.2}
		self.assertEqual(act, exp)

		act = t[numpy.int64(1)]
		self.assertEqual(act, exp)

		# Rows.
		act = t[[0, -1]]
		exp = Table({"c1": [1, 3], "c2": [1.1, 3.3]})
		self.assertEqual(act, exp)

		act = t[numpy.array([0, -1])]
		self.assertEqual(act, exp)

		# Column.
		act = t["c1"]
		exp = numpy.array([1, 2, 3])
		self.assertTrue((act == exp).all())

		act = t[numpy.str_("c1")]
		self.assertTrue((act == exp).all())

		# Columns.
		def chkCols(x, y): # Numpy arrays are annoying sometimes...
			if len(x) != len(y): return False
			for i in range(len(x)):
				if not (x[i] == y[i]).all(): return False
			return True

		act = t[["c2", "c1"]]
		exp = [[1.1, 2.2, 3.3], [1, 2, 3]]
		self.assertTrue(chkCols(act, exp))

		act = t[numpy.array(["c2", "c1"])]
		self.assertTrue(chkCols(act, exp))

		# Mask.
		act = t[[True, False, True]]
		exp = Table({"c1": [1, 3], "c2": [1.1, 3.3]})
		self.assertEqual(act, exp)

		act = t[numpy.array([True, False, True])]
		self.assertEqual(act, exp)

		# Slice.
		act = t[:]
		exp = t.copy()
		self.assertEqual(act, exp)

		act = t[0:2]
		exp = t.getRow([0, 1])
		self.assertEqual(act, exp)

		act = t[0:3:2]
		exp = t.getRow([0, 2])
		self.assertEqual(act, exp)

		# Keyed.
		t.key("c1")
		act = t[1]
		exp = {"c1": 1, "c2": 1.1}
		self.assertEqual(act, exp)

		tt = Table({"key": ["abc", "def"], "abc": [1, 2], "def": [3, 4]})
		act = tt["def"]
		exp = numpy.array([3, 4])
		self.assertTrue((act == exp).all())

		tt.key("key")
		act = tt["def"]
		exp = {"key": "def", "abc": 2, "def": 4}
		self.assertEqual(act, exp)

		# Undefined.
		t.unkey()
		self.assertRaisesRegex(TableException, "Unimplemented getitem type: <class 'tuple'>", t.__getitem__, (1, 2))

	def test_len(self):
		t = Table({})
		self.assertEqual(len(t), 0)

		t["x"] = []
		self.assertEqual(len(t), 0)

		t = Table({"x": [1, 2, 3]})
		self.assertEqual(len(t), 3)

	def test_setitem(self):
		# Row.
		t = Table({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
			"z": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]})

		# Rows.
		act = t.copy()
		row = (40, 44.4, "D")
		act[4] = row
		exp = t.setRow(4, row, inPlace=False)
		self.assertEqual(act, exp)

		act = t.copy()
		act[numpy.int(4)] = row
		self.assertEqual(act, exp)

		# Column.
		act = t.copy()
		col = [x*10 for x in t["x"]]
		act["x"] = col
		exp = t.setCol("x", col, inPlace=False)
		self.assertEqual(act, exp)

		act = t.copy()
		act[numpy.str_("x")] = col
		self.assertEqual(act, exp)

		# Columns.
		act = t.copy()
		col = [[y-1 for y in t["y"]], [z.upper() for z in t["z"]]]
		act[["y", "z"]] = col
		exp = t.setCol(["y", "z"], col, inPlace=False)
		self.assertEqual(act, exp)

		act = t.copy()
		act[numpy.array(["y", "z"])] = col
		self.assertEqual(act, exp)

		act = t.copy()
		act["x"] = 10
		exp = t.setCol("x", 10, inPlace=False)
		self.assertEqual(act, exp)

		# Mask.
		idx = [True, False, True] + ([False]*7)
		row = [(-1, -1.1, "ABC"), (-2, -2.2,  "DEF")]
		act = t.copy()
		act[idx] = row
		exp = t.setRow(idx, row, inPlace=False)
		self.assertEqual(act, exp)

		idx = numpy.array(idx)
		act = t.copy()
		act[idx] = row
		self.assertEqual(act, exp)

		# Slice.
		act = t.copy()
		row = [(-1, -1.1, "X"), (-2, -2.2, "Y"), (-3, -3.3, "Z")]
		act[8:3:-2] = row
		exp = t.setRow(slice(8, 3, -2), row, inPlace=False)
		self.assertEqual(act, exp)

		# Keyed.
		act = t.key("z", inPlace=False)
		exp = act.copy()
		act["g"] = (10, 3.14)
		exp.setRow(6, (10, 3.14), ["x", "y"])
		self.assertEqual(act, exp)
		
		# Undefined.
		self.assertRaisesRegex(TableException, "Unimplemented setitem type: <class 'dict'>", t.__setitem__, {}, [])

	def test_iter(self):
		# Not keyed.
		table = Table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
		i = 0

		for row in table:
			self.assertEqual(row, table[i])
			i += 1

		# Keyed.
		table.key("x")
		i = 0

		for row in table:
			self.assertEqual(row, table.getRow(i))
			i += 1

	def test_distinct(self):
		t = Table({"x": [1, 1, 1, 2, 2, 2], "y": ["a", "b", "a", "c", "c", "c"]})
		act = t.distinct()
		ex = Table({"x": [1, 1, 2], "y": ["a", "b", "c"]})
		self.assertEqual(act, ex)

	def test_by(self):
		# Single simple.
		t = Table({"x": [1, 2, 1, 1, 2], "y": [10, 20, 30, 40, 50], "z": ["a", "b", "c", "d", "e"]})
		act = t.by("x", "y")
		exp = Table({"x": [1, 2], "y": numpy.array([numpy.array([10, 30, 40]), numpy.array([20, 50])], dtype=object)})
		exp.key("x")
		self.assertEqual(act.takeCol("x", inPlace=False), exp.takeCol("x", inPlace=False))
		ay = act.getCol("y")
		ey = exp.getCol("y")
		for i in range(len(act)): self.assertTrue((ay[i] == ey[i]).all())

		# Multi simple.
		tt = Table({"k1": ["a", "a", "b", "b", "c", "c"], "k2": [1, 1, 1, 2, 2, 2],
			"v1": [10, 20, 30, 40, 50, 60], "v2": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6], "v3": 3*[True, False]})
		act = tt.by(["k2", "k1"], ["v1", "v2", "v3"])
		exp = Table({"k2": [1, 1, 2, 2], "k1": ["a", "b", "b", "c"],
			"v1": numpy.array([numpy.array(x) for x in [[10, 20], [30], [40], [50, 60]]], dtype=object),
			"v2": numpy.array([numpy.array(x) for x in [[1.1, 2.2], [3.3], [4.4], [5.5, 6.6]]], dtype=object),
			"v3": numpy.array([numpy.array(x) for x in [[True, False], [True], [False], [True, False]]], dtype=object)})
		exp.key(["k1", "k2"])
		self.assertEqual(act.takeCol(["k1", "k2"], inPlace=False), exp.takeCol(["k1", "k2"], inPlace=False))

		for c in ["v1", "v2", "v3"]:
			a = act.getCol(c)
			e = exp.getCol(c)
			for i in range(len(act)): self.assertTrue((a[i] == e[i]).all())

		# Complex.
		avg = lambda x: sum(x)/len(x)
		last = lambda x: x[-1]
		upper = lambda x: x.upper()
		times = lambda x, y: x*y
		plus = lambda x, y: x+y
		act = t.by(
			{"x1": [plus, "x", -1]},
			{
				"avg": [avg, "y"],
				"Z": [upper, [last, "z"]],
				"lit": ["constant"],
				"sum": [sum, [times, [plus, "x", 1], "y"]],
				"rename": "y"
			}
		)
		exp = Table({"x1": [0, 1], "avg": [80/3, 35], "Z": ["D", "E"], "lit": ["constant"]*2,"sum": [160, 210], 
			"rename": numpy.array([numpy.array([10, 30, 40]), numpy.array([20, 50])], dtype=object)})
		exp.key("x1")
		self.assertEqual(act.deleteCol("rename", inPlace=False), exp.deleteCol("rename", inPlace=False))
		ay = act.getCol("rename")
		ey = exp.getCol("rename")
		for i in range(len(ay)): self.assertTrue((ay[i] == ey[i]).all())

		# Ragged cols.
		drop = lambda x: x[:-1]
		act = t.by("x", {"r": [drop, "y"], "s": [sum, "y"]}, raggedCols="r")
		exp = Table({
			"x": [1, 2],
			"r": numpy.array([numpy.array([10, 30]), numpy.array([20])], dtype=object),
			"s": [80, 70]
		})
		exp.key("x")
		self.assertEqual(act.deleteCol("r", inPlace=False), exp.deleteCol("r", inPlace=False))
		ar = act.getCol("r")
		er = exp.getCol("r")
		for i in range(len(ar)): self.assertTrue((ar[i] == er[i]).all())

		# Key type error.
		self.assertRaisesRegex(TableException, "Incorrect clause key type: <class 'int'>", t.by, ["x", 1], "y")

	def test_join(self):
		# Normal case.
		base = Table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
		t = base.copy()
		s = Table({"z": [1.2, 3.4, 5.6]})
		t.join(s)
		exp = Table({"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [1.2, 3.4, 5.6]})
		self.assertEqual(t, exp)

		# Overwrite a column.
		t = base.copy()
		s = Table({"z": [1.2, 3.4, 5.6], "x": [False, True, False], "w": [10, 20, 30]})
		t.join(s)
		exp = Table({"x": [False, True, False], "y": ["a", "b", "c"], "z": [1.2, 3.4, 5.6], "w": [10, 20, 30]})
		self.assertEqual(t, exp)

		# Length error.
		t = base.copy()
		s = Table({"z": [1, 2]})
		self.assertRaisesRegex(TableException, "Join length mismatch, self=3 right=2", t.join, s)

		# Not in place.
		t = base.copy()
		s = Table({"z": [1.2, 3.4, 5.6]})
		act = t.join(s, inPlace=False)
		exp = Table({"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [1.2, 3.4, 5.6]})
		self.assertEqual(act, exp)
		self.assertEqual(t, base)

	def test_lj(self):
		# Single key -- explicit.
		left = Table({"x": [1, 2, 3], "y": [10, 20, 30]})
		right = Table({"z": ["a", "B", "C", "d", "e"], "x": [2, 1, 2, 3, 4]})
		left.lj(right, "x")
		exp = Table({"x": [1, 2, 3], "y": [10, 20, 30], "z": ["B", "a", "d"]})
		self.assertEqual(left, exp)		

		# Multi-key -- implicit.
		left = Table({"x": [1, 2, 3], "y": [10, 20, 30]})
		right = Table({"x": [1, 2, 3], "y": [10, 10, 30], "z": [1.1, 2.2, 3.3], "w": ["a", "b", "c"]})
		right.key(["x", "y"])
		left.lj(right)
		exp = Table({"x": [1, 2, 3], "y": [10, 20, 30], "z": [1.1, null.FLOAT, 3.3], "w": ["a", null.STRING, "c"]})
		self.assertEqual(left, exp)

		# Restrict columns + explicit > implicit.
		left = Table({"x": [1, 2, 3], "y": [10, 20, 30]})
		right = Table({"x": [2, 1, 3], "y": [200.1, 100.1, 300.1], "z": [2.2, 1.1, 3.3], "w": ["b", "a", "c"]})
		right.key("y")
		left.lj(right, key="x", col=["y", "z"])
		left.xcol(["x", "y", "z"])
		exp = Table({"x": [1, 2, 3], "y": [100.1, 200.1, 300.1], "z": [1.1, 2.2, 3.3]})
		self.assertEqual(left, exp)

		# Not in place.
		left = Table({"x": [1, 2, 3], "y": [10, 20, 30]})
		right = Table({"z": ["a", "B", "C", "d", "e"], "x": [2, 1, 2, 3, 4]})
		og = left.copy()
		act = left.lj(right, key="x", inPlace=False)
		exp = Table({"x": [1, 2, 3], "y": [10, 20, 30], "z": ["B", "a", "d"]})
		self.assertEqual(act, exp)
		self.assertEqual(left, og)

		# No keys.
		left = Table({"x": [1, 2, 3], "y": [10, 20, 30]})
		right = Table({"z": ["a", "B", "C", "d", "e"], "x": [2, 1, 2, 3, 4]})
		self.assertRaisesRegex(TableException, "Left join: no keys", left.lj, right)

		# Repeat keys (we rely on this).
		left = Table({"x": [1]})
		right = Table({"x": [1, 1], "y": [10, 20]})
		left.lj(right, "x")
		exp = Table({"x": [1], "y": [10]})
		self.assertEqual(left, exp)