########################################################################################################################
# Miscellaneous function tests.
########################################################################################################################

import math, numpy
from unittest import TestCase
from src import mock, misc

class MiscTest(TestCase):
	def tearDown(self):
		mock.resetMocks()

	def test_now(self):
		# Defaults ms.
		now = misc.now()
		self.assertIsInstance(now, numpy.datetime64)
		self.assertRegex(str(now),
			"[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9].[0-9][0-9][0-9]")

		# Different units.
		self.assertRegex(str(misc.now("D")), "[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]")
		self.assertRegex(str(misc.now("us")),
			"[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]")

	def test_today(self):
		today = misc.today()
		self.assertIsInstance(today, numpy.datetime64)
		self.assertRegex(str(today), "[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]")

	def test_timeDelta(self):
		self.assertEqual(misc.timeDelta(), numpy.timedelta64(0))
		self.assertEqual(misc.timeDelta(Y=10), numpy.timedelta64(10, "Y"))
		self.assertEqual(misc.timeDelta(ms=1000), numpy.timedelta64(1000, "ms"))
		self.assertEqual(misc.timeDelta(h=6, s=45, us=92),
			numpy.timedelta64(6, "h") + numpy.timedelta64(45, "s") + numpy.timedelta64(92, "us"))

	def test_plurals(self):
		self.assertEqual(misc.ps(1), "")
		self.assertEqual(misc.ps(2), "s")
		self.assertEqual(misc.pes(1), "")
		self.assertEqual(misc.pes(-1), "es")
		self.assertEqual(misc.pies(1), "y")
		self.assertEqual(misc.pies(0), "ies")

	def test_distinct(self):
		# List.
		l = ["a", "list", "a", "with", "with", "a", "repeats"]
		self.assertEqual(misc.distinct(l), ["a", "list", "repeats", "with"])

		# Tuple.
		l = (1, 1, 5, 2, 5, 5, 2, 3)
		self.assertEqual(misc.distinct(l), (1, 2, 3, 5))
		numpy.unique

		#
		# Array.
		#
		l = numpy.array([1, 1, 2, 1, 2])
		self.assertTrue((misc.distinct(l) == numpy.array([1, 2])).all())

	def test_getVal(self):
		d = {"x": 10, "y": 20}
		self.assertEqual(misc.getVal(d, "x", 100), 10)
		self.assertEqual(misc.getVal(d, "z", 100), 100)

	def test_mkList(self):
		self.assertEqual(misc.mkList(1), [1]) # Enlist int
		self.assertEqual(misc.mkList("string"), ["string"]) # Enlist string
		self.assertEqual(misc.mkList([1, 2]), [1, 2]) # Enlist list (does nothing)
		self.assertEqual(misc.mkList((1, 2)), [(1, 2)]) # Enlist tuple

	def test_asof(self):
		self.assertEqual(misc.asof([], 1), -1) # Empty
		self.assertEqual(misc.asof([1, 4, 10], 0), -1) # Too small
		self.assertEqual(misc.asof([1], 2), 0) # Search one element
		self.assertEqual(misc.asof([1], 0), -1) # Search one element but too small
		self.assertEqual(misc.asof("aepqz", "b"), 0) # First
		self.assertEqual(misc.asof("aepqz", "a"), 0) # Almost first
		self.assertEqual(misc.asof("abcdef", "f"), 5) # Last
		self.assertEqual(misc.asof([1, 2, 3, 4, 5], 100), 4) # Beyond
		self.assertEqual(misc.asof([1, 10, 20, 40, 50, 60 , 70], 59), 4) # General
		self.assertEqual(misc.asof([1, 10, 20, 40, 50, 60 , 70], 60), 5) # General -- exact
		self.assertEqual(misc.asof([1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10], 3), 2) # Multi

	def test_where(self):
		self.assertEqual(misc.where([True, True, True]), [0, 1, 2]) # All true
		self.assertEqual(misc.where([False, False]), []) # All true
		self.assertEqual(misc.where([False, True, True, False, True]), [1, 2, 4]) # Mix
		self.assertEqual(misc.where([]), []) # Empty

	def test_remove(self):
		self.assertEqual(misc.remove([], []), []) # Both empty
		self.assertEqual(misc.remove([1, 2, 3], []), [1, 2, 3]) # Right empty
		self.assertEqual(misc.remove([], [1, 2]), []) # Left empty
		self.assertEqual(misc.remove([1, 2, 3], [4, 5]), [1, 2, 3]) # No intersection
		self.assertEqual(misc.remove([1, 2, 3, 4, 5], [1, 3]), [2, 4, 5]) # Non-trivial intersection
		self.assertEqual(misc.remove([1, 2, 3], [10, 1, 20, 3, 2]), []) # Complete overlap
		self.assertEqual(set(misc.remove("abc", "b")), set(["c", "a"])) # General case with strings (order not guaranteed)
		self.assertEqual(misc.remove(["abc", "def", "gh"], ["abc", "gh"]), ["def"]) # General case with lists of strings
		self.assertEqual(set(misc.remove([1, 2, 3], 2)), set([1, 3])) # Remove atom