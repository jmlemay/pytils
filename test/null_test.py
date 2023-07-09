########################################################################################################################
# Null tests.
########################################################################################################################

from src import null
import numpy
from unittest import TestCase

class NullTest(TestCase):
	def test_nulls(self):
		# Different types.
		self.assertTrue(null.isNull(null.getNull(numpy.array([1]).dtype.name))) # Int
		self.assertTrue(null.isNull(null.getNull(numpy.array([1.2]).dtype.name))) # Float
		self.assertTrue(null.isNull(null.getNull(numpy.array([object]).dtype.name))) # Object
		self.assertTrue(null.isNull(null.getNull(
			numpy.array([numpy.datetime64("2021-04-11 09:00:00")]).dtype.name))) # Datetime
		self.assertTrue(null.isNull(null.getNull(numpy.array(["awef"]).dtype.name))) # Time delta
		self.assertTrue(null.isNull(null.getNull(numpy.array([True]).dtype.name))) # Boolean

		# Unknown.
		self.assertRaisesRegex(Exception, "No null defined for type=dummy", null.getNull, "dummy")
		self.assertRaisesRegex(Exception, "Unrecognized type for null check, type=<class 'list'> val=\[\]", null.isNull, [])