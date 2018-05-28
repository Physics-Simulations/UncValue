from __future__ import division
import numpy as np
import sys
#import re

#PROG = re.compile(r'^(%d).%d')

if sys.version_info[0] > 2:
	xrange = lambda x: range(x)

_TOLERANCE = 5e-7

def _value_binary_operator(a, b, f, uf):
	val = 0
	if isinstance(b, Value):
		val = b
	elif isinstance(b, np.ndarray):
		r = np.empty_like(b, dtype=object)
		for i in xrange(len(b)):
			r[i] = f(a, b[i])
		return r
	else:
		val = Value(b, 0)

	return Value(f(a.x, val.x), uf(a, val))

def _compare(a, b, f):
	val = 0
	if isinstance(b, Value):
		val = b.x
	elif isinstance(b, np.ndarray):
		r = np.empty_like(b)
		for i in xrange(len(b)):
			r[i] = f(a, b[i])
		return r
	else:
		val = b

	return f(a, val)

def _equals(x, y, tol=_TOLERANCE):
	return abs(y - x) < tol

def val(b):
	"""
		returns the value of a Value type or np.ndarray of Values
	"""

	if isinstance(b, Value):
		return b.x
	elif isinstance(b, np.ndarray):
		r = np.empty_like(b)
		for i in xrange(len(b)):
			r[i] = val(b[i])
		return r
	
	return b

def unc(b):
	"""
		returns the uncetainty of a Value type or np.ndarray of Values
	"""

	if isinstance(b, Value):
		return b.ux
	elif isinstance(b, np.ndarray):
		r = np.empty_like(b)
		for i in xrange(len(b)):
			r[i] = unc(b[i])
		return r
	
	return 0

class Value:

	def __init__(self, x, ux):
		self.x = x
		self.ux = ux

	def __add__(self, y):
		return _value_binary_operator(self, y, 
			lambda a,b: a+b, 
			lambda a,b: np.hypot(a.ux, b.ux))

	def __radd__(self, y):
		return self.__add__(y)

	def __iadd__(self, y): # self += y
		a = self.__add__(self, y)
		self.x = a.x
		self.ux = a.ux

	def __sub__(self, y): # self - y
		return _value_binary_operator(self, y, 
			lambda a,b: a-b, 
			lambda a,b: np.hypot(a.ux, b.ux))

	def __rsub__(self, y): # y - self
		return _value_binary_operator(self, y, 
			lambda a,b: b-a, 
			lambda a,b: np.hypot(a.ux, b.ux))

	def __isub__(self, y): # self -= y
		a = self.__sub__(self, y)
		self.x = a.x
		self.ux = a.ux

	def __mul__(self, y):
		return _value_binary_operator(self, y, 
			lambda a,b: a*b, 
			lambda a,b: np.hypot(b.x * a.ux, a.x * b.ux))

	def __rmul__(self, y):
		return self.__mul__(y)

	def __imul__(self, y): # self *= y
		a = self.__mul__(self, y)
		self.x = a.x
		self.ux = a.ux

	def __div__(self, y): # self / y
		return _value_binary_operator(self, y, 
			lambda a,b: a//b, 
			lambda a,b: np.hypot(a.ux // b.x, a.x * b.ux // (b.x * b.x)))

	def __truediv__(self, y): # self // y
		return _value_binary_operator(self, y, 
			lambda a,b: a/b, 
			lambda a,b: np.hypot(a.ux / b.x, a.x * b.ux / (b.x * b.x)))

	def __rdiv__(self, y):
		return _value_binary_operator(self, y, 
			lambda a,b: b/a, 
			lambda a,b: np.hypot(b.ux / a.x, b.x * a.ux / (a.x * a.x)))

	def __idiv__(self, y): # self /= y
		a = self.__div__(self, y)
		self.x = a.x
		self.ux = a.ux

	def __pow__(self, y): # self**y
		return _value_binary_operator(self, y, 
			lambda a,b: np.power(a,b), 
			lambda a,b: np.hypot(np.log(np.abs(a.x))*np.power(a.x,b.x)*a.ux, b.x*np.power(a.x,b.x-1)*b.ux))

	def __rpow__(self, y): # y**self
		return _value_binary_operator(self, y, 
			lambda a,b: np.power(b,a), 
			lambda a,b: np.hypot(np.log(np.abs(b.x))*np.power(b.x,a.x)*b.ux, a.x*np.power(b.x,a.x-1)*a.ux))

	def __neg__(self):
		return Value(-self.x, self.ux)

	def __abs__(self):
		return Value(abs(self.x), self.ux)

	def __invert__(self):
		a = self.x
		self.x = 1 / self.x
		self.ux = np.abs(self.ux / (a*a))
		return self

	def __getitem__(self, key):
		if isinstance(key, int):
			if key == 0:
				return self.x
			elif key == 1:
				return self.ux
		elif isinstance(key, slice):
			return (self.x, self.ux)

		raise ValueError('Invalid index %s in Value getitem' % key)

	def __setitem__(self, key, val):
		if isinstance(key, int):
			if key == 0:
				self.x = val
			elif key == 1:
				self.ux = val
		elif isinstance(key, slice) and len(val) == 2:
			self.x, self.ux = val[0], val[1]

	def __lt__(self, y): # x < y
		return _compare(self.x, y, lambda a,b: a < b)

	def __le__(self, y): # x <= y
		return _compare(self.x, y, lambda a,b: a <= b)

	def __eq__(self, y): # x == y
		return _compare(self.x, y, lambda a,b: a == b)

	def __ne__(self, y): # x <> y (not equals)
		return _compare(self.x, y, lambda a,b: a != b)

	def __gt__(self, y): # x > y
		return _compare(self.x, y, lambda a,b: a > b)

	def __ge__(self, y): # x >= y
		return _compare(self.x, y, lambda a,b: a >= b)

	def __repr__(self):
		if isinstance(self.x, np.ndarray):
			return '%s +/- %s' % (self.x, self.ux)

		return '%.5g +/- %.5g' % (self.x, self.ux)

	def __str__(self):
		if isinstance(self.x, np.ndarray):
			return '%s +/- %s' % (self.x, self.ux)

		return '%.5g +/- %.5g' % (self.x, self.ux)

	def log(self):
		a = self.x
		return Value(np.log(a), np.abs(self.ux / a))

	def exp(self):
		return Value(np.exp(self.x), np.abs(self.x * self.ux))

	def sin(self):
		a = self.x
		return Value(np.sin(a), np.abs(np.cos(a) * self.ux))

	def cos(self):
		a = self.x
		return Value(np.cos(a), np.abs(np.sin(a) * self.ux))

	def tan(self):
		a = self.x
		return Value(np.tan(a), np.abs(self.ux / np.power(np.cos(a), 2)))

	def sqrt(self):
		a = self.x
		self.x = np.sqrt(a)
		self.ux = self.ux / (2 * a)
		return self

	def arcsin(self):
		a = self.x
		self.x = np.arcsin(a)
		self.ux = self.ux / np.sqrt(1 - np.square(a))
		return self

	def arccos(self):
		a = self.x
		self.x = np.arccos(a)
		self.ux = self.ux / np.sqrt(1 - np.square(a))
		return self

	def arctan(self):
		a = self.x
		return Value(np.arctan(a), self.ux / (1 + np.square(a)))