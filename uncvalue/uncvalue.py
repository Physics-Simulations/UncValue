from operator import truediv
from numbers import Number

import numpy as np


def _value_binary_operator(a, b, f, uf):
    val = None
    if isinstance(b, Value):
        val = b
    elif isinstance(b, np.ndarray):
        r = np.empty_like(b, dtype=object)
        for idx, x in np.ndenumerate(b):
            r[idx] = _value_binary_operator(a, x, f, uf)
        return r
    else:
        val = Value(b, 0)

    return Value(f(a.x, val.x), uf(a, val))


def _compare(a, b, f):
    val = 0
    if isinstance(b, Value):
        val = b
    elif isinstance(b, np.ndarray):
        r = np.empty_like(b, dtype=np.bool_)
        for idx, x in np.ndenumerate(b):
            r[idx] = _compare(a, x, f)
        return r
    else:
        val = Value(b, 0)

    # TODO: use uncertainty to compare values
    return f(a.x, val.x)


def val(b):
    """
        returns the value of a Value type or np.ndarray of Values
    """

    if isinstance(b, Value):
        return b.x
    elif isinstance(b, np.ndarray):
        r = np.empty_like(b, dtype=float)
        for idx, x in np.ndenumerate(b):
            r[idx] = val(x)
        return r

    return b


def unc(b):
    """
        returns the uncetainty of a Value type or np.ndarray of Values
    """

    if isinstance(b, Value):
        return b.ux
    elif isinstance(b, np.ndarray):
        r = np.empty_like(b, dtype=float)
        for idx, x in np.ndenumerate(b):
            r[idx] = unc(x)
        return r

    return 0


def set_unc(x, ux):
    """
        sets the uncetainty of a Value or number type
    """
    if isinstance(x, list):
        x = np.array(x)

    if isinstance(x, Value):
        return Value(x.x, ux)
    elif isinstance(x, np.ndarray):
        if isinstance(ux, list):
            ux = np.array(ux)
        if not isinstance(ux, np.ndarray):
            ux = np.full(x.shape, ux)

        if x.shape != ux.shape:
            raise ValueError(f'x and ux must have the same shape: {x.shape} vs {ux.shape}')

        r = np.empty_like(x, dtype=object)
        for idx, v in np.ndenumerate(x):
            r[idx] = Value(val(v), ux[idx])
        return r

    return Value(x, ux)


class Value(Number):

    def __init__(self, x, ux):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x = np.array(x)
            if isinstance(ux, list) or isinstance(ux, np.ndarray):
                ux = np.array(ux)
            else:
                ux = np.full(x.shape, ux)

            if x.shape != ux.shape:
                raise ValueError(f'x and ux must have the same shape: {x.shape} vs {ux.shape}')

        if np.any(ux < 0):
            raise ValueError(f'Negative uncertainty is non-sense: {ux}')
        if np.any(np.iscomplex(x)) or np.any(np.iscomplex(ux)):
            raise ValueError('Value must be real')

        self.x = x
        self.ux = ux

    @property
    def val(self):
        return val(self)

    @property
    def unc(self):
        return unc(self)

    def __add__(self, y):
        return _value_binary_operator(
            self, y,
            lambda a, b: a + b,
            lambda a, b: np.hypot(a.ux, b.ux)
        )

    def __radd__(self, y):
        return _value_binary_operator(
            self, y,
            lambda a, b: b + a,
            lambda a, b: np.hypot(b.ux, a.ux)
        )

    def __iadd__(self, y):  # self += y
        a = self.__add__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __sub__(self, y):  # self - y
        return _value_binary_operator(
            self, y,
            lambda a, b: a - b,
            lambda a, b: np.hypot(a.ux, b.ux)
        )

    def __rsub__(self, y):  # y - self
        return _value_binary_operator(
            self, y,
            lambda a, b: b - a,
            lambda a, b: np.hypot(b.ux, a.ux)
        )

    def __isub__(self, y):  # self -= y
        a = self.__sub__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __mul__(self, y):
        return _value_binary_operator(
            self, y,
            lambda a, b: a * b,
            lambda a, b: np.hypot(b.x * a.ux, a.x * b.ux)
        )

    def __rmul__(self, y):
        return _value_binary_operator(
            self, y,
            lambda a, b: b * a,
            lambda a, b: np.hypot(a.x * b.ux, b.x * a.ux)
        )

    def __imul__(self, y):  # self *= y
        a = self.__mul__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __div__(self, y):  # py2: self / y
        return _value_binary_operator(
            self, y,
            lambda a, b: a / b,
            lambda a, b: np.hypot(a.ux / b.x, a.x * b.ux / (b.x * b.x))
        )

    def __floordiv__(self, y):  # py3: self // y
        return _value_binary_operator(
            self, y,
            lambda a, b: a // b,
            lambda a, b: np.hypot(a.ux // b.x, a.x * b.ux // (b.x * b.x))
        )

    def __truediv__(self, y):  # py3: self / y
        return _value_binary_operator(
            self, y,
            lambda a, b: truediv(a, b),
            lambda a, b: np.hypot(truediv(a.ux, b.x), truediv(a.x * b.ux, b.x * b.x)))

    def __rdiv__(self, y):  # py2: y / self
        return _value_binary_operator(
            self, y,
            lambda a, b: b / a,
            lambda a, b: np.hypot(b.ux / a.x, b.x * a.ux / (a.x * a.x))
        )

    def __rfloordiv__(self, y):  # py3: y // self
        return _value_binary_operator(
            self, y,
            lambda a, b: b // a,
            lambda a, b: np.hypot(b.ux // a.x, b.x * a.ux // (a.x * a.x))
        )

    def __rtruediv__(self, y):  # py3: y / self
        return _value_binary_operator(
            self, y,
            lambda a, b: truediv(b, a),
            lambda a, b: np.hypot(truediv(b.ux, a.x), truediv(b.x * a.ux, a.x * a.x))
        )

    def __idiv__(self, y):  # self /= y
        a = self.__div__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __pow__(self, y):  # self**y
        return _value_binary_operator(
            self, y,
            lambda a, b: np.power(a, b),
            lambda a, b: np.hypot(np.log(np.abs(a.x)) * np.power(a.x, b.x) * b.ux, b.x * np.power(a.x, b.x - 1) * a.ux)
        )

    def __rpow__(self, y):  # y**self
        return _value_binary_operator(
            self, y,
            lambda a, b: np.power(b, a),
            lambda a, b: np.hypot(np.log(np.abs(b.x)) * np.power(b.x, a.x) * a.ux, a.x * np.power(b.x, a.x - 1) * b.ux)
        )

    def __neg__(self):
        return Value(-self.x, self.ux)

    def __abs__(self):
        return Value(abs(self.x), self.ux)

    def __invert__(self):
        return Value(1 / self.x, np.abs(self.ux / self.x**2))

    def __lt__(self, y):
        """Smaller than comparison between `self` and `y`"""
        return _compare(self, y, lambda a, b: a < b)

    def __le__(self, y):
        """Smaller or equal than comparison between `self` and `y`"""
        return _compare(self, y, lambda a, b: a <= b)

    def __eq__(self, y):  # x == y
        """Equal than comparison between `self` and `y`"""
        return _compare(self, y, lambda a, b: a == b)

    def __ne__(self, y):
        """Not equal to comparison between `self` and `y`"""
        return _compare(self, y, lambda a, b: a != b)

    def __gt__(self, y):  # x > y
        """Greater than comparison between `self` and `y`"""
        return _compare(self, y, lambda a, b: a > b)

    def __ge__(self, y):
        """Greater or equal than comparison between `self` and `y`"""
        return _compare(self, y, lambda a, b: a >= b)

    def __contains__(self, y):
        """Extension of `in` keyword. Checks if the `y` value is within the uncertainty of this value"""
        return self.x - self.ux <= val(y) <= self.x + self.ux

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """String represrntation of a number with its uncertainty.

        The number is rounded as to match the precision given by the two most significant digits of the uncetainty
        """
        if isinstance(self.x, np.ndarray):
            return '%s ± %s' % (self.x, self.ux)

        p = self.precision() - 1
        fix_x = np.around(self.x * 10**(-p), 0) / 10
        fix_ux = np.around(self.ux * 10**(-p), 0) / 10
        p += 1

        if p == 0:
            return '%.1f ± %.1f' % (fix_x, fix_ux)

        return '(%.1f ± %.1f)·10^%d' % (fix_x, fix_ux, p)

    def __round__(self, **kwargs):
        return np.around(self.x, decimals=-self.precision())

    # https://docs.python.org/3/reference/datamodel.html#object.__contains__
    def __trunc__(self):
        p = self.precision()
        return np.trunc(self.x * 10**(-p)) * 10**(p)

    def __floor__(self):
        p = self.precision()
        return np.floor(self.x * 10**(-p)) * 10**(p)

    def __ceil__(self):
        p = self.precision()
        return np.ceil(self.x * 10**(-p)) * 10**(p)

    def __complex__(self):
        """returns the complex representation of the value without uncertainty"""
        return complex(self.x)

    def __float__(self):
        """returns the float representation of the value without uncertainty"""
        return float(self.x)

    def __int__(self):
        """returns the int representation of the value without uncertainty"""
        return int(self.x)

    def __bool__(self):
        """returns true if the value is different from 0 and false otherwise"""
        return self.x != 0

    def copy(self):
        return Value(self.x, self.ux)

    def precision(self):
        """Gives the position of the least significant digit, counting as powers of 10."""
        return int(np.floor(np.log10(self.ux)))

    def log(self):
        return Value(np.log(self.x), np.abs(self.ux / self.x))

    def log2(self):
        return Value(np.log2(self.x), self.ux / np.abs(self.x * np.log(2)))

    def log10(self):
        return Value(np.log10(self.x), self.ux / np.abs(self.x * np.log(10)))

    def log1p(self):
        return Value(np.log1p(self.x), self.ux / np.abs(self.x + 1))

    def exp(self):
        a = np.exp(self.x)
        return Value(a, a * self.ux)

    def exp2(self):
        a = np.exp2(self.x)
        return Value(a, a * self.ux * np.log(2))

    def expm1(self):
        return Value(np.expm1(self.x), np.exp(self.x) * self.ux)

    def sin(self):
        return Value(np.sin(self.x), np.abs(np.cos(self.x) * self.ux))

    def cos(self):
        return Value(np.cos(self.x), np.abs(np.sin(self.x) * self.ux))

    def tan(self):
        return Value(np.tan(self.x), np.abs(self.ux / np.cos(self.x)**2))

    def arcsin(self):
        return Value(np.arcsin(self.x), self.ux / np.sqrt(1 - self.x**2))

    def arccos(self):
        return Value(np.arccos(self.x), self.ux / np.sqrt(1 - self.x**2))

    def arctan(self):
        return Value(np.arctan(self.x), self.ux / (1 + self.x**2))

    def sinh(self):
        return Value(np.sinh(self.x), np.abs(np.cosh(self.x) * self.ux))

    def cosh(self):
        return Value(np.cosh(self.x), np.abs(np.sinh(self.x) * self.ux))

    def tanh(self):
        return Value(np.tanh(self.x), np.abs(self.ux / np.cosh(self.x)**2))

    def arcsinh(self):
        return Value(np.arcsinh(self.x), self.ux / np.sqrt(1 + self.x**2))

    def arccosh(self):
        return Value(np.arccosh(self.x), self.ux / np.sqrt(self.x**2 - 1))

    def arctanh(self):
        return Value(np.arctanh(self.x), self.ux / (1 - self.x**2))

    def sqrt(self):
        s = np.sqrt(self.x)
        return Value(s, self.ux / (2 * s))

    def cbrt(self):
        c = np.cbrt(self.x)
        return Value(c, self.ux / (3 * c**2))

    def min(self, **kwargs):
        return self.x - self.ux

    def max(self, **kwargs):
        return self.x + self.ux
