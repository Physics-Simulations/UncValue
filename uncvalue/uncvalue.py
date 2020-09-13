"""
Simple python class to evaluate the uncertainty for complex or very long calculations
given the initial values together with its uncertainty.

Classes
-------
   Value

Functions
---------
    val
    unc
    set_unc

"""
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
    Returns the values of input parameter without the uncertainty as pure Python numbers.

    Parameters
    ----------
    b: int or float or tuple or list or numpy.ndarray or Value
        the parameter to extract the value from

    Returns
    -------
    value: int or float or numpy.ndarray
        the value without the uncertainty,
        if the input parameter is an array it will return an array of floats with the same shape

    See Also
    --------
    uncvalue.unc

    Examples
    --------
    >>> from uncvalue import Value, val
    >>> val(Value(3.14, 0.25))
    3.14
    >>> val([Value(3.14, 0.25), Value(2.17, 0.03)])
    array([3.14, 2.17])
    >>> val(Value([3.14, 2.17], [0.25, 0.03]))
    array([3.14, 2.17])
    """
    if isinstance(b, Value):
        return b.x

    if isinstance(b, list) or isinstance(b, tuple) or isinstance(b, np.ndarray):
        b = np.array(b)
        # TODO: extract type from original array
        r = np.empty_like(b, dtype=float)
        for idx, x in np.ndenumerate(b):
            r[idx] = val(x)
        return r

    return b


def unc(b):
    """
    Returns the uncertainty of input parameter without the value as pure Python numbers.

    Parameters
    ----------
    b: int or float or tuple or list or numpy.ndarray or Value
        the parameter to extract the uncertainty from

    Returns
    -------
    value: int or float or numpy.ndarray
        the uncertainty without the value,
        if the input parameter is already a pure python number it return 0,
        if the input parameter is an array it will return an array of floats with the same shape

    See Also
    --------
    uncvalue.val

    Examples
    --------
    >>> from uncvalue import Value, unc
    >>> unc(Value(3.14, 0.25))
    0.25
    >>> unc([Value(3.14, 0.25), Value(2.17, 0.03)])
    array([0.25, 0.03])
    >>> unc(Value([3.14, 2.17], [0.25, 0.03]))
    array([0.25, 0.03])
    """
    if isinstance(b, Value):
        return b.ux

    if isinstance(b, list) or isinstance(b, tuple) or isinstance(b, np.ndarray):
        b = np.array(b)
        r = np.empty_like(b, dtype=float)
        for idx, x in np.ndenumerate(b):
            r[idx] = unc(x)
        return r

    return 0


def set_unc(x, ux):
    """
    Set the given uncertainty to the specified value.

    Parameters
    ----------
    x : int or float or tuple or list or numpy.ndarray or Value
        the parameter to set the uncertainty to
    ux : int or float or tuple or list or numpy.ndarray
        the uncertainty to assciate to the given value,
        if `ux` is a list or similar it must have the same shape as `x`

    Returns
    -------
    value : Value or numpy.ndarray
        the value with the uncertainty,
        if the input parameter is an array it will return an array of Values with the same shape

    Notes
    -----
    For a single value this is the same as initialising a new instance of Value, i.e. `set_unc(3.14, 0.25)` and
    `Value(3.14, 0.25)` return the same result.

    Examples
    --------
    >>> from uncvalue import Value, set_unc
    >>> set_unc(3.14, 0.25)
    (31.4 ± 2.5)·10^-1
    >>> set_unc([3.14, 2.17], [0.25, 0.03])
    array([(31.4 ± 2.5)·10^-1, (217.0 ± 3.0)·10^-2], dtype=object)
    """
    if isinstance(x, Value):
        return Value(x.x, ux)

    if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
        x = np.array(x)
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
    """A class to keep track of the uncertainty of measured values.

    Parameters
    ----------
    x : int or float or tuple or list or numpy.ndarray
        the value or list of values
    ux : int or float or tuple or list or numpy.ndarray
        the uncertainty of the value,
        if `x` is a list and `ux` a single number,
        an array filled with `ux` will be created to match the dimensions of `x`,
        if both are list then the dimension must match

    Raises
    ------
    ValueError
        if any uncertainty is negative,
        if any value passed is complex,
        if dimensions of `x` and `ux` do not match when both are arrays
    """

    def __init__(self, x, ux):
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
            x = np.array(x)
            if isinstance(ux, list) or isinstance(x, tuple) or isinstance(ux, np.ndarray):
                ux = np.array(ux)
            else:
                ux = np.full(x.shape, ux)

            if x.shape != ux.shape:
                raise ValueError(f'x and ux must have the same shape: {x.shape} vs {ux.shape}')

        if np.any(ux < 0):
            raise ValueError(f'Negative uncertainty is non-sense: {ux}')
        if np.any(np.iscomplex(x)) or np.any(np.iscomplex(ux)):
            raise ValueError('Value and uncertainty must be real')

        self.x = x
        self.ux = ux

    @property
    def val(self):
        """returns the value of the given instance

        See Also
        --------
        uncvalue.val
        """
        return val(self)

    @property
    def unc(self):
        """returns the uncertainty of the given instance

        See Also
        --------
        uncvalue.unc
        """
        return unc(self)

    def __add__(self, y):
        """addition (+) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` + `y`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: a + b,
            lambda a, b: np.hypot(a.ux, b.ux)
        )

    def __radd__(self, y):
        """addition (+) of a number and the Value

        Returns
        -------
        Value
            `y` + `self`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: b + a,
            lambda a, b: np.hypot(b.ux, a.ux)
        )

    def __iadd__(self, y):
        """self addition (+=) of the number and a float or another Value

        Returns
        -------
        Value
            `self` += `y`
        """
        a = self.__add__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __sub__(self, y):
        """substraction (-) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` - `y`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: a - b,
            lambda a, b: np.hypot(a.ux, b.ux)
        )

    def __rsub__(self, y):
        """substraction (-) of a number and the Value

        Returns
        -------
        Value
            `y` - `self`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: b - a,
            lambda a, b: np.hypot(b.ux, a.ux)
        )

    def __isub__(self, y):
        """self substraction (-=) of the number and a float or another Value

        Returns
        -------
        Value
            `self` -= `y`
        """
        a = self.__sub__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __mul__(self, y):
        """product (\*) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` \* `y`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: a * b,
            lambda a, b: np.hypot(b.x * a.ux, a.x * b.ux)
        )

    def __rmul__(self, y):
        """product (\*) of a number and the Value

        Returns
        -------
        Value
            `y` \* `self`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: b * a,
            lambda a, b: np.hypot(a.x * b.ux, b.x * a.ux)
        )

    def __imul__(self, y):
        """self product (\*=) of the number and a float or another Value

        Returns
        -------
        Value
            `self` \*= `y`
        """
        a = self.__mul__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __div__(self, y):
        """division (/) of the Value and a number or another Value

        .. deprecated:: 3.0
          `__div__` has been removed from Python > 2.7 and replaced by `__truediv__`

        Returns
        -------
        Value
            `self` / `y`

        See Also
        --------
        uncvalue.Value.__truediv__
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: a / b,
            lambda a, b: np.hypot(a.ux / b.x, a.x * b.ux / (b.x * b.x))
        )

    def __truediv__(self, y):
        """true division (/) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` / `y`

        See Also
        --------
        operator.__truediv__
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: truediv(a, b),
            lambda a, b: np.hypot(truediv(a.ux, b.x), truediv(a.x * b.ux, b.x * b.x)))

    def __floordiv__(self, y):
        """integer division (//) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` // `y`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: a // b,
            lambda a, b: np.hypot(a.ux // b.x, a.x * b.ux // (b.x * b.x))
        )

    def __rdiv__(self, y):
        """right division (/) of a number and the Value

        .. deprecated:: 3.0
          `__rdiv__` has been removed from Python > 2.7 and replaced by `__rtruediv__`

        Returns
        -------
        Value
            `y` / `self`

        See Also
        --------
        uncvalue.Value.__rtruediv__
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: b / a,
            lambda a, b: np.hypot(b.ux / a.x, b.x * a.ux / (a.x * a.x))
        )

    def __rtruediv__(self, y):
        """right true division (/) of a number and the Value

        Returns
        -------
        Value
            `y` / `self`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: truediv(b, a),
            lambda a, b: np.hypot(truediv(b.ux, a.x), truediv(b.x * a.ux, a.x * a.x))
        )

    def __rfloordiv__(self, y):
        """right integer division (//) of a number and the Value

        Returns
        -------
        Value
            `y` // `self`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: b // a,
            lambda a, b: np.hypot(b.ux // a.x, b.x * a.ux // (a.x * a.x))
        )

    def __idiv__(self, y):
        """self division (/=) of the Value and a number or another Value

        .. deprecated:: 3.0
          `__idiv__` has been removed from Python > 2.7 and replaced by `__itruediv__`

        Returns
        -------
        Value
            `self` /= `y`

        See Also
        --------
        uncvalue.Value.__itruediv__
        """
        a = self.__div__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __itruediv__(self, y):
        """self true division (/=) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` /= `y`
        """
        a = self.__truediv__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __ifloordiv__(self, y):
        """self integer division (//=) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` //= `y`
        """
        a = self.__floordiv__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __pow__(self, y):
        """power (\*\*) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` \*\* `y`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: np.power(a, b),
            lambda a, b: np.hypot(np.log(np.abs(a.x)) * np.power(a.x, b.x) * b.ux, b.x * np.power(a.x, b.x - 1) * a.ux)
        )

    def __rpow__(self, y):
        """power (\*\*) of a number and the Value

        Returns
        -------
        Value
            `y` \*\* `self`
        """
        return _value_binary_operator(
            self, y,
            lambda a, b: np.power(b, a),
            lambda a, b: np.hypot(np.log(np.abs(b.x)) * np.power(b.x, a.x) * a.ux, a.x * np.power(b.x, a.x - 1) * b.ux)
        )

    def __ipow__(self, y):
        """self power (\*\*=) of the Value and a number or another Value

        Returns
        -------
        Value
            `self` \*\*= `y`
        """
        a = self.__pow__(y)
        self.x = a.x
        self.ux = a.ux
        return self

    def __neg__(self):
        """negation of the Value

        Returns
        -------
        Value
            `-x`
        """
        return Value(-self.x, self.ux)

    def __abs__(self):
        """absolute value of the Value

        Returns
        -------
        Value
            `|x|`
        """
        return Value(abs(self.x), self.ux)

    def __invert__(self):
        """invert value with respect to the multiplication of the Value

        Returns
        -------
        Value
            `1 / x`
        """
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

    def is_(self, y):
        """Extension of `is` keyword. Returns true if `self` is equal to `y` in the value"""
        return self.__eq__(y)

    def is_not(self, y):
        """Extension of `is not` keyword. Returns true if `self` is not  equal to `y` in the value"""
        return not self.__eq__(y)

    def __contains__(self, y):
        """Extension of `in` keyword. Checks if the `y` value is within the uncertainty of this value"""
        return self.x - self.ux <= val(y) <= self.x + self.ux

    def __repr__(self):
        return '{} ± {}'.format(self.x, self.ux)

    def __str__(self):
        """String represrntation of a number with its uncertainty.

        The number is rounded as to match the precision given by the two most significant digits of the uncertainty
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
        """rounds the number to match the precision dictated by the uncertainty

        Returns
        -------
        float or numpy.ndarray
            the value rounded to 2 significant digits of precision

        See Also
        --------
        numpy.around
        """
        return np.around(self.x, decimals=-self.precision())

    def __trunc__(self):
        """truncates the number to match the precision dictated by the uncertainty

        Returns
        -------
        float or numpy.ndarray
            the value truncated to 2 significant digits of precision

        See Also
        --------
        numpy.trunc
        """
        p = self.precision()
        return np.trunc(self.x * 10**(-p)) * 10**p

    def __floor__(self):
        """floors the number to match the precision dictated by the uncertainty

        Returns
        -------
        float or numpy.ndarray
            the value floored to 2 significant digits of precision

        See Also
        --------
        numpy.floor
        """
        p = self.precision()
        return np.floor(self.x * 10**(-p)) * 10**p

    def __ceil__(self):
        """ceils the number to match the precision dictated by the uncertainty

        Returns
        -------
        float or numpy.ndarray
            the value ceiled to 2 significant digits of precision

        See Also
        --------
        numpy.ceil
        """
        p = self.precision()
        return np.ceil(self.x * 10**(-p)) * 10**p

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
        """returns a new instance with the same value and uncertainty."""
        return Value(self.x, self.ux)

    def precision(self):
        """Gives the position of the least significant digit, counting as powers of 10.

        Returns
        -------
        int or numpy.ndarray of ints
            the precision of the number

        Examples
        --------
        Here are some examples of the output given by this function:

            ============  ===========
             Uncertainty  Precision
            ============  ===========
             1,45e-04     -7
             0,001456     -3
             0,0006666    -4
             0,123        -1
             1            0
             22,22        1
             7684,65      3
             17,8e8       9
            ============  ===========
        """
        return int(np.floor(np.log10(self.ux)))

    def log(self):
        """computes the natural logarithm of the value"""
        return Value(np.log(self.x), np.abs(self.ux / self.x))

    def log2(self):
        """computes the binary logarithm of the value"""
        return Value(np.log2(self.x), self.ux / np.abs(self.x * np.log(2)))

    def log10(self):
        """computes the decimal logarithm of the value"""
        return Value(np.log10(self.x), self.ux / np.abs(self.x * np.log(10)))

    def log1p(self):
        """computes the natural logarithm plus 1 of the value"""
        return Value(np.log1p(self.x), self.ux / np.abs(self.x + 1))

    def exp(self):
        """computes the exponentiation of the value"""
        a = np.exp(self.x)
        return Value(a, a * self.ux)

    def exp2(self):
        """computes the base two power of the value"""
        a = np.exp2(self.x)
        return Value(a, a * self.ux * np.log(2))

    def expm1(self):
        """computes the exponentiation minus 1 of the value"""
        return Value(np.expm1(self.x), np.exp(self.x) * self.ux)

    def sin(self):
        """computes the sinus of the value"""
        return Value(np.sin(self.x), np.abs(np.cos(self.x) * self.ux))

    def cos(self):
        """computes the cosinus of the value"""
        return Value(np.cos(self.x), np.abs(np.sin(self.x) * self.ux))

    def tan(self):
        """computes the tangent of the value"""
        return Value(np.tan(self.x), np.abs(self.ux / np.cos(self.x)**2))

    def arcsin(self):
        """computes the inverse sinus of the value"""
        return Value(np.arcsin(self.x), self.ux / np.sqrt(1 - self.x**2))

    def arccos(self):
        """computes the inverse cosinus of the value"""
        return Value(np.arccos(self.x), self.ux / np.sqrt(1 - self.x**2))

    def arctan(self):
        """computes the inverse tangent of the value"""
        return Value(np.arctan(self.x), self.ux / (1 + self.x**2))

    def sinh(self):
        """computes the hyperbolic sinus of the value"""
        return Value(np.sinh(self.x), np.abs(np.cosh(self.x) * self.ux))

    def cosh(self):
        """computes the hyperbolic cosinus of the value"""
        return Value(np.cosh(self.x), np.abs(np.sinh(self.x) * self.ux))

    def tanh(self):
        """computes the hyperbolic tangent of the value"""
        return Value(np.tanh(self.x), np.abs(self.ux / np.cosh(self.x)**2))

    def arcsinh(self):
        """computes the inverse hyperbolic sinus of the value"""
        return Value(np.arcsinh(self.x), self.ux / np.sqrt(1 + self.x**2))

    def arccosh(self):
        """computes the inverse hyperbolic cosinus of the value"""
        return Value(np.arccosh(self.x), self.ux / np.sqrt(self.x**2 - 1))

    def arctanh(self):
        """computes the inverse hyperbolic tangent of the value"""
        return Value(np.arctanh(self.x), self.ux / (1 - self.x**2))

    def sqrt(self):
        """computes the square root of the value"""
        s = np.sqrt(self.x)
        return Value(s, self.ux / (2 * s))

    def cbrt(self):
        """computes the cubic root of the value"""
        c = np.cbrt(self.x)
        return Value(c, self.ux / (3 * c**2))

    def min(self, **kwargs):
        """returns the minimum value in the range covered by the uncertainty

        Returns
        -------
        float or numpy.ndarray
            `x` - `ux`
        """
        return self.x - self.ux

    def max(self, **kwargs):
        """returns the maximum value in the range covered by the uncertainty

        Returns
        -------
        float or numpy.ndarray
            `x` + `ux`
        """
        return self.x + self.ux
