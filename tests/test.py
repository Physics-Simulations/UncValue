import pytest

from math import trunc, ceil, floor

import numpy as np

from uncvalue import Value, val, unc, set_unc

ϵ = 1e-8

a = Value(3.1415, 0.0012)
b = Value(-1.618, 0.235)
c = Value(3.1264e2, 1.268)

A = np.array([[a, a], [b, b], [c, c]])
B = Value([a.x] * 5, a.ux)
C = Value([b.x] * 5, [b.ux] * 5)


@pytest.mark.parametrize('v, x', [
    (a, a.x),
    (A, np.array([[a.x, a.x], [b.x, b.x], [c.x, c.x]])),
    (B, a.x),
    (a.x, a.x)],
    ids=['Single', 'Array of values', 'Value array', 'Number'])
def test_val(v, x):
    assert np.all(val(v) == x)


@pytest.mark.parametrize('v, x', [
    (a, a.ux),
    (A, np.array([[a.ux, a.ux], [b.ux, b.ux], [c.ux, c.ux]])),
    (B, a.ux),
    (a.x, 0)],
    ids=['Single', 'Array of values', 'Value array', 'Number'])
def test_unc(v, x):
    assert np.all(unc(v) == x)


def test_set_unc():
    v = set_unc(0.234, 0.0052)
    assert isinstance(v, Value)
    assert v.x == 0.234
    assert v.ux == 0.0052

    v = set_unc(a, 0.0052)
    assert isinstance(v, Value)
    assert v.x == a.x
    assert v.ux == 0.0052

    v = set_unc([0.234] * 8, 0.0052)
    assert isinstance(v, np.ndarray)
    assert v.shape == (8, )
    assert np.mean(unc(v)) == 0.0052

    v = set_unc([0.234] * 8, [0.0052] * 8)
    assert isinstance(v, np.ndarray)
    assert v.shape == (8, )
    assert np.mean(unc(v)) == 0.0052

    with pytest.raises(ValueError):
        set_unc(np.random.random((3, 2, 1)), np.random.random((4, 2, 1)))


def test_constructor():
    v = Value(3.1415, 0.0012)
    assert v.x == 3.1415 == v.val
    assert v.ux == 0.0012 == v.unc

    with pytest.raises(ValueError):
        Value(3.14, -0.28)

    V = Value([3.1415] * 8, 0.0012)
    assert V.x.shape == (8, )
    assert V.ux.shape == (8, )
    assert np.mean(V.ux) == 0.0012

    V = Value([3.1415] * 8, [0.0012] * 8)
    assert V.x.shape == (8, )
    assert V.ux.shape == (8, )
    assert np.mean(V.ux) == 0.0012

    with pytest.raises(ValueError):
        Value(np.random.random((3, 2, 1)), np.random.random((4, 2, 1)))

    with pytest.raises(ValueError):
        Value(1j, 0)
        Value(1, 2j)


@pytest.mark.parametrize('x, y, r', [
    (a.x, a, False),
    (a, a.x, False),
    (a, Value(a.x, a.ux * 5), False),
    (b, a, True),
    (a, a - 0.0001, False),
    (A, A, False),
    (B, C, False)],
    ids=['Right', 'Left', 'Both', 'Different', 'Within unc', 'Array eq', 'Array dif'])
def test_smaller(x, y, r):
    assert np.all((x < y) == r)


@pytest.mark.parametrize('x, y, r', [
    (1, a, Value(a.x + 1, a.ux)),
    (a, 1, Value(a.x + 1, a.ux)),
    (a, b, Value(a.x + b.x, np.hypot(a.ux, b.ux))),
    (1, A, np.array([[a+1, a+1], [b+1, b+1], [c+1, c+1]])),
    (a, A, np.array([[a+a, a+a], [b+a, b+a], [c+a, c+a]])),
    (1, B, Value(1 + B.x, B.ux)),
    (a, B, Value(a.x + B.x, np.hypot(a.ux, B.ux))),
    (A, A, np.array([[a+a, a+a], [b+b, b+b], [c+c, c+c]])),
    (B, C, Value(B.x + C.x, np.hypot(B.ux, C.ux))),
    ],
    ids=['Right', 'Left', 'Both', 'Number + Array', 'Value + Array', 'Array of values',
         'Number + Valued array', 'Value + Valued array', 'Valued array'])
def test_add(x, y, r):
    z = x + y
    assert np.all(val(z) == val(r))
    assert np.all(unc(z) == unc(r))


@pytest.mark.parametrize('x, y, r', [
    (1, a, Value(a.x + 1, a.ux)),
    (a.copy(), 1, Value(a.x + 1, a.ux)),
    (a.copy(), b, Value(a.x + b.x, np.hypot(a.ux, b.ux))),
    (B.copy(), C, Value(B.x + C.x, np.hypot(B.ux, C.ux))),
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_iadd(x, y, r):
    x += y
    assert isinstance(x, Value)
    assert np.all(x.x == r.x)
    assert np.all(x.ux == r.ux)


@pytest.mark.parametrize('x, y, r', [
    (1, a, Value(1 - a.x, a.ux)),
    (a, 1, Value(a.x - 1, a.ux)),
    (a, b, Value(a.x - b.x, np.hypot(a.ux, b.ux))),
    (A, A, np.array([[a-a, a-a], [b-b, b-b], [c-c, c-c]])),
    (B, C, Value(B.x - C.x, np.hypot(B.ux, C.ux))),
    ],
    ids=['Right', 'Left', 'Both', 'Array of values', 'Valued array'])
def test_sub(x, y, r):
    z = x - y
    assert np.all(val(z) == val(r))
    assert np.all(unc(z) == unc(r))


@pytest.mark.parametrize('x, y, r', [
    (1, a, Value(1 - a.x, a.ux)),
    (a.copy(), 1, Value(a.x - 1, a.ux)),
    (a.copy(), b, Value(a.x - b.x, np.hypot(a.ux, b.ux))),
    (B.copy(), C, Value(B.x - C.x, np.hypot(B.ux, C.ux))),
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_isub(x, y, r):
    x -= y
    assert isinstance(x, Value)
    assert np.all(x.x == r.x)
    assert np.all(x.ux == r.ux)


@pytest.mark.parametrize('x, y, r', [
    (2, a, Value(2 * a.x, 2 * a.ux)),
    (a, 2, Value(a.x * 2, 2 * a.ux)),
    (a, b, Value(a.x * b.x, np.hypot(a.ux * b.x, a.x * b.ux))),
    (A, A, np.array([[a*a, a*a], [b*b, b*b], [c*c, c*c]])),
    (B, C, Value(B.x * C.x, np.hypot(B.ux * C.x, B.x * C.ux))),
    ],
    ids=['Right', 'Left', 'Both', 'Array of values', 'Valued array'])
def test_mul(x, y, r):
    z = x * y
    assert np.all(val(z) == val(r))
    assert np.all(unc(z) == unc(r))


@pytest.mark.parametrize('x, y, r', [
    (2, a, Value(2 * a.x, 2 * a.ux)),
    (a.copy(), 2, Value(a.x * 2, 2 * a.ux)),
    (a.copy(), b, Value(a.x * b.x, np.hypot(a.ux * b.x, a.x * b.ux))),
    (B.copy(), C, Value(B.x * C.x, np.hypot(B.ux * C.x, B.x * C.ux))),
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_imul(x, y, r):
    x *= y
    assert isinstance(x, Value)
    assert np.all(x.x == r.x)
    assert np.all(x.ux == r.ux)


@pytest.mark.parametrize('x, y, r', [
    (2, a, Value(2 / a.x, 2 * a.ux / a.x**2)),
    (a, 2, Value(a.x / 2, a.ux / 2)),
    (a, b, Value(a.x / b.x, np.hypot(a.ux / b.x, a.x * b.ux / b.x**2))),
    (B, C, Value(B.x / C.x, np.hypot(B.ux / C.x, B.x * C.ux / C.x**2))),
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_div(x, y, r):
    z = x / y
    assert isinstance(z, Value)
    assert np.all(z.x == r.x)
    assert np.all(z.ux == r.ux)


@pytest.mark.parametrize('x, y, r', [
    (2, a, Value(2 // a.x, 2 * a.ux // a.x**2)),
    (a, 2, Value(a.x // 2, a.ux // 2)),
    (a, b, Value(a.x // b.x, np.hypot(a.ux // b.x, a.x * b.ux // b.x**2))),
    (B, C, Value(B.x // C.x, np.hypot(B.ux // C.x, B.x * C.ux // C.x**2))),
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_floordiv(x, y, r):
    z = x // y
    assert isinstance(z, Value)
    assert np.all(z.x == r.x)
    assert np.all(z.ux == r.ux)


@pytest.mark.parametrize('x, y, r', [
    (2, a, Value(2 / a.x, 2 * a.ux / a.x**2)),
    (a.copy(), 2, Value(a.x / 2, a.ux / 2)),
    (a.copy(), b, Value(a.x / b.x, np.hypot(a.ux / b.x, a.x * b.ux / b.x**2))),
    (B.copy(), C, Value(B.x / C.x, np.hypot(B.ux / C.x, B.x * C.ux / C.x**2))),
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_idiv(x, y, r):
    x /= y
    assert isinstance(x, Value)
    assert np.all(x.x == r.x)
    assert np.all(x.ux == r.ux)


@pytest.mark.parametrize('x, y, r', [
    (2, a, Value(2 ** a.x, abs(2**a.x * np.log(2) * a.ux))),
    (a, 2, Value(a.x ** 2, abs(2 * a.x**(2 - 1)) * a.ux)),
    (a, b, Value(a.x ** b.x, np.hypot(b.x * a.x**(b.x-1) * a.ux, a.x**b.x * np.log(np.abs(a.x)) * b.ux))),
    (B, C, Value(B.x ** C.x, np.hypot(C.x * B.x**(C.x-1) * B.ux, B.x**C.x * np.log(np.abs(B.x)) * C.ux)))
    ],
    ids=['Right', 'Left', 'Both', 'Array'])
def test_pow(x, y, r):
    z = x**y
    assert isinstance(z, Value)
    assert np.all(z.x == r.x)
    assert np.all(z.ux == r.ux)


@pytest.mark.parametrize('x, r', [
    (a, Value(-a.x, a.ux)),
    (A, np.array([[-a, -a], [-b, -b], [-c, -c]])),
    (B, Value(-B.x, B.ux))
    ], ids=['Value', 'Array of values', 'Value array'])
def test_neg(x, r):
    z = -x
    assert np.all(val(z) == val(r))
    assert np.all(unc(z) == unc(r))


@pytest.mark.parametrize('x, r', [
    (b, Value(abs(b.x), b.ux)),
    (A, np.array([[a, a], [-b, -b], [c, c]])),
    (B, Value(B.x, B.ux))
    ], ids=['Value', 'Array of values', 'Value array'])
def test_abs(x, r):
    z = abs(x)
    assert np.all(val(z) == val(r))
    assert np.all(unc(z) == unc(r))


@pytest.mark.parametrize('x, r', [
    (a, Value(1 / a.x, a.ux / a.x**2)),
    (A, np.array([[1 / a, 1 / a], [1 / b, 1 / b], [1 / c, 1 / c]])),
    (B, Value(1 / B.x, B.ux / B.x**2))
    ], ids=['Value', 'Array of values', 'Value array'])
def test_invert(x, r):
    z = ~x
    assert np.all(val(z) == val(r))
    assert np.all(unc(z) == unc(r))


@pytest.mark.parametrize('x, y, r', [
    (a.x, a, True),
    (a, a.x, True),
    (a, Value(a.x, a.ux * 5), True),
    (a, b, False),
    (a, a + 0.0001, False),
    (A, A, True),
    (B, C, False)],
    ids=['Right', 'Left', 'Both', 'Different', 'Within unc', 'Array eq', 'Array dif'])
def test_equality(x, y, r):
    assert np.all((x == y) == r)
    assert np.all((x != y) != r)


@pytest.mark.parametrize('x, y, r', [
    (a.x, a, True),
    (a, a.x, True),
    (a, Value(a.x, a.ux * 5), True),
    (b, a, False),
    (a, a - 0.0001, True),
    (A, A, True),
    (B, C, True)],
    ids=['Right', 'Left', 'Both', 'Different', 'Within unc', 'Array eq', 'Array dif'])
def test_greater_equal(x, y, r):
    assert np.all((x >= y) == r)


@pytest.mark.parametrize('x, y, r', [
    (a.x, a, False),
    (a, a.x, False),
    (a, Value(a.x, a.ux * 5), False),
    (b, a, False),
    (a, a - 0.0001, True),
    (A, A, False),
    (B, C, True)],
    ids=['Right', 'Left', 'Both', 'Different', 'Within unc', 'Array eq', 'Array dif'])
def test_greater(x, y, r):
    assert np.all((x > y) == r)


@pytest.mark.parametrize('x, y, r', [
    (a.x, a, True),
    (a, a.x, True),
    (a, Value(a.x, a.ux * 5), True),
    (b, a, True),
    (a, a - 0.0001, False),
    (A, A, True),
    (B, C, False)],
    ids=['Right', 'Left', 'Both', 'Different', 'Within unc', 'Array eq', 'Array dif'])
def test_smaller_equal(x, y, r):
    assert np.all((x <= y) == r)


@pytest.mark.parametrize('x, y, r', [
    (1, Value(1, 2), True),
    (1, Value(0.75, 0.05), False),
    (0.8, Value(0.75, 0.08), True),
    (0.8, Value(0.75, 0.05), True),
    (0.7, Value(0.75, 0.05), True),
    (Value(0.8, 0.2), Value(0.7, 0.04), False)],
    ids=['Inside', 'Outside', 'Inside float', 'Over upper limit', 'Over lower limit', 'Outside second value'])
def test_contains(x, y, r):
    assert (x in y) == r


@pytest.mark.parametrize('x, s', [
    (Value(2, 2), '2.0 ± 2.0'),
    (Value(0.2, 2), '0.2 ± 2.0'),
    (Value(0.2, 0.002385), '(200.0 ± 2.4)·10^-3'),
    (Value(0.02414, 0.002345), '(24.1 ± 2.3)·10^-3'),
    (Value(0.02415, 0.002365), '(24.2 ± 2.4)·10^-3')])
def test_print(x, s):
    assert str(x) == s


@pytest.mark.parametrize('x, p, r', [
    (0.145e-6, -8, 0.14e-6),
    (0.001456, -4, 0.0015),
    (0.0006666, -5, 0.00067),
    (0.123, -2, 0.12),
    (1, -1, 1.0),
    (22.22, 0, 22.0),
    (7684.65, 2, 77e2),
    (17.8e8, 8, 18e8)])
def test_round(x, p, r):
    v = Value(x, 10.0**p)
    assert abs(round(v) - r) < ϵ


@pytest.mark.parametrize('x, p, r', [
    (0.145e-6, -8, 0.14e-6),
    (0.001456, -4, 0.0014),
    (0.0006666, -5, 0.00066),
    (0.123, -2, 0.12),
    (1, -1, 1.0),
    (22.22, 0, 22.0),
    (7684.65, 2, 76e2),
    (17.8e8, 8, 17e8)])
def test_trunc(x, p, r):
    v = Value(x, 10.0**p)
    assert abs(trunc(v) - r) < ϵ


@pytest.mark.parametrize('x, p, r', [
    (0.145e-6, -8, 0.14e-6),
    (0.001456, -4, 0.0014),
    (0.0006666, -5, 0.00066),
    (0.123, -2, 0.12),
    (1, -1, 1.0),
    (22.22, 0, 22.0),
    (7684.65, 2, 76e2),
    (17.8e8, 8, 17e8)])
def test_floor(x, p, r):
    v = Value(x, 10.0**p)
    assert abs(floor(v) - r) < ϵ


@pytest.mark.parametrize('x, p, r', [
    (0.145e-6, -8, 0.15e-6),
    (0.001456, -4, 0.0015),
    (0.0006666, -5, 0.00067),
    (0.123, -2, 0.13),
    (1, -1, 1.0),
    (22.22, 0, 23.0),
    (7684.65, 2, 77e2),
    (17.8e8, 8, 18e8)])
def test_ceil(x, p, r):
    v = Value(x, 10.0**p)
    assert abs(ceil(v) - r) < ϵ


def test_convert_to_number():
    assert complex(a) == a.x + 0j
    assert float(a) == a.x
    assert int(a) == 3
    assert bool(a)

    assert not bool(Value(0, 0.0028))


def test_copy():
    v = a.copy()
    assert v.x == a.x
    assert v.ux == a.ux


@pytest.mark.parametrize('ux, acc', [
    (0.145e-6, -7),
    (0.001456, -3),
    (0.0006666, -4),
    (0.123, -1),
    (1, 0),
    (22.22, 1),
    (7684.65, 3),
    (17.8e8, 9)])
def test_precision(ux, acc):
    v = Value(1, ux)
    assert v.precision() == acc


@pytest.mark.parametrize('x', [
    a, abs(A), B
], ids=['Value', 'Array of values', 'Valued array'])
def test_log(x):
    z = np.log(x)
    assert np.all(val(z) == np.log(val(x)))
    assert np.all(unc(z) == unc(x) / val(x))


@pytest.mark.parametrize('x', [
    a, abs(A), B
], ids=['Value', 'Array of values', 'Valued array'])
def test_log2(x):
    z = np.log2(x)
    assert np.all(val(z) == np.log2(val(x)))
    assert np.all(unc(z) == unc(x) / (val(x) * np.log(2)))


@pytest.mark.parametrize('x', [
    a, abs(A), B
], ids=['Value', 'Array of values', 'Valued array'])
def test_log10(x):
    z = np.log10(x)
    assert np.all(val(z) == np.log10(val(x)))
    assert np.all(unc(z) == unc(x) / (val(x) * np.log(10)))


@pytest.mark.parametrize('x', [
    a, abs(A), B
], ids=['Value', 'Array of values', 'Valued array'])
def test_log1p(x):
    z = np.log1p(x)
    assert np.all(val(z) == np.log1p(val(x)))
    assert np.all(unc(z) == unc(x) / (val(x) + 1))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_exp(x):
    z = np.exp(x)
    assert np.all(val(z) == np.exp(val(x)))
    assert np.all(unc(z) == unc(x) * np.exp(val(x)))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_exp2(x):
    z = np.exp2(x)
    assert np.all(val(z) == np.exp2(val(x)))
    assert np.all(unc(z) == unc(x) * np.exp2(val(x)) * np.log(2))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Nagetive', 'Array of values', 'Valued array'])
def test_expm1(x):
    z = np.expm1(x)
    assert np.all(val(z) == np.expm1(val(x)))
    assert np.all(unc(z) == unc(x) * np.exp(val(x)))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_sin(x):
    z = np.sin(x)
    assert np.all(val(z) == np.sin(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) * np.cos(val(x))))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_cos(x):
    z = np.cos(x)
    assert np.all(val(z) == np.cos(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) * np.sin(val(x))))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_tan(x):
    z = np.tan(x)
    assert np.all(val(z) == np.tan(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / np.cos(val(x))**2))


@pytest.mark.parametrize('x', [
    a / 10, b / 10, abs(A) / 1000, B / 10
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_arcsin(x):
    z = np.arcsin(x)
    assert np.all(val(z) == np.arcsin(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / np.sqrt(1 - val(x)**2)))


@pytest.mark.parametrize('x', [
    a / 10, b / 10, abs(A) / 1000, B / 10
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_arccos(x):
    z = np.arccos(x)
    assert np.all(val(z) == np.arccos(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / np.sqrt(1 - val(x)**2)))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_arctan(x):
    z = np.arctan(x)
    assert np.all(val(z) == np.arctan(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / (1 + val(x)**2)))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_sinh(x):
    z = np.sinh(x)
    assert np.all(val(z) == np.sinh(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) * np.cosh(val(x))))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_cosh(x):
    z = np.cosh(x)
    assert np.all(val(z) == np.cosh(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) * np.sinh(val(x))))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_tanh(x):
    z = np.tanh(x)
    assert np.all(val(z) == np.tanh(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / np.cosh(val(x))**2))


@pytest.mark.parametrize('x', [
    a / 10, b / 10, abs(A) / 1000, B / 10
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_arcsinh(x):
    z = np.arcsinh(x)
    assert np.all(val(z) == np.arcsinh(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / np.sqrt(1 + val(x)**2)))


@pytest.mark.parametrize('x', [
    a, abs(A), B
], ids=['Value', 'Array of values', 'Valued array'])
def test_arccosh(x):
    z = np.arccosh(x)
    assert np.all(val(z) == np.arccosh(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / np.sqrt(val(x)**2 - 1)))


@pytest.mark.parametrize('x', [
    a / 10, b / 10, abs(A) / 1000, B / 10
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_arctanh(x):
    z = np.arctanh(x)
    assert np.all(val(z) == np.arctanh(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / (1 - val(x)**2)))


@pytest.mark.parametrize('x', [
    a, abs(A), B
], ids=['Value', 'Array of values', 'Valued array'])
def test_sqrt(x):
    z = np.sqrt(x)
    assert np.all(val(z) == np.sqrt(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / (2 * np.sqrt(val(x)))))


@pytest.mark.parametrize('x', [
    a, b, abs(A), B
], ids=['Value', 'Negative', 'Array of values', 'Valued array'])
def test_cbrt(x):
    z = np.cbrt(x)
    assert np.all(val(z) == np.cbrt(val(x)))
    assert np.all(unc(z) == np.abs(unc(x) / (3 * np.cbrt(val(x))**2)))


@pytest.mark.parametrize('x, y, res', [
    (2, Value(3, 2), 2),
    (2, Value(1, 0.2), 1),
    (Value(2, 1), Value(1, 0.2), 1)],
    ids=['Number', 'Value', "Two values"])
def test_min(x, y, res):
    assert min(x, y) == res


@pytest.mark.parametrize('x, y, res', [
    (2, Value(3, 2), 3),
    (2, Value(1, 0.2), 2),
    (Value(2, 1), Value(1, 0.2), 2)],
    ids=['Number', 'Value', "Two values"])
def test_max(x, y, res):
    assert max(x, y) == res
