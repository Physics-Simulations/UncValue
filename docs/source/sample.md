Examples
--------

Imagine you measured a rectangular table whose sides measure `1.5m` and `80cm` with ruler with precision up to `1mm`. Then, you can initialize the `Value` as
* Python:
```python
from uncvalue import Value
L1 = Value(1.5, 1e-3)
L2 = Value(0.8, 1e-3)
```
* Julia:
```julia
using UncValue
L1 = Value(1.5, 1e-3)
L2 = Value(0.8, 1e-3)
```

You now want to calculate the area of the table, so you multiply both lengths
* Python:
```python
A = L1 * L2
print(A)
```
* Julia:
```julia
A = L1 * L2
println(A)
```
and obtain as outcome `(1200.0 ± 1.7)·10^-3`.

Of course, it is possible to perform more complex operations like
* Python (`numpy` required, functions from python `math` will only compute the value)
```python
import numpy as np
print(L1 ** L2)  # power -> (13831.6 ± 9.3)·10^-4
print(np.sin(L1))  # sinus -> (99749.5 ± 7.1)·10^-5
print(np.exp(L2))  # exponential -> (2225.5 ± 2.2)·10^-3
```
* Julia
```julia
println(L1^L2)  # power -> (13831.6 ± 9.3)·10^-4
println(sin(L1))  # sinus -> (99749.5 ± 7.1)·10^-5
println(exp(L2))  # exponential -> (2225.5 ± 2.2)·10^-3
```

You can also take a look at the [sample floder](https://github.com/Physics-Simulations/UncValue/tree/master/sample)
containing real use and more complex examples.
