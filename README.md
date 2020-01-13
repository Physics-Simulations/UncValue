# Uncertainty-Value-Calculator
Simple python class to evaluate the uncertainty for complex or very long calculations given the initial values together with its uncertainty.

# How-To
The way it works is simple, first import the script as
```python
from uncvalue import *
```
then initialise your `Value` variables (numbers, lists, matrices...) as
```python
pi = Value(3.14159, 0.00011) # number variable 3.14159 +/- 0.00011
A = np.array([pi, Value(2.718, 0.036), Value(1.61803398875, 29e-11)]) # numpy array with 3 elements
M = Value(np.random.rand(3,5), np.random.rand(3,5)*0.056) # 3x5 matrix
```

- `pi` is just a number variable with uncertainty
- `A` is a list of values, each one with each own uncertainty
- `M` is a 3x5 value matrix (not a matrix of values) where the uncertainty is separated from the value, so this class only works as a container for keeping them together but some operations will not work properly (like multiplication). To initialize the matrix of values correctly we should do it as the list (an example of this is inside `test.py`).

Perform any operation you want between Value(s):
- Binary operators: `+`, `-`, `*`, `/`, `**`
- Unary operators (all with numpy): `abs`, `exp`, `log`, `sqrt`, `sin(h)`, `cos(h)`, `tan(h)`, `arcsin(h)`, `arccos(h)`, `arctan(h)`
- Comparison: `>=`, `>`, `=`, `<`, `<=`

It's important that, for the unary operators, you use `numpy` as your base class for math. Operations made with the built-in `math` python library will result in terrible errors that for sure will end up destroying our and other universes.

For more examples visit take a look at [test.py](/test.py).

# ToDo
- Compare values with the uncertainty
- Add to pip

# License
      Copyright 2018 labay11

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.
