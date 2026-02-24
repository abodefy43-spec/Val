# Val

Val is a query-based pipeline flow language. Data flows left-to-right through transformations using pipes (`|>`). It has declarative query expressions for data manipulation.

## Installation

- **Requirements**: Python 3.9 or later
- **Dependencies**: None for the core interpreter

## Quick Start

Run a Val script:

```bash
python val.py script.val
```

Start the interactive REPL (no arguments):

```bash
python val.py
```

## Syntax Reference

### Variables and Literals

```val
name = "Val"
age = 3
ready = yes
nums = 1, 2, 3, 4, 5
```

- **Numbers**: `42`, `3.14`
- **Strings**: `"hello"`
- **Booleans**: `yes`, `no`
- **Lists**: comma-separated values `1, 2, 3`
- **Range**: `1..10` yields `[1, 2, ..., 10]` (inclusive)

### Pipeline Operator

Data flows left-to-right. `a |> f |> g` evaluates as `g(f(a))`.

```val
5 |> double |> + 3 |> print

1, 2, 3, 4, 5
    |> filter > 2
    |> map * 2
    |> sum
    |> print
```

### Query Expressions

```val
from users where age >= 18 select name |> each |> print

find numbers in 1, 2, 3, 4, 5 where it > 2
```

- `from X where Y select Z` — filter and project
- `find item in X where Y` — filter with named item

### Functions

```val
add = (a, b) => a + b
double = x => x * 2
greet = (name, prefix = "Hello") => "{prefix}, {name}!"
```

- **Default parameters**: `(a, b = 0) => a + b`
- **Block bodies**: `fn = () => { x = 1; y = 2; x + y }`

### Loops

```val
for n in 1, 2, 3 do print(n)
for n in nums do { total = total + n }

while i < 5 do { count = count + 1; i = i + 1 }
```

### Control Flow

```val
result = check age
    when > 18 then "adult"
    when > 12 then "teen"
    default "child"

x = if ready then "go" else "stop"
```

### String Interpolation

```val
name = "Val"
print("Hello {name}!")
print("1 + 2 = {1 + 2}")
```

### Custom Types and OOP

```val
Point = { x: number, y: number }
Point.distance = (self, other) => (self.x - other.x) * (self.x - other.x) + (self.y - other.y) * (self.y - other.y)

p = Point(x: 5, y: 10)
d = p.distance(q)

-- Inheritance
Circle extends Point { r: number }
c = Circle(x: 0, y: 0, r: 5)
```

### Imports

```val
import "io"
import "math"
import "io" as io
```

### Operators

- **Membership**: `x in list`, `"a" in "Val"`
- **Comments**: `-- line comment`

### Python / ML Interop

Val can call Python libraries (NumPy, PyTorch, sklearn) via the `python:` import prefix:

```val
import "python:numpy" as np
import "python:torch"
import "python:torch.nn" as nn

arr = array(1, 2, 3, 4, 5)
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
```

- **`import "python:module"`** — loads a Python module
- **`py_import(name)`** — dynamic import, returns PyRef
- **`py_call(obj, method, *args)`** — call Python method
- **`array(data)`** — create NumPy array (requires numpy)

**ML dependencies**: `pip install -r requirements.txt` (numpy, torch, scikit-learn)

## File Layout

```
val.py           # Entry point (CLI + REPL)
lexer.py         # Tokenizer
parser.py        # Parser + AST
interpreter.py   # Evaluator
std/             # Standard library
  io.val
  math.val
  string.val
examples/        # Example programs
  hello.val
  query.val
  types.val
```

## Running Examples

```bash
python val.py examples/hello.val
python val.py examples/query.val
python val.py examples/types.val
```

## Standard Library

- **io** — I/O helpers
- **math** — arithmetic helpers (double, square, add, sub, mul, div)
- **string** — string utilities

## Built-in Functions

`print`, `input`, `len`, `type`, `split`, `join`, `read`, `write`, `range`, `each`, `map`, `filter`, `reduce`, `sort`, `reverse`, `keys`, `values`, `sum`, `not_null`
