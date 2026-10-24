# 01 — The `.hsg` language

A `.hsg` file declares **what** to compute: inputs, trainable parameters,
datasets, helper functions, the model and the loss. It contains no commands
and no training settings. Those come from CLI flags (decision D11 in
`00-architecture.md`).

The language is small, first-order, statically shape-checked, and has no
recursion. Every program unrolls into a finite graph of scalar operations.

---

## 1. Examples

### 1.1 `intro.hsg`: micrograd's first example

```
-- Scalar expression; used to explore derivatives.
input a : Real = 2.0
input b : Real = -3.0
input c : Real = 10.0

let e = a * b
let d = e + c
let L = d * -2.0
```

```
hsgrad derive examples/intro.hsg --of L --wrt a,b,c --simplify --format text
∂L/∂a = -2·b  = 6
∂L/∂b = -2·a  = -4
∂L/∂c = -2    = -2
```

### 1.2 `neuron.hsg`: vectors, a function with shape variables

```
input x : Vec 3 = [1.0, -2.0, 0.5]
param w : Vec 3 ~ uniform(-1, 1)
param b : Real  = 0

fn neuron(w : Vec n, b : Real, x : Vec n) : Real =
  tanh(dot(w, x) + b)

let y = neuron(w, b, x)
```

`hsgrad derive examples/neuron.hsg --of y --wrt w --dot neuron-grad.dot` plots the
gradient graph. Each `∂y/∂w[i]` has the shape `(1 − tanh(…)²) · x[i]`, and
all three share the `1 − tanh(…)²` node.

### 1.3 `xor.hsg`: inline dataset, model and loss

```
data xor : { x : Vec 2, y : Real } = rows [
  ([0, 0], -1),
  ([0, 1],  1),
  ([1, 0],  1),
  ([1, 1], -1)
]

param W1 : Mat 4 2 ~ uniform(-1, 1)
param b1 : Vec 4   = 0
param w2 : Vec 4   ~ uniform(-1, 1)
param b2 : Real    = 0

model net(x : Vec 2) : Real =
  dot(w2, tanh(W1 @ x + b1)) + b2

loss = mean [ (net(x) - y) ^ 2 | (x, y) <- xor ]
```

### 1.4 `moons.hsg`: micrograd's full demo

```
data moons : { x : Vec 2, y : Real } =
  csv "data/moons.csv" { x = [x1, x2], y = label }

param W1 : Mat 16 2  ~ uniform(-1, 1)
param b1 : Vec 16    = 0
param W2 : Mat 16 16 ~ uniform(-1, 1)
param b2 : Vec 16    = 0
param W3 : Mat 1 16  ~ uniform(-1, 1)
param b3 : Vec 1     = 0

fn layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m = W @ x + b

model mlp(x : Vec 2) : Real =
  let h1 = relu(layer(W1, b1, x))  in
  let h2 = relu(layer(W2, b2, h1)) in
  layer(W3, b3, h2)[0]

-- SVM max-margin loss + L2 regularisation
loss =
  let data_loss = mean [ relu(1 - y * mlp(x)) | (x, y) <- moons ] in
  let reg       = 1e-4 * sum [ sum(p ^ 2) | p <- params ] in
  data_loss + reg
```

```
hsgrad train examples/moons.hsg --steps 100 --lr 1.0 --lr-schedule linear:0.1 --seed 1337
```

These four files, and three more (a linear regression, a calculus
playground, a broken program), are in `examples/`; `17-examples.md`
walks through all of them with real output.

---

## 2. Lexical structure

| Element | Form |
|---|---|
| Line comment | `-- …` to end of line |
| Block comment | `{- … -}`, nestable |
| Identifier | `[a-zA-Z_][a-zA-Z0-9_']*`, not a keyword or type name |
| Natural | `0`, `16` (used for dimensions and indices) |
| Number | `digits ( "." digits )? ( ("e" \| "E") ("+" \| "-")? digits )?`: `2`, `2.5`, `1e-4`, `3.0E2`. No leading `.` and no sign: `-` is an operator |
| String | `"…"` on one line, with escapes `\"` and `\\`. Only used for CSV paths |
| Keywords | `input param data fn let in model loss rows csv params` |
| Type names | `Real Vec Mat` (reserved like keywords) |
| Operators | `+ - * / ^ @ = : ~ <- \|` |
| Punctuation | `( ) [ ] { } ,` |

Whitespace and newlines aren't significant. Each top-level declaration
starts with a keyword, so declarations need no separators.

## 3. Declarations

Every name must be declared **before** it is used (D9). The only exception
is the built-in collection `params`, which always means *all* parameters in
the file.

### `input`: values supplied from outside

```
input a : Real = 2.0
input x : Vec 3              -- no default: must be given via --set x=[1,2,3]
```
The default must be a literal: a number, or a vector or matrix of numbers.
`--set` overrides the default.

### `param`: trainable parameters

```
param W : Mat 16 2 ~ uniform(-1, 1)    -- random: uniform(lo, hi) or normal(mean, std)
param b : Vec 16   = 0                 -- constant; a Real literal fills the whole shape
```
`~` means "sampled from a distribution" and `=` means "constant". The seed
comes from `--seed`.

### `data`: datasets

```
data xor : { x : Vec 2, y : Real } = rows [ ([0,0], -1), ([0,1], 1), … ]
data moons : { x : Vec 2, y : Real } = csv "data/moons.csv" { x = [x1, x2], y = label }
```
- The **schema** lists fields in order. Field types are `Real` or `Vec n`.
- `rows`: each row is a tuple with one literal per field. For a schema with
  one field, the bare literal is allowed.
- `csv`: the file must have a header row. The mapping tells which column
  feeds each field: a single column for `Real`, a bracketed list of
  exactly `n` columns for `Vec n`; every field is mapped once. Paths are
  relative to the `.hsg` file, so `examples/moons.hsg` reads
  `examples/data/moons.csv`.
- The CSV format: UTF-8 text, comma-separated, a header line of column
  names, then one line per row with one cell per column. Spaces around a
  cell don't count; a cell may be quoted (`"x1"`, with `""` for a quote
  inside). Blank lines may only end the file. Mapped columns must hold
  finite decimal numbers (`2`, `-0.5`, `.5`, `1e-4`); other columns may
  hold anything.
- A dataset is **not** a value. It can only appear as the source of a
  comprehension (§5.6).

### `fn`: functions

```
fn layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m = W @ x + b
fn sq(z) = z ^ 2                                  -- annotations are optional
```
- Parameters and the result may be annotated. Lowercase identifiers in
  dimension positions (`m`, `n`) are **shape variables**, scoped to that
  signature.
- Bodies may refer to earlier top-level declarations (params, inputs, lets,
  functions).
- Functions are first-order: they can be called, not passed around.

### `let`: named expressions

```
let L = d * -2.0
let h : Vec 4 = tanh(W1 @ x + b1)     -- optional annotation, checked
```

### `model`: the predictor

```
model mlp(x : Vec 2) : Real = …
```
A `model` is a function with **fully concrete** annotations (no shape
variables), because `hsgrad predict` has to know the input shape. There is
at most one per file. It can be called like any function.

### `loss`: the training objective

```
loss = mean [ … | (x, y) <- moons ] + …
```
At most one per file, and it must have type `Real`. `hsgrad train` requires
it.

## 4. Types and shapes

```
type ::= Real | Vec d | Mat r c        d, r, c ::= natural | shape-variable
```

- `Real`: a scalar.
- `Vec n`: a vector of `n` reals, `n ≥ 1`.
- `Mat r c`: an `r × c` matrix, stored row-major.

There are also internal types that can't be written in annotations:

- **Seq t**: the result of a comprehension. It can only be consumed by `sum`
  or `mean`.
- **Dataset schema**: only usable as a comprehension source.
- **function**: only usable by calling it.

### 4.1 Broadcasting

Elementwise binary operations (`+ - * /`, `max`, `min`) accept:

1. two operands of the **same** shape → the result has that shape;
2. a `Real` and any shape → the scalar is applied to every element.

Nothing else is broadcast (there's no `Vec n` against `Mat r n`).
`*` between two vectors or two matrices is **elementwise** (Hadamard). Use
`@` for matrix products and `dot` for inner products.

### 4.2 Shape variables and checking (D8)

At each call `f(a1, …, ak)`:

1. Check each argument's type against the parameter annotation, binding
   shape variables as they're met. `Vec n` against `Vec 2` binds `n = 2`.
   A later `Vec n` against `Vec 3` is error E0303, which points at both
   arguments.
2. Check the body with parameters bound to the concrete argument types.
3. Check the body's type against the result annotation, if there is one.

Errors in the body are reported at the body location, with a note
"while checking the call at file:line:col". Each function is checked once
per distinct list of argument shapes.

A function whose parameters all have concrete types is checked once, at
its declaration, like a `model`. A generic one is also checked at its
declaration, with the shapes that involve shape variables unknown: that
reports names, kinds, arities and every shape error that doesn't depend
on them, even if the function is never called.

## 5. Expressions

### 5.1 Precedence (highest first)

| Level | Form | Associativity |
|---|---|---|
| 1 | `f(…)`, `e[i]`, `e[i, j]`, literals, `( e )` | — |
| 2 | `e ^ k` (k a constant) | non-associative |
| 3 | unary `-e` | prefix |
| 4 | `a @ b` | left |
| 5 | `a * b`, `a / b` | left |
| 6 | `a + b`, `a - b` | left |
| 7 | `let x = e in body` | extends as far right as possible |

So `-x ^ 2` is `-(x²)`, `2 * W @ x` is `2 * (W @ x)`, and `d * -2.0` is fine.

### 5.2 Literals

- Numbers: `2`, `-3.5` (unary minus applied to a literal).
- Vectors: `[e1, …, en]`, where every `ei : Real`. The result is `Vec n`.
- Matrices: `[[…], …, […]]`, a vector of rows that all have the same length
  (E0312 otherwise). The result is `Mat r c`.

Vector elements may be arbitrary expressions, so `[a * b, c]` is valid.

### 5.3 Indexing

`v[i] : Real` for `v : Vec n`. `M[i] : Vec c` (a row) and `M[i, j] : Real`
for `M : Mat r c`. Indices are **natural-number literals**, 0-based, and
checked against the shape statically (E0305).

### 5.4 Powers

`e ^ k` is applied elementwise. `k` must be a numeric literal, optionally
negative: `x ^ 2`, `x ^ -1`, `x ^ 0.5`. Writing `x ^ y` with a variable
exponent is error E0105, which suggests `exp(y * log(x))`.

### 5.5 `let … in`

```
let h1 = relu(layer(W1, b1, x)) in body
```
`let … in` shares a value: `h1` becomes **one** node, however many times
`body` uses it. Local names may shadow outer ones, which produces warning
W0201.

### 5.6 Comprehensions

```
[ body | (x, y) <- moons ]      -- over a dataset; the pattern matches the schema arity
[ body | p <- params ]          -- over every parameter, in declaration order
```
- The type is `Seq t`, where `t` is the type of `body`. Every iteration must
  give the same `t` (E0310). For `params` the body is checked once per
  parameter, with that parameter's shape.
- A `Seq` can only be the argument of `sum` or `mean` (E0311). Such a call
  may drop its parentheses: `mean [ … ]` is `mean([ … ])`.
- There is one generator per comprehension, and comprehensions don't nest
  over datasets (E0314), not even through a function call.
  Comprehensions over `params` may nest.

## 6. Built-in functions

`s` means any shape. "Lowering" shows how the compiler expresses each
builtin in terms of core ops (see `03-core-op.md`).

| Function | Type | Lowering |
|---|---|---|
| `exp(e)`, `log(e)`, `tanh(e)`, `relu(e)` | `s → s` | core op, elementwise |
| `step(e)` | `s → s` | core op (Heaviside; the derivative of `relu`) |
| `sigmoid(e)` | `s → s` | `1 / (1 + exp(-e))` |
| `sqrt(e)` | `s → s` | `e ^ 0.5` |
| `abs(e)` | `s → s` | `relu(e) + relu(-e)` |
| `max(a, b)`, `min(a, b)` | broadcasting | `a + relu(b - a)`, `a - relu(a - b)` |
| `sum(e)` | `Real → Real`, `Vec n → Real`, `Mat r c → Real`, `Seq t → t` | chain of `+` |
| `mean(e)` | same as `sum` | `sum(e) / count` |
| `dot(u, v)` | `Vec n, Vec n → Real` | `Σ u[i]·v[i]` |
| `transpose(M)` | `Mat r c → Mat c r` | re-indexing only, no new nodes |
| `a @ b` | `Mat m n, Vec n → Vec m`; `Mat m n, Mat n k → Mat m k` | sums of products |

In `param` declarations, `uniform(lo, hi)` and `normal(mean, std)` are
initialisers, not expressions.

## 7. Static semantics, summarised

1. **Names**: there is one namespace. Declarations are visible from the
   point of declaration onward. Duplicate top-level names are an error,
   and so is a top-level name that repeats a built-in function. Locals
   (`let … in`, comprehension patterns) that shadow another name give a
   warning. Function parameters don't: a signature is its own scope, and
   `fn neuron(w, b, x)` called as `neuron(w, b, x)` is the normal style.
2. **Kinds of names**: values, functions and datasets. Using one as another
   is an error (E0204, E0313).
3. **Shapes**: §4 and §5. All checks are static, and nothing is checked at
   runtime except non-finite numbers.
4. **Declarations**: at most one `model` (fully concrete signature) and at
   most one `loss` (type `Real`).
5. **Unused**: an input, parameter, dataset, function or local that
   nothing uses gives warning W0202. A top-level `let` and the `model`
   never do: they are outputs (`eval`, `derive --of`, `predict`). Names
   starting with `_` are exempt, so `(_x, y) <- d` says a field is not
   needed. A parameter that can't affect `loss` gives warning W0203 (its
   gradient is structurally zero).
6. The checker reports **all** independent errors. After an error in a
   declaration, names that depend on it are marked "poisoned" and don't
   produce follow-up errors.

## 8. Lowering to the scalar graph (D5, D10)

Compile-time values are `Real NodeId`, `Vec [NodeId]` or `Mat [[NodeId]]`.

| Source | Graph |
|---|---|
| `input a : Real` | input leaf `a` |
| `input x : Vec 3` | input leaves `x[0] x[1] x[2]` |
| `param W : Mat 2 2` | param leaves `W[0,0] … W[1,1]` |
| `let x = e` / `let … in` | `e` compiled **once**; the name refers to the same node ids |
| function call | body inlined with parameters bound to argument node ids |
| `[ body \| (x, y) <- moons ]` | `B` copies of `body`, slot `i` bound to input leaves `moons#i.x[j]`, `moons#i.y` |
| `[ body \| p <- params ]` | one copy of `body` per parameter |
| elementwise op on `Vec n` | `n` scalar nodes |

`B` (slots per dataset) is the batch size: `--batch`, defaulting to the
dataset size. Every node records the span of the source expression that
created it (D4).

## 9. Grammar (EBNF)

```
program      ::= decl*

decl         ::= "input" ident ":" type ( "=" literal )?
               | "param" ident ":" type ( "=" literal | "~" ident "(" signed "," signed ")" )
               | "data"  ident ":" schema "=" source
               | "fn"    ident "(" binders? ")" ( ":" type )? "=" expr
               | "model" ident "(" binders? ")" ":" type "=" expr
               | "let"   ident ( ":" type )? "=" expr
               | "loss"  "=" expr

schema       ::= "{" ident ":" type ( "," ident ":" type )* "}"
source       ::= "rows" "[" row ( "," row )* "]"
               | "csv" string "{" colmap ( "," colmap )* "}"
row          ::= "(" literal ( "," literal )* ")" | literal
colmap       ::= ident "=" ( ident | "[" ident ( "," ident )* "]" )

binders      ::= binder ( "," binder )*
binder       ::= ident ( ":" type )?
type         ::= "Real" | "Vec" dim | "Mat" dim dim
dim          ::= natural | ident

expr         ::= "let" ident ( ":" type )? "=" expr "in" expr
               | additive
additive     ::= multiplicative ( ( "+" | "-" ) multiplicative )*
multiplicative ::= matmul ( ( "*" | "/" ) matmul )*
matmul       ::= unary ( "@" unary )*
unary        ::= "-" unary | power
power        ::= postfix ( "^" signed )?
postfix      ::= atom ( "[" natural ( "," natural )? "]" )*
atom         ::= number
               | ident ( "(" ( expr ( "," expr )* )? ")" | comprehension )?
               | "(" expr ")"
               | "[" expr ( "," expr )* "]"
               | comprehension
comprehension ::= "[" expr "|" pattern "<-" ( ident | "params" ) "]"
pattern      ::= ident | "(" ident ( "," ident )* ")"

literal      ::= signed | "[" literal ( "," literal )* "]"
signed       ::= "-"? number
```

`ident comprehension` is a call with the comprehension as its only
argument: `mean [ e | x <- d ]` means `mean([ e | x <- d ])`. A `[` right
after a name is therefore either an index or such a comprehension; it is an
index when a number follows it and then `]` or `,` (`v[0]`, `M[1, 2]`), and
a comprehension otherwise (`sum [ 0 | p <- params ]`).

Function application always uses parentheses (`f(x)`, never `f x`); the
one exception is a call with a single comprehension argument, which reads
better without them (`sum [ … ]`, as in the examples of §1). Together
with keyword-led declarations, this makes the grammar unambiguous without
layout rules or separators, which also helps produce good parse errors.

## 10. Diagnostics catalogue

Every diagnostic has a code, a primary span and, where useful, secondary
spans and notes. The format is shown in `00-architecture.md` §6.

### E01xx: syntax

| Code | Meaning |
|---|---|
| E0101 | Unexpected token (lists what was expected at that point) |
| E0102 | Unterminated block comment or string |
| E0103 | Malformed number literal |
| E0104 | `let` expression without `in` (hint: top-level `let` needs no `in`) |
| E0105 | Exponent must be a numeric constant (hint: `exp(y * log(x))`) |

### E02xx: names and declarations

| Code | Meaning |
|---|---|
| E0201 | Undefined name, with a "did you mean `…`?" suggestion by edit distance |
| E0202 | Duplicate declaration (points at both) |
| E0203 | Used before declaration: "declarations must come before use; recursion is not supported" |
| E0204 | Kind mismatch: calling a value, or using a function as a value |
| E0205 | Wrong number of arguments (shows the signature) |
| E0206 | More than one `model` / `loss` |
| E0207 | Comprehension pattern arity doesn't match the dataset schema |

### E03xx: shapes

| Code | Meaning |
|---|---|
| E0301 | Elementwise operands with incompatible shapes |
| E0302 | Matrix product with mismatched inner dimensions, or `@` on vectors (hint: `dot`) |
| E0303 | Shape variable bound inconsistently (shows both bindings) |
| E0304 | Argument doesn't match the parameter annotation |
| E0305 | Index out of bounds |
| E0306 | Indexing a `Real` |
| E0307 | `loss` isn't `Real` |
| E0308 | Body doesn't match the declared result / `let` annotation |
| E0309 | Type not allowed where it is written: a shape variable outside a `fn` (in a `model` signature, `input`, `param`, `data` or top-level `let`), an unannotated `model` parameter, a dimension 0, a `Mat` dataset field |
| E0310 | Comprehension body has different shapes across iterations, or no iteration at all (`params` in a file without `param`) |
| E0311 | A comprehension used outside `sum` / `mean` |
| E0312 | Malformed vector or matrix literal: rows of different lengths, numbers mixed with rows, a matrix inside |
| E0313 | Dataset used outside a comprehension source |
| E0314 | Comprehension over a dataset inside another one (directly, or through a function call) |

### E04xx: data

| Code | Meaning |
|---|---|
| E0401 | File not found or unreadable: a CSV file, or a file named on the command line (the `.hsg` file, `--params`) |
| E0402 | Column named in the mapping is missing from the header, or appears in it more than once |
| E0403 | Non-numeric cell, or a malformed CSV line (wrong number of cells, bad quoting, a blank line before the end), at file:row:column |
| E0404 | Inline row or CSV column mapping doesn't match the schema |
| E0405 | Empty dataset, or `--batch` larger than the dataset |

### E05xx: runtime

| Code | Meaning |
|---|---|
| E0501 | Non-finite value (NaN/∞), reported at the source span of the producing node |
| E0502 | Input has no value (no default, no `--set`) |
| E0503 | `--set` value has the wrong shape |
| E0504 | Non-finite gradient (NaN/∞) during backprop, reported at the node that passes it on |
| E0505 | `--params` file doesn't match the program: a parameter missing, unknown or of another shape, a random initialiser instead of values, or a declaration other than `param` |

### Warnings

| Code | Meaning |
|---|---|
| W0201 | Local name shadows an outer one |
| W0202 | Unused declaration |
| W0203 | Parameter can't affect `loss` |

## 11. Open points (to revisit when we get there)

The language-level ones; `00-architecture.md` §10 lists all open points
of the project.

- Generic checking of function bodies independent of call sites (D8 upgrade).
- `Mat` fields in dataset schemas.
- Several generators in one comprehension (for example `(x, y) <- d, p <- params`).
- CSV columns whose names are not identifiers (`x 1`, a keyword such as
  `data`) can't be mapped; a quoted form `x = ["x 1"]` would allow it.
- Integer-valued metrics such as accuracy for `--report-every`. For now:
  `sign(prediction) == label` when the model returns `Real`.
