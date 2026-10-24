# hsgrad

A Haskell re-implementation of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) — scalar reverse-mode
automatic differentiation and a tiny neural-network trainer — extended
with:

- a small declarative language, `.hsg` files, that describes inputs,
  parameters, datasets, the model and the loss;
- symbolic gradients: derivatives as graphs, simplified and printed as
  formulas or drawn with Graphviz;
- error messages that point into the source, for everything from a
  typo to a `NaN` at training step 50.

It is a learning project: every module has a page in [`docs/`](docs)
with the theory, the design decisions and a code walkthrough.
[`docs/README.md`](docs/README.md) lists them in reading order; start
with [`docs/00-architecture.md`](docs/00-architecture.md), the language
spec [`docs/01-language.md`](docs/01-language.md), or the guided tour
of the examples, [`docs/17-examples.md`](docs/17-examples.md).

All sixteen steps of the roadmap are done. Open points and ideas for
future work are collected in
[`docs/00-architecture.md`](docs/00-architecture.md) §10.

## Contents

1. [Building](#building)
2. [A quick tour](#a-quick-tour)
3. [Writing a `.hsg` file: a tutorial](#writing-a-hsg-file-a-tutorial)
4. [The `.hsg` language at a glance](#the-hsg-language-at-a-glance)
5. [Command-line reference](#command-line-reference)
6. [Recipes](#recipes)
7. [Error codes](#error-codes)
8. [The examples](#the-examples)
9. [License](#license)

## Building

You need GHC 9.10.3 and cabal 3.16 (for example via
[ghcup](https://www.haskell.org/ghcup/)). Graphviz (`dot`) is optional,
for pictures.

```bash
cabal build all
```

```bash
cabal test
```

The tests take about 8 seconds. They include end-to-end tests that run
the built `hsgrad` on every example and check that every command shown
in [`docs/17-examples.md`](docs/17-examples.md) prints exactly what the
page shows. micrograd's full moons run (about 10 seconds) is one of
them, skipped by default; run it too with

```bash
HSGRAD_SLOW=1 cabal test
```

Run the program through cabal:

```bash
cabal run hsgrad -- --help
```

or install it, which puts `hsgrad` on your `PATH`:

```bash
cabal install exe:hsgrad
```

The examples below are written as `hsgrad …`.

## A quick tour

micrograd's first example, [`examples/intro.hsg`](examples/intro.hsg):

```
input a : Real = 2.0
input b : Real = -3.0
input c : Real = 10.0

let e = a * b
let d = e + c
let L = d * -2.0
```

Check it, evaluate it, differentiate it:

```
$ hsgrad check examples/intro.hsg
…
examples/intro.hsg: ok, 7 nodes

$ hsgrad eval examples/intro.hsg --set a=3
e = -9
d = 1
L = -2

$ hsgrad derive examples/intro.hsg --of L --wrt a,b,c --simplify
∂L/∂a = -2·b  = 6
∂L/∂b = -2·a  = -4
∂L/∂c = -2    = -2
```

Draw a graph with its values and gradients (micrograd's `draw_dot`), or
the graph of a gradient:

```
$ hsgrad plot examples/neuron.hsg --of y --dot neuron.dot
$ dot -Tsvg neuron.dot -o neuron.svg
$ hsgrad derive examples/neuron.hsg --of y --wrt w --simplify --dot neuron-grad.dot
```

Train the XOR net of [`examples/xor.hsg`](examples/xor.hsg), save the
parameters, and predict with them:

```
$ hsgrad train examples/xor.hsg --steps 200 --seed 1 --report-every 50 --save-params xor.params
train examples/xor.hsg: 200 steps, lr 0.1 (constant), seed 1, xor: 4 of 4 rows per step
step   0  loss 1.22558      accuracy 50%
step  50  loss 0.543008     accuracy 100%
step 100  loss 0.0432655    accuracy 100%
step 150  loss 0.000801179  accuracy 100%
step 199  loss 1.22675e-5   accuracy 100%
wrote xor.params

$ hsgrad predict examples/xor.hsg --params xor.params --set "x=[0, 1]"
net([0, 1]) = 0.9963848175526087
```

micrograd's full demo, a 2-16-16-1 MLP on the two moons
([`examples/moons.hsg`](examples/moons.hsg)):

```
$ hsgrad train examples/moons.hsg --steps 100 --lr 1.0 --lr-schedule linear:0.1
…
step 99  loss 0.0202695    accuracy 100%    lr 0.109
```

Mistakes are reported where they are, in the file or on the command
line:

```
$ hsgrad eval examples/neuron.hsg --set x=[1,2]
error[E0503]: the value for input `x` has the wrong shape
 --> <command line>:1:41
  |
1 | hsgrad eval examples/neuron.hsg --set x=[1,2]
  |                                         ^^^^^ a `Vec 2`
  |
 ::: examples/neuron.hsg:1:7
  |
1 | input x : Vec 3 = [1.0, -2.0, 0.5]
  |       - declared as a `Vec 3` here
  |
  = hint: write a `Vec 3`, like `x=[1, 2, 3]`

1 error
```

## Writing a `.hsg` file: a tutorial

A `.hsg` file says **what** to compute: the data, the trainable
parameters, the model and the loss. It contains no training settings;
those are command-line options (learning rate, steps, seed, …). This
section builds a small classifier from scratch, one declaration at a
time.

### 1. The data

We want to learn which side of a line a point lies on. Six labelled
points go straight into the file as an inline dataset:

```
-- Logistic regression: which side of a line is a point on?
-- Labels are -1 and +1, so `train` can report sign accuracy.

data pts : { x : Vec 2, y : Real } = rows [
  ([ 2.0,  1.0],  1),
  ([ 1.0,  3.0],  1),
  ([ 3.0,  2.5],  1),
  ([-1.0, -2.0], -1),
  ([-2.0,  0.5], -1),
  ([ 0.5, -3.0], -1)
]
```

- `{ x : Vec 2, y : Real }` is the **schema**: the fields of a row, in
  order. Fields are `Real` or `Vec n`.
- Each row is a tuple with one literal per field.
- For more data, read a CSV file instead:
  `csv "data/points.csv" { x = [x1, x2], y = label }` maps the CSV columns
  `x1`, `x2` to the vector `x` and the column `label` to `y`. The path is
  relative to the `.hsg` file.

### 2. The parameters

```
param w : Vec 2 ~ normal(0, 0.1)
param b : Real  = 0
```

- `~` samples the initial value from a distribution: `normal(mean, std)`
  or `uniform(lo, hi)`. The seed comes from `--seed` (default 1337), so
  runs are reproducible.
- `=` sets a constant; a single number fills a whole vector or matrix
  (`param b1 : Vec 16 = 0`).

### 3. The model

```
-- The model: a score whose sign is the predicted class.
model score(x : Vec 2) : Real = dot(w, x) + b
```

A `model` is a function with **fully concrete** types: `predict` needs
to know the shape of its argument. Its body can use any parameter,
input or earlier declaration. There is at most one model per file.

### 4. The loss

```
-- Logistic loss log(1 + e^(-y·f)), plus a little L2 regularisation.
loss =
  let data_loss = mean [ log(1 + exp(-y * score(x))) | (x, y) <- pts ] in
  let reg       = 0.001 * sum [ sum(p ^ 2) | p <- params ] in
  data_loss + reg
```

- `[ body | (x, y) <- pts ]` is a **comprehension**: the body once per
  row, with the row's fields bound to `x` and `y`. It can only be passed
  to `sum` or `mean`, which may drop their parentheses.
- `p <- params` runs over every parameter, whatever its shape, so the
  regularisation term doesn't need to change when you add a layer.
- `let … in` names a value inside an expression; it is computed once,
  however often it is used.
- The loss must be a `Real`. `train` minimises it.

### 5. Check, train, predict

`check` parses, type-checks, loads the data and compiles, and prints
what it found:

```
$ hsgrad check classify.hsg
data pts : { x : Vec 2, y : Real }
param w : Vec 2
param b : Real
model score : Vec 2 → Real
loss : Real
classify.hsg: ok, 91 nodes
```

`train` runs gradient descent and can save the result:

```
$ hsgrad train classify.hsg --steps 50 --lr 0.5 --save-params classify.params
train classify.hsg: 50 steps, lr 0.5 (constant), seed 1337, pts: 6 of 6 rows per step
step  0  loss 0.719409     accuracy 50%
step 10  loss 0.06179      accuracy 100%
step 20  loss 0.0361643    accuracy 100%
step 30  loss 0.0270307    accuracy 100%
step 40  loss 0.0224855    accuracy 100%
step 49  loss 0.0200445    accuracy 100%
wrote classify.params
```

The accuracy column appears because the loss calls the `model` on each
row and the dataset's last field is a `Real`: a prediction counts as
correct when its sign matches the label's. Use labels −1 and +1 for
classification.

The parameter file is itself a small `.hsg` file, readable and
editable:

```
-- Parameters of classify.hsg, saved by `hsgrad train`.
-- train classify.hsg: 50 steps, lr 0.5 (constant), seed 1337, pts: 6 of 6 rows per step
-- last step 49 loss 0.0200445 accuracy 100%
param w : Vec 2 = [1.6619692740190497, 1.4357630177279321]
param b : Real  = -0.4747457690617862
```

`predict` evaluates the model with those parameters:

```
$ hsgrad predict classify.hsg --params classify.params --set "x=[1, 1]"
score([1, 1]) = 2.6229865226851956
$ hsgrad predict classify.hsg --params classify.params --set "x=[-1, -1]"
score([-1, -1]) = -3.572478060808768
```

### 6. Growing it

Typical next steps, all shown in [`examples/`](examples):

- **Hidden layers**: a helper function with shape variables, reused for
  every layer ([`moons.hsg`](examples/moons.hsg)):

  ```
  fn layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m = W @ x + b

  model mlp(x : Vec 2) : Real =
    let h1 = relu(layer(W1, b1, x))  in
    let h2 = relu(layer(W2, b2, h1)) in
    layer(W3, b3, h2)[0]
  ```

- **Values to explore** rather than train: `input` declarations with
  defaults and top-level `let`s, then `eval` and `derive`
  ([`intro.hsg`](examples/intro.hsg), [`calculus.hsg`](examples/calculus.hsg)).
- **Regression**: a `Real` target and mean squared error
  ([`linreg.hsg`](examples/linreg.hsg)).

## The `.hsg` language at a glance

The full specification, with the grammar, is
[`docs/01-language.md`](docs/01-language.md). This is the part you need
day to day.

### Declarations

Each declaration starts with a keyword, so they need no separators.
Every name must be declared **before** it is used, which also means
there is no recursion.

| Declaration | Form | Notes |
|---|---|---|
| `input` | `input x : Vec 3 = [1, 0, -1]` | A value from outside. The default is optional; `--set x=…` overrides it. |
| `param` | `param W : Mat 4 2 ~ uniform(-1, 1)` | Trainable. `~ uniform(lo, hi)`, `~ normal(mean, std)`, or `= literal`. |
| `data` | `data d : { x : Vec 2, y : Real } = rows [ … ]` | A dataset, inline (`rows`) or from a CSV file (`csv`). |
| `fn` | `fn sq(z) = z ^ 2` | A helper function. Types are optional and may use shape variables. |
| `let` | `let h : Vec 4 = tanh(W1 @ x + b1)` | A named value; the type annotation is optional and checked. |
| `model` | `model net(x : Vec 2) : Real = …` | The predictor, for `predict`. Concrete types only; at most one. |
| `loss` | `loss = mean [ … \| (x, y) <- d ]` | The training objective. Must be `Real`; at most one. |

### Types and shapes

| Type | Meaning |
|---|---|
| `Real` | a number |
| `Vec n` | a vector of `n` numbers |
| `Mat r c` | an `r × c` matrix, row-major |

Every shape is checked before anything runs. In `fn` signatures,
lowercase names in a dimension are **shape variables**:
`fn layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m` works for every
`m` and `n`, and a call where the shapes don't fit is reported with the
binding that failed (`n = 3` from `W`, `n = 2` from `x`).

### Expressions

From the highest precedence to the lowest:

| Form | Meaning |
|---|---|
| `f(a, b)`, `v[i]`, `M[i, j]`, `M[i]`, `(e)` | call, element, row; indices are number literals from 0 |
| `e ^ k` | power, elementwise; `k` must be a number (`x ^ 2`, `x ^ -1`, `x ^ 0.5`) |
| `-e` | negation (`-x ^ 2` is `-(x²)`) |
| `A @ B` | matrix × vector or matrix × matrix |
| `a * b`, `a / b` | elementwise |
| `a + b`, `a - b` | elementwise |
| `let x = e in body` | a local name; extends as far right as possible |

Literals: numbers (`2`, `-3.5`, `1e-4`), vectors (`[a, b * 2, c]`) and
matrices (`[[1, 2], [3, 4]]`). Comprehensions: `[ body | (x, y) <- d ]`
over a dataset, `[ body | p <- params ]` over the parameters.

**Broadcasting**: elementwise operations take two values of the same
shape, or a `Real` and any shape. Nothing else is broadcast.

**Three kinds of product**, easy to mix up:

| You want | Write | Shapes |
|---|---|---|
| matrix product | `W @ x` | `Mat m n`, `Vec n` → `Vec m` |
| inner product | `dot(u, v)` | `Vec n`, `Vec n` → `Real` |
| elementwise product | `u * v` | same shapes → same shape |

### Built-in functions

| Function | Type | Notes |
|---|---|---|
| `exp`, `log`, `tanh`, `relu`, `step` | any shape → same | elementwise; `step` is the derivative of `relu` |
| `sigmoid`, `sqrt`, `abs` | any shape → same | built from the above: `1/(1+exp(-e))`, `e ^ 0.5`, `relu(e) + relu(-e)` |
| `max(a, b)`, `min(a, b)` | broadcasting | elementwise |
| `sum(e)`, `mean(e)` | `Vec`/`Mat` → `Real`; a comprehension → its body's type | |
| `dot(u, v)` | `Vec n`, `Vec n` → `Real` | |
| `transpose(M)` | `Mat r c` → `Mat c r` | |

### Comments and names

- `-- to the end of the line` and `{- nestable block comments -}`.
- Names: letters, digits, `_` and `'`, starting with a letter or `_`.
  Keywords (`input param data fn let in model loss rows csv params`) and
  type names (`Real Vec Mat`) are reserved, and so are built-in function
  names.
- An unused input, parameter, dataset, function or local gives a
  warning; names starting with `_` never do, so `(_x, y) <- d` says a
  field is not needed. Top-level `let`s and the model are outputs and
  never count as unused.

### CSV files

A header line of column names, then one line per row, comma-separated.
Spaces around a cell are ignored; a cell may be quoted (`"x1"`, with `""`
for a quote inside). Mapped columns must hold finite decimal numbers;
other columns may hold anything and are ignored. Blank lines may only
end the file.

```
x1,x2,label
0.909158,-0.135722,-1
0.082757,0.700558,1
```

### Common mistakes

| You wrote | Problem | Write instead |
|---|---|---|
| `x ^ n` with `n` a value | exponents must be numbers (E0105) | `exp(n * log(x))` |
| `u @ v` for two vectors | `@` needs a matrix on the left (E0302) | `dot(u, v)` |
| `f x` | calls need parentheses (E0101) | `f(x)` |
| `let y = x * 2` inside an expression | an inner `let` needs `in` (E0104) | `let y = x * 2 in …` |
| `model m(x : Vec n) : Real` | the model's types must be concrete (E0309) | `model m(x : Vec 2) : Real` |
| `loss = W @ x` | the loss must be a `Real` (E0307) | reduce it: `sum(…)`, `dot(…)`, `[i]` |
| `sum(d)`, `d` a dataset | datasets only appear in comprehensions (E0313) | `sum [ y \| (x, y) <- d ]` |
| using a name above its declaration | define before use (E0203) | move the declaration up |

## Command-line reference

```
hsgrad COMMAND FILE [OPTIONS]
```

| Command | Does |
|---|---|
| `check` | Parse, check, load the data and compile; print the declarations and the graph size; report every problem. |
| `eval` | Evaluate values: every top-level `let` and the loss, or the `--show` names. |
| `derive` | Differentiate a value symbolically, as formulas with values or as a Graphviz graph. |
| `plot` | Draw the graph of a value as Graphviz DOT, with values and gradients. |
| `train` | Minimise the `loss` with gradient descent; optionally save the parameters. |
| `predict` | Evaluate the `model` on given arguments, with saved parameters. |

`hsgrad --help` lists the commands, `hsgrad COMMAND --help` shows every
option of a command with examples, and `hsgrad --version` prints the
version.

### Options every command takes

| Option | Meaning |
|---|---|
| `FILE` | the `.hsg` file (always the first argument after the command) |
| `--color auto\|always\|never` | colour in error messages; `auto` (the default) colours when stderr is a terminal and the `NO_COLOR` environment variable is not set |
| `-h`, `--help` | the command's help |

### Where values come from

`eval`, `derive` and `plot` evaluate the graph; `train` and `predict`
run it too. They give the graph's leaves their values like this:

| Leaves | Value |
|---|---|
| inputs | `--set NAME=VALUE`, else the declared default; an input that is needed and has neither is error E0502 |
| parameters | the `--params` file, else the initialisers, sampled with `--seed` (default 1337) |
| dataset rows | every row (`eval`, `derive`, `plot`); a random batch of `--batch` rows per step (`train`) |
| model arguments (`predict`) | `--set NAME=VALUE` |

Options for that, shared by the commands that use them:

| Option | Commands | Meaning |
|---|---|---|
| `--set NAME=VALUE` | `eval`, `derive`, `plot`, `train`, `predict` | Give an input (or, for `predict`, a model argument) a value. Repeatable. The value is a literal of the language: `3`, `[1, 2]`, `[[1, 2], [3, 4]]`; its shape must match the declaration (E0503). |
| `--params PARAMS` | `eval`, `derive`, `plot`, `predict` (required) | Read the parameters from a file saved by `train --save-params`. |
| `--seed N` | `eval`, `derive`, `plot`, `train` | Seed for the random initial parameters (and, in `train`, the batches). Default 1337. |

Quote values with spaces or brackets for your shell: `--set "x=[1, 0, -1]"`.

### Naming values

`--show`, `--of` and `--wrt` take a name: an input, a parameter, a
top-level `let`, or `loss`.

- A vector or matrix name stands for all its elements: `--wrt W1` means
  `W1[0,0]`, `W1[0,1]`, … in row-major order.
- One element can be named on its own: `W1[1,0]`, `x[2]`. Quote it for
  the shell: `--wrt "W1[1,0]"`.
- `--wrt` takes a comma-separated list, split only at commas outside
  brackets: `--wrt "W1[1,0],b2"` is two names.
- A misspelt name is reported with a suggestion ("did you mean `b1`?").

### Output, files and exit codes

- Results go to **stdout**, diagnostics to **stderr**. `wrote FILE`
  messages go to stderr too.
- An output file of `-` means stdout, so DOT can be piped into Graphviz:
  `hsgrad plot f.hsg --of y --dot - | dot -Tsvg > y.svg`.
- Exit code **0**: the command worked (warnings or not). **1**: there
  were errors, or the command line was wrong. **2**: an internal error
  (a bug in hsgrad).

### `check`

```
hsgrad check FILE [--color auto|always|never]
```

Everything short of running: parse, check names and shapes, load the
datasets, compile. It prints each declaration with its type and the
number of nodes in the graph, and reports every problem it finds, not
just the first.

```
$ hsgrad check examples/xor.hsg
data xor : { x : Vec 2, y : Real }
param W1 : Mat 4 2
param b1 : Vec 4
param w2 : Vec 4
param b2 : Real
model net : Vec 2 → Real
loss : Real
examples/xor.hsg: ok, 154 nodes
```

[`examples/errors.hsg`](examples/errors.hsg) is broken on purpose; `hsgrad
check examples/errors.hsg` shows four errors and a warning in one run
([`docs/17-examples.md`](docs/17-examples.md) §7).

### `eval`

```
hsgrad eval FILE [--set NAME=VALUE ...] [--params PARAMS] [--seed N] [--show NAME ...]
```

| Option | Default | Meaning |
|---|---|---|
| `--show NAME` | every top-level `let`, then the loss | Show this value: an input, parameter, `let`, `loss`, or an element like `W1[0,1]`. Repeatable; values print in the order given. |

Vectors and matrices print as literals, numbers as the shortest decimal
that reads back exactly.

```
$ hsgrad eval examples/intro.hsg
e = -6
d = 4
L = -8

$ hsgrad eval examples/xor.hsg --show "W1[0,1]" --show b2 --show loss
W1[0,1] = 0.18280262295967586
b2      = 0
loss    = 1.2026332715553636
```

The second run uses the initial parameters (seed 1337) and the whole
dataset; add `--params xor.params` to evaluate trained ones.

### `derive`

```
hsgrad derive FILE --of NAME --wrt NAME[,NAME...] [--simplify] [--dot OUT] [--format dot|text]
                   [--set NAME=VALUE ...] [--params PARAMS] [--seed N]
```

| Option | Default | Meaning |
|---|---|---|
| `--of NAME` | required | The value to differentiate: `loss`, a `let`, … A vector gives one block of gradients per element. |
| `--wrt NAMES` | required | Differentiate with respect to these: inputs, parameters, or intermediate `let`s. Vectors and matrices expand to their elements. |
| `--simplify` | off | Simplify the formulas: gather like terms, merge powers, fold constants. |
| `--format text\|dot` | `text` | Print formulas with their values, or print the gradient graph as DOT. |
| `--dot OUT` | — | Also write the gradient graph as DOT to `OUT` (`-` for stdout). |

Each line is `∂out/∂x = formula  = value`. The derivatives are computed
symbolically, as graphs, so you see the formula and not only the
number:

```
$ hsgrad derive examples/calculus.hsg --of cube --wrt x
∂cube/∂x = 3·x²  = 6.75

$ hsgrad derive examples/calculus.hsg --of p --wrt x
∂p/∂x = -x + (-x) + (x + 1 + (x + 1))  = 2

$ hsgrad derive examples/calculus.hsg --of p --wrt x --simplify
∂p/∂x = 2  = 2
```

`--wrt` also accepts intermediate values: the derivative with respect
to a `let` is how much the output changes when that value does.

```
$ hsgrad derive examples/intro.hsg --of L --wrt d,e
∂L/∂d = -2  = -2
∂L/∂e = -2  = -2
```

Subexpressions used more than once are named `t1`, `t2`, … and listed
under `where`:

```
$ hsgrad derive examples/neuron.hsg --of y --wrt w,b --simplify
∂y/∂w[0] = x[0]·t1  = 0.6952308208745206
∂y/∂w[1] = x[1]·t1  = -1.3904616417490412
∂y/∂w[2] = x[2]·t1  = 0.3476154104372603
∂y/∂b    = t1       = 0.6952308208745206
where
  t1 = 1 - tanh(x[0]·w[0] + x[1]·w[1] + x[2]·w[2] + b)²
```

Values come from the unsimplified gradient graph, so they are exactly
backprop's numbers. If an input the formulas need has no value, the
formulas are still printed, without values.

Derivatives of a loss over a dataset mention the dataset's rows by
name, `pts#0.x[1]` being field `x[1]` of row 0; for a big dataset that
is a long formula, and `--dot` with Graphviz is easier to read.

### `plot`

```
hsgrad plot FILE --of NAME --dot OUT [--cluster] [--set NAME=VALUE ...] [--params PARAMS] [--seed N]
```

| Option | Default | Meaning |
|---|---|---|
| `--of NAME` | required | The value whose graph to draw. |
| `--dot OUT` | required | Write the graph as Graphviz DOT to `OUT` (`-` for stdout). |
| `--cluster` | off | Group the nodes of each source expression in a labelled box: useful for vector code, where one line becomes many scalar nodes. |

Every node shows its value and, for a `Real` output, its gradient,
like micrograd's `draw_dot`:

```
$ hsgrad plot examples/intro.hsg --of L --dot - | head -5
digraph hsgrad {
  rankdir=LR;
  n0 [shape=record, label="{ a | data 2.0000 | grad 6.0000 }"];
  n1 [shape=record, label="{ b | data -3.0000 | grad -4.0000 }"];
  n2 [shape=record, label="{ c | data 10.0000 | grad -2.0000 }"];

$ hsgrad plot examples/xor.hsg --of loss --dot - --cluster | dot -Tsvg > xor.svg
```

### `train`

```
hsgrad train FILE [--steps N] [--lr X] [--lr-schedule SCHEDULE] [--batch N] [--seed N]
                  [--report-every N] [--save-params OUT] [--set NAME=VALUE ...]
```

| Option | Default | Meaning |
|---|---|---|
| `--steps N` | 100 | Number of gradient-descent steps. |
| `--lr X` | 0.1 | Learning rate (the base rate of the schedule). |
| `--lr-schedule S` | `const` | How the rate changes over the run; see below. |
| `--batch N` | every row | Rows per step, sampled at random without replacement. Error E0405 if larger than the dataset. |
| `--seed N` | 1337 | Seed for the initial parameters and the batches. |
| `--report-every N` | 10 | Print a report every `N` steps, and after the last one. |
| `--save-params OUT` | — | Save the trained parameters to `OUT`. |

The file needs a `loss` declaration. Each step evaluates the loss on a
batch, computes the gradient of every parameter, and moves each
parameter against its gradient: `p ← p − lr · ∂loss/∂p`.

**Learning-rate schedules**, for step `t` of `N` steps and base rate
`lr`:

| Schedule | Rate at step `t` | Example |
|---|---|---|
| `const` | `lr` | `--lr 0.1` |
| `linear:END` | `lr + (END − lr) · t / N`, from `lr` down (or up) to `END` | `--lr 1 --lr-schedule linear:0.1` (micrograd's) |
| `step:K:F` | `lr · F^⌊t / K⌋`: multiplied by `F` every `K` steps | `--lr 0.5 --lr-schedule step:50:0.5` |

**The report line** shows the step, the loss on that step's batch
(before the update), the sign accuracy when there is one, and the rate
when the schedule changes it. Accuracy is reported when the loss calls
the `model` on each row of a dataset whose last field is a `Real`; a
prediction is correct when its sign matches the label's.

```
$ hsgrad train examples/xor.hsg --steps 200 --lr 0.5 --lr-schedule step:50:0.5 --seed 1 --report-every 50
train examples/xor.hsg: 200 steps, lr 0.5 (times 0.5 every 50 steps), seed 1, xor: 4 of 4 rows per step
step   0  loss 1.22558      accuracy 50%     lr 0.5
step  50  loss 0.437134     accuracy 75%     lr 0.25
step 100  loss 0.000351957  accuracy 100%    lr 0.125
step 150  loss 5.49701e-6   accuracy 100%    lr 0.0625
step 199  loss 7.2473e-7    accuracy 100%    lr 0.0625
```

With `--batch`, every step sees a different random sample, so the loss
and accuracy jump around:

```
$ hsgrad train examples/moons.hsg --steps 30 --lr 0.5 --batch 16 --seed 3 --report-every 10
train examples/moons.hsg: 30 steps, lr 0.5 (constant), seed 3, moons: 16 of 100 rows per step
step  0  loss 1.94513      accuracy 12.5%
step 10  loss 0.142786     accuracy 100%
step 20  loss 0.299826     accuracy 87.5%
step 29  loss 0.337273     accuracy 81.2%
```

If a value or gradient stops being finite — usually because the
learning rate is too large — training stops at that step, and the error
points at the expression that overflowed, with a hint to try a smaller
`--lr`.

### `predict`

```
hsgrad predict FILE --params PARAMS [--set NAME=VALUE ...]
```

| Option | Default | Meaning |
|---|---|---|
| `--params PARAMS` | required | The parameters, saved by `train --save-params`. |
| `--set NAME=VALUE` | — | A model argument (or an input the model uses). Repeatable. |

The file needs a `model` declaration. The output shows the call:

```
$ hsgrad predict examples/xor.hsg --params xor.params --set "x=[1, 1]"
net([1, 1]) = -0.9954410498241719
```

If a model argument has the same name as an `input`, `--set` sets the
argument.

### Parameter files

`train --save-params OUT` writes, and `--params FILE` reads, a small
`.hsg` file of `param` declarations with explicit values. Numbers are
written as the shortest decimal that reads back as exactly the same
number, so saving and loading loses nothing.

```
-- Parameters of examples/xor.hsg, saved by `hsgrad train`.
-- train examples/xor.hsg: 200 steps, lr 0.1 (constant), seed 1, xor: 4 of 4 rows per step
-- last step 199 loss 1.22675e-5 accuracy 100%
param W1 : Mat 4 2 = [
  [1.565136577214541, -0.7738212582684185],
  [1.3753745646130369, 1.5741263538315122],
  [-0.19142568511824803, -0.4067464023639376],
  [1.2994860916889452, -1.8474781008125076]
]
param b1 : Vec 4   = [0.22869202081909057, -0.05370492873825431, 0.16877987956341362, -0.682371006080259]
param w2 : Vec 4   = [-1.7449438930150662, 1.7918077820789862, 0.9325426444406195, 1.5596239477396878]
param b2 : Real    = 0.2588148707269265
```

- You can write or edit one by hand: comments, any layout, and `= 0` to
  fill a whole shape are all allowed.
- It must match the program exactly: every parameter, at its declared
  shape, and nothing else. Random initialisers (`~`) and other
  declarations are not allowed. Mismatches are error E0505, with a
  "did you mean" for misspelt names.
- The declarations can be pasted into the program itself, replacing the
  `~` initialisers.

## Recipes

**Explore a derivative by hand.** Declare `input`s with defaults and the
expression as a `let`, then:

```bash
hsgrad derive f.hsg --of y --wrt x --simplify --set x=0.5
```

**Second derivatives.** `derive` differentiates once. Write the first
derivative as a `let` and differentiate that, as in
[`examples/calculus.hsg`](examples/calculus.hsg):

```bash
hsgrad derive examples/calculus.hsg --of dcube --wrt x --simplify
```

**See the computation.** Draw the forward graph with values and
gradients, or the graph of the gradient itself:

```bash
hsgrad plot examples/neuron.hsg --of y --dot - | dot -Tsvg > neuron.svg
```

```bash
hsgrad derive examples/neuron.hsg --of y --wrt w --simplify --format dot | dot -Tsvg > neuron-grad.svg
```

**Train, inspect, predict.**

```bash
hsgrad train examples/linreg.hsg --save-params linreg.params
```

```bash
hsgrad eval examples/linreg.hsg --params linreg.params --show w --show b
```

```bash
hsgrad predict examples/linreg.hsg --params linreg.params --set x=10
```

**Compare seeds or rates.** Runs are deterministic for a given `--seed`,
so a change in the result comes from the change in the options.

**Training diverges** (an E0501 error, or a loss that grows): lower
`--lr`, or start high and decay it with `--lr-schedule linear:END` or
`step:K:F`.

**Large datasets are slow.** The whole dataset is unrolled into the
graph for every step; `--batch 32` trains on a sample per step and is
much faster.

## Error codes

Every problem is reported with a code, the place in the file (or on the
command line), and often a hint. The full catalogue is in
[`docs/01-language.md`](docs/01-language.md) §10.

| Codes | Stage | Examples |
|---|---|---|
| E01xx | syntax | unexpected token, unterminated comment, `let` without `in`, non-constant exponent |
| E02xx | names | undefined name (with a suggestion), duplicate, used before declaration, wrong number of arguments |
| E03xx | shapes | incompatible shapes, matrix product dimensions, inconsistent shape variable, index out of range, loss not `Real` |
| E04xx | data | missing file, missing CSV column, non-numeric cell, row not matching the schema, `--batch` too large |
| E05xx | running | non-finite value or gradient, input without a value, `--set` of the wrong shape, parameter file not matching |
| W02xx | warnings | shadowing, unused declaration, parameter that can't affect the loss |

## The examples

| File | Shows |
|---|---|
| [`intro.hsg`](examples/intro.hsg) | micrograd's first example: values, gradients, pictures |
| [`neuron.hsg`](examples/neuron.hsg) | vectors, a function with shape variables, shared subexpressions |
| [`calculus.hsg`](examples/calculus.hsg) | symbolic derivatives, second derivatives, what `--simplify` does |
| [`linreg.hsg`](examples/linreg.hsg) | an inline dataset, mean squared error, training and prediction |
| [`xor.hsg`](examples/xor.hsg) | a small neural network, parameter files |
| [`moons.hsg`](examples/moons.hsg) | micrograd's demo: a CSV dataset, a 2-16-16-1 MLP, mini-batches |
| [`errors.hsg`](examples/errors.hsg) | a broken program and its diagnostics |

[`docs/17-examples.md`](docs/17-examples.md) walks through each of them
with real output, and [`docs/16-cli.md`](docs/16-cli.md) explains how
the command line is built.

## License

MIT. See [LICENSE](LICENSE).
