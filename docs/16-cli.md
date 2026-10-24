# 16 — `Driver` and the CLI: from a command line to results

Fifteen steps built the parts: a graph and its evaluation (Core), two
ways to differentiate it, a simplifier and a printer, pictures, a
trainer, a language with a parser, a checker and a compiler, datasets.
This step puts them behind one program, `hsgrad`, with the commands of
`00-architecture.md` §5:

```
hsgrad check   FILE
hsgrad eval    FILE [--set NAME=VALUE ...] [--show NAME ...]
hsgrad derive  FILE --of NAME --wrt NAME[,NAME...] [--simplify] [--dot OUT] [--format dot|text]
hsgrad plot    FILE --of NAME --dot OUT [--cluster]
hsgrad train   FILE [--steps N] [--lr X] [--lr-schedule const|linear:END|step:K:F]
                    [--batch N] [--seed N] [--report-every N] [--save-params OUT]
hsgrad predict FILE --params PARAMS [--set NAME=VALUE ...]
```

Four modules do it:

- `HsGrad.Driver`: the **pipeline** file → parse → check → load datasets
  → compile → run a command, written so that every stage returns its
  result and its diagnostics as values. It also turns the runtime errors
  of the core (a missing input, a `NaN`) into diagnostics that point at
  the source expression they came from.
- `HsGrad.Driver.ParamsFile`: the file that `train --save-params` writes
  and `--params` reads.
- `HsGrad.Cli`: the command line, described with
  `optparse-applicative`: commands, flags, help texts with examples.
- `app/Main.hs`: the IO around them: print, write files, exit.

The milestone of the roadmap is "all CLI commands work". The first
example of the language spec is now a real run:

```
$ hsgrad derive examples/intro.hsg --of L --wrt a,b,c --simplify --format text
∂L/∂a = -2·b  = 6
∂L/∂b = -2·a  = -4
∂L/∂c = -2    = -2
```

and so is training, here the XOR net of step 9, from its `.hsg` file:

```
$ hsgrad train examples/xor.hsg --steps 200 --seed 1 --report-every 50
train examples/xor.hsg: 200 steps, lr 0.1 (constant), seed 1, xor: 4 of 4 rows per step
step   0  loss 1.22558      accuracy 50%
step  50  loss 0.543008     accuracy 100%
step 100  loss 0.0432655    accuracy 100%
step 150  loss 0.000801179  accuracy 100%
step 199  loss 1.22675e-5   accuracy 100%
```

Errors in values given on the command line are shown like errors in a
file, because to the driver the command line *is* a small source file:

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

---

## 1. Theory

### 1.1 A pipeline of stages

Running a `.hsg` file is a sequence of stages, each of which needs the
result of the one before:

```
read FILE ─► parse ─► check ─► read datasets ─► compile ─► run the command
   E0401     E01xx    E02xx      E04xx                     E05xx, values
                      E03xx
```

Two rules from earlier steps decide how errors flow through it:

- **Within** a stage, report everything. The checker collects all
  independent errors (step 12), the data loader all problems of a stage
  (step 13), the command line every bad `--set` value. A user who fixes
  one error should not have to run again to find the next.
- **Between** stages, stop at the first stage that fails. Checking a
  program that doesn't parse, or compiling one that doesn't check, would
  only produce follow-up noise (and the later stages rely on the earlier
  ones' guarantees: the compiler assumes every shape fits).

Warnings never stop anything; they are collected and printed with the
rest.

### 1.2 Writer and Either: a monad for stages

"A result or a stop, plus the diagnostics so far" is a combination of
two well-known monads. `Either e` stops at the first `Left`; `Writer w`
accumulates a log `w` next to a result. Together they give:

```haskell
data Outcome a = Outcome [Diagnostic] (Result a)
data Result a = Ok a | Stopped | Internal Text

Outcome ds r >>= f = case r of
  Ok a -> let Outcome ds' r' = f a in Outcome (ds <> ds') r'
  Stopped -> Outcome ds Stopped
  Internal t -> Outcome ds (Internal t)
```

`>>=` runs the next stage only after a success, and keeps the
diagnostics of both: a checker warning is still printed when the command
fails at step 50 of training. The pipeline then reads like the picture:

```haskell
loaded >>= run opts        -- where run = do c <- compileSession …; env <- environment …; …
```

The library `transformers` has this as `ExceptT e (Writer w)`, but the
type is four lines and the instances fifteen, so it is written out where
it can be read. (`Internal` is the third way to end: a broken promise of
an earlier stage, such as the compiler failing on a checked program. It
is a bug, not the user's fault, so it is not a diagnostic.)

`>>=` stops at the first failing stage. For two stages that don't depend
on each other — the names in `--show` and the values in `--set` — that
would hide the second stage's problems, so there is also

```haskell
alongside :: Outcome a -> Outcome b -> Outcome (a, b)
```

which runs both and keeps the diagnostics of both. That is exactly what
an `Applicative` that is not a `Monad` can do and a `Monad` can't: with
`>>=`, the second computation may depend on the first result, so it
can't run when there is none.

### 1.3 A pure core in an IO shell

Most of a run is pure: parsing, checking, compiling, evaluating,
training, formatting the output. Only a few things touch the world —
reading the `.hsg` file, the CSV files and the parameter file, printing,
writing output files, the exit code. The driver keeps those at the
edges:

```
          IO                        pure                          IO
 ┌─────────────────────┐   ┌──────────────────────────┐   ┌──────────────────┐
 │ load: read the file, │──►│ run: compile, evaluate,  │──►│ Main: print lines,│
 │ its CSVs, --params   │   │ train, format the output │   │ write files, exit │
 └─────────────────────┘   └──────────────────────────┘   └──────────────────┘
```

This is sometimes called "functional core, imperative shell". The
payoff is in the tests: a whole command, from an argument list to its
output lines and diagnostics, runs in the test suite as a function call
(`execute`), without a process, a terminal or a temporary file.

### 1.4 Laziness carries the progress

Training prints a report every few steps *while* it runs. In a language
with mutation that is a print statement inside the loop. Here the
training loop is pure (step 9): it returns a lazy stream of reports,
`Step report rest`, ending in `Done params` or `Failed error`. The
driver maps it, still lazily, to a stream of output lines:

```haskell
data Stream = Line Text Stream | End (Outcome [(FilePath, Text)])
```

`Main` prints each `Line` as soon as it is evaluated, and evaluating it
is what runs the training step that produces it. Nothing is computed
ahead, nothing is kept: the stream is consumed as it is produced, in
constant memory. The `End` carries what is left to do — the files to
write (the saved parameters) — or the diagnostics of a failed step.

### 1.5 Errors at runtime point into the source

Most errors are found before anything runs. A few can only be found
with numbers: `log(x)` of a negative input, `exp` that overflows, an
input that nobody gave a value. The core reports them by **node**
(`NonFiniteValue node args value`, step 4), and the graph knows where
every node came from (D4): the compiler recorded the span of the
innermost expression that created it (step 14). So the driver can say
exactly where:

```
error[E0501]: non-finite value: `log` of a negative number
 --> test/golden/run/log-negative.hsg:3:13
  |
3 | let y = 3 * log(x)
  |             ^^^^^^ `log(-2)` is NaN
  |
  = hint: `log` is defined only for positive numbers
```

The node's op and its argument values also say *what* went wrong, so the
message names the problem (`log` of a negative number, a division by
zero, `exp` overflowing) and the label shows the computation. In a
gradient graph the same machinery reports a non-finite *gradient*
(E0504) at the expression whose derivative it is.

### 1.6 The command line as a source

Values on the command line have the same problems as values in a file:
`--set x=[1,,2]` doesn't parse, `--set x=[1, 2]` has the wrong shape,
`--of Lx` names nothing. The driver treats the command line as a tiny
source file named `<command line>`: it re-assembles the arguments into
one line of text, gives every value a span in it, parses `--set` values
with the language's own literal parser, and reports problems as
diagnostics with snippets. Everything above — the renderer, the
"did you mean" suggestions, secondary labels in the `.hsg` file — then
works for the command line too, with no new code.

The re-assembled line is not exactly what was typed (the shell has
already removed quotes, and the options appear in a fixed order), but it
holds the same words. Doing this inside `optparse-applicative` is a
small exercise in applicative functors, §3.6.

---

## 2. Design

### 2.1 Modules

| Module | Role |
|---|---|
| `HsGrad.Driver` | Options types, `Outcome`, `load` (the IO of reading), `run` (pure), runtime diagnostics, output formats, `execute` for tests |
| `HsGrad.Driver.ParamsFile` | Write and read parameter files |
| `HsGrad.Cli` | The `optparse-applicative` description of the command line → `Options` |
| `app/Main.hs` | Parse, load, run; print, write, exit |

`HsGrad.Cli` is a library module, not part of `app/`, so that the tests
can run the real parser on argument lists with `execParserPure`. The
executable itself is the IO shell of §1.3, about a hundred lines.

### 2.2 Options

`Cli` turns the arguments into one value:

```haskell
data Options = Options
  { optCommandLine :: Text        -- the re-assembled command line (§1.6)
  , optColour :: Colour           -- --color auto|always|never
  , optCommandWord :: Arg         -- "train", for "train needs a loss"
  , optFile :: Arg                -- the .hsg file
  , optCommand :: Command }

data Command = Check | Eval EvalOptions | Derive DeriveOptions
             | Plot PlotOptions | Train TrainOptions | Predict PredictOptions

data Arg = Arg { argText :: Text, argSpan :: Span }   -- a value with its place
```

Every value a diagnostic may point at is an `Arg`; flags that are only
numbers (`--steps`) are checked by the parser itself and are plain
values. The commands that evaluate the graph share one record for where
the leaves get their values:

```haskell
data Bindings = Bindings
  { bindSet :: [Arg]          -- --set NAME=VALUE
  , bindParams :: Maybe Arg   -- --params FILE
  , bindSeed :: Seed }        -- --seed N, for the initialisers
```

### 2.3 Where leaves get their values

The graph's leaves are inputs, parameters and dataset slots (D10). A
command that evaluates the graph binds them like this:

| Leaves | Value |
|---|---|
| inputs | `--set NAME=VALUE`, else the declared default; no value → E0502 if the output needs it |
| parameters | the `--params` file, else the initialisers with `--seed` (default 1337, micrograd's) |
| dataset slots (`eval`, `derive`, `plot`) | the full batch: slot `i` holds row `i` |
| dataset slots (`train`) | a new batch of `--batch` rows every step (step 9) |
| model arguments (`predict`) | `--set NAME=VALUE` |

A `--set` value is a literal of the language — `3`, `[1, 2]`,
`[[1, 2], [3, 4]]` — parsed by `parseLiteral` (step 11) and checked
against the declared shape: a `Real` for a `Vec 3` is E0503, not a
silent fill. For `predict`, a name is looked up among the model's
arguments first and then among the inputs, since an argument may shadow
an input of the same name (`15-compile.md` §2.3).

### 2.4 Names: whole values and elements

`--of`, `--show` and `--wrt` take a name: an input, a parameter, a
top-level `let` or `loss`. A vector or matrix name stands for all its
elements, in row-major order, named like their leaves: `--wrt W1` is
`W1[0,0]`, `W1[0,1]`, … An element can be named on its own, `W1[1,0]`
(spaces inside the brackets don't matter), and `--wrt` splits its list
only at commas outside brackets, so `--wrt W1[1,0],b2` is two names.

`--of` of a vector differentiates every element; `--wrt` of a `let`
gives the derivative with respect to that intermediate value (step 7's
gradient nodes exist for every node, not only leaves). A name that
doesn't exist is E0201 with a suggestion, a function or the model E0204,
an element out of range E0305.

### 2.5 Parameter files

`train --save-params OUT` writes the trained parameters; `--params FILE`
reads them back. The format is a small `.hsg` program: one `param`
declaration with explicit values per parameter.

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

- It is **human-readable and writable**: a header comment says where it
  came from, a matrix has one row per line, and a hand-written file may
  use everything the language allows in a `param` declaration with `=`:
  comments, any layout, `= 0` to fill a whole shape.
- Numbers are printed with `showConstant` (step 8): the shortest decimal
  that reads back as **exactly** the same `Double`. The parser reads a
  decimal exactly and rounds once (step 11), so saving and loading loses
  nothing (a property test checks it, §5).
- **Reading it needs no new parser**: it is parsed and checked by the
  language's own `parseProgram` and `checkProgram`, with their
  diagnostics (a malformed literal is E0312 in the parameter file).
  What is left is comparing it with the program (§3.5).
- The declarations can be pasted into the program itself, replacing the
  `~ uniform(…)` initialisers.

### 2.6 Runtime errors as diagnostics

| From the core | Diagnostic | At |
|---|---|---|
| `UnboundLeaves` of input leaves | E0502, once per input or model argument | its name in the declaration |
| `UnboundLeaves` of parameter or slot leaves | internal error | — |
| `NonFiniteValue n args v` | E0501 (E0504 in a gradient graph) | the span of node `n` |
| `NonFiniteGrad n args g locals` | E0504 | the span of node `n` |
| `StepFailed t e` (training) | the above, with a note "in training step `t`" | |
| `EmptyFeed`, `BatchTooLarge`, `RowWidthMismatch` | internal error (the data loader checked) | — |
| `CompileError` | internal error (the parser rejects `1e999`, E0103) | — |

In `derive`, values are computed in the gradient graph, which extends
the program's graph (step 7): a node whose id is past the forward graph
computes a gradient, so a non-finite value there is a non-finite
gradient (E0504), and `x ^ -0.5` at 0 is reported as the infinite slope
of `sqrt` it is.

### 2.7 Exit codes and colour

`hsgrad` exits with 0 when the command worked (warnings or not), 1 when
there were errors or the command line was wrong, and 2 on an internal
error. Diagnostics go to stderr, results to stdout, so
`hsgrad plot f.hsg --of y --dot - | dot -Tsvg` works.

`--color auto` (the default) colours diagnostics when stderr is a
terminal and the environment has no `NO_COLOR` (the informal standard of
no-color.org); `always` and `never` force it.

---

## 3. Command reference

Every command takes the file as its first argument, `--color
auto|always|never`, and `--help`. Wherever a command evaluates the
graph, `--set NAME=VALUE` (repeatable), `--params FILE` and `--seed N`
(default 1337) say where the leaves get their values (§2.3).

### 3.1 `check FILE`

Everything short of running: parse, check, load the datasets, compile.
Prints the declarations with their types and the size of the graph;
reports every problem.

```
$ hsgrad check examples/intro.hsg
input a : Real
input b : Real
input c : Real
let e : Real
let d : Real
let L : Real
examples/intro.hsg: ok, 7 nodes
```

For `moons.hsg` the last line is `examples/moons.hsg: ok, 65715 nodes`
(the full batch of 100 rows, `15-compile.md` §5.1).

### 3.2 `eval FILE [--set NAME=VALUE ...] [--show NAME ...]`

Evaluate values, one line each: the `--show` names in order, or every
top-level `let` and then the loss. Vectors and matrices print as
literals; numbers as the shortest exact decimal.

```
$ hsgrad eval examples/intro.hsg
e = -6
d = 4
L = -8
$ hsgrad eval examples/intro.hsg --set a=3 --show L --show e
L = -2
e = -9
$ hsgrad eval examples/neuron.hsg --set x=[1,0,-1] --show y --show w
y = -0.6983719513613043
w = [-0.45852706058072323, 0.18280262295967586, 0.4055883215369599]
```

`w` comes from `uniform(-1, 1)` with seed 1337; `--seed` or `--params`
change it.

### 3.3 `derive FILE --of NAME --wrt NAME[,NAME...] [--simplify] [--dot OUT] [--format dot|text]`

Differentiate `--of` with respect to each `--wrt` name, symbolically
(step 7): the derivatives are graphs, printed as formulas (step 8) with
their values.

| Option | Meaning |
|---|---|
| `--of NAME` | the value to differentiate; a vector gives one block per element |
| `--wrt NAMES` | comma-separated; vectors and matrices expand to their elements |
| `--simplify` | simplify the gradients (gather like terms, fold constants) |
| `--format text` | (default) print `∂out/∂x = formula  = value` lines |
| `--format dot` | print the gradient graph as DOT instead |
| `--dot OUT` | also write the gradient graph as DOT to `OUT` (`-`: stdout) |

```
$ hsgrad derive examples/neuron.hsg --of y --wrt w,b --simplify
∂y/∂w[0] = x[0]·t1  = 0.6952308208745206
∂y/∂w[1] = x[1]·t1  = -1.3904616417490412
∂y/∂w[2] = x[2]·t1  = 0.3476154104372603
∂y/∂b    = t1       = 0.6952308208745206
where
  t1 = 1 - tanh(x[0]·w[0] + x[1]·w[1] + x[2]·w[2] + b)²
```

Shared subexpressions are named (`t1`) and listed under `where`, in an
order where each uses only the ones before it (step 8's `renderNodes`).
Values are those of the unsimplified gradient graph, the exact numbers
of backprop (step 7); if an input the gradients need has no value, the
values are left out rather than being an error: a formula is useful on
its own. The DOT picture highlights every gradient with its name
(`∂y/∂w[0]`), next to the forward expression it belongs to.

### 3.4 `plot FILE --of NAME --dot OUT [--cluster]`

Write the forward graph of a value as DOT (step 6), micrograd's
`draw_dot`: every node with its value and, for a `Real` output, its
gradient, when every input has a value. `--cluster` groups the scalar
nodes of each source expression into a box labelled with its span.
`OUT` may be `-`.

```
$ hsgrad plot examples/neuron.hsg --of y --dot neuron.dot
wrote neuron.dot
$ dot -Tsvg neuron.dot -o neuron.svg
```

### 3.5 `train FILE [options]`

Gradient descent on the `loss` (step 9).

| Option | Default | Meaning |
|---|---|---|
| `--steps N` | 100 | number of steps (≥ 1) |
| `--lr X` | 0.1 | learning rate (> 0) |
| `--lr-schedule S` | `const` | `const`, `linear:END` (reaches END after the last step), `step:K:F` (times F every K steps) |
| `--batch N` | every row | rows per step, sampled without replacement (E0405 if larger than a dataset) |
| `--seed N` | 1337 | the initial parameters and the batches |
| `--report-every N` | 10 | a report line every N steps, and after the last |
| `--save-params OUT` | — | write the trained parameters (§2.5) |
| `--set NAME=VALUE` | — | fixed inputs |

A report line shows the step, the loss of that step's batch (before its
update), the sign accuracy when the program has a model whose result is
compared with a `Real` label (`01-language.md` §11), and the rate when
it changes. Numbers are rounded to 6 significant digits.

```
$ hsgrad train examples/moons.hsg --steps 100 --lr 1.0 --lr-schedule linear:0.1 --seed 1337
train examples/moons.hsg: 100 steps, lr 1 (linear to 0.1), seed 1337, moons: 100 of 100 rows per step
step  0  loss 0.858769     accuracy 63%     lr 1
step 10  loss 0.26799      accuracy 90%     lr 0.91
step 20  loss 0.276071     accuracy 90%     lr 0.82
step 30  loss 0.0683658    accuracy 99%     lr 0.73
step 40  loss 0.0781449    accuracy 97%     lr 0.64
step 50  loss 0.0314549    accuracy 99%     lr 0.55
step 60  loss 0.0270361    accuracy 100%    lr 0.46
step 70  loss 0.0238252    accuracy 100%    lr 0.37
step 80  loss 0.021905     accuracy 100%    lr 0.28
step 90  loss 0.0211429    accuracy 100%    lr 0.19
step 99  loss 0.0202695    accuracy 100%    lr 0.109
```

This is micrograd's demo: the same network, loss, schedule and number of
steps, and like micrograd it separates the two moons completely. It
takes about 9 seconds here (65,715 nodes, forward and backward 100
times).

If a step fails — a value or gradient that is not finite — the report
lines so far are printed, then the error with the step:

```
$ hsgrad train test/golden/run/train-overflow.hsg --steps 5 --lr 100
train test/golden/run/train-overflow.hsg: 5 steps, lr 100 (constant), seed 1337
step 0  loss -2.71828
step 1  loss -3.0745e118
error[E0501]: non-finite value: `exp` overflows
 --> test/golden/run/train-overflow.hsg:3:9
  |
3 | loss = -exp(w)
  |         ^^^^^^ `exp(3.0744987277590327e120)` is ∞
  |
  = note: in training step 2
  = hint: `exp(x)` is too large for a Double when x is above about 709.78
  = hint: training may have diverged: try a smaller `--lr`

1 error
```

### 3.6 `predict FILE --params PARAMS [--set NAME=VALUE ...]`

Evaluate the `model` on arguments given with `--set`, with parameters
from a file. The output shows the call:

```
$ hsgrad train examples/xor.hsg --steps 200 --seed 1 --save-params xor.params
…
wrote xor.params
$ hsgrad predict examples/xor.hsg --params xor.params --set x=[0,1]
net([0, 1]) = 0.9963848175526087
$ hsgrad predict examples/xor.hsg --params xor.params --set x=[1,1]
net([1, 1]) = -0.9954410498241719
```

The file must match the program: every parameter, at its shape, and no
other (E0505), explicit values only.

```
error[E0505]: the program has no parameter `scal`
 --> test/golden/run/mismatch.params:3:7
  |
3 | param scal : Real = 2
  |       ^^^^ unknown parameter
  |
  = hint: did you mean `scale`?
```

---

## 4. Code walkthrough

### 4.1 Loading

```haskell
load :: Options -> IO (Sources, Outcome Session)
```

`load` reads the `.hsg` file (an unreadable one is E0401 at the file name
on the command line), parses and checks it (`checkSource`), reads its
datasets with step 13's `loadDatasets`, and the `--params` file if the
command has one. It returns the texts of everything it read, as the
`Sources` the renderer needs, with the command line under
`<command line>`, and a `Session`: the checked program, its datasets,
and the parameter file's text. Datasets are loaded for every command,
also `check` (a broken CSV file is a problem worth knowing about before
training) and `derive` (a comprehension over a CSV dataset needs the
number of rows to be compiled, `15-compile.md` §2.4).

### 4.2 Compiling

```haskell
compileSession :: Maybe Int -> Session -> Outcome Compiled
compileSession batch s =
  case compileProgram (batchSizes batch (sessionDatasets s)) (sessionChecked s) of
    Left e -> internal ("the compiler failed: " <> …)
    Right c -> do
      report [d' | Just b <- [batch], d <- sessionDatasets s, dataName d `Map.member` compiledSlots c, d' <- checkBatch b d]
      pure c
```

Training passes `--batch`; every other command compiles the full batch.
`--batch` is checked (E0405) only against the datasets the graph
actually iterates over.

### 4.3 `--set`

```haskell
settings :: [Settable] -> [Text] -> [Arg] -> Outcome (Map Name Double)
```

A `Settable` is an input of the program or an argument of the model,
with its `InputSpec` (step 14: name, shape, leaves) and its declaration.
For each `NAME=VALUE`, `settings` finds the settable (else E0201, with
"did you mean", or a special message for a parameter), parses the value,
checks the shape, and zips the numbers with the leaves. The value is
parsed on its own, so its spans start at column 1; `shiftSpan` moves
them (and those of any parse error) to where the value is on the command
line. All arguments are checked, and all their problems reported
together (`collect`).

### 4.4 Values and runtime diagnostics

```haskell
data Runtime = Runtime
  { rtGraph :: Graph, rtSettables :: [Settable]
  , rtIsGradient :: NodeId -> Bool, rtFallback :: Span }

evaluated :: Runtime -> Env -> [NodeId] -> Outcome Values
evalFailure :: Runtime -> EvalError -> Outcome a
gradFailure :: Runtime -> BackpropError -> Outcome a
```

A `Runtime` is what a diagnostic needs to know about the graph being
run: the graph (for the failing node's op and span), the settables (to
turn unbound leaves into "input `x` has no value"), and which nodes are
gradients. `opProblem` reads the op and its argument values to name the
problem and pick a hint; `computation` prints the failing application
(`log(-2)`, `1 / 0`, `(-4) ^ 0.5`).

### 4.5 Commands

Each command is a function `Session -> … -> Outcome Stream`. `derive`
shows the pattern:

```haskell
runDerive s o = do
  c <- compileSession Nothing s
  ((outs, wrts), env) <- alongside (alongside (names s c [deriveOf o]) (names s c (deriveWrt o)))
                                   (environment s c (deriveBindings o))
  (g, named) <- gradientGraph c outs wrts
  values <- optionalValues rt env (map snd named)
  (shown, shownIds) <- if deriveSimplify o then simplified g ids else pure (g, ids)
  let Rendered bindings roots = renderNodes defaultPrettyOptions shown shownIds
  …
```

`gradientGraph` runs step 7's `backwardNodes` once per output in one
`extend` of the program's graph, and names every gradient `∂out/∂x`;
step 8's `simplify` and `renderNodes` do the rest. The three-column
layout (`equations`) pads names and formulas, except that a formula
longer than 40 characters doesn't widen the value column for the
others.

`train` maps step 9's lazy `Run` to a `Stream`, one line every
`--report-every` steps and after the last, ending in the parameter file
to write:

```haskell
go lastReport = \case
  Loop.Step r rest
    | Loop.reportStep r `mod` every == 0 || isLast rest -> Line (reportLine width sched r) (go (Just r) rest)
    | otherwise -> go (Just r) rest
  Loop.Done ps -> End (save c ps lastReport)
  Loop.Failed e -> End (trainFailure rt e)
```

### 4.6 The command line: `Compose Parser Layout`

An `optparse-applicative` parser has type `Parser a`, and it is an
`Applicative`: `(,) <$> p <*> q` parses what `p` and `q` parse. To give
every value a span, each parser should also *place* its words on the
re-assembled line. Placing is a state computation — "the next free
column, the words so far":

```haskell
type Layout = State (Int, [Text])

word :: Text -> Layout Span          -- place a word, return its span
```

A parser of `Layout a` values does both: optparse parses, and the
result, once run, places. `Data.Functor.Compose` makes the pair an
applicative of its own:

```haskell
type CliParser = Compose Parser Layout

valueArg :: Text -> Mod OptionFields String -> CliParser Arg
valueArg flag mods = Compose (flagged flag . Text.pack <$> strOption mods)
-- flagged flag v = word flag *> (Arg v <$> word v)
```

The composition of two applicatives is always an applicative: `<*>` of
`Compose` combines the parsers with optparse's `<*>` and the layouts
with `State`'s `<*>`, so the words are placed in the order the parsers
are written. Every command's parser is written in `CliParser`:

```haskell
deriveOptions =
  DeriveOptions
    <$> bindings
    <*> valueArg "--of" (long "of" <> metavar "NAME" <> help "…")
    <*> wrt
    <*> plain (switch (long "simplify" <> help "…"))
    …
```

and `finish` runs the layout of the parsed result: the text of the line,
and the options pointing into it. Two details: `plain` lifts a parser
that places nothing (`--steps`, `--simplify`), and `many`/`optional`
need hand-written versions (`manyArgs`, `optionalArg`), because the
generic ones of `Compose` would recurse forever through optparse's
`Parser`, whose `many` is special.

Help texts have an "Examples:" footer per command:

```
$ hsgrad derive --help
Usage: hsgrad derive FILE [--set NAME=VALUE] [--params PARAMS] [--seed N]
                     --of NAME --wrt NAME[,NAME...] [--simplify] [--dot OUT]
                     [--format dot|text] [--color auto|always|never]

  Differentiate a value with respect to inputs, parameters or other values, as
  formulas or a graph.
…
Examples:
  hsgrad derive examples/intro.hsg --of L --wrt a,b,c --simplify
  hsgrad derive examples/neuron.hsg --of y --wrt w,b --simplify --dot neuron-grad.dot
  hsgrad derive examples/xor.hsg --of loss --wrt "W1[0,1]" --format dot
```

### 4.7 Reading a parameter file

```haskell
readParams :: FilePath -> Text -> [ParamInfo] -> Either [Diagnostic] Params
```

Three stages, each only if the one before found nothing: the file parses
and holds only `param` declarations with values (E01xx, E0505); it passes
the checker, without its warnings ("`W1` is never used" means nothing in
a parameter file); every parameter of the program is there, at its
shape, and nothing else (E0505, with "did you mean" for a misspelt
name). A missing parameter is reported at its declaration in the
program, since the file has no place for it.

### 4.8 `Main`

```haskell
main = do
  mapM_ (`hSetEncoding` utf8) [stdout, stderr]
  opts <- customExecParser cliPrefs cliInfo
  …
  (sources, loaded) <- load opts
  case loaded >>= run opts of
    Outcome ds (Ok stream) -> do
      printDiagnostics render sources ds          -- warnings, before any output
      Outcome ds' result <- printLines stream     -- lines as they come
      printDiagnostics render sources ds'
      … write the files of the result, "wrote OUT" on stderr …
      finish (ds <> ds') …
    Outcome ds result -> …
```

Output is UTF-8 whatever the locale (`∂`, `·`), stdout is line-buffered
so training reports appear as they are made, and the summary line
(`1 error, 1 warning`) comes once, at the end.

---

## 5. Design decisions

### 5.1 Every problem is a diagnostic

Problems with the command line could have been plain messages
("hsgrad: unknown name Lx"). Making them diagnostics, with the command
line as a source, costs one re-assembled line and a span per value, and
buys the whole machinery of steps 10–12: snippets, suggestions, labels
in the `.hsg` file next to labels on the command line, colour, golden
tests. It also means one output format for everything that can go
wrong, which is what `00-architecture.md` §6 asked for.

The codes come from the catalogue where one fits: an unknown name is
E0201 (with its "did you mean"), a function where a value is needed
E0204, an element out of range E0305, a malformed `--set` value E01xx or
E0312 like the same literal in a file, a command without the
declaration it needs (`train` without `loss`, `predict` without `model`)
E0201 at the command word. Two changes to the catalogue
(`01-language.md` §10):

- **E0401** now covers any unreadable file named on the command line
  (the `.hsg` file, `--params`), not only CSV files.
- **E0505** is new: a `--params` file that doesn't match the program.

### 5.2 The parameter file is a `.hsg` subset

Alternatives were a format of its own (`W1[0,1] = 0.25` per line, which
needs a parser and loses the shapes) or JSON (a new dependency, and
not pleasant to write by hand). A program of `param` declarations is
readable, shaped, commented, reuses the parser and checker with their
error messages, and can be pasted back into the program. Shortest exact
decimals make it lossless.

### 5.3 Values are optional in `derive` and `plot`

`derive` without the values its formulas need still prints the formulas;
`plot` still draws the names. Symbolic results are useful on their own,
and making them depend on inputs would make `derive model.hsg --of loss
--wrt W` fail for every program whose inputs have no defaults. A value
that is not finite is still an error: then the formulas are evaluated at
a point where they are wrong, and the user should know.

### 5.4 More than §5 listed

The commands are those of `00-architecture.md` §5; a few additions came
from using them:

- `eval`, `derive` and `plot` take `--set`, `--params` and `--seed`
  (§2.3); without them, programs with random parameters could only be
  evaluated at seed-dependent values nobody chose.
- `-` as an output file means stdout (pipes into Graphviz).
- Element names (`W1[0,1]`) in `--of`, `--show` and `--wrt`.
- `check` prints the declarations and the graph size.

The defaults: 100 steps, rate 0.1, constant schedule, seed 1337, a report
every 10 steps. They train `xor.hsg` to accuracy 1 (step 9 found seeds
1–5 all reach it in 200 steps at rate 0.1).

### 5.5 Spec changes

- `01-language.md` §1.1: the `derive` output shows the real spacing (two
  spaces before the value column instead of a hand-aligned column).
- §10: E0401 widened, E0505 added (§5.1).
- The open point of step 14, a literal like `1e999` that compiles to an
  infinite constant, needed nothing: the parser already rejects it
  (E0103 "number is too large", step 11), also in `--set` values and
  parameter files. A `GraphFailure` from the compiler is therefore an
  internal error.

---

## 6. Tests

`test/HsGrad/DriverSpec.hs`, `test/HsGrad/CliSpec.hs`, and the golden
files in `test/golden/run/`. The Graphviz helpers of step 6 (`withDot`,
`toSvg`) moved from `DotSpec` to `HsGrad.Test.Graphviz`, so the driver's
pictures are checked the same way.

### 6.1 Commands

Each test parses an argument list with the real parser and runs it with
`execute`, the same pipeline as the program, collecting output lines,
files and diagnostics:

- **derive**: the §1.1 output of `intro.hsg` exactly, simplified and
  not; `neuron.hsg` with the shared `t1` under `where`; `--wrt W1`
  expanded to eight elements in row-major order; `W1[1,0],b2`; an
  unrelated leaf gives `0`; values left out when an input has none;
  `--set` changes the values; `--format dot` and `--dot` give the same
  DOT, every gradient highlighted, and Graphviz accepts it.
- **eval**: the lets of `intro.hsg`; `--set` and `--show`; vectors,
  elements and matrices; the loss of `xor.hsg` over its whole dataset;
  parameters from a file.
- **plot**: `neuron.hsg` with values and gradients, `xor.hsg` with
  clusters, both accepted by Graphviz; names only without input values.
- **train and predict**: `xor.hsg` with seed 1 reaches accuracy 100% and
  a loss below `10⁻³` in 200 steps, reporting at steps 0, 10, …, 190
  and 199; `--report-every` and a changing rate; `moons.hsg` on
  mini-batches of 8; the saved parameters, read by `predict`, give every
  XOR row the sign of its label; a hand-written parameter file.
- **check**: the summary of `xor.hsg` (154 nodes); a missing file is
  E0401.
- **Parameter files**: a property — random values of every magnitude
  (down to `10⁻³⁰⁰`, up to `10³⁰⁰`), saved and read back, are exactly the
  same `Double`s; the layout (columns, one matrix row per line); a single
  number fills a shape.

### 6.2 Golden runs

Each `test/golden/run/*.hsg` starts with a line `-- args: COMMAND
OPTIONS`; the harness (step 10's `goldenFiles`) runs `hsgrad COMMAND
FILE OPTIONS` and compares the transcript — output lines, files that
would be written, rendered diagnostics — with the `.err` file. A test
checks that every E05xx code appears in some `.err` file.

| File | What it pins down |
|---|---|
| `missing-input` | E0502 at the input's name, with a `--set` hint of its shape |
| `predict-missing-arg` | E0502 for a model argument |
| `bad-set-shape` | E0503 twice: a `Vec 3` for a `Vec 2`, a matrix for a `Real` |
| `set-syntax` | E0101 and E0103 inside `--set` values, at the right columns |
| `set-unknown` | E0201 for a parameter and for an unknown input |
| `names` | E0305, E0204, E0201 with "did you mean", no `loss`, and a bad `--set` in the same run |
| `unknown-wrt` | E0201 for names inside a `--wrt` list |
| `log-negative` | E0501 at `log(x)` |
| `sqrt-zero` | E0504 in `derive`: the infinite slope of `sqrt` at 0 |
| `train-sqrt` | E0504 from backprop at step 0 |
| `train-overflow` | E0501 at step 2, after two report lines, with the divergence hint |
| `params-mismatch` | E0505: wrong shape, unknown (did you mean), two missing |
| `params-random` | E0505: a non-`param` declaration and a random initialiser |
| `missing-params-file` | E0401 at the file name on the command line |
| `no-loss`, `no-model` | E0201 at the command word |
| `batch-too-large` | E0405 for `--batch` |
| `predict`, `predict-shadow` | `predict` output; `--set x` sets the model argument that shadows `input x` |

### 6.3 The command line

`CliSpec` runs the parser on argument lists (`execParserPure`): every
command with its options, the defaults, options in any order; the
re-assembled line and the spans of the command word, the file, `--of`,
each `--wrt` name and `--set` (each span slices exactly its value out of
the line); failures with a message (steps below 1, `--lr NaN`, bad
schedules, `--set` without `=`, a missing `--of` or `--params`, an
unknown format or command); an "Examples:" footer in the help of every
command; the schedule reader and the bracket-aware comma split.

### 6.4 Testing the tests

Deliberate bugs, each tried on its own; every one makes the suite fail:

| Bug | Caught by |
|---|---|
| `equations` without padding | the §1.1 output |
| `alongside` keeps only the first stage's problems | `names` golden |
| command-line spans one column off | `set-syntax`, `bad-set-shape` goldens |
| `--set` prefers inputs over model arguments | `predict-shadow` golden |
| no slot bindings outside training | `eval` of the XOR loss |
| defaults override `--set` | the `eval --set` tests |
| gradient nodes reported as E0501 | `sqrt-zero` golden |
| the last report not printed | the XOR training test |
| missing parameters not reported | `params-mismatch` golden |
| parameters saved with 7 characters | the round-trip property |
| no step note in training errors | `train-sqrt`, `train-overflow` goldens |
| `--wrt` split at every comma | `splitNames`, the `W1[1,0],b2` test |
| elements paired with nodes in reverse | `eval --show x[2]` |
| `--simplify` ignored | the §1.1 and `neuron.hsg` tests |
| a wrong-shape `--set` accepted | `bad-set-shape` golden |

The whole suite has 747 tests and runs in about 4 seconds.

---

## 7. What step 16 builds on

- `execute :: Options -> IO Transcript` and `parseArgs` run any command
  line in a test; `renderTranscript` gives its text as printed.
- End-to-end checks of the examples from the command line: `moons.hsg`
  trains with micrograd's settings to accuracy 100% in 100 steps (§3.5;
  about 9 s at the full batch, so a test may prefer `--batch` or fewer
  steps), `xor.hsg` round-trips through `--save-params` and `predict`.
- `test/golden/run/` takes new whole-run cases with a `-- args:` line.
