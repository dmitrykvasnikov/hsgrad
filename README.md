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

```bash
cabal run hsgrad -- --help
```

`cabal install exe:hsgrad` puts `hsgrad` on your `PATH`; the examples
below are written as `hsgrad …`.

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

Every command and option is described in
[`docs/16-cli.md`](docs/16-cli.md) §3, and in `hsgrad COMMAND --help`.

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

[`docs/17-examples.md`](docs/17-examples.md) walks through each of them.
