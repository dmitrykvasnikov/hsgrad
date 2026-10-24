# 00 — Architecture

hsgrad is a learning project: a Haskell re-implementation of
[micrograd](https://github.com/karpathy/micrograd) (scalar reverse-mode
automatic differentiation plus a tiny neural-network trainer), extended with:

- a small **declarative language** (`.hsg` files) that describes inputs,
  parameters, datasets, the model and the loss;
- **symbolic differentiation**: gradients are produced as graphs that can be
  simplified and rendered with Graphviz;
- **good error messages** for everything that can go wrong in a `.hsg` file.

The project is built one module at a time. Every module gets its own page in
`docs/` explaining the theory, the design decisions and the code.

---

## 1. The big picture

```
                       ┌──────────────────── Lang ─────────────────────┐
  model.hsg ──────────►│ Parser ──► Syntax ──► Check ──► Compile       │
                       │   (megaparsec)   (names, shapes)  (unrolling) │
                       └──────────────────────────────────────┬────────┘
  data/*.csv ──► Data ─────────────────────────────────┐      │ Builder actions
                                                       ▼      ▼
                       ┌──────────────────── Core ─────────────────────┐
                       │ Graph (DAG of scalar ops, hash-consed)        │
                       │   ├─► Eval      forward pass  → values        │
                       │   ├─► Backprop  reverse pass  → numeric grads │
                       │   ├─► Symbolic  reverse pass  → gradient graph│
                       │   └─► Simplify  canonical form, like terms    │
                       └───────────────┬───────────────────────────────┘
                                       │
           ┌───────────────────────────┼──────────────────────┐
           ▼                           ▼                      ▼
     Train (SGD loop)            Viz.Dot (Graphviz)      Diagnostics
           ▲                           ▲                      ▲
           └──────────── Driver / CLI (flags = "how to run") ─┘
```

The `.hsg` file says **what** is computed (model, loss, data). The command
line says **how** to run it (which command, learning rate, steps, seed, output
file).

## 2. Layers and the dependency rule

```
  app/Main ──► Driver ──► Lang ──► Core
                  │        │
                  ├──► Train ──► Core
                  ├──► Viz ────► Core
                  └──► Data
```

- **Core** knows nothing about the language, files, or training. It is a pure
  library for building, evaluating and differentiating scalar graphs.
- **Lang** turns text into Core graphs. It depends on Core, never the reverse.
- **Train, Viz, Data** are small libraries on top of Core.
- **Driver** wires everything together; **Cli** describes the command
  line, and **app/Main** only does the IO (`16-cli.md`).

Two edges are not in the picture. `Data.Dataset` reads the checker's
dataset declarations (`Lang.Check`), and `Lang.Compile`, the back end of
Lang, calls the naming functions of `Train.Param` and `Data.Dataset`
(`elementNames`, `slotNames`) so that leaf names cannot disagree
(`15-compile.md` §4.1). The front end (`Syntax`, `Parser`, `Check`)
depends only on Core, and there are no module cycles.

This rule keeps each step testable in isolation: Core is finished and tested
before a single line of parser code exists.

## 3. Module catalogue

| Module | Responsibility | Doc |
|---|---|---|
| `HsGrad.Core.Op` | The fixed set of scalar operations: arity, forward function, local derivative rule | 03 |
| `HsGrad.Core.Graph` | `NodeId`, `Node`, `Graph`; the `Builder` monad; hash-consing; provenance | 04 |
| `HsGrad.Core.Eval` | Forward pass: graph + leaf values → value of every node | 05 |
| `HsGrad.Core.Backprop` | Numeric reverse-mode AD (micrograd's `backward()`) | 06 |
| `HsGrad.Viz.Dot` | Export any graph (forward or gradient) to Graphviz DOT | 07 |
| `HsGrad.Core.Symbolic` | Reverse-mode AD that *emits graph nodes*: the gradient as a graph | 08 |
| `HsGrad.Core.Simplify` | Simplifier: identities, constant folding, gathering like terms | 09 |
| `HsGrad.Core.Pretty` | Textual rendering of expressions (used by tests and `--format text`) | 09 |
| `HsGrad.Train.Param` | Parameter store and initialisation (`uniform`, `normal`, constants) | 10 |
| `HsGrad.Train.Optim` | SGD with learning-rate schedules | 10 |
| `HsGrad.Train.Loop` | The training loop | 10 |
| `HsGrad.Lang.Span` | Source positions and spans | 11 |
| `HsGrad.Lang.Diagnostic` | Error/warning type and the renderer with source snippets | 11 |
| `HsGrad.Lang.Syntax` | Surface AST, every node annotated with a span | 12 |
| `HsGrad.Lang.Parser` | megaparsec parser for `.hsg` | 12 |
| `HsGrad.Lang.Shape` | Types (`Real`, `Vec d`, `Mat r c`), dimensions, shape variables, matching | 13 |
| `HsGrad.Lang.Check` | Name resolution, define-before-use, shape checking | 13 |
| `HsGrad.Data.Csv` | Minimal numeric CSV reader | 14 |
| `HsGrad.Data.Dataset` | Datasets (from CSV or inline rows) checked against their schema | 14 |
| `HsGrad.Lang.Compile` | Checked AST → `Builder` actions (inlining, unrolling vectors into scalars) | 15 |
| `HsGrad.Driver` | Pipeline: file → parse → check → compile → run a command; runtime diagnostics | 16 |
| `HsGrad.Driver.ParamsFile` | Parameter files for `--save-params` / `--params` | 16 |
| `HsGrad.Cli` | The `optparse-applicative` description of the command line | 16 |
| `app/Main.hs` | The IO around the driver: print, write files, exit | 16 |

## 4. Key design decisions

Each decision gets a short rationale. Module docs refer back to these by ID.

### D1. The graph is pure data built in a monad

micrograd uses mutable `Value` objects whose `grad` field is updated in place.
In Haskell we instead build an immutable graph:

- a `Builder` monad (a `State` over the graph under construction) hands out a
  fresh `NodeId` for every new node;
- nodes refer to their arguments by `NodeId`, so **sharing is explicit**: using
  `a` twice means referring to the same id twice, and the graph is a DAG, not a
  tree;
- values and gradients live *outside* the graph, in `IntMap Double` maps.

This keeps evaluation, backprop and symbolic differentiation as pure functions
that are easy to test.

### D2. Node ids are already a topological order

A node can only refer to nodes that already exist, so its arguments always
have smaller ids. Walking the graph in ascending id order is therefore a valid
forward order, and descending order a valid backward order. micrograd needs an
explicit DFS topological sort; we get it from how the graph is built. We still
compute the set of nodes **reachable** from the output, so we don't evaluate
dead nodes.

### D3. Hash-consing

The builder keeps a table `(Op, [NodeId]) → NodeId`. Asking for a node that
already exists returns the existing id. This gives common-subexpression
elimination for free, which matters a lot for symbolic gradients and
simplification: both create many structurally identical nodes.

### D4. Every node carries provenance

Each node records where it came from: a source span in the `.hsg` file and an
optional human label (for example `W1[3,0]`). Provenance is ignored by
hash-consing, but it is used for:

- runtime diagnostics ("`log` of a negative number at `model.hsg:12:9`");
- Graphviz clusters that group the thousands of unrolled scalar nodes back
  into the source expressions they came from.

### D5. Scalar core, shape-typed language (option "c")

- **Core** is purely scalar, exactly like micrograd. Derivative rules stay
  trivial.
- **The language** has `Real`, `Vec d` and `Mat r c` with static shapes. The
  checker verifies every shape, and the compiler **unrolls** vector and matrix
  code into scalar nodes.
- A tensor backend can be added later (see the roadmap) without changing the
  language.

Cost: the unrolled graph is large. A 2-16-16-1 MLP over 100 samples has about
60k nodes. That's fine for learning, but readable symbolic output and plots
are only realistic for small examples until a tensor backend exists.

### D6. Two kinds of reverse-mode differentiation

- `Core.Backprop` computes **numbers**: one reverse sweep accumulating
  `∂out/∂node` in a map. This is micrograd's algorithm.
- `Core.Symbolic` runs the **same sweep**, but each step emits new nodes into
  the builder instead of computing a number. The result maps each leaf to a
  `NodeId` that *is* `∂out/∂leaf` as a graph. It shares subexpressions with the
  forward graph and with other gradients, so it costs the same as backprop.
  It is also not "textbook" symbolic differentiation, which would blow up
  exponentially.

Tests check that both agree with each other and with finite differences.

This requires the op set to be **closed under differentiation**: the
derivative of every op must be expressible with ops from the same set. For
example `d tanh(x) = 1 − tanh(x)²` reuses the existing `tanh` node, and
`d relu(x) = step(x)` is why `Step` is in the op set.

### D7. The simplifier gathers like terms without expanding

`Core.Simplify` rewrites each node into a canonical *sum of products*:

- a sum is a constant plus a map *monomial → coefficient*;
- a monomial is a map *atom → integer exponent*;
- atoms are leaves and non-polynomial applications (`tanh`, `exp`, `log`,
  `relu`, `step`, non-integer powers) whose arguments are simplified
  recursively.

This gives `x + x → 2·x`, `x·x → x²`, `2·x·3 → 6·x`, constant folding and the
0/1 identities. It deliberately does **not** distribute products over sums
(`(a+b)(c+d)`), since that can blow up exponentially. Simplification is
memoised per `NodeId` to preserve sharing, and the rebuilt graph is hash-consed
again.

Caveat: `x · x⁻¹ → 1` is only valid for `x ≠ 0`. This is documented, and the
simplifier's soundness tests sample inputs away from zero.

### D8. Shape checking is "template style", with shape variables

Function signatures may use shape variables:
`fn layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m`.
Because the language has no recursion and every call is inlined, the checker
can work like C++ templates:

1. At each call site, match the declared parameter types against the concrete
   argument types. This binds the variables (`m = 16, n = 2`) and reports
   inconsistent bindings with both locations.
2. Check the function body with those concrete shapes.

This avoids full Hindley–Milner inference while still giving precise errors.
Trade-off: shape errors inside a function that is never called aren't
reported if they depend on its shape variables. Name and arity errors,
shape errors that don't depend on them, and everything in a function with
a concrete signature are still reported at the declaration
(`13-checker.md` §2.3).
Generic (call-independent) checking of function bodies is a possible later
upgrade.

### D9. No recursion, define-before-use, first-order

Every name must be declared before it is used. This alone rules out
recursion, so every program unrolls into a finite graph. Functions aren't
values: there are no lambdas and no higher-order functions.

### D10. Data enters the graph as input leaves

Dataset values aren't baked in as constants. A reduction over `B` rows of a
dataset creates `B` *slots* of input leaves (`moons#3.x[1]`). The graph is
built once, and every training step only rebinds the slots to new rows
(mini-batches). Batch size is fixed for a run (`--batch`, default: the whole
dataset).

### D11. The file says *what*, the CLI says *how*

The `.hsg` file contains no training settings and no commands. Learning rate,
schedule, steps, seed, batch size, output paths and which derivatives to
compute are CLI flags.

## 5. Command-line interface

```
hsgrad check  FILE                               parse + check, report all errors
hsgrad eval   FILE [--set NAME=VALUE ...] [--show NAME ...]
hsgrad derive FILE --of NAME --wrt NAME[,NAME...] [--simplify] [--dot OUT] [--format dot|text]
hsgrad plot   FILE --of NAME --dot OUT [--cluster]
hsgrad train  FILE [--steps N] [--lr X] [--lr-schedule const|linear:END|step:K:F]
                   [--batch N] [--seed N] [--report-every N] [--save-params OUT]
hsgrad predict FILE --params PARAMS [--set NAME=VALUE ...]
```

`train` requires a `loss` declaration, and `predict` requires a `model`
declaration (see `01-language.md`). `eval`, `derive` and `plot` also take
`--set`, `--params` and `--seed`; every command takes `--color
auto|always|never`. The full reference, with the parameter file format,
is `16-cli.md` §3.

## 6. Error handling

All user-facing errors go through one `Diagnostic` type: a code
(`E0302`), a severity, a message, a primary span with a label, optional
secondary spans, and notes. One renderer prints them Rust/Elm style
(`11-diagnostics.md`):

```
error[E0302]: shape mismatch in matrix product
  --> examples/moons.hsg:14:17
   |
14 |   let h2 = relu(W2 @ h1 + b2) in
   |                 ^^^^^^^ Mat 16 16 @ Vec 8
   |
   = note: h1 : Vec 8 because W1 : Mat 8 2 (declared at 6:7)
```

The checker collects **all** independent errors instead of stopping at the
first. Error codes are listed in `01-language.md` §10.

## 7. Testing strategy

- **hspec** for unit tests, **QuickCheck** for properties.
- **Gradient check**: on random graphs and random points, numeric backprop ≈
  central finite differences.
- **Symbolic vs numeric**: evaluating the symbolic gradient graph equals the
  numeric gradient.
- **Simplifier soundness**: evaluating before and after simplification gives
  the same value on random inputs (away from singularities).
- **Parser round-trip**: pretty-print the AST, parse it back, get the same AST.
- **Golden tests** for diagnostics: each `test/golden/<stage>/*.hsg` has an
  expected `*.err` output, so error message quality is under test
  (`11-diagnostics.md` §5 explains the harness).
- **End-to-end**: `moons.hsg` trains to ≥ 95% accuracy with a fixed seed.

## 8. Dependencies

Kept small on purpose. Most things are written by hand because this is a
learning project.

| Package | Used for |
|---|---|
| `base`, `containers`, `mtl`, `text` | everything |
| `megaparsec`, `parser-combinators` | parser, operator precedence |
| `optparse-applicative` | CLI |
| `random` | parameter initialisation, mini-batch sampling |
| `filepath` | CSV paths relative to the `.hsg` file |
| `hspec`, `QuickCheck` | tests |

Hand-written: the Graphviz DOT writer, the CSV reader, the diagnostics
renderer, and the simplifier.

Toolchain: GHC 9.10.3 and cabal 3.16 (via ghcup).

## 9. Roadmap

Every step = one module (or a tiny pair), its tests, its doc page, one commit,
and a review before moving on.

| Step | Modules | Doc | Milestone |
|---|---|---|---|
| 0 | — | 00, 01 | Architecture and language agreed |
| 1 | cabal skeleton, test suite | 02 | `cabal build && cabal test` pass |
| 2 | `Core.Op` | 03 | Every op has a forward and a derivative rule |
| 3 | `Core.Graph` | 04 | Builder with sharing and hash-consing |
| 4 | `Core.Eval` | 05 | Forward pass on hand-built graphs |
| 5 | `Core.Backprop` | 06 | Gradients match finite differences |
| 6 | `Viz.Dot` | 07 | micrograd-style `draw_dot` pictures |
| 7 | `Core.Symbolic` | 08 | Gradient graphs agree with numeric backprop |
| 8 | `Core.Simplify`, `Core.Pretty` | 09 | `∂L/∂a = -2·b` shows up simplified |
| 9 | `Train.*` | 10 | An MLP learns XOR, driven from Haskell code |
| 10 | `Lang.Span`, `Lang.Diagnostic` | 11 | Snippet renderer, golden tests |
| 11 | `Lang.Syntax`, `Lang.Parser` | 12 | All example files parse, good parse errors |
| 12 | `Lang.Shape`, `Lang.Check` | 13 | Name and shape errors with all locations |
| 13 | `Data.*` | 14 | CSV and inline datasets |
| 14 | `Lang.Compile` | 15 | `.hsg` → graph equals the hand-built graph |
| 15 | `Driver`, `app/Main` | 16 | All CLI commands work |
| 16 | examples, end-to-end tests | 17 | `moons.hsg` trains from the CLI |
| later | tensor backend, Adam, user-defined ops | — | — |
