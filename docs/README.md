# The hsgrad docs

One page per roadmap step (`00-architecture.md` §9), each with the
theory, the design decisions, a code walkthrough and the tests of its
module. The numbers are the order the project was built in, which is
also a good order to read it in: every page builds only on the ones
before it.

## Where to start

- **To use hsgrad**: [`01-language.md`](01-language.md) (the `.hsg`
  language), then [`17-examples.md`](17-examples.md) (a tour of every
  example with real output), and [`16-cli.md`](16-cli.md) §3 for the
  command reference.
- **To learn how it works**: [`00-architecture.md`](00-architecture.md),
  then the pages in order. [`17-examples.md`](17-examples.md) is a good
  first or last stop: it shows what all the parts do together.

## Reading order

| Page | Module(s) | What it teaches |
|---|---|---|
| [00 — Architecture](00-architecture.md) | — | The big picture, layers, design decisions D1–D11, testing, the roadmap, open points (§10) |
| [01 — The `.hsg` language](01-language.md) | — | The language spec: declarations, shapes, expressions, grammar, the diagnostics catalogue |
| [02 — Project setup](02-project-setup.md) | cabal skeleton | The package, its components, warnings, pinned dependencies |
| [03 — `Core.Op`](03-core-op.md) | `HsGrad.Core.Op` | The scalar operations, forward functions and derivative rules |
| [04 — `Core.Graph`](04-core-graph.md) | `HsGrad.Core.Graph` | Graphs built in a monad: sharing, hash-consing, provenance |
| [05 — `Core.Eval`](05-core-eval.md) | `HsGrad.Core.Eval` | The forward pass, and why `NaN` is always an error |
| [06 — `Core.Backprop`](06-core-backprop.md) | `HsGrad.Core.Backprop` | Reverse-mode differentiation with numbers (micrograd's `backward`), gradient checks |
| [07 — `Viz.Dot`](07-viz-dot.md) | `HsGrad.Viz.Dot` | Drawing graphs with Graphviz |
| [08 — `Core.Symbolic`](08-core-symbolic.md) | `HsGrad.Core.Symbolic` | Reverse mode that builds the gradient as a graph |
| [09 — `Core.Simplify`](09-core-simplify.md) | `HsGrad.Core.Simplify`, `HsGrad.Core.Pretty` | Canonical sums of products, printing formulas |
| [10 — `Train`](10-train.md) | `HsGrad.Train.*` | Parameters, SGD and schedules, the training loop |
| [11 — Diagnostics](11-diagnostics.md) | `HsGrad.Lang.Span`, `HsGrad.Lang.Diagnostic` | Error messages with source snippets, golden tests |
| [12 — Parser](12-parser.md) | `HsGrad.Lang.Syntax`, `HsGrad.Lang.Parser` | The AST and a megaparsec parser with error recovery |
| [13 — Checker](13-checker.md) | `HsGrad.Lang.Shape`, `HsGrad.Lang.Check` | Names and shapes, template-style shape variables |
| [14 — Data](14-data.md) | `HsGrad.Data.Csv`, `HsGrad.Data.Dataset` | CSV files and inline datasets against a schema |
| [15 — Compile](15-compile.md) | `HsGrad.Lang.Compile` | Unrolling a checked program into a scalar graph |
| [16 — Driver and CLI](16-cli.md) | `HsGrad.Driver`, `HsGrad.Cli`, `app/Main.hs` | The pipeline, runtime errors, every command |
| [17 — Examples](17-examples.md) | `examples/`, end-to-end tests | A guided tour of every example, with real output |

Pictures are in [`img/`](img); each page says how its pictures were
made.
