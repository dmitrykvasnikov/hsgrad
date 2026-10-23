# 02 — Project setup

This step adds the cabal project skeleton: a library, an executable and a
test suite, all of which build and pass before any real code exists. Every
later step adds one module (and its spec) to this skeleton.

There is no theory in this step. The page explains what each file is for,
why each setting was chosen, and the commands used day to day.

---

## 1. Layout

```
hsgrad/
├── cabal.project        how to build: compiler, index snapshot, tests on
├── hsgrad.cabal         what to build: the package and its three components
├── .gitignore
├── docs/                one page per roadmap step (this is 02)
├── src/                 the library
│   └── HsGrad.hs        placeholder top-level module (package version)
├── app/
│   └── Main.hs          the `hsgrad` executable (placeholder usage message)
└── test/
    ├── Main.hs          test-suite entry point: runs every spec
    └── HsGradSpec.hs    tests for HsGrad
```

Three directories, three **components** of one package:

| Component | Directory | Depends on | Purpose |
|---|---|---|---|
| library `hsgrad` | `src/` | `base` | All real code: `HsGrad.Core.*`, `HsGrad.Lang.*`, … |
| executable `hsgrad` | `app/` | library | The CLI. Only parses flags (layer rule, `00-architecture.md` §2) |
| test-suite `spec` | `test/` | library, `hspec`, `QuickCheck` | Unit tests and properties |

Putting everything into the library, and keeping the executable and the tests
as thin clients of it, is what makes the code testable: the tests import the
same modules the CLI uses, and nothing lives only inside `Main`.

## 2. `hsgrad.cabal`, line by line

### 2.1 Package description

```cabal
cabal-version:      3.4
name:               hsgrad
version:            0.1.0.0
...
build-type:         Simple
tested-with:        GHC == 9.10.3
extra-doc-files:    docs/*.md
```

- **`cabal-version: 3.4`** is the version of the *file format*, not of the
  cabal tool. It must be the first line, and it decides which fields and
  syntax the rest of the file may use. `common` stanzas, `import` and the
  leading-comma list style (`, hsgrad`) need at least 2.2. 3.4 is a
  modern, widely supported format that covers everything here; the newer
  formats that cabal 3.16 understands add features we don't need.
- **`version: 0.1.0.0`** follows the Haskell
  [PVP](https://pvp.haskell.org/): `A.B.C.D`, where `A.B` is the major
  version. It stays at 0.1 while the project is under construction.
- **`build-type: Simple`**: no custom `Setup.hs` build logic.
- **`tested-with`** is documentation for humans and CI tools: the compiler
  this package is known to build with.
- **`extra-doc-files`** ships the `docs/` pages with the source package, so
  the design notes travel with the code. A glob like `docs/*.md` picks up
  each new page automatically.

`synopsis`, `description`, `category`, `author` and `maintainer` are
metadata only. There is no `license` field yet: the project has no license
file, and choosing one is a separate decision.

### 2.2 The `common` stanza

```cabal
common shared
  default-language: GHC2021
  ghc-options:
    -Wall
    ...
  build-depends:    base ^>=4.20
```

A `common` stanza is a named block of settings. Each component pulls it in
with `import: shared` as its first line, so the language, warnings and the
`base` dependency are written once and every component agrees on them.

#### Language: `GHC2021`

`default-language` picks the set of language extensions that are on in every
module. `GHC2021` is a curated set of 46 stable, uncontroversial extensions
on top of Haskell 2010. The ones this project is most likely to use:

| Extension | What it enables | Where it helps us |
|---|---|---|
| `ImportQualifiedPost` | `import Data.Map qualified as Map` | Readable imports everywhere |
| `ScopedTypeVariables` | Type variables from a signature are visible in the body | Helpers inside `where` blocks |
| `GeneralisedNewtypeDeriving` | Deriving a newtype's instances from the wrapped type | `newtype NodeId = NodeId Int deriving (Eq, Ord)` |
| `BangPatterns` | Strict bindings: `go !acc (x:xs) = …` | Strict accumulators in Eval/Backprop |
| `TypeApplications` | `read @Double "2.5"` | Disambiguating numeric code |
| `NamedFieldPuns` | `f Node{op, args} = …` | Short record pattern matches |
| `TupleSections` | `(,x)` | Small conveniences |

Why not `GHC2024` (also available in GHC 9.10)? It adds `GADTs`,
`DataKinds`, `LambdaCase`, `DerivingStrategies` and a few others. We won't
need type-level programming, and `GADTs` switches on `MonoLocalBinds`, which
changes how local definitions are generalised and can produce confusing type
errors. `GHC2021` is the smaller, better-documented baseline. A module that
wants `LambdaCase` turns it on with a `{-# LANGUAGE LambdaCase #-}` pragma,
which also makes the choice visible where it is used.

#### Warnings

`-Wall` turns on most useful warnings: unused bindings and imports, missing
type signatures, incomplete pattern matches (including partial lambda and
`let` patterns like `\(Just x) -> …`, since GHC 9.2), name shadowing, and
more. The other flags add warnings that `-Wall` leaves out:

| Flag | Warns about | Why we want it |
|---|---|---|
| `-Wcompat` | code that will break in upcoming GHC versions | Cheap future-proofing |
| `-Widentities` | useless conversions such as `fromIntegral :: Int -> Int` | Numeric code has many conversions |
| `-Wredundant-constraints` | constraints in a signature that are never used | Keeps signatures honest |
| `-Wpartial-fields` | record fields that don't exist in every constructor | They make partial selector functions |
| `-Wmissing-export-lists` | modules without an explicit export list | Every module states its public API |
| `-Wunused-packages` | `build-depends` entries no module imports | Enforces "only the dependencies we need" (§4) |

Name shadowing deserves a remark: `-Wall` warns when a local name hides an
outer one. That is exactly the bug the `.hsg` checker warns about too
(W0201), so we keep it on in our own code.

**No `-Werror` in the cabal file.** Making warnings fatal in the package
description breaks the build for anyone using a newer GHC that adds a
warning. During development you can make them fatal locally instead, in an
ignored `cabal.project.local`:

```cabal
package hsgrad
  ghc-options: -Werror
```

#### `base ^>=4.20`

`^>=4.20` is the PVP "caret" bound, shorthand for `>=4.20 && <4.21`: any
4.20.x version, nothing newer. `base` is tied to the compiler (GHC 9.10 ships
`base-4.20`), so this bound also says "GHC 9.10".

### 2.3 The library

```cabal
library
  import:           shared
  hs-source-dirs:   src
  exposed-modules:  HsGrad
  other-modules:    Paths_hsgrad
  autogen-modules:  Paths_hsgrad
```

- **`hs-source-dirs: src`**: module `HsGrad.Core.Op` will live at
  `src/HsGrad/Core/Op.hs`.
- **`exposed-modules`**: the public modules. Every new library module must be
  added here. cabal does not find modules by itself, and GHC reports a missing
  one as "module … is not listed in the .cabal file" or an unresolved import.
- **`other-modules`**: modules that are compiled into the library but not
  exported to its users.
- **`Paths_hsgrad`** is a module cabal *generates* for every package. It
  contains `version :: Version`, taken from the `version:` field above. It
  must be listed both in `other-modules` (to be compiled) and in
  `autogen-modules` (so cabal knows not to look for a source file). We use it
  so the version number exists in exactly one place: the cabal file.

The placeholder module itself:

```haskell
module HsGrad
  ( version
  ) where

import Data.Version (showVersion)
import Paths_hsgrad qualified as Paths

version :: String
version = showVersion Paths.version
```

Later steps will put the real code in `HsGrad.Core.*` and the other module
families (`00-architecture.md` §3). `HsGrad` may then become a convenience
module that re-exports the most commonly used parts.

### 2.4 The executable

```cabal
executable hsgrad
  import:           shared
  hs-source-dirs:   app
  main-is:          Main.hs
  build-depends:    hsgrad
  ghc-options:      -rtsopts
```

- **`build-depends: hsgrad`** depends on the library of the same package.
  The executable sees only the library's `exposed-modules`.
- **`-rtsopts`** allows passing options to the GHC runtime system on the
  command line, e.g. `hsgrad train … +RTS -s` prints time, allocation and GC
  statistics. That will be useful when training on large unrolled graphs
  (D5). Without the flag, the RTS refuses most `+RTS` options.

`app/Main.hs` is a placeholder: `--help` (or no arguments) prints usage,
`--version` prints the version, anything else prints an error plus the usage
to stderr and exits with a failure code. Step 15 replaces it with the
`optparse-applicative` CLI described in `00-architecture.md` §5.

### 2.5 The test suite

```cabal
test-suite spec
  import:           shared
  type:             exitcode-stdio-1.0
  hs-source-dirs:   test
  main-is:          Main.hs
  other-modules:    HsGradSpec
  build-depends:
    , hsgrad
    , hspec       ^>=2.11
    , QuickCheck  ^>=2.19
```

- **`type: exitcode-stdio-1.0`**: the test suite is an ordinary program;
  exit code 0 means "passed". hspec's `hspec` function exits with a non-zero
  code if any test fails.
- **`other-modules`** lists every spec module. Like the library, cabal has to
  be told about each one.
- **Dependencies**: `hspec` for the test framework, `QuickCheck` for property
  tests. The leading-comma style (`, hsgrad`) makes adding or removing a line
  a one-line diff.

## 3. `cabal.project`

The `.cabal` file describes the *package*. `cabal.project` describes *how to
build* it in this checkout.

```cabal
packages: .
with-compiler: ghc-9.10.3
tests: True
index-state: 2026-10-20T17:20:44Z
```

- **`packages: .`**: the project has one package, in this directory.
- **`with-compiler: ghc-9.10.3`**: use exactly this compiler, even if `ghc`
  on the `PATH` points to another version. ghcup installs every GHC as
  `ghc-X.Y.Z`, so this name always resolves.
- **`tests: True`**: include test suites when resolving dependencies and in
  `cabal build all`. Without it, `cabal build all` silently skips the tests,
  and `cabal test` has to re-plan the build first.
- **`index-state`**: cabal resolves dependency versions against its local
  copy of the Hackage index, which changes on every `cabal update`. Pinning a
  timestamp makes cabal pretend the index stopped there, so the same versions
  get picked on every machine and every day, until we bump the line on
  purpose. (The timestamp must not be newer than the local index; if cabal
  complains, run `cabal update`.)

## 4. Dependencies

Only what this step needs:

| Component | Package | Why now |
|---|---|---|
| all | `base` | Prelude, `Data.Version` |
| test-suite | `hspec`, `QuickCheck` | The test framework |

The other packages in `00-architecture.md` §8 (`containers`, `mtl`, `text`,
`megaparsec`, `optparse-applicative`, `random`, …) are added by the step that
first imports them. `-Wunused-packages` would flag them if they were added
early.

**Version bounds.** Every dependency gets a caret bound on the major version
that was resolved when it was added (`hspec-2.11.18`, `QuickCheck-2.19.0.0`
today). By the PVP, a new major version may break the API, so the bound says
"this code was written against 2.11; a 2.12 needs a look first".

Bounds and `index-state` (§3) do different jobs:

- **Bounds** (in `hsgrad.cabal`) say which versions the code is *compatible*
  with. They are a property of the code.
- **`index-state`** (in `cabal.project`) picks *one* concrete set of versions
  within those bounds and keeps it stable. It is a property of this checkout.

Adding a dependency in a later step: add it to `build-depends` without a
bound, build, look up the version cabal picked (in the `Building …-X.Y.Z`
lines of the build output, or in `dist-newstyle/cache/plan.json`), then
write `^>=X.Y`. If the dependency
needs a version newer than the pinned index, bump `index-state` after a
`cabal update`.

## 5. How the tests are organised

```
test/Main.hs                    runs all specs
test/HsGradSpec.hs              tests for src/HsGrad.hs
test/HsGrad/Core/OpSpec.hs      (step 2) tests for src/HsGrad/Core/Op.hs
…
```

Convention: **one spec module per library module**, at the mirrored path,
with `Spec` appended. Each spec module exports a single `spec :: Spec`:

```haskell
module HsGradSpec
  ( spec
  ) where

import HsGrad (version)
import Test.Hspec (Spec, it, shouldSatisfy)
import Test.Hspec.QuickCheck (prop)
import Test.QuickCheck ((===))

spec :: Spec
spec = do
  it "has a non-empty version string" $
    version `shouldSatisfy` (not . null)

  prop "reversing a list twice gives it back" $ \xs ->
    reverse (reverse xs) === (xs :: [Int])
```

- `it "…" $ assertion` is a unit test (an hspec *example*).
- `prop "…" $ \x -> …` is a QuickCheck property: QuickCheck generates 100
  random inputs and reports the smallest counterexample it can find
  (*shrinking*) if one fails. `===` is like `==`, but on failure it prints
  both sides. These two tests mostly prove that the tooling works; real
  properties start in step 2.

`test/Main.hs` collects the specs explicitly:

```haskell
main :: IO ()
main = hspec $ do
  describe "HsGrad" HsGradSpec.spec
```

**Adding the tests for a new module** `HsGrad.Core.Op` takes three edits:

1. create `test/HsGrad/Core/OpSpec.hs` exporting `spec`;
2. add `HsGrad.Core.OpSpec` to the test suite's `other-modules`;
3. add `import HsGrad.Core.OpSpec qualified` and
   `describe "HsGrad.Core.Op" HsGrad.Core.OpSpec.spec` to `test/Main.hs`.

(There is a tool, `hspec-discover`, that generates `test/Main.hs` from the
file names and saves step 3. We keep the list explicit: it is one more line
per module, it needs no extra build tool, and the entry point stays a plain
Haskell file that does what it says.)

Golden tests for diagnostics (`00-architecture.md` §7) will add a
`test/golden/` directory in step 10.

## 6. Everyday commands

All commands run from the repository root.

```bash
cabal build all                  # library + executable + tests
cabal test                       # build and run the test suite
cabal test --test-show-details=direct   # stream hspec's output live
cabal run hsgrad -- --help       # run the executable; args after --
cabal repl hsgrad                # GHCi with the library loaded
cabal repl spec                  # GHCi with the tests loaded
```

Notes:

- The first build downloads and compiles `hspec`, `QuickCheck` and their
  dependencies into cabal's global store (`~/.local/state/cabal/store`);
  later builds reuse them. Build products of *this* project go into
  `dist-newstyle/` (ignored by git).
- **Running one group of tests**: hspec's `--match` (`-m`) option selects
  tests by path:
  `cabal test --test-options='-m "HsGrad/has a non-empty"'`.
- **Reproducing a QuickCheck failure**: hspec prints the seed it used;
  rerun with `--test-options='--seed N'`.
- **In the REPL**: `:r` reloads after editing, `:t expr` shows a type,
  `:i Name` shows a definition's info, `:q` quits. `cabal repl spec` then
  `main` runs the whole test suite from inside GHCi, which is the fastest
  edit–test loop.
- `cabal clean` deletes `dist-newstyle/` if the build ever gets confused.

## 7. `.gitignore`

| Pattern | What it is |
|---|---|
| `dist-newstyle/` | All cabal build output |
| `cabal.project.local*` | Machine-local cabal settings (e.g. `-Werror` above), and their backups |
| `.ghc.environment.*` | Package environment files for bare `ghc`, written only if asked for (`--write-ghc-environment-files`) |
| `*.prof`, `*.hp`, `*.eventlog` | Profiling and heap output from the GHC runtime (`+RTS -p/-h/-l`) |
| `*.dot`, `*.svg`, `*.png` | Graphviz output from `hsgrad derive/plot` |

`cabal.project` itself and a `cabal.project.freeze`, if one is ever
created, are committed: they are part of how the project builds.
