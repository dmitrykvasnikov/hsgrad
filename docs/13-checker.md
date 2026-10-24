# 13 — `Lang.Shape` and `Lang.Check`: names and shapes

The parser (step 11) turns text into a tree; it knows nothing about what
the names mean or whether the shapes fit. This step adds the **checker**,
the last stage that can reject a program. Two modules:

- `HsGrad.Lang.Shape`: the **types** of the language (`Real`, `Vec d`,
  `Mat r c`), templates with **shape variables** (`Vec n`), **matching**
  argument shapes against templates, the shape rules of the operators
  (broadcasting, `@`, indexing), and the types of the **built-in
  functions**. It knows nothing about names or source positions.
- `HsGrad.Lang.Check`: walks a parsed program, resolves every **name**,
  gives every expression a concrete **shape**, checks every **call** of a
  function "template style" (D8), and reports every independent error
  with codes E0201–E0207 and E0301–E0314, plus the warnings W0201–W0203.
  Its output is a **checked program**: resolved, fully typed, with every
  function instance spelled out, so that the compiler (step 14) can lower
  it to a graph without asking any question twice.

The milestone of the roadmap is "name and shape errors with all
locations". The four example programs check without a single diagnostic,
with the expected types (`model mlp : Vec 2 → Real`, `loss : Real`), and
the error of the architecture document now comes from the checker itself
(`test/golden/check/matrix-product.err`):

```
error[E0302]: shape mismatch in matrix product
  --> matrix-product.hsg:12:17
   |
12 |   let h2 = relu(W2 @ h1 + b2) in
   |                 ^^^^^^^ `Mat 16 16 @ Vec 8`
   |
   = note: h1 : Vec 8 because W1 : Mat 8 2 (declared at 3:7)
```

The note is computed: the checker traces the 8 of `h1`'s type back through
`relu`, `+` and `@` to the declaration of `W1`.

---

## 1. Theory

### 1.1 What a checker does

A parser checks **form**: `let y = x +` is not a program. A checker checks
**meaning**: `let y = x + w` is a perfectly formed program that may still
be wrong, because `x` doesn't exist, or `w` is a function, or `x` is a
`Vec 3` and `w` a `Vec 2`. These are the language's *static semantics*
(01-language §7): rules about a program that can be decided without
running it.

A checker could stop at "yes" or "no". Ours does more: on "yes" it hands
over an **elaborated** program, the same program with everything the
checker found out written into it. That is common in compilers (GHC's
type checker produces a typed core language, for instance) and it has a
practical reason here: the compiler must know, at every node, the shape of
the value, which declaration a name refers to, whether `sum` gets a vector
or a comprehension, and which body a call inlines. Each of those is a
question the checker already answered.

### 1.2 Shapes are types

In most languages a vector's length is a run-time property. Here it is
part of the type (decision D5): `Vec 3` and `Vec 2` are different types,
and `x + w` with those types is rejected before anything runs, just like
`1 + "a"` in a typed language. Types that mention values (here: sizes) are
called *dependent* or *indexed* types. Ours are the simplest kind: every
size is a natural number known at check time, so checking needs nothing
but arithmetic on those numbers.

The rules come in three forms:

- **Fixed rules** for the operators: `Mat m n @ Vec n = Vec m`;
  `e[i] : Real` for `e : Vec n` if `i < n`.
- **Broadcasting** for elementwise operations (§4.1): equal shapes, or a
  `Real` against anything. Nothing else: `Vec n + Mat r n` is an error,
  not a guess.
- **Signatures** for functions, possibly with shape variables:
  `layer(W : Mat m n, b : Vec m, x : Vec n) : Vec m`.

### 1.3 Shape variables and matching

A signature with shape variables describes a whole family of types: one
for every value of `m` and `n`. To check a call, the checker must find
the member of the family that fits the arguments. With arguments
`Mat 16 2, Vec 16, Vec 2`:

```
Mat m n  against  Mat 16 2   →  m = 16, n = 2
Vec m    against  Vec 16     →  m = 16   ✓ (already 16)
Vec n    against  Vec 2      →  n = 2    ✓
result   Vec m               →  Vec 16
```

This is **matching**: one side (the template) has variables, the other
(the argument) is concrete, and variables get values by reading them off.
It is the one-sided special case of *unification*, the algorithm at the
heart of Hindley–Milner type inference, where both sides may contain
variables. One-sided matching can't fail in subtle ways: a variable is
either unbound (bind it), or bound to the same size (fine), or bound to a
different one, and then two specific arguments disagree. That is E0303,
and it can name both:

```
error[E0303]: shape variable `n` bound inconsistently
  --> shape-variable.hsg:12:22
   |
7  | fn neuron(w : Vec n, b : Real, x : Vec n) : Real =
   |               -----                ----- both parameters have length `n`
...
12 | let y = neuron(w, b, x)
   |                -     ^ `Vec 2`, so `n = 2` here
   |                |
   |                `Vec 3`, so `n = 3` here
```

To point at the first argument, a binding remembers which argument made
it (`Binding { bindingSize, bindingArgument }`).

### 1.4 Template-style checking

Matching gives the result type of a call, but the body of the function
must be checked too, and its shapes depend on the shape variables. There
are two classic ways:

1. **Check the body once, generically**, with `m` and `n` as unknowns,
   proving it correct for all sizes. This needs reasoning about symbolic
   sizes ("is `m` equal to `n + 1`?"), which is where the real complexity
   of dependent types lives.
2. **Check the body once per use**, with the variables replaced by the
   sizes of that call. This is how C++ templates work: a template is
   checked when it is *instantiated*.

D8 picks the second. The language has no recursion (D9) and every call
is inlined anyway (01-language §8), so there are finitely many
instantiations, each with concrete sizes, and checking a body with
concrete sizes needs nothing but the fixed rules. One instance is checked
per distinct list of argument shapes: `moons.hsg` calls `layer` three
times, with three different shape lists, so it has three **instances**;
calling it again with `Mat 16 2, Vec 16, Vec 2` reuses the first.

An error in an instance is found while checking a call, but it is
*located* in the body, so the message points at the body and adds where
the instance came from, one note per call on the way
(`test/golden/check/instance.err`):

```
error[E0302]: shape mismatch in matrix product
 --> instance.hsg:7:12
  |
7 | fn sq(z) = z @ z
  |            ^^^^^ `Vec 3 @ Vec 3`
  |
  = note: z : Vec 3 because y : Vec 3 (declared at 5:7)
  = note: while checking the call at instance.hsg:8:15
  = note: while checking the call at instance.hsg:11:9
  = hint: for the inner product of two vectors write `dot(u, v)`
```

The known weakness of template checking is the uncalled template: its
errors go unnoticed. §2.3 shows how much of that this checker recovers.

### 1.5 Many errors, no cascades

Reporting every error in one run (01-language §7.6) raises the same
problem the parser had (12 §1.6): one mistake must not cause a stream of
follow-up errors. If `W` has a broken type, then `W @ x` has no shape,
and neither does `tanh(W @ x + b)`, nor `sum` of it; none of those are
new mistakes.

The standard cure in type checkers is an **error type** (also called
*poison*): a type that is compatible with everything and silences every
check it takes part in. Here it is the `Unknown` case of what the checker
knows about an expression (§2.4). A name whose declaration failed is
**poisoned**: its kind (value, function, dataset) is still known, so
calling it is still an error, but its shape is `Unknown`, and every
expression that uses it is `Unknown` without a word.

Annotations work in the other direction, as **firewalls**: a declaration
whose body is broken but whose type is written down still has that type.
`fn layer(v : Vec 3) : Vec 4 = v @ v` is wrong inside, but every call of
it is a `Vec 4`, so the rest of the program is still checked
(`test/golden/check/no-cascade.err`). Only the errors that remain
independent are reported.

### 1.6 Explaining a shape

"`Mat 16 16 @ Vec 8`" says what is wrong, but not why `h1` is a `Vec 8`,
which is the real question when the mistake is elsewhere. Every shape
rule says where each result dimension comes from: the rows of `W @ x` are
the rows of `W`; the length of `a + b` is that of either operand; `relu`
keeps its argument's dimensions. Following these rules backwards from a
name to a declaration that *states* a size (a `param`, an `input`, an
annotated parameter) answers the question. This is a small form of what
type-error research calls *blame tracking*: carrying, for each part of a
type, the source location responsible for it.

## 2. Design

### 2.1 One `Shape` type for sizes and templates

```haskell
data Shape d = Real | Vec d | Mat d d          -- Functor, Foldable, Traversable

data Dimension = Fixed Int | Variable Text
type Template = Shape Dimension                 -- Vec n, Mat 16 n
```

A concrete shape is a `Shape Int`; a declared type is a `Template`. The
parameter makes the conversions one-liners: `fmap Fixed` turns a shape
into a template, `traverse` turns a template without variables back into
a shape (`toShape`), and `substitute` is a `traverse` that looks variables
up. The constructor `Real` lives in the namespace of data constructors,
so it doesn't clash with the Prelude class `Real`.

Besides shapes there are the internal sorts of 01-language §4: a
comprehension's result `Seq t` (`Operand = Value s | Seq s`, which only
the built-in functions see), the kinds of names (`Kind`), and function
signatures (`Signature`).

### 2.2 The checked program

This is the contract with step 14, so it is designed for the compiler,
not for the checker.

```haskell
checkProgram :: Program Span -> ([Diagnostic], Maybe Checked)

data Checked = Checked
  { checkedDecls     :: [CheckedDecl]              -- source order
  , checkedInstances :: Map InstanceKey Instance   -- every function body that is called
  }

data CheckedDecl
  = CInput InputInfo | CParam ParamInfo | CData DatasetInfo | CLet LetInfo
  | CFunction FunctionInfo | CModel ModelInfo | CLoss LossInfo
```

The diagnostics are sorted; the `Checked` program is there exactly when
none of them is an error (warnings don't prevent it). Declarations stay
in source order because a `let` may refer to earlier ones; accessors
(`checkedParams`, `checkedModel`, `checkedLoss`, …) pick one sort.

Each declaration record has what the compiler and the driver need and
nothing they would have to recompute:

| Record | Fields |
|---|---|
| `InputInfo` | name, name span, declaration span, `Shape Int`, default as row-major `[Rational]` |
| `ParamInfo` | name, spans, shape, `Init` = `InitFill c` / `InitValues [c]` / `InitUniform lo hi` / `InitNormal mean std` |
| `DatasetInfo` | name, spans, fields `[(Text, Shape Int)]`, the unloaded `DataSource Span` |
| `LetInfo` | name, spans, shape, typed expression |
| `FunctionInfo` | name, span, `Signature`, the `InstanceKey`s it is called with |
| `ModelInfo` | name, span, parameters with shapes, result shape, its `InstanceKey` |
| `LossInfo` | keyword span, declaration span, typed expression (always `Real`) |

Every expression is a **typed tree**:

```haskell
data TExpr = TExpr { texprSpan :: Span, texprShape :: Shape Int, texprNode :: Node }

data Node
  = NNum Rational
  | NRef Ref                       -- RInput | RParam | RLet | RLocal, by name
  | NVector [TExpr]                -- Vec of Reals, or Mat of rows (the shape says which)
  | NIndex TExpr Int (Maybe Int)
  | NNegate TExpr
  | NPower TExpr Rational
  | NBinary Operator TExpr TExpr   -- @, or elementwise with a Real broadcast
  | NBuiltin Builtin [TExpr]       -- exp, max, dot, transpose, sum of a value, …
  | NReduce Builtin Iteration      -- sum / mean of a comprehension
  | NCall InstanceKey [TExpr]      -- inline that instance's body
  | NLet Text TExpr TExpr

data Iteration
  = OverData Text [Text] TExpr     -- dataset, one local per field, body
  | OverParams Text [(Text, TExpr)] -- local, body checked once per parameter

data InstanceKey = InstanceKey { instanceFunction :: Text, instanceArguments :: [Shape Int] }
data Instance = Instance { instanceParams :: [Text], instanceBody :: TExpr, instanceResult :: Shape Int }
```

Some choices worth explaining:

- **A separate tree, not annotations on `Expr`.** Step 12's doc
  suggested `Expr (Span, Shape)`. That would type every node, but it would
  leave the questions the tree can't express: a `Var` would still be text
  (input? local? which?), `Call` would not say "built-in" or "this
  instance", and `sum [ … | p <- params ]` would have one body for six
  different parameter shapes. The typed tree has one constructor per thing
  the compiler does differently.
- **Calls name an instance.** `NCall key args` says "inline the body of
  `checkedInstances ! key` with its parameters bound to `args`". Bodies are
  shared between calls with the same shapes, so the output doesn't grow
  with the number of calls, only with the number of distinct
  instantiations.
- **Locals are referred to by name.** Lexical scoping is the same in the
  compiler as in the checker: a `NLet x`, a comprehension or an instance
  binds `x` in an environment, the innermost binding wins. Unique ids would
  only matter if the compiler moved code between scopes, which it never
  does.
- **`params` is already expanded.** `OverParams "p" [("W1", body₁), …]` has
  the body once per parameter, each typed with that parameter's shape, in
  declaration order, and *all* parameters (even ones declared after the
  comprehension, 01-language §3).
- **Data stays out.** A dataset's rows are step 13's business; the checker
  only checks the schema and passes the source through.

A test helper, `lint` in `CheckSpec`, recomputes every node's shape from
its children with the rules of `Lang.Shape` and checks references and
instances. The checked examples and hundreds of mutated programs pass it
(§5.3), so the compiler may trust every `texprShape`.

### 2.3 Functions: checked at the declaration, and per instance

A `fn` whose parameters all have concrete types is the easy case: there
is only one possible instance, so it is checked **at its declaration**,
like a `model`, and that check is stored as its instance. Calls only match
their arguments against the signature.

A generic function is also checked at its declaration, with the
parameters whose types involve shape variables (or have no type) bound to
`Unknown`. Thanks to poisoning (§1.5) that check produces no false errors:
every expression that depends on an unknown shape is silently unknown.
What it does find is everything else: undefined names, kinds, arities,
patterns, shadowing, unused locals, and shape errors among known shapes
(`[1, 2] + [1, 2, 3]` is wrong for every instance). So D8's trade-off
shrinks to: *errors that depend on shape variables are only found at
calls*. (`00-architecture.md` D8 now says so.)

At a call with known argument shapes and no matching problem, the
instance is looked up in a memo (`stInstances`, keyed by `InstanceKey`) or
checked now. A function whose declaration had errors is never
instantiated: its calls get its result annotation, if it has a concrete
one, and nothing otherwise.

Because names were checked at the declaration, an instance check is
**quiet**: it reports only errors that can differ between instances (shape
errors, and E0314), never a name error or a warning a second time. Quiet
scopes also attach their **context notes** ("while checking the call at
…") to exactly those errors.

### 2.4 What the checker knows about an expression

```haskell
data Typed = Unknown | ShapeOnly (Shape Int) | Full TExpr
```

`Unknown` is the poison of §1.5. `ShapeOnly` is the firewall: the shape is
known from an annotation although the tree below is broken (a call of a
broken function, an annotated `let` with an error inside). `Full` is a
finished typed tree. A node is built with

```haskell
node :: Span -> Shape Int -> Maybe Node -> Typed
node sp s = maybe (ShapeOnly s) (Full . TExpr sp s)
```

so a node is `Full` exactly when all its parts are. Without errors every
expression ends up `Full`; `checkProgram` treats anything else as a bug
in the checker (it calls `error`) rather than silently returning no
program.

### 2.5 Scopes

- **Globals** are threaded through the declarations in order: a
  declaration sees exactly the names declared before it (D9). A function
  keeps the globals of its declaration (`fnGlobals`), so its instances see
  the same names whenever they are checked.
- **Locals** (function parameters, `let … in`, comprehension variables)
  are a map from name to shape, searched first.
- **Built-in functions** come last. A top-level declaration can't reuse
  their names (E0202); a local may shadow one (W0201).
- Before the walk, `fileInfo` collects every top-level name with its
  declaration, so a name that is used too early gets E0203 ("declared
  later, here") instead of a misleading E0201, and every `param`, for
  `params`.

### 2.6 Warnings

- **W0201 (shadowing)** for `let … in` and comprehension variables that
  hide any other name. Not for function parameters: `neuron.hsg` declares
  `fn neuron(w : Vec n, b : Real, x : Vec n)` next to the globals `w`, `b`,
  `x`, and calls `neuron(w, b, x)`. A signature is its own scope, and
  warning about the normal style would teach people to ignore warnings
  (01-language §7.1 changed accordingly; step 10's golden `several.err`
  already showed it this way).
- **W0202 (unused)** for inputs, parameters, datasets, functions, and
  locals that nothing refers to. Never for a top-level `let` or the
  `model`: they are outputs (`eval --show L`, `derive --of L`, `predict`),
  and `intro.hsg` would otherwise warn about `L`. Names starting with `_`
  are exempt, which is how a comprehension pattern says "this field is not
  needed": `(_x, y) <- d`.
- **W0203 (can't affect the loss)** is decided here, see §4.4.

## 3. Code walkthrough

### 3.1 `Lang.Shape`: matching

```haskell
matchArguments :: [Maybe Template] -> [Maybe (Shape Int)] -> (Bindings, [MatchProblem])
```

Arguments are matched left to right. `Nothing` as a template means "no
annotation, anything goes"; `Nothing` as a shape means "unknown, skip it"
(poison). For each argument, the template and the shape are paired up
dimension by dimension (`pairUp`); a different kind of shape or a wrong
fixed dimension is `ArgumentMismatch i` (E0304) and binds nothing, since
the sizes of a wrong argument are not to be trusted. Each variable is then
bound, or checked against its binding, giving `InconsistentVariable v
first conflicting` (E0303). Every problem is collected.

`matchShape` is the one-shape version used for result and `let`
annotations, where there is no argument position to remember.

### 3.2 `Lang.Shape`: operations and built-ins

`broadcast`, `matMul` and `indexShape` are direct transcriptions of
01-language §4.1, §6 and §5.3; they return `Maybe` or an `IndexProblem`
and leave the wording of messages to the checker.

The built-in functions are typed in four ways:

```haskell
data BuiltinType
  = Elementwise                          -- exp, log, tanh, relu, step, sigmoid, sqrt, abs
  | Broadcasting                         -- max, min
  | Reduction                            -- sum, mean
  | Matching [(Text, Template)] Template -- dot, transpose
```

`dot(u : Vec n, v : Vec n) : Real` is an ordinary signature and is checked
by `matchArguments`, like a user function; the other three are not
expressible as signatures (`exp` works on *any* shape) and are rules.
`applyBuiltin :: Builtin -> [Operand] -> Either [CallProblem] (Shape Int)`
applies them. A test reads the table of 01-language §6 and checks that it
names exactly these functions.

### 3.3 `Lang.Check`: the declarations

`checkProgram` runs `checkDeclarations` in a `State` monad that collects
diagnostics, counts errors, records used names and memoises instances.
Each declaration:

1. `declare`s its name: E0202 if it repeats an earlier name or a
   built-in function, in which case the first declaration stays the one
   that counts (the second is still checked, for its own errors).
2. Checks its annotations (`annotation`, `concreteType`): E0309 for a
   dimension 0, and for shape variables where the type must be concrete
   (`input`, `param`, `data`, a top-level `let`, a `model`).
3. Checks its body, default or initialiser.
4. Adds itself to the globals, poisoned if its shape is unknown.

`LossDecl` and `ModelDecl` additionally check that there is at most one of
each (E0206, pointing at both); the loss must be `Real` (E0307).

### 3.4 Expressions

`checkExpr :: Scope -> Expr Span -> Check Typed` has one case per form.
The interesting ones:

- `Var`: `resolve` looks the name up (locals, globals, built-ins) and
  records the use; a function or a built-in used as a value is E0204, a
  dataset E0313. A name that isn't in scope gets E0203 if it is declared
  later in the file (with the recursion note when it is the declaration
  being checked), and otherwise E0201 with a suggestion (§3.8).
- `Binary`: `matMul` for `@` (E0302, with hints for `Vec @ Vec`, a
  scalar, or a vector on the left), `broadcast` for the rest (E0301,
  with a hint when `*` was meant to be `@`).
- `Index`: `indexShape`; E0305 points at the offending index, E0306 at
  the brackets.
- `Vector`: `vectorShape` decides between `Vec n` and `Mat r c`, or
  reports E0312 (rows of different lengths, numbers mixed with rows, a
  matrix inside). Literals in declarations use the same function.
- `Let`: the bound expression, the annotation (matched with the shape
  variables of the enclosing function, if any), then the body with the
  new local (`withLocals`, which also produces W0201 and W0202).

### 3.5 Calls

`checkCall` resolves the callee first, because it decides how the
arguments are read: `sum` and `mean` with a single comprehension argument
are a `reduction`; any other comprehension argument is E0311. Then:

- a built-in function goes to `callBuiltin` (arity E0205, then
  `applyBuiltin`);
- a `fn` or the `model` goes to `callFunction`: arity, `matchArguments`,
  and if everything is known and fine, the instance:

```haskell
outcome <- case Map.lookup key instances of
  Just o -> pure o
  Nothing -> do
    o <- runBody (scopeFile scope) (Calling notes) fn (map Just shapes) origins (bindingSize <$> bindings)
    modify' (\st -> st {stInstances = Map.insert key o (stInstances st)})
    pure o
```

`runBody` is shared by declarations (`Declaring`) and calls
(`Calling notes`): it binds the parameters, checks the body, and matches
the body's shape against the result annotation, whose variables may
already have values from the arguments (`Vec m` with `m = 16`) or get one
from the body (a result variable that no parameter mentions). A mismatch
is E0308, phrased "expected `Vec m`, which is `Vec 16` here".

`reportMatch` turns `MatchProblem`s into E0303 / E0304 for user and
built-in functions alike. For a user function it points at the parameter
annotations; a built-in function has none, so its signature goes in a
note.

### 3.6 Comprehensions

`comprehension` checks the source and the pattern and returns the body's
shape (the `t` of `Seq t`) and the typed `Iteration`.

- **Over a dataset**: the source must be a dataset (E0204 otherwise), the
  pattern must bind one name per field (E0207, with the right pattern as
  a hint), and the fields' shapes become the locals' shapes. Afterwards
  `nestedData` looks for another comprehension over a dataset in the typed
  body, also inside the instances it calls (E0314, §4.5).
- **Over `params`**: the body is checked once per parameter. Only the
  first iteration reports name errors and warnings; the others are quiet.
  After the first iteration with an error the rest are skipped (they would
  mostly repeat it), and that error carries the note "while checking the
  body with `p` = `W1 : Mat 16 2`". If the iterations give different
  shapes, E0310 names two parameters that differ.

### 3.7 Explanations

```haskell
dimOrigins :: Scope -> Map InstanceKey Outcome -> TExpr -> [Maybe Origin]
```

returns, for each dimension of an expression, the declaration it comes
from (§1.6). It walks the typed tree with the rules of §3.2 read
backwards (`take 1` of the left operand for the rows of `@`, `reverse`
for `transpose`, either operand for elementwise operations), through
`let`s and into instance bodies. Locals store their origins when they are
bound, as a **lazy** field: the list is only ever computed if an error
needs it.

`becauseNotes` turns that into the note, for operands that are names
whose shape came from somewhere else:
`h1 : Vec 8 because W1 : Mat 8 2 (declared at 3:7)`. Which dimension is
explained depends on the error: the inner dimensions for `@`, the first
differing one for elementwise operations, the dimension where the
variable sits for E0303.

### 3.8 Suggestions

`editDistance` is the **optimal string alignment** distance: insertions,
deletions, substitutions and swaps of neighbouring characters each cost 1
(`tnah` → `tanh` is 1, not 2). It is the textbook dynamic programme over a
table indexed by prefixes; the table is a lazy `Data.Map` whose cells
refer to each other, so each cell is computed once, on demand.
`suggestName` picks the closest candidate within a third of the name's
length (at least 1, and fewer edits than the name has characters, so `y`
never suggests `x`). Candidates are the locals, the globals in scope and
the built-in functions; names declared later are not candidates, they get
E0203.

### 3.9 W0203

`lossInfluence` computes the set of parameters the loss depends on, from
the typed tree: a `RParam` reference, a `let` (its set, computed once per
`let` in order), a local (the set of what it was bound to), a call (the
instance body with its parameters bound to the arguments' sets), and a
comprehension over `params`, where the body for parameter `p` depends on
`p` only if it uses the variable. Every used parameter outside the set
gets W0203; an unused one already has W0202.

## 4. Design decisions

### 4.1 Two modules, and where the line is

`Lang.Shape` is pure rules: shapes in, shapes or problems out. It has no
spans and no names beyond shape variables, so it can be tested
exhaustively without writing programs (every built-in on every kind of
shape, `broadcast` symmetric, …). Everything about locations and words is
in `Lang.Check`. The problem types (`MatchProblem`, `CallProblem`,
`IndexProblem`) are the interface: precise enough for the checker to
choose a message, with no text in them.

### 4.2 A memo of instances keyed by argument shapes

Two calls with the same argument shapes have the same instance, so the
key is just the function and the shapes. It makes repeated calls cheap,
reports an instance's errors once, and is exactly the information the
compiler needs. An instance that failed is memoised too (as `Failed`,
with the result shape if an annotation gives it), so its errors are not
reported again either.

### 4.3 Explanations computed lazily

The alternative to §3.7 is to carry an origin inside every dimension
(`Shape (Int, Origin)`) and let every rule propagate it. That makes the
`Shape` functions polymorphic in what a dimension is, and makes every
equality test of sizes ignore half of the data. Since origins are only
needed when an error is reported, the checker keeps plain `Shape Int`
everywhere and recomputes origins from the typed tree on demand. The cost
is that the walk mirrors the shape rules: a new operator needs a line
there too, or its errors simply lose their "because" note.

### 4.4 W0203 here, at the level of declarations

The question "can this parameter affect the loss?" has two natural
places to be answered:

- **After compiling**: a parameter leaf not in `reachable g [loss]` has a
  structurally zero gradient. This is exact per *element*: it would also
  notice that only row 0 of a matrix is ever used.
- **In the checker**, on the typed tree, per *declaration*.

The checker is cheaper (no graph, the same walk as everything else), can
point at the declaration with a normal diagnostic, and runs for `check`,
which never compiles. So W0203 is done here, per declaration. The
element-level question is left to the compiler, which can answer it with
`reachable` if a later step wants it.

### 4.5 Codes for the corners

The catalogue had no code for a few cases the spec does forbid; they were
fitted to the nearest code, and the catalogue text was widened to say so:

- **E0309** ("type not allowed here") covers every bad annotation: shape
  variables outside `fn` signatures and bodies, an unannotated `model`
  parameter, a dimension 0 (`Vec n` needs `n ≥ 1`, §4), and a `Mat`
  dataset field (§3: fields are `Real` or `Vec n`).
- **E0310** also covers a comprehension over `params` in a file with no
  `param`: with no iteration, the body has no shape.
- **E0312** covers every malformed vector literal, not only rows of
  different lengths.
- **E0201** also reports an unknown initialiser (`~ gauss(0, 1)`), with a
  suggestion from `uniform` and `normal`.
- **E0314** is new: "comprehensions don't nest over datasets" (§5.6) was a
  rule without a code. Nesting is found on the typed tree, so it is also
  caught through a function call (`mean [ g(x) | x <- d ]` where `g` itself
  iterates over a dataset).

### 4.6 Changes to the language spec

Genuine gaps found while writing the checker, fixed minimally in
`01-language.md`:

- **`sum` and `mean` of a `Real`** (§6). The table only had `Vec`, `Mat`
  and `Seq`, so the regularisation idiom of `moons.hsg`,
  `sum [ sum(p ^ 2) | p <- params ]`, failed for any file with a `Real`
  parameter, such as `xor.hsg`'s `b2`. The sum of one number is now that
  number.
- **Shadowing** (§7.1). The spec warned about function parameters that
  shadow globals, which `neuron.hsg` itself does (§2.6). Now only
  `let … in` and comprehension variables warn, and a top-level name that
  repeats a built-in function is an error.
- **Function checking** (§4.2): concrete functions are checked at their
  declaration, generic ones partly (§2.3).
- **Nesting** (§5.6): E0314 added; comprehensions over `params` may nest.
- The catalogue texts of E0309, E0310 and E0312 (§4.5), and the
  `codeSummary` lines in `Lang.Diagnostic` that go with them.

## 5. Tests

`test/HsGrad/Lang/ShapeSpec.hs`, `test/HsGrad/Lang/CheckSpec.hs`, and the
golden files in `test/golden/check/`.

### 5.1 Unit tests

- **Shape**: printing (shapes, templates, `Seq`, signatures, arrows);
  matching (bindings and which argument made them, E0303 between and
  within arguments, E0304, wrong arguments bind nothing, unknown and
  unannotated arguments); `matchShape`, `substitute`, `bindTemplate`;
  `broadcast` (cases, and as properties: symmetric, `s` with itself and
  with `Real`); `matMul`; `indexShape` with both errors; every built-in
  function on every kind of shape, arities, sequences only in `sum` /
  `mean`, signatures; the built-in table against 01-language §6.
- **Check, examples**: each example checks without any diagnostic;
  `describeChecked` gives the expected type of every declaration
  (`model net : Vec 2 → Real`, `fn neuron(…)` with its one instance, the
  three instances of `layer`); resolved names in `intro.hsg`; initialisers,
  defaults, schemas, the model's instance; the structure of `moons.hsg`'s
  loss, down to the six parameter bodies of `params`.
- **Check, rules**: define-before-use in function bodies, recursion,
  `params` including later parameters, shadowing rules, `_` names, outputs
  never unused, D8 (concrete functions checked uncalled, generic ones only
  at calls, independent errors at the declaration, one report per
  instance, nested call notes), firewalls, shape variables in local
  annotations, result variables bound by the body, E0314 through a call,
  W0203 through `let`s, calls and `params`, and the distance and
  suggestion functions.

### 5.2 Golden files

`test/golden/check/` has one file per group of codes, rendered exactly as
the driver will. A test checks that every E02xx, E03xx and W02xx code
appears in some `.err` file, and another that the checker reproduces the
hand-built E0201 of step 10 (`tabs.err`) character for character.

| File | What it pins down |
|---|---|
| `undefined-name` | E0201 with and without a suggestion, an unknown initialiser |
| `duplicate` | E0202 for declarations, fields, parameters, pattern variables, a built-in name |
| `before-declaration` | E0203, and recursion |
| `kinds` | E0204: a value called, a function and a built-in as values, a non-dataset source |
| `arity` | E0205 with the signature, user and built-in |
| `model-loss-twice` | E0206 for both |
| `pattern` | E0207 for a dataset and for `params` |
| `elementwise` | E0301 with a "because" note, `max`, the `*`-for-`@` hint |
| `matrix-product` | E0302, the architecture's example, and the `dot` / `*` hints |
| `shape-variable` | E0303 at a call, in `dot`, and within one argument (`Mat n n`) |
| `argument` | E0304, user and built-in |
| `index` | E0305 (vector, row, column, with a note) and E0306 |
| `loss-not-real` | E0307 |
| `annotation` | E0308 for defaults, initialisers, function results, `let`s, inside an instance |
| `not-concrete` | E0309 in all its forms |
| `params-shapes`, `no-params` | E0310, with the per-parameter note |
| `comprehension` | E0311 as a value and as an argument |
| `rows` | E0312 in literals and expressions; a broken default keeps its type |
| `dataset-value` | E0313 as a value, called, as an argument |
| `nested` | E0314, directly and through a call |
| `instance` | errors in instances, nested call notes, one report per instance |
| `warnings` | W0201, W0202, W0203 in a program that checks |
| `no-cascade` | five independent errors; poisoned names and firewalls |

### 5.3 Properties

- **Mutated examples.** A generator applies one to three random edits to
  an example file: delete a line, swap two lines (which breaks
  define-before-use), replace a name by another name of the file, change a
  number (sizes, indices). For every mutant that still parses: the checker
  doesn't crash (the rendered diagnostics are forced), it returns a
  checked program exactly when there is no error, and that program passes
  `lint`. 400 samples per run.
- **`lint`** recomputes the shape of every node of a checked program from
  its children (§2.2): references against declarations, `NCall` against
  the instance and its argument shapes, `OverParams` against the list of
  parameters. It also runs on the examples.
- **Edit distance** is symmetric and zero exactly for equal texts.

### 5.4 Testing the tests

Deliberate bugs, each tried on its own:

| Bug | Failing tests |
|---|---|
| `broadcast` compares element counts (`Vec 4 + Mat 2 2` accepted) | 2 |
| matching ignores fixed dimensions | 3 |
| `@` ignores the inner dimension of `Mat @ Vec` | 2 |
| index bound off by one | 2 |
| quiet scopes report name problems and warnings again | 1 |
| no memo of instances | 2 |
| an undefined name is a `Real` instead of unknown | 3 |
| explanations prefer the right operand | 1 |
| function parameters warn when they shadow | 7 |
| `params` means only the parameters declared so far | 1 |
| W0203 ignores instance bodies | 4 |
| E0314 not found through calls | 2 |
| result annotation checked without the arguments' bindings | 1 |
| no E0203, every early use is E0201 | 3 |

All are killed. The first run left two survivors, which led to two tests
(a warning in a generic function reported once; a result variable bound
by an argument and contradicted by the body) and to one simplification:
the quiet-scope mutant survived because `withLocals` also skipped its
warnings in quiet scopes, a second mechanism for the same thing. Only the
one in `report` is left. The `broadcast` mutant was at first only caught
by a random property; it now has a fixed case too.

The full suite has 557 tests and runs in about 3 seconds.

## 6. What later steps build on

- **Step 13 (`Data.*`)**: `checkedDatasets` gives each dataset's fields in
  schema order (`Real` or `Vec n` only) and its `DataSource Span`. The
  checker does **not** look inside the source: inline rows against the
  schema (E0404) and the CSV column mapping (a mapping for every field, no
  unknown field, `n` columns for a `Vec n`) are step 13's.
- **Step 14 (`Lang.Compile`)**: lower `checkedDecls` in order. Inputs and
  parameters become leaves named by their elements (`x[0]`, `W[1,0]`);
  `let`s are compiled once and referred to by name (`RLet`); an expression
  is lowered by `texprNode`, under `withSpan texprSpan`, with every shape
  already known: `NBinary MatMul` is matrix-vector or matrix-matrix by the
  operands' shapes, other `NBinary`s broadcast a `Real` operand, `NVector`
  is a `Mat` exactly when `texprShape` is. `NCall key args` inlines
  `instanceBody` of `checkedInstances ! key` with `instanceParams` bound to
  the compiled arguments. `NReduce b (OverData d xs body)` makes one copy
  of `body` per slot with `xs` bound to that slot's leaves (D10);
  `OverParams x bodies` one copy per parameter. `sum`/`mean` of a `Real`
  is the value itself. Never check anything again: `lint` (in the tests)
  is the contract.
- **Step 15 (`Driver`)**: `checkProgram` after a successful parse; render
  all diagnostics (sorted already), stop on `hasErrors`. `ParamInfo`'s
  `Init` maps onto `Train.Param`: `InitFill` → `Constant`, `InitUniform` /
  `InitNormal` directly; `InitValues` (an explicit literal) had no
  `Train.Param.Init` yet (step 14 added `Values`). `inputDefault` gives defaults for `eval`;
  `checkedModel` gives `predict` its inputs; `describeChecked` is a
  ready-made `check --verbose`.
