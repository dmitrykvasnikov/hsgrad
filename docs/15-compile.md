# 15 — `Lang.Compile`: from a checked program to a scalar graph

The checker (step 12) has done all the thinking: every name is resolved,
every expression has a concrete shape, every function call names the
instance it inlines. What is left is mechanical, and that is the point of
this step: `HsGrad.Lang.Compile` **lowers** a checked program into the
scalar graph of `Core.Graph` (decision D5, `01-language.md` §8).

- Every expression becomes a **compile-time value**: one node id for a
  `Real`, a list of node ids for a `Vec n`, a list of rows for a
  `Mat r c`. Vector and matrix code is **unrolled**: `a + b` on two
  `Vec 3` is three scalar `+` nodes, `W @ x` is a sum of products per row.
- `input` and `param` declarations become **leaves** named by their
  elements (`x[0]`, `W[1,0]`); a comprehension over a dataset becomes `B`
  copies of its body over **slots** of input leaves (`moons#3.x[1]`,
  D10); a comprehension over `params` one copy per parameter.
- `let`s are compiled **once** and referred to by node id; function calls
  are **inlined**, with the parameters bound to the arguments' node ids.
- The built-in functions become core ops or the small formulas of §6:
  `sigmoid(e)` is `1 / (1 + exp(-e))`, `max(a, b)` is `a + relu(b - a)`.
- Every node records the span of the expression that created it, and the
  results of top-level `let`s carry their names as labels (D4).

The output is a `Compiled` record: the graph, and every handle the
commands of step 15 need — the nodes of each named value, the loss, the
parameters in the form `Train.Param` wants, the inputs with their
defaults, the dataset slots, accuracy pairs, and the model compiled into
a graph of its own for `predict`.

The milestone of the roadmap is "`.hsg` → graph equals the hand-built
graph". It holds literally: compiling `intro.hsg` and `neuron.hsg` gives
**exactly** (`==`, ids, spans and labels included) the graphs that steps
6–8 built by hand in `HsGrad.Test.Examples`, so `L = -8` and the
gradients `6, -4, -2` come out of the `.hsg` file now. `xor.hsg` gives a
graph structurally equal to step 9's hand-built XOR net and trains to
accuracy 1 through `Train.Loop`, with no CLI involved yet. `moons.hsg`
compiles to `647·B + 1015` nodes for a batch of `B` rows: 65,715 at the
full batch of 100, in about 0.2 seconds.

---

## 1. Theory

### 1.1 Lowering and unrolling

A compiler translates from a language that is convenient to write into
one that is convenient to run. Here the source has vectors, matrices,
functions, local names and comprehensions; the target, Core's graph, has
none of them: only scalar nodes, each an op applied to earlier nodes
(D1, D5). *Lowering* removes the high-level features one by one:

| Source feature | Removed by |
|---|---|
| vectors, matrices | **unrolling**: one node per element |
| names (`let`, parameters, patterns) | binding the name to node ids at compile time |
| functions | **inlining**: compiling the body at every call |
| comprehensions | one copy of the body per slot or parameter |
| built-in functions | core ops, or a formula of core ops |

Because the language has no recursion (D9) and every shape is known
statically (D8), each of these terminates and gives a finite graph. That
is the whole reason for those two restrictions: there is no runtime left
over, nothing that needs to loop or branch.

The compiler is therefore an **interpreter at compile time**. It walks
the typed tree the way an evaluator would, but where an evaluator would
compute a number, it creates a node. Its "values" are node ids:

```haskell
data Value
  = Real NodeId
  | Vec [NodeId]
  | Mat [[NodeId]]
```

`W @ x` for `W : Mat 2 2` and `x : Vec 2` is, at compile time, nothing
more than list processing: take the rows of `W`'s value, multiply them
elementwise with `x`'s value, add up the products. What comes out is
`Vec [n1, n2]`, two new nodes that the rest of the program refers to.

### 1.2 Environments: what a name means

The interpreter needs to know what each name stands for. A name of the
source means one of four things (the checker's `Ref`):

- an `input` or a `param`: its leaves, made at the declaration;
- a top-level `let`: the value it was compiled to;
- a **local** — a function parameter, a `let … in` name, a comprehension
  variable: whatever value it is bound to in the current scope.

Globals live in one map that grows as declarations are compiled, in
order. Locals live in a *scope* that is passed down the tree: `let x = e
in body` compiles `e` to a value `v` and compiles `body` with `x ↦ v`
added; a call compiles the arguments to values `v1 … vk` and compiles
the instance's body in a fresh scope `p1 ↦ v1, …, pk ↦ vk` (a function
body only sees its parameters and the globals, as in the checker, so the
caller's locals must not leak in); a comprehension binds its pattern
once per copy.

### 1.3 Sharing: compile once, refer by id

Binding a name to a **value** (node ids) rather than to an **expression**
is what makes sharing work. Compare two ways of compiling

```
let t = tanh(a) in t * t
```

- *Substitution* would replace each `t` by `tanh(a)` and compile
  `tanh(a) * tanh(a)`: the bound expression is compiled once per use.
- *Binding* compiles `tanh(a)` once, gets a node id `n`, and compiles the
  body with `t ↦ n`: the product's arguments are `[n, n]`.

Hash-consing (D3) would actually merge the two `tanh(a)` nodes of
substitution again, so the *graph* is the same either way. The
difference is the **work**. Substitution copies the bound expression into
every use, and a chain of lets that each use the previous one twice,

```
let a1 = a0 + a0 in let a2 = a1 + a1 in … in a60
```

would be compiled `2⁶⁰` times over. With binding it is 60 nodes, built
in microseconds (a test checks exactly this). The same holds for
function arguments: `sq(exp(a))` with `fn sq(z) = z * z` compiles
`exp(a)` once, then the body with `z ↦` that node.

So the language's promise of §5.5 ("`h1` becomes one node, however many
times `body` uses it") is kept twice: by construction in the compiler,
and again by hash-consing in the builder.

### 1.4 Inlining

There are no function nodes in the graph. A call is replaced by its
body, compiled with the arguments bound to the parameters. The checker
already did the "template" part of D8: for every list of argument shapes
a function is called with, there is an `Instance` with the body typed at
those shapes, and every call names its instance (`NCall key args`).
Inlining is then just a lookup and a recursive compile.

Inlining makes the graph larger than the source — `layer` in `moons.hsg`
is written once and inlined three times per slot — but it is exactly
what a scalar autodiff engine needs: every arithmetic step is a node, so
backprop (step 5) and the symbolic gradients (step 7) see everything.

### 1.5 Comprehensions and slots

A comprehension over a dataset,

```
mean [ (net(x) - y) ^ 2 | (x, y) <- xor ]
```

is a sum (or mean) over the dataset's rows. The rows are data, not part
of the program, and decision D10 says how they get into a graph built
once: as **input leaves**. For a batch of `B` rows the compiler makes `B`
**slots**, each a set of input leaves named after the dataset, the slot
and the column:

```
xor#0.x[0]  xor#0.x[1]  xor#0.y
xor#1.x[0]  xor#1.x[1]  xor#1.y
…
```

and compiles the body once per slot, with the pattern's variables bound
to that slot's leaves (`x ↦ Vec [xor#i.x[0], xor#i.x[1]]`,
`y ↦ Real xor#i.y`). The copies are added up with a chain of `+` and,
for `mean`, divided by `B`. Training (step 9) binds the slots to a new
batch of rows in every step; the graph never changes.

A comprehension over `params` is the same idea with no data: one copy of
the body per parameter, in declaration order, with the variable bound to
that parameter's leaves.

### 1.6 Desugaring built-ins

Core has a deliberately tiny op set (step 2), closed under
differentiation (D6). The language has a few more functions, which the
compiler expresses with core ops (§6):

| Built-in | Lowering |
|---|---|
| `exp`, `log`, `tanh`, `relu`, `step` | the core op, per element |
| `sigmoid(e)` | `1 / (1 + exp(-e))` |
| `sqrt(e)` | `e ^ 0.5` |
| `abs(e)` | `relu(e) + relu(-e)` |
| `max(a, b)` | `a + relu(b - a)` |
| `min(a, b)` | `a - relu(a - b)` |
| `sum(e)` | a chain of `+` from the left; the value itself for a `Real` |
| `mean(e)` | `sum(e) / count`; the value itself for a `Real` |
| `dot(u, v)` | `Σ u[i]·v[i]`: all products, then their sum |
| `transpose(M)` | re-indexing only: no new nodes |
| `a @ b` | a `dot` per row (and column) |

Such a translation is called **desugaring**: `sigmoid` is "syntactic
sugar" for its formula. The derivative comes for free: backprop of
`1 / (1 + exp(-e))` with the rules of the core ops *is* the derivative of
the sigmoid, `σ(e)(1 − σ(e))`, computed through the nodes of the formula.
No new derivative rule is needed, which is the reason to desugar rather
than add ops.

`max` and `min` are written with one `relu` so they stay in the op set;
`a + relu(b - a)` is `b` when `b > a` and `a` otherwise. At `a = b` the
gradient goes entirely to `a` (`step(0) = 0`), a choice like any other
at a kink.

### 1.7 Provenance

Every node records where it came from (D4). The compiler wraps the code
for each expression in `withSpan` with that expression's span, so a node
gets the span of the **innermost** expression that created it: in
`tanh(dot(w, x) + b)` the products and the sums of `dot` get the span of
`dot(w, x)`, the `+ b` the span of `dot(w, x) + b`, the `tanh` the whole
call. Leaves get the span of their **declaration** — the place a
runtime error about an input should point to. Slot leaves get the span
of the dataset's declaration.

Labels name the nodes a user can ask for: the result of a top-level
`let` is labelled with its name (`L`, or `h[0]`, `h[1]` for a vector),
the loss with `loss`, the model's result in its own graph with the
model's name. These labels are what step 7's `gradientName` prints
(`∂L/∂a`), and what pictures show inside the boxes of interior nodes.

---

## 2. Design

### 2.1 The compiled program

```haskell
data Compiled = Compiled
  { compiledGraph :: !Graph
  , compiledInputs :: ![InputSpec]          -- input declarations, in order
  , compiledParams :: ![Train.ParamSpec]    -- param declarations, in order
  , compiledNames :: !(Map Text Value)      -- every input, param, top-level let
  , compiledLets :: ![(Text, Value)]        -- the top-level lets, in order
  , compiledLoss :: !(Maybe NodeId)
  , compiledSlots :: !(Map Text [[Name]])   -- per dataset iterated over
  , compiledAccuracy :: ![(NodeId, NodeId)] -- per slot: prediction, label
  , compiledPredictor :: !(Maybe Predictor)
  }

data InputSpec = InputSpec
  { inputSpecName :: !Text, inputSpecShape :: !(Shape Int)
  , inputSpecLeaves :: ![Name], inputSpecDefault :: !(Maybe [Double]) }

data Predictor = Predictor
  { predictorName :: !Text, predictorGraph :: !Graph
  , predictorArguments :: ![InputSpec], predictorOutput :: !Value
  , predictorSlots :: !(Map Text [[Name]]) }
```

Each field answers a question of a command:

| Command | Needs | From |
|---|---|---|
| `eval --show NAME` | the nodes of a name; input values | `lookupName`, `inputDefaults`, `compiledInputs` (for `--set`) |
| `derive --of NAME --wrt NAMES` | an output node; leaf nodes | `lookupName` (`compiledLoss` for `loss`) |
| `plot` | the graph, an output | `compiledGraph`, `lookupName` |
| `train` | loss, parameters, fixed inputs, feeds, accuracy | `compiledLoss`, `compiledParams`, `inputDefaults`, `compiledSlots`, `compiledAccuracy` |
| `predict` | the model as a function of its arguments | `compiledPredictor` |

`compiledParams` is a list of `Train.ParamSpec`s, so `train` is
`initParams seed (compiledParams c)` with no conversion, and the
parameter leaves are named by `elementNames` of those very specs.
`InputSpec` is the same for inputs: the name the user writes in `--set`,
the shape to check the value against (E0503), the leaves to bind, the
default. `compiledSlots`, paired with a dataset's rows, is a
`Train.Loop.Feed` (`toFeed`).

Everything is plain data with `Eq` and `Show`, like the graph itself
(D1), so the tests compare whole records.

### 2.2 The graph holds every leaf

The main graph contains the leaves of **every** input and parameter,
made at their declarations, whether anything uses them or not. Unused
leaves are dead nodes: evaluation only visits what the outputs reach
(D2), so they cost nothing, and `derive --wrt b` of a parameter the
output doesn't depend on still finds `b`'s leaf (and a zero gradient)
instead of an unknown name.

Functions and the model are compiled only where they are called, and a
dataset only where a comprehension runs over it: those need arguments or
a batch size, and an unused one has neither.

### 2.3 The model: a separate graph

`predict` evaluates the model on argument values given on the command
line. There are two ways to offer that:

1. **A separately compiled graph**: the model's body compiled once, in a
   graph of its own, with its arguments as input leaves.
2. **A function** `[Value] -> Builder Value` that inlines the model into
   any builder on demand.

The compiler does the first. The predictor is then plain data like the
main graph: it can be evaluated with `Core.Eval`, drawn with `Viz.Dot`
(`plot` of the model itself), differentiated, compared in tests with
`==`. A function would put a closure into `Compiled` (no `Eq`, no
`Show`), expose the compiler's internals to the driver, and buy nothing
the CLI needs: `predict` evaluates one model on one input. If evaluation
on many rows is ever wanted, the predictor graph can be evaluated once
per row, or a later step can add a dataset of its own.

The predictor graph contains the leaves of every input and parameter
under the same names as the main graph, so the same parameter store (a
trained one, read back from `--save-params`) and the same input values
bind it. The model's **arguments** are new input leaves, named after the
model: `net.x[0]`, `net.x[1]`. They can't be called just `x[0]`, because
the model's parameter `x` may shadow an `input x` that a function called
by the model still refers to (function parameters may shadow globals,
`01-language.md` §7.1); both would want the leaf `x[0]`. The prefix
makes the clash impossible, and `predictorArguments` tells the driver
which leaves `--set x=…` fills.

Top-level `let`s that the model's body refers to are compiled into the
predictor graph **on demand**, on the first reference: the predictor
contains what the model needs and nothing else.

### 2.4 Batch sizes are an input

The number of slots `B` of a dataset is the batch size: `--batch`, or,
by default, the dataset's size (D10). The compiler never reads data
(`derive` and `plot` must work without the CSV file), so it takes the
slot counts as an argument:

```haskell
compileProgram :: Map Text Int -> Checked -> Either CompileError Compiled
batchSizes :: Maybe Int -> [Dataset] -> Map Text Int
```

`batchSizes` is the default rule for a driver that has loaded the data.
A dataset missing from the map gets one slot per row if its rows are
inline (`rows [ … ]` says how many there are); a CSV dataset can't, and
is a `MissingBatchSize` error. Several comprehensions over the same
dataset get the **same** slot leaves: they are created by name, and
hash-consing returns the existing leaf. So in one training step every
comprehension sees the same rows, and the driver has one feed per
dataset.

### 2.5 Accuracy pairs

`Train.Loop` can report a sign accuracy (`01-language.md` §11) given,
per slot, the model's prediction node and the label's leaf. Which node
is "the prediction" is a question about the program, so the compiler
answers it: inside a comprehension over a dataset, the **first call of
the model** in a slot is that slot's prediction, and the leaf of the
dataset's **last field** is its label, when both are `Real`. For
`xor.hsg` and `moons.hsg` that is `net(x)`/`mlp(x)` and `y`.

Only pairs the loss depends on are kept: training evaluates only what the
loss reaches, so a prediction made in a `let` the loss doesn't use has
no value to compare. A program without such pairs gets `[]`: no accuracy.

### 2.6 Errors

```haskell
data CompileError
  = MissingBatchSize Text
  | BadBatchSize Text Int
  | GraphFailure GraphError
  | InternalError Text
```

A checked program always compiles. `MissingBatchSize` and `BadBatchSize`
are the caller's mistakes (a CSV dataset without a slot count, a count
below 1). `InternalError` means the checked program broke one of the
checker's promises (a name that doesn't resolve, operands of different
shapes): a bug, reported instead of crashing. `GraphFailure` wraps a
builder error; the only one a user can provoke is a number literal too
large for a `Double` (`1e999`), whose constant would be infinite (see
§4.6).

---

## 3. Code walkthrough

The module is `src/HsGrad/Lang/Compile.hs`.

### 3.1 The compiler's monad

```haskell
type Compile = ReaderT Context (StateT St (ExceptT CompileError Builder))

data Context = Context
  { ctxChecked :: Checked, ctxBatch :: Map Text Int
  , ctxDatasets :: Map Text DatasetInfo, ctxLets :: Map Text LetInfo
  , ctxModel :: Maybe InstanceKey }

data St = St
  { stGlobals :: Map Text Value
  , stSlots :: Map Text [[Name]]
  , stPredictions :: Map (Text, Int) (NodeId, NodeId) }
```

Four layers, innermost first: the graph `Builder` itself (step 3), an
error, the state the compiler accumulates (globals compiled so far, the
slots made, the predictions seen), and the read-only context (the
checked program and the batch sizes). The builder's operations are
lifted through the three layers with `liftB = lift . lift . lift`.
`withSpan` takes a builder action and runs it under a span; to use it on
a whole compiler step, the step is "mapped" through each layer down to
the builder:

```haskell
spanned :: Span -> Compile a -> Compile a
spanned sp = mapReaderT (mapStateT (mapExceptT (withSpan sp)))
```

`Span` is Core's `SrcSpan` (step 10), so the checker's spans go straight
into `withSpan` with no conversion.

### 3.2 Declarations

```haskell
compileProgram batch c = do
  (losses, st, g) <- runCompile ctx (mapM declare decls)
  predictor <- traverse (compilePredictor ctx) (checkedModel c)
  …

declare = \case
  CInput i -> Nothing <$ declareInput i
  CParam p -> Nothing <$ declareParam p
  CLet l -> Nothing <$ compileLet l
  CLoss l -> Just <$> compileLoss l
  CData _ -> pure Nothing
  CFunction _ -> pure Nothing
  CModel _ -> pure Nothing
```

Declarations are compiled in source order, so node ids follow the file:
the leaves of `intro.hsg`'s `a`, `b`, `c` are nodes 0–2, `e` is 3, and so
on, exactly like the hand-built graph. Inputs and parameters make their
leaves under the declaration's span:

```haskell
leaves :: (Name -> Builder NodeId) -> Span -> Text -> Shape Int -> Compile Value
leaves make sp x s = do
  ids <- spanned sp (liftB (mapM make (leafNamesOf x s)))
  shaped s ids
```

`leafNamesOf` is `Train.leafNames`, the function behind `elementNames`
(§4.1), and `shaped` cuts the row-major list into a `Value` of the
shape. A `let` compiles its expression in the empty scope, labels the
result and records it:

```haskell
compileLet l = do
  v <- compileExpr topScope (letExpr l)
  labelValue (letName l) v
  setGlobal (letName l) v
  pure v
```

### 3.3 Expressions

```haskell
compileExpr :: Scope -> TExpr -> Compile Value
compileExpr scope (TExpr sp _ n) = spanned sp $ case n of
  NNum r -> Real <$> number r
  NRef r -> reference scope r
  NVector es -> mapM (compileExpr scope) es >>= vector
  NIndex e i j -> compileExpr scope e >>= index i j
  NNegate (TExpr _ _ (NNum r)) -> Real <$> number (negate r)
  NNegate e -> compileExpr scope e >>= elementwise (unary Neg)
  NPower e k -> compileExpr scope e >>= elementwise (unary (PowC k))
  NBinary op l r -> do
    a <- compileExpr scope l
    b <- compileExpr scope r
    operator op a b
  NBuiltin b args -> mapM (compileExpr scope) args >>= builtin b
  NReduce b it -> reduce scope b it
  NCall key args -> mapM (compileExpr scope) args >>= inline scope key
  NLet x bound body -> do
    v <- compileExpr scope bound
    compileExpr (bindLocal x v scope) body
```

One case per form of the checker's `Node`, each wrapped in the
expression's span. A few things to notice:

- The type in `TExpr` is ignored. Every decision that depends on a shape
  can be read off the compiled values: a `Mat` times a `Vec` is a
  matrix-vector product, a vector literal of rows is a matrix. The
  checker guaranteed the shapes fit; a value of the wrong form is an
  `InternalError`.
- `-2.0` is one constant `-2`, not `Neg` of `2`: `01-language.md` §5.2
  reads a minus applied to a literal as a negative number, and step 6's
  hand-built `intro` has a single node `f = -2` (micrograd's `f`). The
  constant gets the span of the whole `-2.0`.
- Indexing and `transpose` only pick node ids out of lists: no nodes.
- Arguments are compiled **before** the call, left to right, and the
  order of node creation is the order of evaluation in the source. That
  is what makes the ids of the compiled examples match the hand-built
  ones.

References look in the scope for locals and in the globals otherwise. A
`let` that isn't compiled yet is compiled on the spot — this only
happens in the predictor's graph (§2.3):

```haskell
reference scope = \case
  RLocal x -> maybe (internal ("unknown local " <> x)) pure (Map.lookup x (scopeLocals scope))
  RInput x -> global x
  RParam x -> global x
  RLet x ->
    gets (Map.lookup x . stGlobals) >>= \case
      Just v -> pure v
      Nothing -> asks (Map.lookup x . ctxLets) >>= maybe (internal ("unknown let " <> x)) compileLet
```

### 3.4 Elementwise operations and broadcasting

```haskell
elementwise :: (NodeId -> Builder NodeId) -> Value -> Compile Value

broadcasting :: (NodeId -> NodeId -> Builder NodeId) -> Value -> Value -> Compile Value
broadcasting f a b = case (a, b) of
  (Real x, _) -> elementwise (f x) b
  (_, Real y) -> elementwise (`f` y) a
  (Vec xs, Vec ys) -> Vec <$> pairwise xs ys
  (Mat xs, Mat ys) -> …
```

Broadcasting (§4.1) is one line per case: a `Real` on either side is
applied to every element of the other operand, **keeping the operand
order** (`b - u` is `b - u[i]`, not `u[i] - b`); otherwise the operands
have the same shape and are zipped. The four arithmetic operators and
`max`/`min` all go through `broadcasting`.

### 3.5 Products and sums

```haskell
matMul (Mat rows) (Vec xs) = Vec <$> mapM (`dotNodes` xs) rows
matMul (Mat rows) (Mat others) = do
  let columns = transpose others
  Mat <$> mapM (\r -> mapM (dotNodes r) columns) rows

dotNodes us vs = liftB (zipWithM (binary Mul) us vs) >>= sumNodes

sumNodes (x : xs) = liftB (foldM (binary Add) x xs)
```

A dot product makes all its products first and then adds them from the
left, `(p0 + p1) + p2`: the shape of step 6's hand-built `neuron`, and of
a Python `sum`. A one-element sum is the element itself, so `sum` of a
`Real` makes no node.

### 3.6 Built-ins

```haskell
builtin b args = case (b, args) of
  (BExp, [v]) -> elementwise (unary Exp) v
  …
  (BSigmoid, [v]) -> elementwise sigmoid v
  (BSqrt, [v]) -> elementwise (unary (PowC (1 / 2))) v
  (BAbs, [v]) -> elementwise absolute v
  (BMax, [u, v]) -> broadcasting maximum2 u v
  (BMin, [u, v]) -> broadcasting minimum2 u v
  (BSum, [v]) -> Real <$> sumNodes (valueNodes v)
  (BMean, [Real x]) -> pure (Real x)
  (BMean, [v]) -> …  -- sum, then divide by the number of elements
  (BDot, [Vec us, Vec vs]) -> Real <$> dotNodes us vs
  (BTranspose, [Mat rows]) -> pure (Mat (transpose rows))
  where
    sigmoid x = do                       -- 1 / (1 + exp(-e))
      e <- unary Neg x >>= unary Exp
      one <- constant 1
      binary Add one e >>= binary Div one
    absolute x = do                      -- relu(e) + relu(-e)
      r <- unary Relu x
      r' <- unary Neg x >>= unary Relu
      binary Add r r'
    maximum2 x y = binary Sub y x >>= unary Relu >>= binary Add x   -- a + relu(b - a)
    minimum2 x y = binary Sub x y >>= unary Relu >>= binary Sub x   -- a - relu(a - b)
```

Each formula is the one in §6, with the operands in the order written
there (hash-consing is structural, `x + y` is not `y + x`, so the order
is part of the lowering). The constant `1` of `sigmoid` is one shared
node however many sigmoids there are.

### 3.7 Calls

```haskell
inline scope key args = do
  inst <- … Map.lookup key instances …
  let inner = Scope (Map.fromList (zip (instanceParams inst) args)) (scopeSlot scope)
  v <- compileExpr inner (instanceBody inst)
  model <- asks ctxModel
  case (v, scopeSlot scope) of
    (Real p, Just (d, i, Just y)) | model == Just key -> notePrediction d i p y
    _ -> pure ()
  pure v
```

The body is compiled in a scope of its own: the parameters bound to the
argument values, nothing else. The call's own span is current while the
body compiles, but every node of the body is made under an inner span —
the body's — so in `neuron.hsg` no node has the span of the call
`neuron(w, b, x)`, exactly as the hand-built graph has it.

The scope also carries the current **slot**, if any, which a call passes
on (it is not a name, so lexical scoping doesn't apply). A call of the
model in a slot is noted as that slot's prediction; `notePrediction`
keeps the first one.

### 3.8 Comprehensions

```haskell
reduce scope b it = do
  copies <- case it of
    OverData d xs body -> overData scope d xs body
    OverParams x bodies -> forM bodies $ \(p, body) -> do
      v <- global p
      compileExpr (bindLocal x v scope) body
  total <- … foldM (broadcasting (binary Add)) …
  case b of
    BSum -> pure total
    BMean -> elementwise (divideBy (length copies)) total
```

The copies are added up elementwise (a comprehension may produce
vectors), then divided by the count for `mean`. For `params`, the
checker has already typed the body once per parameter, so the compiler
just binds the variable to each parameter's leaves in turn.

```haskell
overData scope d xs body = do
  …
  let fields = datasetFields info
      slots = slotNames d fields b
  modify' (\st -> st {stSlots = Map.insert d slots (stSlots st)})
  forM (zip [0 ..] slots) $ \(i, names) -> do
    ids <- spanned (datasetDeclSpan info) (liftB (mapM input names))
    values <- splitFields fields ids
    let labelLeaf = …   -- the last field's leaf, if it is a Real
        inner = Scope (Map.union (Map.fromList (zip xs values)) (scopeLocals scope)) (Just (d, i, labelLeaf))
    compileExpr inner body
```

Slot names come from `Data.Dataset.slotNames` (step 13), the same
function the training feed uses. Slot `i`'s leaves are made right before
copy `i` of the body, under the dataset declaration's span, and split
into one value per field. The body sees the enclosing locals too (a
comprehension in a function body may use the function's parameters).

### 3.9 The predictor

```haskell
compilePredictor ctx m = do
  ((args, out), st, g) <- runCompile ctx $ do
    mapM_ declareLeaf (checkedDecls (ctxChecked ctx))
    args <- forM (modelParams m) $ \(x, s) ->
      leaves input (modelNameSpan m) (modelName m <> "." <> x) s
    out <- inline topScope (modelInstance m) args
    labelValue (modelName m) out
    pure (args, out)
  …
```

A second, independent compilation: the input and parameter leaves in
declaration order (so they have the same names and spans as in the main
graph), the argument leaves under the model's name span, and the
model's instance inlined on them.

---

## 4. Design decisions

### 4.1 Reuse the libraries' naming functions

Three modules must agree on leaf names: the compiler creates the leaves,
`Train.Param` binds parameter values to them, the training loop binds
dataset rows to the slots. Steps 9 and 13 each wrote the convention down
(`elementNames`, `slotNames`) and asked the compiler to follow it. The
compiler **calls those functions** rather than repeating the convention,
so a mismatch can't happen: `Lang.Compile` imports `Train.Param` and
`Data.Dataset`.

That adds edges to the layer picture of `00-architecture.md` §2:
`Lang.Compile` sits above Train and Data, where the language meets the
libraries' conventions. The front end (`Syntax`, `Parser`, `Check`) still
depends only on Core (and `Data.Dataset` on `Lang.Check`, for the
dataset declarations), and there is no module cycle. The alternative,
moving the naming functions into a Core module, would have moved
`ParamSpec` too, since `compiledParams` returns specs; Train is the
natural home for both.

`Train.Param` gained a small, general `leafNames :: Text -> Shape ->
[Name]` (`elementNames` is now `leafNames` of a spec's name and shape),
which also names input leaves and the labels of `let` vectors.

### 4.2 Explicit parameter values in `Train.Param`

`param w : Vec 3 = [1, 2, 3]` is valid source (`InitValues` in the
checker), but `Train.Param.Init` had only `Uniform`, `Normal` and
`Constant`. Rather than leaving the driver to patch the store after
`initParams`, `Init` gained a fourth case:

```haskell
  | Values ![Double]   -- = [1, 2, 3]: one value per element, row-major
```

and `sampleInit` now produces the values of all of a parameter's
elements at once (`sampleInit :: Init -> Int -> State StdGen [Double]`),
since an explicit list can't be sampled one element at a time. Random
elements are drawn in the same order as before, so every seed gives the
same parameters as in step 9; `Values`, like `Constant`, draws nothing.

### 4.3 The model in its own graph

Discussed in §2.3: plain data over a closure, prefixed argument leaves
over a possible clash, lets compiled on demand. The cost is compiling
the model body once more (979 nodes for `moons.hsg`, against 65,715 for
the training graph).

### 4.4 Labels only for names the user can ask for

Top-level `let`s, the loss and the model's result get labels. Locals
(`let … in`, function parameters, comprehension variables) don't: a
model inlined 100 times would put 100 different nodes under the label
`h1[3]`, and a label is meant to identify a node (it becomes the `L` of
`∂L/∂a`). Leaves keep their leaf names. A node that already has a label
keeps it (first label wins, step 3), so `let z = L` doesn't rename `L`.

### 4.5 Fixing the hand-built `neuron`

The milestone test compares the compiled `neuron.hsg` with step 6's
hand-built graph, ids and spans included. It found two wrong spans in
the hand-built version: the declarations `input x : Vec 3 = [1.0, -2.0,
0.5]` and `param w : Vec 3 ~ uniform(-1, 1)` end at columns 35 and 33
(exclusive), not 36 and 34; they had been counted by hand before the
parser existed. `HsGrad.Test.Examples` and the DOT cluster labels in
`DotSpec` (`neuron.hsg:1:1-35`) are corrected. With that, both examples
are equal as whole graphs.

### 4.6 Not done here

- **W0203** (a parameter that can't affect `loss`) is already reported by
  the checker, per declaration (`13-checker.md` §4.4); step 12 did not
  defer it. The element-level version (row 3 of a matrix that nothing
  reads) could be computed from `reachable (compiledGraph c) [loss]` and
  the parameter leaves, if it is ever wanted.
- **A literal too large for a `Double`** (`1e999`) becomes an infinite
  constant, which the builder rejects (`GraphFailure (NonFiniteConst
  Infinity)`). It is a user mistake without a diagnostic code; the
  natural place for it is the parser (E0103, "malformed number literal",
  could cover "out of range"). Left for the driver step to decide.

---

## 5. Tests

`test/HsGrad/Lang/CompileSpec.hs`. The helpers `copyWith` and
`sameExpressions` (structural equality of expressions in two graphs: copy
both into one builder and let hash-consing decide) moved from
`SimplifySpec` to the shared `HsGrad.Test.Graphs`.

### 5.1 The milestone

- `intro.hsg` compiles to **exactly** step 6's hand-built graph (`==`:
  ids, ops, arguments, spans, labels); `L` is node `introL`; evaluated
  with the defaults, `L = -8` and the input gradients are `6, -4, -2`.
- `neuron.hsg` compiles to exactly the hand-built graph; `y = tanh(1.5)`
  with `neuronEnv`.
- `xor.hsg` with 4 slots: its loss is **structurally** equal to step 9's
  `xorNet 4`, and so are its accuracy pairs; its parameters are
  `xorSpecs`, its slots `xorSlots 4`. The node ids differ — the compiler
  computes `W1 @ x` for all rows before adding `b1`, the hand-built net
  row by row — which is why the comparison is structural here.
- `xor.hsg` **trains** through `Train.Loop`, with the rows loaded by
  `Data.Dataset` from the file and every piece of the `Setup` taken from
  `Compiled`: 200 steps, seed 1337, rate 0.1; the loss starts above 1,
  ends below `10⁻³`, the accuracy ends at 1, and every step's loss equals
  the hand-built net's within `10⁻⁹` (same graph up to node order, same
  seed).
- `moons.hsg` with `B` slots has `647·B + 1015` nodes (checked for `B = 4`
  and `B = 7`), one feed `moons` with `slotNames`' slots, `B` accuracy
  pairs, a predictor of 979 nodes, and a loss that evaluates to a
  plausible number on the first 4 rows of the CSV file.

The node count, worked out:

| Part | Nodes |
|---|---|
| parameter leaves: 32 + 16 + 256 + 16 + 16 + 1 | 337 |
| per slot: 3 leaves | 3 |
| per slot: `relu(layer(W1, b1, x))`: 16·(2 + 1) + 16 + 16 | 80 |
| per slot: `relu(layer(W2, b2, h1))`: 16·(16 + 15) + 16 + 16 | 528 |
| per slot: `layer(W3, b3, h2)[0]`: (16 + 15) + 1 | 32 |
| per slot: `relu(1 - y * mlp(x))` | 3 |
| the constant `1`, shared by all slots | 1 |
| `mean`: `B − 1` additions, the constant `B`, a division | B + 1 |
| `p ^ 2` for every parameter element | 337 |
| `sum(p ^ 2)` per parameter: 31 + 15 + 255 + 15 + 15 + 0 | 331 |
| `sum [ … ]` over the 6 parameters | 5 |
| `1e-4 * …`: a constant, a product | 2 |
| `data_loss + reg` | 1 |
| **total** | **647·B + 1015** |

(For `B = 1` the count is one less: the constant `B = 1` is the shared
constant `1`.) At the full batch, `B = 100`, that is 65,715 nodes,
matching the "about 60k" estimate of D5; compiling it takes about 0.2 s.
The predictor's 979 nodes are the 337 parameter leaves, 2 argument
leaves and the 640 nodes of one `mlp`.

### 5.2 Sharing

- A top-level `let` used twice: one node, used as both arguments of the
  `+`.
- `let t = tanh(a) in t * t`: three nodes, and `t`'s node has the span of
  `tanh(a)`.
- A function argument used twice in the body: compiled once.
- A chain of 60 nested `let … in` doublings: 61 nodes, value `2⁶⁰`,
  compiled instantly (substitution would never finish).
- Two comprehensions over one dataset share its slot leaves.

### 5.3 Lowerings and built-ins

- Leaf names and spans for inputs and parameters; `B` copies over a
  dataset, with slot names and the dataset's span on every slot leaf;
  one slot per inline row by default; one copy per parameter over
  `params`; nested inlined calls; negative literals as one constant;
  labels on `let` vectors and the loss; a span on every node of
  `moons.hsg`.
- **Properties against Haskell**: for every built-in and operator, a
  program `let y = <expr>` over inputs `a, b : Real`, `u, v : Vec 3`,
  `M : Mat 2 3` is evaluated at random input values (100 samples each)
  and compared with the same computation written directly in Haskell:
  `exp`, `log`, `tanh`, `relu`, `step`, `sigmoid`, `sqrt`, `abs`, `max`,
  `min`, `^` (3, −1, 0.5), `+ - * /` on vectors, broadcasting a `Real` in
  both operand positions, `sum`/`mean` of vectors, matrices and `Real`s,
  `dot`, matrix-vector and matrix-matrix products, `transpose`,
  indexing, vector and matrix literals.
- **The desugarings are the formulas of §6**: `sigmoid`, `sqrt`, `abs`,
  `max`, `min`, `mean` and `dot` compile to graphs structurally equal to
  their formulas built by hand.
- `transpose`, indexing and `sum`/`mean` of a `Real` make no nodes.

### 5.4 Handles

Input specs and defaults by leaf name; parameter specs including explicit
values (`Values`), and `initParams` on them; the predictor's argument
leaves (`net.x[0]`), and — on every XOR row — the predictor's output
equals the training graph's prediction in that row's slot; lets compiled
into the predictor on demand; `MissingBatchSize` and `BadBatchSize`;
`batchSizes`; the first model call of a slot as its prediction, and no
pairs when the loss doesn't depend on the predictions.

`Train.Param` has one new property: explicit values are exact, in
element order, and draw nothing.

### 5.5 Testing the tests

Deliberate bugs in `Lang.Compile`, each tried on its own:

| Bug | Caught by |
|---|---|
| a broadcast `Real` applied on the wrong side | the broadcasting property, the milestone |
| sums added from the right | the intro/neuron graphs, the `dot` formula |
| `mean` of a `Real` divides by 1 | "no nodes for … mean of a Real" |
| `-2.0` compiled as `Neg 2` | the intro graph |
| no labels on `let`s | the intro graph |
| slot leaves without the dataset's span | the slot-span test |
| matrix-matrix product without transposing | the matrix-matrix property |
| the last model call of a slot is the prediction | the first-call test |
| accuracy pairs not filtered by the loss | the reachability test |
| model arguments named `x[0]` instead of `net.x[0]` | the predictor test |
| `sigmoid` as `1 / (1 - exp(-e))` | the sigmoid property and formula |
| `abs` as `relu(e) + relu(e)` | the abs property and formula |
| `max` built like `min` | the max property and formula |
| `mean [ … ]` without the division | the xor milestone |
| comprehension bodies don't see enclosing locals | the local-in-comprehension test |
| a `let … in` body that doesn't see the bound name | the sharing tests |
| no default slot count for inline rows | the xor milestone |
| the loss not labelled | the label test |

All are killed. The comprehension-scope mutant survived the first run —
no test had a comprehension using a local from outside — and got its own
test.

The full suite has 680 tests and runs in about 4 seconds.

---

## 6. What later steps build on

- **Step 15 (`Driver`, `app/Main`)**:
  - After a clean check, `compileProgram slots checked`. For commands that
    read data (`train`), load the datasets first and pass
    `batchSizes batch datasets`, after `checkBatch`; for the others, an
    empty map works for inline datasets, and a CSV dataset that a
    comprehension uses needs either the data or a slot count.
    `MissingBatchSize` / `BadBatchSize` then mean the driver passed too
    little; `InternalError` is a bug; `GraphFailure` is (in practice) a
    literal like `1e999` (§4.6).
  - **eval**: `env = Env (inputDefaults c ∪ --set values) Map.empty`
    (parameters from `initParams seed (compiledParams c)` or a saved
    file); `--set x=…` checks the literal's shape against
    `inputSpecShape` (E0503) and zips it with `inputSpecLeaves`; an
    input leaf left unbound is E0502. `--show NAME` is `lookupName`, all
    lets in order are `compiledLets`.
  - **derive --of NAME --wrt NAMES**: the output is `lookupName` (a
    `Real`), or `compiledLoss c` for `loss`; the `--wrt` leaves are
    `valueNodes` of `lookupName` of each name; then step 7/8's
    pipeline. Names for a matrix element: `leafNames`, e.g. `W1[0,1]`.
  - **train**: `Setup { setupGraph = compiledGraph c, setupLoss = loss,
    setupInputs = inputDefaults c (with --set), setupFeeds = [toFeed d
    (compiledSlots c ! dataName d) | d used], setupAccuracy =
    compiledAccuracy c }`, `initParams seed (compiledParams c)`.
  - **predict**: `compiledPredictor`; bind `predictorArguments` from
    `--set`, parameters from `--params`, inputs from their defaults;
    evaluate `predictorOutput` in `predictorGraph`. `predict` on a file
    without a model is an error the driver reports.
  - `--save-params`: `paramValues spec store` for each spec of
    `compiledParams` (explicit `Values` parameters included).
- **Later**: a tensor backend (D5) would replace unrolling here and only
  here: `Value` would hold one node per tensor, and the rest of the
  pipeline would see bigger ops.
