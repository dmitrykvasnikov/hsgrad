# 05 — `Core.Eval`: the forward pass

This step adds `HsGrad.Core.Eval`, the first module that computes numbers
from a graph. Given

- a `Graph` (step 3),
- an **environment**: values for the input and parameter leaves, by name,
- the **outputs** whose values are wanted,

it computes the value of every node the outputs depend on, and of no other
node, in ascending id order. The values are kept outside the graph, in an
`IntMap Double` keyed by node index (D1).

This is micrograd's forward pass. micrograd computes it eagerly, while the
expression is being written (`e = a * b` immediately stores `-6.0` in
`e.data`); here the graph is built first and evaluated afterwards, as many
times as needed with different leaf values.

Two things can go wrong, and both are reported as values: a leaf without a
value (the core side of error **E0502**) and a NaN or infinite result
(**E0501**). For the second, the error names the node where the problem
*starts*, so that a diagnostic can later point at its source span.

---

## 1. Theory

### 1.1 Evaluation in topological order

The value of a node is its op applied to the values of its arguments:

```
value(n) = forward(op n, value(arg₁), …, value(argₖ))
```

with leaves as the base case: a `Const` carries its value, an `Input` or
`Param` gets it from outside. Read as a program, this definition is a
recursive function, and that function is correct, but on a DAG it is
**exponentially slow**: a node used twice is computed twice, and each of
those computes *its* shared arguments twice, and so on. A chain of `n`
doublings,

```
d₀ = x,   dₖ₊₁ = dₖ + dₖ
```

has `n + 1` nodes but `2ⁿ` paths from the top to `x`; the naive recursion
walks every path.

The fix is to compute each node **once**, after its arguments, and remember
the result. "After its arguments" means: in a topological order. D2 gives us
one for free, because ids are handed out in creation order and arguments are
always created first. So the forward pass is a single loop:

```
for each needed node n, in ascending id order:
    value[n] = forward(op n, value[arg₁], …, value[argₖ])
```

Each `value[argᵢ]` is already in the map when it is needed, because
`argᵢ < n`. The loop does one step per node, so the cost is linear in the
number of needed nodes (times the `IntMap` cost per lookup), however much
sharing there is.

This is what micrograd does as well, only spread out over time: every
`Value` is computed once, when its Python expression runs, and later uses
read `.data`.

### 1.2 Only the needed nodes

A graph can contain nodes the outputs do not depend on: an unused `let`, the
`model` while only the `loss` is wanted, a gradient graph's forward part
(step 7). Computing them would waste time and, worse, could *fail*: an
unused `log(x)` with `x < 0` is NaN, and an unused input may have no value.
So the forward pass only evaluates `reachable g outs` (step 3, §1.4): the
outputs and everything they depend on. A set of ids in ascending order is
still a topological order.

### 1.3 Leaves and environments

Leaves are where values enter the graph. `Const c` carries `c` in the graph
itself. `Input n` and `Param n` are **named holes**: the graph says *that* a
value goes there, not *which*. The environment fills the holes.

Keeping values out of the graph is what makes training cheap (D10). A
training run builds the graph once, and every step only changes the
environment:

- the **parameters** get new values from the optimiser;
- the **dataset slots** (`moons#3.x[1]`, `01-language.md` §8) are rebound
  to the rows of the next mini-batch.

Neither needs the graph to be rebuilt, or even re-inspected.

### 1.4 Non-finite values and where they start

IEEE arithmetic does not stop on errors. `log(-1)` is NaN, `1 / 0` is ∞,
`exp(1000)` overflows to ∞, and from then on the bad value **spreads**: NaN
plus anything is NaN, `tanh(NaN)` is NaN, and in a training loss one NaN
makes the whole loss NaN, then every gradient, then every parameter after
the next update. The forward functions of step 2 follow these rules on
purpose (`docs/03-core-op.md` §2.2): they never throw, and they don't know
where they are in the source anyway.

The forward pass is the place to notice, because it knows which node it is
computing. And because it computes nodes in topological order, the **first**
node with a non-finite value is special: all its arguments were computed
before it and were finite, so this node *created* the bad value instead of
inheriting it. That is the node a diagnostic should point at
(`log(x)` at `model.hsg:12:9`, with `x = -1`), not the loss at the end of
the chain that the NaN spread to.

## 2. Code walkthrough

The module lives at `src/HsGrad/Core/Eval.hs`. It has three parts: the
environment, the schedule, and the evaluation itself.

### 2.1 The environment

```haskell
data Env = Env
  { envInputs :: !(Map Name Double)
  , envParams :: !(Map Name Double)
  }

emptyEnv  :: Env
bindInput :: Name -> Double -> Env -> Env
bindParam :: Name -> Double -> Env -> Env
```

Two maps, because `Input "w"` and `Param "w"` are different leaves
(`docs/03-core-op.md` §3.4): binding one must not give the other a value.
The constructor and fields are exported; an `Env` has no invariant to
protect. That keeps the two typical updates of a training step one-liners:

```haskell
env { envParams = newParams }                -- after an optimiser step
foldr (uncurry bindInput) env slotValues     -- the next mini-batch
```

Names the graph does not use are ignored. Whether a `--set x=…` names a
declared input is a question about the *source* file, and the checker and
driver (steps 12, 15) answer it with a proper diagnostic.

### 2.2 The schedule

```haskell
data Schedule = Schedule
  { schedNodes :: ![(NodeId, Node)]    -- every reachable node, ascending
  , schedLeaves :: ![(NodeId, Leaf)]   -- the reachable Input and Param leaves
  }

schedule :: Graph -> [NodeId] -> Schedule
schedule g outs = Schedule steps leaves
  where
    steps = [(i, nodeAt g i) | i <- Set.toAscList (reachable g outs)]
    leaves = [(i, l) | (i, Node {nodeOp = Leaf l}) <- steps, needsValue l]
    needsValue (Const _) = False
    needsValue _ = True
```

A schedule is everything about an evaluation that depends only on the graph
and the outputs: which nodes to compute, in which order, and which leaves
will need a value. It is computed once and can then be evaluated with any
number of environments (§3.1). The node list holds the `Node`s themselves,
not just their ids, so evaluation doesn't look anything up in the graph.

The second list comprehension uses a pattern on the left of `<-`: elements
that don't match `Node {nodeOp = Leaf l}` are skipped, so it picks the leaf
nodes out of the list in one line.

`Schedule` is abstract, and read with `scheduleNodes` (backprop, step 5,
walks it in reverse) and `scheduleLeaves`.

### 2.3 Values and errors

```haskell
type Values = IntMap Double

data EvalError
  = UnboundLeaves (NonEmpty (NodeId, Leaf))
  | NonFiniteValue NodeId [Double] Double
```

`Values` is a plain synonym for D1's `IntMap Double`, keyed by `nodeIndex`.
Two accessors take a `NodeId` instead of an `Int`, in the style of
`lookupNode` and `nodeAt`:

```haskell
lookupValue :: NodeId -> Values -> Maybe Double
valueAt     :: Values -> NodeId -> Double     -- partial: error if not evaluated
```

The errors:

- **`UnboundLeaves`** lists *every* reachable input and parameter leaf
  without a value, in id order. `NonEmpty` (from `base`) is a list with at
  least one element, which is what "some leaves are unbound" means; the type
  rules out the meaningless `UnboundLeaves []`.
- **`NonFiniteValue i args v`** is the first node in evaluation order whose
  value `v` is NaN or ±∞, together with its argument values `args`, which
  are all finite (§1.4). The arguments are there for the message: "`log` of
  `-1`" says more than "`log` gave NaN". The source span is
  `provSpan (nodeProv (nodeAt g i))`.

### 2.4 Evaluation

```haskell
evalSchedule :: Schedule -> Env -> Either EvalError Values
evalSchedule s env = do
  leafVals <- bindLeaves env (schedLeaves s)
  sweep leafVals (schedNodes s)

evaluate :: Graph -> Env -> [NodeId] -> Either EvalError Values
evaluate g env outs = evalSchedule (schedule g outs) env
```

Two phases, in the `Either` monad, so the second only runs if the first
succeeded. `evaluate` is the convenience for a one-off evaluation.

**Phase 1** looks up every leaf of the schedule:

```haskell
bindLeaves :: Env -> [(NodeId, Leaf)] -> Either EvalError Values
bindLeaves env leaves =
  case nonEmpty [(i, l) | (i, l, Nothing) <- looked] of
    Just missing -> Left (UnboundLeaves missing)
    Nothing -> Right (IntMap.fromList [(nodeIndex i, v) | (i, _, Just v) <- looked])
  where
    looked = [(i, l, leafValue env l) | (i, l) <- leaves]
```

Every lookup is done; the ones that failed become the error, and if none
failed the values become the starting map. `nonEmpty` turns a list into
`Maybe (NonEmpty a)`, `Nothing` for the empty list, which is exactly the
case split needed here.

**Phase 2** is the loop of §1.1:

```haskell
sweep :: Values -> [(NodeId, Node)] -> Either EvalError Values
sweep !vals [] = Right vals
sweep !vals ((i, n) : rest)
  | isFinite v = sweep (IntMap.insert (nodeIndex i) v vals) rest
  | otherwise = Left (NonFiniteValue i argVals v)
  where
    argVals = map (valueAt vals) (nodeArgs n)
    v = case (nodeOp n, argVals) of
      (Leaf (Const c), []) -> c
      (Leaf _, []) -> valueAt vals i -- bound by 'bindLeaves'
      (Un op, [x]) -> forwardUn op x
      (Bin op, [x, y]) -> forwardBin op x y
      _ -> error ("HsGrad.Core.Eval: wrong number of arguments at " <> show i)
```

Some details:

- **The finiteness check is on every node**, leaves included, so an input
  bound to NaN (a bad CSV cell, say) is reported at the input. A `Const` is
  never non-finite (step 3 rejects those), so the check costs it nothing but
  a comparison.
- **`valueAt` can't fail here.** An argument has a smaller id than its node
  and is reachable whenever the node is, so it was computed earlier in the
  sweep; input and parameter leaves were put in the map by phase 1.
- **The `error` case can't happen either**: the builder guarantees that a
  node has `arity op` arguments. Matching on the op and the argument list
  together is still the clearest way to take the list apart, and the
  fallback states the invariant instead of hiding it.
- **The bang patterns** (`!vals`, `BangPatterns` is part of GHC2021) force
  the map at every step. Without them, the loop could build up a chain of
  unevaluated `insert`s, one per node, and only evaluate it at the end.
  `Data.IntMap.Strict` additionally forces each *value* when it is inserted,
  so no `Double` is left as a thunk either.

The first non-finite node stops the loop: everything after it that depends
on it would be NaN as well, and reporting those adds nothing.

Using it, micrograd's first example (`intro.hsg`, `01-language.md` §1.1):

```haskell
let Right ((a, b, c, l), g) = build $ do
      a <- input "a"; b <- input "b"; c <- input "c"
      e <- binary Mul a b
      d <- binary Add e c
      l <- binary Mul d =<< constant (-2)
      pure (a, b, c, l)
    env = bindInput "a" 2 (bindInput "b" (-3) (bindInput "c" 10 emptyEnv))
evaluate g env [l]
-- Right, in id order: a = 2, b = -3, c = 10, e = -6, d = 4, the constant -2, L = -8
```

## 3. Design decisions

### 3.1 A reusable schedule

The simplest interface would be `evaluate g env outs` alone. But a training
run evaluates the same outputs of the same graph thousands of times, with
only the environment changing (D10). Everything that doesn't depend on the
environment, the reachability search and the extraction of the leaves, can
be done once. Splitting `schedule` from `evalSchedule` makes that explicit
in the types: the training loop (step 9) holds a `Schedule`, and each step
is just `evalSchedule s env'`.

The saving is a constant factor, not a better complexity (reachability
costs about as much as the sweep), but the split also documents the D10
idea where it happens, and backprop (step 5) can walk the same schedule
backwards instead of recomputing it.

### 3.2 Leaves by name, values by id

The environment is keyed by **name**, values by **node id**. The two keys
serve different callers:

- Names are what the outside world knows: the driver has `--set a=2`, the
  parameter store has `W1[3,0]`, the dataset has `moons#3.x[1]`. None of
  them should have to know node ids.
- Ids are what the graph knows, and later Core steps (backprop, gradient
  checks) ask for values of arbitrary nodes, which have no names.

Each name is looked up once per evaluation, in phase 1, and turned into an
id-keyed value; the sweep itself never touches a name.

An `Env` keyed by node id was the alternative: slightly faster, but every
caller would first have to find its leaves' ids (`findNode (Leaf (Param n))
[] g`), and an environment would only make sense for one graph. With names,
the same parameter environment works for the forward graph and for its
simplified version (step 8), whose ids are different.

### 3.3 All unbound leaves, but only the first non-finite value

The two errors are reported differently on purpose:

- Whether a leaf has a value is **static**: it can be decided for all leaves
  before computing anything, and one missing value says nothing about the
  others. So phase 1 checks them all and reports all of them, like the
  checker reports all independent errors (`00-architecture.md` §6). A user
  who forgot three `--set`s learns about all three at once.
- A non-finite value is **dynamic**: it is only known once its node is
  computed, and everything downstream of it is garbage. Only the first one
  is a separate problem.

Phase 1 also runs first, so a graph with a missing input and a would-be NaN
reports the missing input: the NaN might well be a consequence of the value
the user forgot.

### 3.4 Non-finite detection is always on

The check could be optional, for speed or to allow intermediate
infinities. It is always on, for three reasons:

1. **It costs almost nothing.** Two floating-point tests per node, next to
   an `IntMap` insert and a list of lookups.
2. **Late detection is much worse than early.** Checking only the outputs
   (or only every few training steps) would report the loss, not the
   `log` or `/` that went wrong; that information is gone once the NaN has
   spread. During training, an unnoticed NaN destroys every parameter at
   the next update.
3. **Non-finite intermediate values are never harmless here.** One could
   argue for allowing `1 / exp(1000) = 1 / ∞ = 0`, which ends finite. But the
   graph is evaluated in order to be differentiated, and the derivative
   passes through the same nodes: `d exp(x) = exp(x) = ∞` times an upstream
   gradient of `0` gives NaN in backprop (step 5). An infinity on the way
   *is* a bug in the model, and the place to report it is where it appears.
   (The cure is in the model: a stable formulation of the function, not a
   switch in the evaluator.)

If a use for unchecked evaluation turns up later, an options argument can
be added then; nothing planned so far needs one.

### 3.5 Errors as values

As in the builder (`docs/04-core-graph.md` §3.7), errors are returned in an
`Either`, not thrown. Unlike the builder's errors, these are **user**
errors: a missing `--set`, a `log` of a negative input. They have to reach
the diagnostics renderer (step 10) with enough information to point into
the source, so they carry node ids (for provenance) and the values
involved, and the driver turns them into E0501/E0502 with spans.

Core stays independent of the language (`00-architecture.md` §2): the error
knows the node, the node knows its `SrcSpan`, and producing the message is
the driver's job.

### 3.6 `Values` is a synonym, not a newtype

D1 says values live in an `IntMap Double`, and later steps want exactly
that (backprop builds a second `IntMap Double` of gradients next to it).
A newtype would hide `IntMap`'s whole API behind wrapper functions for
little gain. The two accessors `lookupValue` and `valueAt` cover the common
case of asking by `NodeId`.

## 4. Tests

`test/HsGrad/Core/EvalSpec.hs` has five groups:

- **micrograd examples**: `intro.hsg` gives `L = -8` with every
  intermediate value (`e = -6`, `d = 4`); micrograd's neuron
  `tanh(x1·w1 + x2·w2 + b)` gives `1/√2`; every op on known values; a
  constants-only graph with the empty environment.
- **Sharing and reachability**: `L = d * d` computes `d` once; a chain of
  40 doublings (`2⁴⁰` paths, 41 nodes) evaluates instantly to `2⁴⁰`, with
  41 values; a dead `log(-1)` and a dead unbound input are not evaluated and
  cause no error; several outputs share their common part; no outputs, or
  ids of another graph, give no values.
- **Environments and schedules**: one schedule evaluated with different
  dataset rows and a replaced parameter map (D10); the leaves that need
  values, in id order, without constants; unused names are ignored; the
  last binding of a name wins.
- **Errors**: all unbound leaves at once; `Input "w"` is not a value for
  `Param "w"`; NaN from `log(-1)` reported at the `log` with argument `-1`;
  ∞ from `1/0` and from `exp(1000)`; in `tanh(√x + 1) · 2` with `x < 0`, the
  error names `√x`, not a later node; `1 / exp(1000)` is an error although
  its value is `0` (§3.4); a NaN-bound input is reported at the input;
  unbound leaves are reported before a NaN; the failing node's provenance
  leads back to the span it was built under.
- **Properties over random graphs**, using the random `Recipe`s of step 3
  (`docs/04-core-graph.md` §4.1), now moved to the shared test module
  `test/HsGrad/Test/Graphs.hs`.

### 4.1 A reference evaluator

The main property compares the forward pass with a **reference evaluator**
that is the definition of §1.1 read as a program:

```haskell
refValue :: Graph -> Env -> NodeId -> Double
refValue g env i = case (nodeOp n, nodeArgs n) of
  (Leaf (Const c), []) -> c
  (Leaf (Input x), []) -> envInputs env Map.! x
  (Leaf (Param x), []) -> envParams env Map.! x
  (Un op, [a]) -> forwardUn op (ref a)
  (Bin op, [a, b]) -> forwardBin op (ref a) (ref b)
  …
```

No order, no map, no reachability, no error handling: obviously right, and
exponentially slow on shared nodes, so the property is limited to recipes of
at most 16 steps. For random graphs, environments and outputs, it checks:

- if every reachable node has a finite reference value, `evaluate` returns
  exactly those values (compared with `==`: both sides perform the same
  `Double` operations on the same inputs, so they agree bit for bit);
- otherwise `evaluate` returns `NonFiniteValue` for the **smallest** id
  whose reference value is non-finite, with that node's argument values
  (all finite) and the same value (NaN compared as equal to NaN).

Random graphs use `x ^ ½`, `exp` and division, so some of them do produce
NaN or ∞. That is only a few percent of cases, so this property runs 500
cases instead of the default 100; QuickCheck's `classify` prints the
percentage.

Two more properties: the schedule and the evaluated values cover exactly
`reachable g outs`, each node once; and with leaves bound at random (half of
them missing, and sometimes `Input "w"` bound instead of `Param "w"`),
`UnboundLeaves` lists exactly the reachable unbound leaves.

As a sanity check of the tests themselves, four deliberate bugs in
`Eval.hs` each make several tests fail: dropping the finiteness check
(7 failures), swapping the arguments of binary ops (3), reporting only the
first unbound leaf (2), and evaluating the whole graph instead of the
reachable part (6).

## 5. What later steps build on

- **Step 5 (`Core.Backprop`)**: take `scheduleNodes s` in reverse
  (descending ids) and the `Values` from `evalSchedule s env`; for each
  node, `localGrads (nodeOp n) (map (valueAt vals) (nodeArgs n)) (valueAt
  vals i)`. Forward values are guaranteed finite, so a non-finite gradient
  can only come from a rule itself (for example `relu`'s kink is fine, but
  `x ^ ½` at `0` gives `∞`).
- **Step 7 (`Core.Symbolic`)**: a gradient graph is evaluated with the same
  `evaluate`, reusing the forward pass's `Env` (names are stable across
  graphs, §3.2).
- **Step 8 (`Core.Simplify`)**: soundness tests compare `evaluate` before
  and after simplification on the same `Env`; they must skip inputs where
  either side reports `NonFiniteValue`.
- **Step 9 (`Train.Loop`)**: build the graph and `schedule g [loss]` once;
  per step, set `envParams` to the parameter store and `bindInput` the
  mini-batch rows into the dataset slots, then `evalSchedule`. Stop training
  on `NonFiniteValue` and report it.
- **Step 15 (`Driver`)**: map `UnboundLeaves` to E0502 (inputs without a
  default or `--set`; an unbound *parameter* is an internal error, since the
  parameter store initialises all of them) and `NonFiniteValue i args v` to
  E0501 at `provSpan (nodeProv (nodeAt g i))`, naming the op and the
  argument values.
