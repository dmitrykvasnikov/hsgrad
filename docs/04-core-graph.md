# 04 — `Core.Graph`: nodes, graphs and the builder

This step adds `HsGrad.Core.Graph`: the data structure every later Core
module works on, and the one way to create it, the `Builder` monad. A graph
is a DAG of scalar operations from `Core.Op`; nodes refer to each other by
`NodeId`. The builder guarantees four invariants that later steps rely on:

- **ids are a topological order** (D2): arguments always have smaller ids
  than the nodes that use them;
- **hash-consing** (D3): the graph never holds two nodes with the same op
  and the same arguments;
- **arity**: a node has exactly `arity op` arguments;
- **finite constants**: no `Const` is NaN or infinite.

Every node also carries **provenance** (D4): an optional source span and an
optional label. The module also computes **reachability** (the nodes some
outputs depend on), which the forward pass, backprop and symbolic
differentiation will use to skip dead nodes.

There is no evaluation yet. Step 4 (`Core.Eval`) is the first consumer.

---

## 1. Theory

### 1.1 Expressions are DAGs, not trees

Take micrograd's running example, slightly shortened:

```python
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
e = a * b
d = e + c
L = d * d
```

As a *tree*, `L` has two copies of `d`, each with its own copy of `e`. But
the program computed `d` once and used it twice, and backprop must know
that: the gradient of `d` is the *sum* of what both uses contribute. The
right picture is a **directed acyclic graph** in which `d` is one node with
two outgoing edges.

micrograd gets this for free from Python object identity: `d` is one
`Value` object, and `L._prev` holds two references to it. In Haskell, values
have no identity, so two equal subtrees can't be told apart from one shared
one. We make identity explicit instead (D1): every node gets a `NodeId`, and
a node lists its arguments by id.

```
id  op       args      (comment)
0   Param a  []
1   Param b  []
2   Param c  []
3   Mul      [0, 1]    e = a * b
4   Add      [3, 2]    d = e + c
5   Mul      [4, 4]    L = d * d   (d used twice: the same id twice)
```

Values and gradients are kept outside the graph, in maps from id to number,
so the graph itself never changes after it is built.

### 1.2 Topological order for free (D2)

A **topological order** of a DAG lists every node after all the nodes it
depends on. The forward pass needs one (compute arguments before the node),
and backprop needs the reverse (finish a node's gradient before passing it
on to its arguments). micrograd computes one with a depth-first search in
`backward()`.

We get one from the construction itself. The builder hands out ids
`0, 1, 2, …` in creation order, and a node can only be created from nodes
that already exist. So every argument id is smaller than the id of the node
using it, and **ascending id order is a topological order**. In the table
above, `5` comes after `4`, which comes after `3` and `2`, and so on.

### 1.3 Hash-consing (D3)

"Hash-consing" is the old Lisp name for building every structure through a
table, so that equal structures are only built once. Here the table maps a
node's **key**, its op and its argument ids, to the node's id:

```
(Leaf (Param "a"), [])      ↦ 0
(Bin Mul,          [0, 1])  ↦ 3
(Bin Mul,          [4, 4])  ↦ 5
…
```

Before creating a node, the builder looks its key up. If it is there, the
existing id is returned and nothing is created. So writing `a * b` twice
gives one node, and so does writing `tanh(a * b)` twice: the first `a * b`
merges, so both `tanh` nodes have the same argument id, so they merge too.
By induction from the leaves up, **two expressions get the same id exactly
when they are structurally equal**. This is common-subexpression
elimination at no extra cost.

It matters most for generated code. Symbolic differentiation (step 7)
emits nodes like `1 · g` or `(-1) · g` over and over, and the simplifier
(step 8) rebuilds whole graphs; with hash-consing, their repeated pieces
collapse into shared nodes.

Hash-consing is **structural**, not mathematical. `x + y` and `y + x` are
different keys, and so are `x · 2` and `2 · x`. Recognising those as equal
needs knowledge of the ops (commutativity, constant folding), which is the
simplifier's job (D7). The simplifier produces a canonical form, and then
hash-consing merges what it made equal.

### 1.4 Reachability

The nodes an output **depends on** are the output, its arguments, their
arguments, and so on down to the leaves. A graph can contain nodes the
output does not depend on: a `let` that is never used, the `model` while
only the `loss` is evaluated, or the forward graph's leftovers when only a
gradient is wanted. `reachable g outs` computes the set of nodes the outputs
`outs` depend on, so later steps only evaluate (or differentiate) those.

Because the result is a set of ids, its ascending order is still a
topological order (1.2): step 4 evaluates `Set.toAscList (reachable g outs)`
and step 5 walks `Set.toDescList`.

## 2. Code walkthrough

The module lives at `src/HsGrad/Core/Graph.hs`.

### 2.1 Node ids

```haskell
newtype NodeId = NodeId Int
  deriving (Eq, Ord, Show)

nodeIndex :: NodeId -> Int
nodeIndex (NodeId i) = i
```

The constructor `NodeId` is **not exported**. Outside this module, the only
way to get an id is from the builder, so every id that exists was handed out
in order, and `i < j` really means "`i` was created before `j`". `nodeIndex`
goes the other way, for code that keeps per-node numbers in an `IntMap`
(values in step 4, gradients in step 5, D1).

### 2.2 Nodes and graphs

```haskell
data Node = Node
  { nodeOp :: !Op
  , nodeArgs :: ![NodeId]
  , nodeProv :: !Provenance
  }

data Graph = Graph
  { graphNodes :: !(IntMap Node)                  -- keyed by nodeIndex: 0 .. graphCount-1
  , graphCount :: !Int                            -- number of nodes = next fresh id
  , graphTable :: !(Map (Op, [NodeId]) NodeId)    -- the hash-consing table
  }
```

A `Node` is what `docs/03-core-op.md` §5 promised: an `Op` plus a list of
`arity op` argument ids, plus provenance (§2.6).

`Graph`'s constructor is not exported either, so a graph can only come out
of a builder, with its invariants intact. Code outside reads it with:

| Function | Type | |
|---|---|---|
| `emptyGraph` | `Graph` | no nodes |
| `nodeCount` | `Graph -> Int` | O(1) |
| `nodes` | `Graph -> [(NodeId, Node)]` | ascending id order (topological) |
| `nodeIds` | `Graph -> [NodeId]` | ascending |
| `lookupNode` | `NodeId -> Graph -> Maybe Node` | `Nothing` for a foreign id |
| `nodeAt` | `Graph -> NodeId -> Node` | partial: `error` for a foreign id |
| `findNode` | `Op -> [NodeId] -> Graph -> Maybe NodeId` | the hash-consing lookup alone |
| `reachable` | `Graph -> [NodeId] -> Set NodeId` | §2.8 |

The argument orders follow `containers`: `lookupNode` takes the key first,
like `Map.lookup`, and `nodeAt` takes the graph first, like `Map.!`.
`nodeAt` is partial on purpose. Every id a caller holds normally came from
building this very graph, so a missing node is a bug, and a clear `error`
message is more useful than a `Maybe` that every caller has to unwrap with
an "impossible" case.

### 2.3 The `Builder` monad

```haskell
newtype Builder a
  = Builder (ReaderT (Maybe SrcSpan) (StateT Graph (Either GraphError)) a)
  deriving (Functor, Applicative, Monad)
```

D1 describes the builder as "a `State` over the graph under construction".
The real thing has two more effects, and each gets its own layer of a
monad-transformer stack (from `mtl`):

| Layer | Effect | Used by |
|---|---|---|
| `ReaderT (Maybe SrcSpan)` | a read-only context that can be changed for a scope | the current source span, `withSpan` (§2.6) |
| `StateT Graph` | state that is threaded through every step | the graph under construction |
| `Either GraphError` | early exit with an error | arity, unknown-node and non-finite-constant errors |

Reading the type from the inside out: an `Either GraphError` computation
that can fail; wrapped in `StateT`, which turns it into
`Graph -> Either GraphError (a, Graph)`; wrapped in `ReaderT`, which adds a
`Maybe SrcSpan` argument. So a `Builder a` is, underneath, a function

```haskell
Maybe SrcSpan -> Graph -> Either GraphError (a, Graph)
```

and the `Monad` instance (derived through the newtype with
`GeneralisedNewtypeDeriving`, which is in GHC2021) plumbs the span and the
graph through each step and stops at the first `Left`.

The newtype also **hides** the stack: `Builder` has no `MonadState`
instance, so nobody outside the module can `put` a graph that breaks the
invariants. The only ways in are the operations below.

### 2.4 `node`: the one place nodes are made

```haskell
node :: Op -> [NodeId] -> Builder NodeId
node op args = Builder $ do
  when (length args /= arity op) $
    throwError (ArityMismatch op args)
  case op of
    Leaf (Const c)
      | isNaN c || isInfinite c -> throwError (NonFiniteConst c)
    _ -> pure ()
  g <- get
  mapM_ (\a -> unless (exists a g) (throwError (UnknownNode a))) args
  sp <- ask
  let prov = Provenance sp Nothing
  case Map.lookup (op, args) (graphTable g) of
    Just i -> do
      put (addProvenance i prov g)
      pure i
    Nothing -> do
      let i = NodeId (graphCount g)
      put
        Graph
          { graphNodes = IntMap.insert (nodeIndex i) (Node op args prov) (graphNodes g)
          , graphCount = graphCount g + 1
          , graphTable = Map.insert (op, args) i (graphTable g)
          }
      pure i
```

Every node in every graph passes through these lines, so they are where the
four invariants are enforced:

1. **Arity**: the argument list must be as long as `arity op`.
2. **Finite constants**: NaN and infinities are rejected *before* the key
   touches the table. Derived `Ord` on a NaN `Double` is inconsistent
   (`docs/03-core-op.md` §3.5), and one bad key in a `Map` can make lookups
   of *other* keys fail.
3. **Arguments exist**: each argument must be an id of this graph. Ids are
   dense (`0 .. graphCount - 1`), so that is a range check. Together with
   the fresh id being `graphCount`, it gives the topological order: every
   argument is smaller than the new id.
4. **Hash-consing**: look the key up; return the old id on a hit, otherwise
   insert the node and its key and bump the counter.

Then a few one-line helpers, which cover almost every use:

```haskell
constant :: Double -> Builder NodeId           -- node (Leaf (Const c)) []
input    :: Name -> Builder NodeId             -- node (Leaf (Input n)) []
param    :: Name -> Builder NodeId             -- node (Leaf (Param n)) []
unary    :: UnOp -> NodeId -> Builder NodeId   -- node (Un op) [x]
binary   :: BinOp -> NodeId -> NodeId -> Builder NodeId
```

`unary` and `binary` take their arguments as separate parameters, so they
can't have the wrong arity: that check only ever fires for the generic
`node`, which code that copies nodes from one graph to another (the
simplifier, step 8) will use as `node op (map translate args)`.

micrograd's example, as a builder:

```haskell
example :: Builder NodeId
example = do
  a <- param "a"
  b <- param "b"
  c <- param "c"
  e <- binary Mul a b
  d <- binary Add e c
  binary Mul d d          -- d used twice: the same id twice
```

`build example` is `Right (NodeId 5, g)`, with the six nodes of §1.1.

### 2.5 Running a builder

```haskell
build  :: Builder a -> Either GraphError (a, Graph)
extend :: Graph -> Builder a -> Either GraphError (a, Graph)
extend g (Builder m) = runStateT (runReaderT m Nothing) g
build = extend emptyGraph

currentGraph :: Builder Graph
```

`build` starts from the empty graph; `extend` adds nodes to an existing
one. Running is just peeling the layers off in order: `runReaderT` with no
current span, then `runStateT` with the starting graph, leaving an `Either`.

`extend` keeps every old node at its old id, and because the hash-consing
table is part of the `Graph`, new nodes merge with old ones too. Step 7
needs exactly that: the gradient graph is built *into* the forward graph,
so that `∂L/∂x` can point at the forward graph's `tanh` node instead of
copying it (D6, `docs/03-core-op.md` §3.2).

`currentGraph` lets a builder look at what it has built so far, read-only.

### 2.6 Provenance

```haskell
data Pos = Pos { posLine :: !Int, posCol :: !Int }

data SrcSpan = SrcSpan
  { spanFile :: !FilePath
  , spanStart :: !Pos     -- inclusive
  , spanEnd :: !Pos       -- exclusive
  }

data Provenance = Provenance
  { provSpan :: !(Maybe SrcSpan)
  , provLabel :: !(Maybe Text)
  }
```

Provenance is the answer to "where did this node come from?" (D4): the
span of the source expression that created it, and an optional human label
such as `W1[3,0]` or `h`. The forward pass uses the span to report a NaN at
the expression that produced it (E0501), and Graphviz uses spans and labels
to group and name nodes. §3.4 explains why Core has its own span type.

**Spans are set for a scope**, not passed to every call:

```haskell
withSpan :: SrcSpan -> Builder a -> Builder a
withSpan sp (Builder m) = Builder (local (const (Just sp)) m)
```

Every node created inside `withSpan sp …` records `sp`, unless an inner
`withSpan` sets another span; afterwards the outer span is back. This is
what the `ReaderT` layer is for: `local` runs a sub-computation with a
modified context and restores it afterwards. The compiler (step 14) will
wrap the code for each source expression in that expression's span, so each
node gets the span of the innermost expression that created it. Vector and
matrix code unrolls into many scalar nodes (D5); they all get the span of
the expression that was unrolled, without the unrolling helpers having to
know about spans at all.

**Labels are given to one node**:

```haskell
label :: Text -> NodeId -> Builder ()
```

A label names one particular node (the result of `let h = …`, the parameter
leaf `W1[3,0]`), so it is attached to that node after it was built, not set
for a scope.

**Which provenance wins when nodes merge.** When hash-consing finds an
existing node, the caller still had a span (and possibly wants a label) in
mind. The rule is: **the first occurrence wins, field by field**. The
existing node keeps its span and label; a field it does not have yet is
filled in from the new request. That is exactly a left-biased `Semigroup`:

```haskell
instance Semigroup Provenance where
  Provenance s1 l1 <> Provenance s2 l2 = Provenance (s1 <|> s2) (l1 <|> l2)
```

`(<|>)` on `Maybe` picks the left `Just` if there is one. On a hit, the
builder stores `old <> new`; `label` does the same with a label-only
provenance. Provenance is never part of the hash-consing key: `x * y`
written on line 3 and on line 7 is one node, and it points at line 3.

### 2.7 Errors

```haskell
data GraphError
  = ArityMismatch Op [NodeId]
  | UnknownNode NodeId
  | NonFiniteConst Double
```

The first error aborts the build, and `build` returns it as a `Left`. All
three are bugs of the *caller* (the compiler, the differentiator, the
simplifier), never mistakes in a user's `.hsg` file: the checker has
rejected those long before a graph is built. `UnknownNode` happens when an
id from one graph is used in another, for example an id from a graph that
was later discarded.

### 2.8 Reachability

```haskell
reachable :: Graph -> [NodeId] -> Set NodeId
reachable g = go Set.empty
  where
    go seen [] = seen
    go seen (i : stack)
      | i `Set.member` seen = go seen stack
      | otherwise = case lookupNode i g of
          Nothing -> go seen stack
          Just n -> go (Set.insert i seen) (nodeArgs n ++ stack)
```

A depth-first search with an explicit stack (a list) of ids to visit and the
set of ids already seen. A node is expanded only the first time it is
popped, so shared nodes cost nothing extra: with `r` reachable nodes, it
does O(r) set operations of O(log r) each, however much sharing there is.
(Without the `seen` check, a chain of `n` nodes that each use the previous
one twice, like `L = d * d`, would be visited 2ⁿ times.)

Ids that are not nodes of the graph are skipped, so the result is always a
subset of the graph's ids.

## 3. Design decisions

### 3.1 Abstract `NodeId` and `Graph`

Both constructors are hidden, so the invariants can only be established by
the builder and can't be broken afterwards. The price is small: code that
needs a plain number has `nodeIndex`; code that needs a node has
`lookupNode`/`nodeAt`. There is no function from `Int` to `NodeId`, which
would let anyone forge ids; no step planned so far needs one.

Hiding does not stop an id from being used with the *wrong* graph (one
built by a different `build`). Ruling that out statically is possible, with
a phantom type parameter in the style of `runST`, but it would add a type
variable to every signature of every Core module. Instead the builder checks
it at runtime (`UnknownNode`), and lookups return `Maybe`.

### 3.2 Storage: `IntMap` plus a counter

Ids are dense, so nodes could live in a list, a `Data.Sequence`, a
`Data.Map NodeId Node` or an `IntMap`. `IntMap Node` was chosen because
lookup and insert are fast, it iterates in ascending key order (which is
the topological order), and the rest of Core keeps its per-node values in
`IntMap`s too (D1), so there is one idiom throughout. `IntMap.size` takes
linear time, so the graph also stores its node count, which doubles as the
next fresh id.

The hash-consing table is a `Data.Map` keyed on `(Op, [NodeId])`, using the
`Ord` instances derived in step 2. A real hash table would be faster but
needs `IO`/`ST` or the `hashable` and `unordered-containers` packages; an
ordered map keeps the builder pure and dependency-free, and at tens of
thousands of nodes (D5) the difference doesn't matter.

### 3.3 The table lives in the graph

The table could be builder state that is thrown away when the build ends,
leaving a smaller `Graph`. Keeping it in the graph costs memory, but makes
`extend` possible: a second builder run can merge with nodes of the first.
Step 7 needs that, and `findNode` comes for free.

### 3.4 A Core-level span type

D4 says every node carries a source span, but the real span type belongs to
the language (`Lang.Span`, step 10), and Core must not depend on Lang
(`00-architecture.md` §2). Three ways out:

- **A type parameter**: `Graph p`, `Node p`, `Builder p a`, with Lang
  choosing `p = Lang.Span`. Core would be fully agnostic, but every
  signature in Eval, Backprop, Symbolic, Simplify and Viz would grow a `p`,
  and the ones that report or draw spans would need class constraints to do
  anything with it.
- **An opaque tag**, say `Maybe Int`, that Lang maps to spans in a table of
  its own. Simple types, but every consumer needs the extra table, and a
  span can't be shown without it.
- **A small concrete span type in Core**: file, start and end line/column.
  That is all a diagnostic needs to point into a file, and Graphviz can
  print it directly.

We take the third. `SrcSpan` is deliberately minimal; step 10's `Lang.Span`
can be this type or wrap it with more (byte offsets, say) and convert when
compiling. Lines and columns are 1-based (as in editors and in megaparsec's
`SourcePos`), and `spanEnd` is exclusive: it is the position just after the
expression, which is what a parser has at hand when the expression ends.

### 3.5 First occurrence wins

When `x * y` appears at two places in the source, the merged node can only
point at one. The first one is the natural choice: in a define-before-use
language (D9) it is where the value is first computed, and the rule is
deterministic and does not depend on later code. Filling in *missing*
fields (instead of ignoring the second request completely) matters for
nodes created without a span, for example a `Const 1` emitted by symbolic
differentiation that is later also written in the source; it then gains the
source span. Labels follow the same rule, so `let a = x * y` followed by
`let b = x * y` labels the shared node `a`.

### 3.6 Spans for a scope, labels for a node

Passing a span to every `unary` and `binary` would be the most explicit
design, but the compiler's helpers that unroll `W @ x` into dozens of
multiplications and additions would then all need a span parameter that
they only pass along. A scoped span (`withSpan`, implemented with
`ReaderT`'s `local`) matches how spans nest in the source and keeps those
helpers span-free. A label, on the other hand, belongs to one node; setting
it for a scope would label every intermediate node inside, so it is
attached explicitly with `label`.

### 3.7 Errors as values, not exceptions

Builder errors are caller bugs, so `error` (an exception) would be
defensible. But returning them in an `Either` makes the failure part of the
type, lets the tests check each error precisely, and lets the driver report
an internal error cleanly instead of crashing. The cost is a `case` at each
`build`, where there is a natural place to handle it anyway.

### 3.8 Constants: no NaN, and `-0.0` merges with `0.0`

Rejecting non-finite constants keeps the `Map` sound (§2.4); the simplifier
must make sure its constant folding never produces one
(`docs/03-core-op.md` §3.5). `Const 0` and `Const (-0)` have equal keys,
because `0 == -0` for `Double`, so they merge into whichever came first.
The only observable difference between the two zeros is the sign of an
infinity (`1 / -0 = -∞`), which is an E0501 error either way.

## 4. Tests

`test/HsGrad/Core/GraphSpec.hs` has five groups:

- **Building**: sharing (`x * x` refers to one id twice), hash-consing
  (`x + y` twice is one node; equal leaves and constants merge; whole
  expressions merge bottom-up), nodes that differ in op, arguments,
  argument order or leaf kind stay apart, `0.0`/`-0.0` merge, ids are
  `0, 1, 2, …` in creation order and a merge uses up no id, `findNode`,
  `extend` keeps old nodes and merges with them.
- **Errors**: too few or too many arguments for each kind of op, an
  argument from another graph, `±∞` and NaN constants, and that the first
  error aborts the build.
- **Provenance**: no span outside `withSpan`, the innermost span wins and
  the outer one comes back, the first span and the first label win on a
  merge, a missing span is filled in, provenance doesn't affect
  hash-consing, and the combination rule of §2.6 on examples.
- **Reachability** on a small graph with a dead node: outputs and their
  dependencies are in, the dead node is out, unions over several outputs,
  the empty case, a lone leaf, foreign ids ignored.
- **Properties over random graphs**.

### 4.1 Random graphs

A random graph is described by a **recipe**: a list of steps, each a leaf,
or a unary or binary op whose arguments are *indices* into the ids produced
by the earlier steps (taken modulo their number). Every list of steps is a
valid recipe, and so is every sublist, so QuickCheck can shrink a failing
recipe by simply dropping steps. The generator uses only six leaves and six
unary ops, so random graphs are full of structurally identical nodes for
hash-consing to merge.

On those graphs, QuickCheck checks the invariants directly:

- ids are exactly `0 .. n-1`, ascending;
- every argument has a smaller id than its node;
- every node has `arity` arguments;
- no two nodes have the same key, and `findNode` finds each node under its
  own key;
- running a recipe twice in one build creates no new nodes and returns the
  same ids; building is deterministic; `extend` keeps the old graph as a
  prefix.

`reachable` is compared with a **reference definition** that is obviously
correct and slow: start from the outputs and keep adding the arguments of
everything in the set until it stops growing. A second property checks that
the result is closed under taking arguments.

As a sanity check of the tests themselves, disabling the table lookup in
`node` (so that nothing is ever merged) makes twelve of them fail, including
the "no two nodes have the same key" property.

## 5. What later steps build on

- **Step 4 (`Core.Eval`)**: iterate `Set.toAscList (reachable g outs)`,
  look nodes up with `nodeAt`, store values in an `IntMap Double` keyed by
  `nodeIndex`. A non-finite value is reported with the node's `provSpan`
  (E0501).
- **Step 5 (`Core.Backprop`)**: the same set in descending order
  (`Set.toDescList`), with `localGrads` from step 2.
- **Step 6 (`Viz.Dot`)**: `nodes g` for every node; `provLabel` and
  `provSpan` for names and clusters.
- **Step 7 (`Core.Symbolic`)**: `extend` the forward graph, so gradient
  nodes share and merge with forward nodes; `DOut` is the forward node's own
  id. `withSpan` can give gradient nodes the span of the forward node they
  differentiate.
- **Step 8 (`Core.Simplify`)**: rebuild with `node op (map translate args)`
  into a fresh `build`; hash-consing merges what the canonical form made
  equal. Folding must never produce a non-finite constant (it would be a
  `NonFiniteConst` error).
- **Step 10 (`Lang.Span`)**: reuse or wrap `SrcSpan`/`Pos` (1-based,
  exclusive end).
- **Step 14 (`Lang.Compile`)**: create leaves at their declarations (so a
  leaf's span is its declaration), wrap each expression's code in
  `withSpan`, and `label` named results.
