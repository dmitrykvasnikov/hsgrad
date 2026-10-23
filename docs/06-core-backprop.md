# 06 — `Core.Backprop`: the reverse pass

This step adds `HsGrad.Core.Backprop`, numeric reverse-mode automatic
differentiation: micrograd's `backward()`. Given

- a `Graph` (step 3) and the forward values of its nodes (step 4),
- one scalar **output** node `out`, typically the loss,

it computes the gradient `∂out/∂n` of the output with respect to every node
`n` the output depends on, in one sweep over the nodes in descending id
order. Each node passes its gradient on to its arguments, multiplied by its
op's local derivatives (`localGrads`, step 2), and each argument **adds up**
what it gets from all of its uses.

Like the values, the gradients live outside the graph, in an
`IntMap Double` keyed by node index (D1). For the training loop (step 9)
there are accessors by name: the gradient of every input and parameter
leaf, in the same shape as the `Env` that gave them their values.

This is the first of the two reverse sweeps of D6. Step 7 (`Core.Symbolic`)
runs the same sweep, but builds graph nodes instead of computing numbers.

---

## 1. Theory

### 1.1 The chain rule, one node at a time

Take micrograd's first example (`intro.hsg`, `01-language.md` §1.1):

```
e = a · b        a = 2, b = -3, c = 10
d = e + c        e = -6, d = 4
L = d · f        f = -2, L = -8
```

We want `∂L/∂a`, `∂L/∂b`, `∂L/∂c`: how much `L` changes, per unit, when one
leaf changes a little. The chain rule says that the effect of a small change
travels along the graph and gets multiplied, at every node, by that node's
**local derivative**: `∂d/∂c = 1` (a change in `c` passes through `+`
unchanged), `∂L/∂d = f` (multiplying by `f` scales it by `f`), and so on.

Reverse mode starts at the output, with `∂L/∂L = 1`, and walks back towards
the leaves. At every node `y = op(x₁, …, xₖ)` whose gradient `∂L/∂y` is
known, it hands each argument its share:

```
∂L/∂xᵢ = ∂L/∂y · ∂y/∂xᵢ
```

For the example:

| Node | Rule | Gradient |
|---|---|---|
| `L` | seed | `∂L/∂L = 1` |
| `d`, `f` | `L = d · f`: local derivatives `f`, `d` | `∂L/∂d = 1 · f = -2`, `∂L/∂f = 1 · d = 4` |
| `e`, `c` | `d = e + c`: local derivatives `1`, `1` | `∂L/∂e = -2`, `∂L/∂c = -2` |
| `a`, `b` | `e = a · b`: local derivatives `b`, `a` | `∂L/∂a = -2 · b = 6`, `∂L/∂b = -2 · a = -4` |

These are the numbers `hsgrad derive intro.hsg` will print. Every step only
needed the local derivative of one op (step 2's rules), the values at that
node (step 4's forward pass), and the gradient of the node itself.

### 1.2 Shared nodes: gradients add up

The table above is a tree: every node is used once. In a DAG (D1) a node can
be used many times, by different nodes or twice by the same one. Then the
rule becomes:

```
∂L/∂n = Σ over every use of n:  ∂L/∂u · ∂u/∂n
```

where `u` is the node that uses `n`, and `∂u/∂n` is `u`'s local derivative
for the argument position where `n` appears. A node used three times gets
three contributions, and its gradient is their **sum**. This is the
multivariable chain rule, and it is the heart of backprop.

Why a sum? Change `n` by a tiny `ε`. Every use of `n` sees the same change
`ε`. Each use, on its own, would change `L` by `(∂L/∂u · ∂u/∂n) · ε`. For a
small enough `ε` the function is effectively linear, and in a linear
function the effects of several small changes simply add up. So the total
change is the sum of the contributions times `ε`, and the gradient is the
sum.

Three small cases show it, and each is a unit test:

- **`a + a`**. Hash-consing makes this one `Add` node with arguments
  `[a, a]`. Each position has local derivative `1`, and both positions are
  `a`, so `∂/∂a = 1 + 1 = 2`. Check: `a + a = 2a`.
- **`a · a`**. Local derivatives `y` and `x`, which are both `a`:
  `∂/∂a = a + a = 2a`, the derivative of `a²`. The product rule
  `(fg)' = f'g + fg'` is nothing but this: the chain rule plus summing over
  the two uses.
- **`a / a`**. The two partials are `1/a` and `-a/a²`, which cancel:
  `∂/∂a = 0`, as it should for the constant function `1`.

micrograd writes this rule as `+=`:

```python
self.grad += other.data * out.grad
other.grad += self.data * out.grad
```

Karpathy's lecture shows what happens with `=` instead of `+=`: for
`b = a + a`, the second line overwrites the first, and `a.grad` comes out
as `1` instead of `2`. Here the same mistake would be one changed line in
`passOn` (§3.2), and it makes ten tests fail (§5.3).

**The path view.** Unfold the sum: `∂L/∂n` is the sum, over every *path*
from `n` up to `L`, of the product of the local derivatives along the path.
A shared node can have exponentially many paths. In the chain of doublings
`d₀ = x`, `dₖ₊₁ = dₖ + dₖ`, there are `2⁴⁰` paths from `d₄₀` down to `x`,
each contributing `1`, so `∂d₄₀/∂x = 2⁴⁰`. Backprop never enumerates the
paths. Every node's gradient is a sum over its uses, and every use's
gradient was already summed further up, so each node does its work once.
This is dynamic programming, the same trick that made the forward pass
linear (`docs/05-core-eval.md` §1.1). The test computes `2⁴⁰` from 41 nodes
in no time.

### 1.3 Why descending ids

The sum in §1.2 creates an ordering constraint. A node may pass its gradient
on only once it is **complete**, i.e. once every use has contributed. If it
passed it on earlier, the contributions that arrive later would never reach
its arguments.

The uses of `n` are nodes that have `n` as an argument, and by D2 every
such node has a larger id than `n`. So visiting nodes in **descending** id
order works: when the sweep reaches `n`, every node with a larger id has
been visited, so all of `n`'s users have already contributed, and `n`'s
gradient is final. The sweep is one loop:

```
grad[out] = 1
for each reachable node n, in descending id order:
    for each argument position i of n:
        grad[argᵢ] += grad[n] · ∂n/∂argᵢ
```

Reversing the forward schedule gives exactly this order. Sweeping in
ascending order instead is one of the deliberate bugs of §5.3: 13 tests
fail.

### 1.4 Why reverse mode

There is also *forward*-mode differentiation: seed one leaf with `1`, push
derivatives along with the values, and read `∂out/∂leaf` at the output.
It gives the derivative of every node with respect to **one** leaf per
sweep. Reverse mode gives the derivative of **one** output with respect to
every node per sweep.

Training has one output (the loss) and many leaves (every parameter: 337
in the 2-16-16-1 MLP of `moons.hsg`). Reverse mode gets all of them for
about the cost of one more pass over the graph; forward mode would need 337
passes. That is why micrograd, PyTorch and this module are reverse mode.

### 1.5 Nodes the output does not depend on

If `out` does not depend on a node, changing that node does not change
`out`, so `∂out/∂n = 0`. The sweep never even visits such a node with a
gradient: contributions only flow from `out` downwards, along arguments,
so they only ever reach nodes `out` depends on. Those are exactly the nodes
that end up with an entry in the result.

That covers both kinds of "unrelated" node: nodes that only *another*
output uses (a graph can have a `model` and a `loss`), and nodes that use
`out` themselves (larger ids, visited first, but with nothing to pass on).

For the language this is warning W0203 ("parameter can't affect `loss`"):
such a parameter has no gradient, so training would never change it.

### 1.6 Non-finite gradients

The forward pass guarantees that every value is finite
(`docs/05-core-eval.md` §1.4). Gradients can still be NaN or infinite,
for three reasons:

1. **An infinite slope.** `√x` at `x = 0` has the finite value `0`, but its
   local derivative `½ · 0^(-½)` is `∞`. The same goes for `x^(1/3)` at
   `0` and for `log x` and `1/x` at tiny `x`.
2. **Overflow in a product.** In `h = x · 10²⁰⁰`, `L = h · 10²⁰⁰` with
   `x = 10⁻²⁰⁰`, every value is finite (`h = 1`, `L = 10²⁰⁰`), but
   `∂L/∂x = 10²⁰⁰ · 10²⁰⁰` is not.
3. **Overflow in a sum.** Two uses each contributing `10³⁰⁸` are each
   finite; their sum is not.

The first case has a twist. `√(relu x)` at `x = -1` is the constant
function `0` near that point, so its derivative is `0`. Backprop instead
multiplies `relu`'s slope `0` by `√`'s slope `∞`, and `0 · ∞` is NaN.
Automatic differentiation differentiates the *program*, one op at a time,
not the mathematical function, and if some intermediate op has no
derivative at the point, the chain rule has nothing to work with.
micrograd gives NaN here too (its `relu` multiplies by `out.data > 0`), and
so do larger systems in similar spots; the classic PyTorch example is
`torch.where(x > 0, torch.sqrt(x), 0)` at `x = 0`. The fix belongs in the
model (for example `√(x + ε)`), so the useful thing to do is to point at
the `√`.

As in the forward pass, a NaN gradient is never harmless: one gradient step
later every parameter it touches is NaN. So the sweep checks every
gradient it writes and stops at the first non-finite one. Because the
sweep goes in descending order, that is the node where the problem starts:
the `√`, not the leaf the NaN would have spread to (§3.3).

## 2. From micrograd to hsgrad

micrograd's version, slightly shortened:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

Every piece has a counterpart here, and two of them are not needed at all:

| micrograd | hsgrad |
|---|---|
| a mutable `grad` field on every `Value` | a fresh `Grads` map, returned by `backward` |
| a `_backward` closure per node, made by each operator | `localGrads (nodeOp n) argVals outVal`: the node's op and argument ids are already in the graph |
| `build_topo`: a depth-first search for a topological order | not needed: ids are a topological order (D2); `reachable` gives the set |
| `self.grad = 1` | the seed `IntMap.singleton (nodeIndex out) 1` |
| `+=` into the arguments' fields | `passOn`: add to the argument's entry in the map |
| `zero_grad()` before every `backward()` | not needed |

**Why no closures.** A `_backward` closure captures three things: which op
the node is, which nodes its arguments are, and the values involved. Our
graph already stores the first two in every `Node`, and the forward pass
stores the values. So "running the closure" is just a function of data we
already have: `localGrads (nodeOp n) argVals (valueAt vals i)`, the
evaluated rule of step 2, times the node's gradient. Keeping rules as data
instead of closures is what lets step 7 interpret the same rules a second
way, as graph nodes (`docs/03-core-op.md` §3.1).

**Why no topological sort.** micrograd's objects only know their children,
and they are created by arbitrary Python code, so `backward()` has to
discover an order by DFS. Our builder hands out ids in creation order, and
a node can only refer to nodes that already exist, so the order is there
from the start (D2). We still need the *set* of reachable nodes, which the
forward pass's `Schedule` already holds.

**Why no `zero_grad`.** micrograd's gradients are fields that live as long
as the `Value`s. Every `backward()` adds into them, so a training loop must
reset them before each step; forgetting to is a classic bug, because the
gradients of all earlier steps then silently pile up. Here accumulation
only happens *within* one sweep, in a map that starts out as just the
seed, so there is nothing to reset and nothing to forget.

## 3. Code walkthrough

The module lives at `src/HsGrad/Core/Backprop.hs`.

### 3.1 Gradients

```haskell
type Grads = IntMap Double

lookupGrad :: NodeId -> Grads -> Maybe Double
gradAt     :: Grads -> NodeId -> Double        -- 0 if not reachable
```

`Grads` is a synonym for D1's `IntMap Double`, keyed by `nodeIndex`, like
`Values` (`docs/05-core-eval.md` §3.6). It has an entry for exactly the
nodes the output depends on, the output itself included. `lookupGrad` says
whether a node has one; `gradAt` returns `0` for a node without one
(§4.3).

### 3.2 The sweep

```haskell
backward :: Schedule -> Values -> NodeId -> Either BackpropError Grads
backward s vals out = case lookupValue out vals of
  Nothing ->
    error ("HsGrad.Core.Backprop.backward: output " <> show out <> " was not evaluated")
  -- The seed: ∂out/∂out = 1.
  Just _ -> sweep vals (IntMap.singleton (nodeIndex out) 1) (reverse (scheduleNodes s))
```

`backward` takes the forward pass's schedule and values, and the output.
The output must be one of the nodes the schedule evaluated; asking for
another node is a programming error, reported with `error` in the style of
`valueAt` (§4.1). The sweep starts from the seed and the schedule in
reverse, i.e. descending ids.

```haskell
sweep :: Values -> Grads -> [(NodeId, Node)] -> Either BackpropError Grads
sweep _ !grads [] = Right grads
sweep vals !grads ((i, n) : rest) = case lookupGrad i grads of
  Nothing -> sweep vals grads rest
  Just g -> do
    grads' <- foldM (passOn g) grads (zip (nodeArgs n) locals)
    sweep vals grads' rest
  where
    argVals = map (valueAt vals) (nodeArgs n)
    locals = localGrads (nodeOp n) argVals (valueAt vals i)
    passOn g acc (a, local)
      | isFinite total = Right (IntMap.insert (nodeIndex a) total acc)
      | otherwise = Left (NonFiniteGrad i argVals g locals)
      where
        total = IntMap.findWithDefault 0 (nodeIndex a) acc + g * local
```

This is the loop of §1.3:

- **A node without a gradient is skipped** (`Nothing`): the output does
  not depend on it (§1.5). This is also why the schedule may have been
  built for more outputs than `out`.
- **A node with a gradient `g`** zips its argument ids with its local
  derivatives (one per argument, none for a leaf) and folds `passOn` over
  the pairs. `foldM` is a left fold in a monad, here `Either`: each step
  returns the updated map or an error, and the first error stops the fold.
- **`passOn` is the `+=`.** The argument's new gradient is its old one (`0`
  if it has none yet) plus `g · local`. For `a + a` the pairs are
  `[(a, 1), (a, 1)]`, so the fold adds to `a`'s entry twice. Nothing
  special-cases repeated arguments; the sum of §1.2 falls out of the fold.
- **The check is on the sum**, so it catches all three cases of §1.6: an
  infinite local derivative, an overflowing product, and an overflowing
  sum.

`localGrads` and `valueAt` are safe here. Every reachable node and every
argument of one was evaluated, and the builder guarantees that `zip` pairs
every argument with its rule (`length (partials op) == arity op`, tested
in step 2).

As in the forward pass, the bangs (`!grads`) and `Data.IntMap.Strict` keep
the map and its values evaluated, so no chain of thunks builds up.

### 3.3 Errors

```haskell
data BackpropError
  = NonFiniteGrad NodeId [Double] Double [Double]
```

`NonFiniteGrad i args g locals` says: passing node `i`'s gradient on made
an argument's gradient non-finite. It carries the node's argument values,
its own gradient `g` (finite, since it was checked when it was written)
and its local derivatives. For `√x` at `0` that is
`NonFiniteGrad i [0] 1 [Infinity]`: everything a message like "the
derivative of `sqrt` is infinite at `0`" needs, plus the node, whose
`provSpan` points into the source.

The error names the node that *passes on*, not the argument that
*receives*. In cases 1 and 2 of §1.6 that is the node whose rule (or whose
large gradient) produced the bad number, which is where the user has to
look. In case 3 no single contribution is wrong, and the error names the
use whose share tipped the sum over.

### 3.4 Forward and backward together

```haskell
data GradError
  = ForwardError EvalError
  | BackwardError BackpropError

forwardBackward :: Schedule -> Env -> NodeId -> Either GradError (Values, Grads)
forwardBackward s env out = do
  vals <- either (Left . ForwardError) Right (evalSchedule s env)
  grads <- either (Left . BackwardError) Right (backward s vals out)
  pure (vals, grads)

gradients :: Graph -> Env -> NodeId -> Either GradError (Values, Grads)
gradients g env out
  | null (scheduleNodes s) = Right (IntMap.empty, IntMap.empty)
  | otherwise = forwardBackward s env out
  where
    s = schedule g [out]
```

The two passes are almost always run together, so there is a convenience,
in the two flavours `Core.Eval` has (`docs/05-core-eval.md` §3.1):

- `forwardBackward` takes a schedule, for the training loop: build
  `schedule g [loss]` once, then call this once per step with the new
  environment.
- `gradients` builds the schedule for one output itself, for one-off uses
  such as `hsgrad derive` and the tests.

Both return the values too: training wants the loss value for its report
and the gradients for its update, from the same forward pass.

`GradError` wraps either pass's error, so one `Either` covers both.
`either (Left . ForwardError) Right` turns an `Either EvalError` into an
`Either GradError` (from `base`, the same thing is `first ForwardError` with
`Data.Bifunctor`; spelling it out shows what happens). `gradients` cannot
reach `backward`'s `error`: an id that is not a node of the graph gives an
empty schedule, and so no values and no gradients.

### 3.5 Leaf gradients by name

```haskell
data LeafGrads = LeafGrads
  { inputGrads :: !(Map Name Double)
  , paramGrads :: !(Map Name Double)
  }

leafGrads :: Schedule -> Grads -> LeafGrads
leafGrads s grads =
  LeafGrads
    { inputGrads = Map.fromList [(n, g) | (i, Input n) <- leaves, Just g <- [lookupGrad i grads]]
    , paramGrads = Map.fromList [(n, g) | (i, Param n) <- leaves, Just g <- [lookupGrad i grads]]
    }
  where
    leaves = scheduleLeaves s
```

The training loop thinks in parameter *names* (`W1[3,0]`), not node ids,
just as it does for values (`docs/05-core-eval.md` §3.2). `LeafGrads` has
the shape of `Env`: one map for inputs, one for parameters, since
`Input "w"` and `Param "w"` are different leaves with different gradients.

Each name has exactly one gradient: hash-consing (D3) makes every leaf a
single node, however often it is used, and that node's gradient is already
the sum over all its uses.

The comprehensions use patterns on the left of `<-` to filter: `(i, Input
n) <- leaves` skips everything that is not an input, and
`Just g <- [lookupGrad i grads]` skips leaves without a gradient (a
one-element list is the usual trick to bind and filter in one step).

A training step is then:

```haskell
case forwardBackward s env loss of
  Right (vals, grads) ->
    let ps = paramGrads (leafGrads s grads)
        update name p = p - lr * Map.findWithDefault 0 name ps
     in ... valueAt vals loss ... Map.mapWithKey update (envParams env) ...
  Left err -> ...
```

## 4. Design decisions

### 4.1 Schedule and values in, gradients out

`backward` takes the forward pass's `Schedule` and `Values` rather than the
graph. The schedule already holds the reachable nodes with their `Node`s,
so backprop needs no second reachability search and no graph lookups, and
the training loop reuses the one schedule for both passes.

The schedule may cover more outputs than the one being differentiated,
because nodes without a gradient are skipped (§3.2). A tested property
checks that differentiating `out` with a schedule for several outputs gives
exactly the same gradients as with `schedule g [out]`.

The precondition (values from the same schedule, `out` evaluated) is not
checked beyond `out` itself: checking that two arguments belong together
would cost as much as the sweep. `forwardBackward` and `gradients`
establish it by construction, and they are what most callers use.

### 4.2 One output at a time

Reverse mode differentiates one scalar per sweep (§1.4), so `backward`
takes one `NodeId`. Gradients of several outputs are several calls. Nothing
planned needs more: training differentiates the loss, and
`hsgrad derive --of NAME` names one output.

### 4.3 A missing gradient is zero, a missing value is unknown

`valueAt` fails for a node that was not evaluated, because its value is
unknown. `gradAt` returns `0` for a node without a gradient, because that
gradient is *known*: the output does not depend on the node, so `∂out/∂n`
is `0` (§1.5). The map still leaves such nodes out instead of storing
zeros, so that "has a gradient" means "the output depends on it", which
`lookupGrad`, `leafGrads` and the visualisation (step 6) can use.

### 4.4 Non-finite gradients are errors, always

The arguments of `docs/05-core-eval.md` §3.4 apply unchanged: the check
costs two comparisons per contribution, a late NaN is much harder to trace
than an early one, and a NaN gradient destroys the parameters at the next
update. There is no switch to turn it off.

The error is reported at the first node in sweep order. Since gradients
flow from the output down, that is the node where the problem *starts*,
just as the forward pass reports the first non-finite node in its own
order.

### 4.5 Constants get gradients too

`Const` nodes are nodes, and nothing in the sweep treats them specially:
`∂L/∂f = 4` for the constant `-2` in `intro.hsg`. Special-casing them would
save a few map entries and make the sweep less uniform. The gradient of a
constant is also meaningful ("how much would the loss change if this
literal were different?"), and micrograd's `draw_dot` shows a gradient for
every `Value`, constants included, which step 6 will imitate. `leafGrads`
leaves them out: they have no name, and nothing updates them.

### 4.6 Values come back with the gradients

`forwardBackward` returns `(Values, Grads)`, not just the gradients. The
forward values are a by-product the caller nearly always wants (the loss
for the training report, the node values for `draw_dot`-style pictures),
and recomputing them would double the forward work.

## 5. Tests

`test/HsGrad/Core/BackpropSpec.hs` has five groups:

- **micrograd examples**: `intro.hsg` (`∂L/∂a = 6`, `∂L/∂b = -4`,
  `∂L/∂c = -2`, and every intermediate gradient), the same leaf gradients by
  name; micrograd's tanh neuron (`x1.grad = -1.5`, `w1.grad = 1`,
  `x2.grad = 0.5`, `w2.grad = 0`); and the expression from micrograd's
  README, translated line by line, with plenty of sharing, `relu`, powers
  and divisions (`g = 24.7041`, `∂g/∂a = 138.8338`, `∂g/∂b = 645.5773`).
- **Sharing**: `a + a`, `a · a`, `a / a` (§1.2); a leaf used by two
  different nodes; a shared interior node (`L = d · d`) passing its
  complete gradient on; the chain of 40 doublings.
- **Reachability**: the output's own gradient is `1`; with a schedule for
  two outputs, the nodes only the other output uses (and the other output
  itself) have no gradient and are not in `leafGrads`; `Input "w"` and
  `Param "w"` get separate gradients; an id of another graph gives nothing.
- **Errors**: `√x` at `0`; `√(relu x)` at `x = -1` (the `0 · ∞` case,
  reported at the `√`); an overflowing product, reported at the node that
  passes it on; an overflowing sum, reported at the use that tips it over;
  forward errors passed through `gradients`; `relu`'s gradient at the kink
  is `0`, as in micrograd.
- **Properties over random graphs**: the gradient check (§5.1); the
  gradients cover exactly `reachable g [out]`; and the gradients don't
  depend on which other outputs the schedule covers (§4.1).

### 5.1 The gradient check

The main property compares backprop with **central finite differences**,
the same approximation step 2 used for single ops
(`docs/03-core-op.md` §4.1), now for whole random graphs. For a random
graph, the node of its last step as output, a random point, and every
input and parameter leaf `v` the output depends on:

```
∂out/∂v  ≈  (out(v + h) − out(v − h)) / 2h,      h = 10⁻⁶ · max(1, |v|)
```

Both `out(v ± h)` are plain forward passes with `v` rebound in the
environment; the finite difference knows nothing about backprop. The two
must agree to a relative tolerance of `10⁻⁵`, absolute below `1`:
`|grad − fd| ≤ 10⁻⁵ · max(1, |fd|)`.

The random graphs come from a new generator, `genSmoothStep` (in
`test/HsGrad/Test/Graphs.hs`, next to step 3's `genStep`), which uses every
op of the set: also `log`, `step`, and the powers `3`, `-1` and `-½` that
step 3's generator doesn't have. It deliberately does *not* avoid singular
points. Instead, the property **discards** the samples where a finite
difference is not a fair reference:

- the forward pass fails at the point or at `v ± h` (outside the domain,
  for example `log` of a negative number);
- some value, at any of the points, or some finite difference is larger
  than `10³` in magnitude (close to a pole like `1/x` at `0`, or an
  overflow in the making);
- some `relu` or `step` argument is `0`, or has different signs at the
  three points: the difference straddles a kink or a jump, where the
  function has no derivative to approximate;
- the output depends on no input or parameter: there is nothing to
  compare.

About 30% of samples are discarded this way. Of the 500 that are checked,
about half contain a node used more than once, so the sum of §1.2 is
exercised, and about 10% contain `relu` or `step`.

### 5.2 Why this tolerance

A finite difference has two sources of error (`docs/03-core-op.md` §4.1):

- **Truncation**: the central difference is off by about `h² · f'''/6`.
  With `h ≤ 3·10⁻⁶` (the leaves are in `[-3, 3]`) that is `10⁻¹²·f'''`,
  negligible unless the third derivative is huge, which the `10³` bounds
  make unlikely.
- **Rounding**: each forward pass rounds every node's value, by about
  `ε ≈ 10⁻¹⁶` relative. A graph of up to 30 nodes with values up to `10³`
  can be off by roughly `30 · 10³ · 10⁻¹⁶ ≈ 3·10⁻¹²` absolute; divided by
  `2h ≈ 2·10⁻⁶`, the difference can be off by about `10⁻⁶`.

In practice the agreement is much better than these worst cases: with
20,000 samples, a tolerance of `10⁻⁷` still passes and `10⁻⁸` fails.
`10⁻⁵` leaves a hundredfold margin for unlucky samples, so the property
does not flake. It is still strict where it matters. The bugs backprop
can have (a lost contribution, partials swapped, a gradient passed on
before it is complete) are off by a whole contribution, not by `10⁻⁵`
of one.

Graphs are limited to 30 steps (`modifyMaxSize 15`): bigger random graphs
mostly overflow or are discarded.

### 5.3 Testing the tests

As in the earlier steps, four deliberate bugs in `Backprop.hs` each make
several tests fail:

| Bug | Failing tests |
|---|---|
| overwrite instead of add (`total = g * local`, micrograd's `=` bug) | 10 |
| sweep in ascending instead of descending order | 13 |
| pair the arguments with their partials in the wrong order | 9 |
| drop the finiteness check | 5 |

The first and third are also caught by the gradient check on its own.

The full suite has 136 tests and runs in about 0.15 s.

## 6. What later steps build on

- **Step 6 (`Viz.Dot`)**: micrograd's `draw_dot` shows `data` and `grad`
  for each node. `gradients g env out` gives both maps; `lookupGrad`
  returning `Nothing` means "the output does not depend on this node",
  which a picture can show as no gradient at all rather than `0`.
- **Step 7 (`Core.Symbolic`)**: the same sweep, in the same order, with
  the same seed and the same accumulation, but in the `Builder`: the seed
  is a `Const 1` node, `g · local` becomes a `Mul` node (local derivatives
  from the node-emitting `DExpr` interpreter), and `+=` becomes an `Add` of
  the old gradient node and the new contribution (or just the contribution,
  for the first use). Its main test: evaluating the gradient graph gives
  `backward`'s numbers. A leaf without an entry has gradient `0`, as here.
- **Step 9 (`Train.Loop`)**: `s = schedule g [loss]` once; per step,
  `forwardBackward s env loss`, then
  `p − lr · Map.findWithDefault 0 name (paramGrads (leafGrads s grads))`
  for every parameter. A missing name is a parameter the loss does not
  depend on (W0203). Stop on any `GradError` and report it.
- **Step 15 (`Driver`)**: `ForwardError` maps as in step 4 (E0501, E0502).
  `BackwardError (NonFiniteGrad i args g locals)` needs a diagnostic of its
  own, at `provSpan (nodeProv (nodeAt g i))`: the language spec only lists
  E0501 for non-finite *values*, so step 15 should either add a code for
  non-finite gradients (E0504, say) or extend E0501's wording. The message
  can name the op and its arguments ("the derivative of `sqrt` is infinite
  at `0`").
