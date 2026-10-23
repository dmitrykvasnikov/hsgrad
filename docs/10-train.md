# 10 — `Train.*`: parameters, SGD and the training loop

This step adds the three modules that turn a differentiable graph into a
trained model:

- `HsGrad.Train.Param`: what a parameter is (name, shape, initialiser),
  the **parameter store** that holds the current value of every parameter
  leaf, and deterministic **initialisation** from a seed;
- `HsGrad.Train.Optim`: plain **stochastic gradient descent** and the
  **learning-rate schedules** of the planned `--lr-schedule` flag;
- `HsGrad.Train.Loop`: the **training loop**. The graph is built once;
  every step binds the parameters and a batch of dataset rows, runs the
  forward and backward passes (steps 4 and 5), updates the parameters, and
  reports the loss and, optionally, the accuracy.

Nothing here knows about the `.hsg` language. The milestone is driven from
Haskell code: a 2-4-1 tanh network, built with the Core `Builder` exactly
as the compiler (step 14) will build `xor.hsg`, learns XOR.

---

## 1. Theory

### 1.1 Gradient descent

Training means finding parameter values that make the loss small. The
gradient `∂L/∂p` (step 5) says how the loss changes when one parameter
changes a little. Taken together, the gradients of all parameters form a
vector that points in the direction in which the loss **grows** fastest.
So a small step the other way makes it smaller:

```
p ← p − η · ∂L/∂p          for every parameter p
```

`η` (eta), the **learning rate**, sets the size of the step. Too small and
training crawls; too large and the step overshoots the valley it was
aiming for, and the loss goes up or even becomes infinite (§5 has a test
that drives `log w` to a negative `w` this way). Why a small step really
lowers the loss: near the current point the loss is almost linear,
`L(p + δ) ≈ L(p) + Σ ∂L/∂p · δ`. With `δ = −η · ∂L/∂p` the change is
`−η · Σ (∂L/∂p)²`, which is negative unless every gradient is zero. One of
the tests checks exactly this on a small graph, at random points.

This is micrograd's update, line for line:

```python
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

### 1.2 Learning-rate schedules

A rate that is good at the start, far from any minimum, is often too
large near the end, where it makes the parameters bounce around the
minimum instead of settling in it. A **schedule** lowers the rate over the
run. The CLI plans three (`00-architecture.md` §5), for a run of `n` steps
numbered `t = 0 … n−1` and a base rate `η₀` (`--lr`):

| Flag | Rate at step `t` |
|---|---|
| `const` | `η₀` |
| `linear:END` | `(1 − t/n) · η₀ + (t/n) · END`: a straight line from `η₀` towards `END` |
| `step:K:F` | `η₀ · F^(t div K)`: multiplied by `F` every `K` steps |

micrograd's demo uses `learning_rate = 1.0 - 0.9*k/100` for 100 steps,
which is `--lr 1.0 --lr-schedule linear:0.1`: the line reaches `END` at
step `n`, just *after* the last step (§4.5).

### 1.3 Stochastic gradient descent and mini-batches

The loss of a dataset is a mean over its rows. Its gradient is the mean of
the rows' gradients, so computing it costs a pass over the whole dataset.
**Stochastic** gradient descent instead uses a random **mini-batch** of `B`
rows per step: the gradient of the batch's mean loss is a noisy but
unbiased estimate of the full gradient, at `B/n` of the cost. The noise
is not only a price: it also helps to escape flat regions and poor local
minima.

micrograd's demo supports both:

```python
if batch_size is None:
    Xb, yb = X, y
else:
    ri = np.random.permutation(X.shape[0])[:batch_size]
    Xb, yb = X[ri], y[ri]
```

We do the same. With `B = n` (the default, `--batch` not given) every step
uses the whole dataset, in order, and no randomness. With `B < n` every
step draws `B` **distinct** rows uniformly at random: sampling without
replacement, like the first `B` entries of a random permutation.

### 1.4 Initialisation: random, but reproducible

Why not start every weight at `0`? Because of **symmetry**. In a layer
whose weights are all equal, every neuron computes the same function,
receives the same gradient and makes the same update, so they stay equal
forever: a layer of 16 neurons learns no more than one. Random initial
weights break the symmetry. Biases may start at `0`, because the random
weights already make the neurons differ. This is why `xor.hsg` declares

```
param W1 : Mat 4 2 ~ uniform(-1, 1)
param b1 : Vec 4   = 0
```

Random, but not arbitrary: a run must be reproducible. Every random number
comes from a pseudo-random generator started from `--seed`, so the same
file, flags and seed always train the same model. Tests depend on that,
and so does anyone comparing two learning rates.

Two distributions are planned. `uniform(lo, hi)` is what the generator
gives directly. `normal(mean, std)` needs a transformation; we use the
**Box–Muller transform**: from two independent uniform numbers `u₁` in
`(0, 1]` and `u₂` in `[0, 1]`,

```
z = √(−2 ln u₁) · cos(2π u₂)
```

is exactly standard normal, and `mean + std · z` has the requested mean and
standard deviation. (Why it works: a pair of independent standard normals,
seen as a point in the plane, has a uniformly distributed angle `2πu₂` and
a squared radius that is exponentially distributed, `−2 ln u₁`. The
transform builds that point and keeps its x coordinate.)

### 1.5 Sign accuracy

The loss tells how wrong a model is, but not in units people think in. For
classifiers with labels `±1` and a real-valued prediction, micrograd also
prints the **accuracy**: the fraction of samples where the prediction is on
the label's side of zero.

```python
accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
```

`01-language.md` §11 plans the same for `--report-every`. The loop computes
it, when asked, on the step's batch, from the forward pass the step already
did, so it costs nothing extra.

### 1.6 The graph is built once

micrograd builds a new expression graph in every step: the forward pass
*is* the construction of `Value` objects. Here the graph is data built
before training (D1), and every step only changes the numbers that flow
through it (D10):

- **parameters** are leaves (`W1[0,0]`, …). Each step binds them to the
  current values of the parameter store;
- **dataset rows** enter through **slots**: a comprehension over a dataset
  with batch size `B` is unrolled into `B` copies of its body, copy `i`
  reading input leaves such as `xor#2.x[0]`, `xor#2.y`. Each step binds the
  `B` slots to the `B` rows of its batch.

The `Schedule` of the loss (step 4) is computed once too. A step is then
"rebind, `forwardBackward`, update": no graph construction, no hash-consing,
no reachability.

## 2. From micrograd to hsgrad

| micrograd | hsgrad |
|---|---|
| `random.uniform(-1, 1)` in `Neuron.__init__` | `Uniform (-1) 1` in a `ParamSpec`, drawn by `initParams` |
| `random.seed(1337)` (global state) | `initParams seed …`, an explicit generator (`State StdGen`) |
| `p.data` | an entry of the parameter store `Params`, keyed by leaf name |
| a new graph per step | one graph; parameters and dataset slots rebound per step (D10) |
| `model.zero_grad()` | nothing: every backward pass returns fresh gradients (step 5) |
| `p.data -= learning_rate * p.grad` | `sgdStep eta grads params` |
| `learning_rate = 1.0 - 0.9*k/100` | `learningRate (Sgd 1 (LinearTo 0.1)) 100 k` |
| `np.random.permutation(n)[:batch_size]` | `sampleBatch n b` (partial Fisher–Yates) |
| `print(f"step {k} loss …")` | a lazy stream of `Report`s; the caller prints |

`zero_grad` deserves a remark. micrograd stores gradients *in* the
`Value` objects and accumulates with `+=`, so they must be reset before
every backward pass; forgetting it is a classic bug (the gradients of all
previous steps leak into the update). Here gradients are a fresh map
returned by `forwardBackward`, so there is nothing to reset and nothing to
forget.

## 3. Code walkthrough

The modules live under `src/HsGrad/Train/`. `random` (1.3) joins the
library's dependencies, for `StdGen`.

### 3.1 Describing parameters

```haskell
data Shape = Scalar | Vector !Int | Matrix !Int !Int

data Init
  = Uniform !Double !Double     -- ~ uniform(lo, hi)
  | Normal !Double !Double      -- ~ normal(mean, std)
  | Constant !Double            -- = c

data ParamSpec = ParamSpec
  { specName :: !Text
  , specShape :: !Shape
  , specInit :: !Init
  }
```

A `ParamSpec` is one `param` declaration, as plain data:
`param W1 : Mat 16 2 ~ uniform(-1, 1)` is
`ParamSpec "W1" (Matrix 16 2) (Uniform (-1) 1)`. The driver (step 15) can
build it from the checked program; Train never imports Lang (layer rule,
`00-architecture.md` §2). `Shape` is deliberately not the language's type
with its shape variables (step 12): by the time a parameter reaches
training, its shape is concrete.

```haskell
elementNames :: ParamSpec -> [Name]
elementNames (ParamSpec base sh _) = map Name $ case sh of
  Scalar -> [base]
  Vector n -> [indexed [i] | i <- [0 .. n - 1]]
  Matrix r c -> [indexed [i, j] | i <- [0 .. r - 1], j <- [0 .. c - 1]]
  where
    indexed is = base <> "[" <> Text.intercalate "," (map (Text.pack . show) is) <> "]"
```

Core only sees scalar leaves. `elementNames` gives their names, row-major,
exactly as `01-language.md` §8 lowers a declaration: `b2`, `w[0]`,
`W1[1,0]`. The compiler must name the parameter leaves the same way (§6).

### 3.2 The store and initialisation

```haskell
type Params = Map Name Double

initParams :: Seed -> [ParamSpec] -> Params
initParams seed specs =
  Map.fromList (evalState (concat <$> mapM initSpec specs) (streamGen seed InitStream))
  where
    initSpec spec = mapM (\n -> (,) n <$> sampleInit (specInit spec)) (elementNames spec)

sampleInit :: Init -> State StdGen Double
sampleInit i = case i of
  Uniform lo hi -> state (uniformR (lo, hi))
  Normal mean std -> (\z -> mean + std * z) <$> standardNormal
  Constant c -> pure c
```

The store is a plain map from leaf name to value. That is the shape of
`envParams` (step 4) and of `paramGrads` (step 5), so binding it is
`env {envParams = ps}` and updating it is a map over the gradients by
name (§4.8).

`random`'s pure API threads the generator by hand:
`uniformR (lo, hi) :: StdGen -> (Double, StdGen)` returns a number *and*
the next generator. Doing that by hand across a list of specs, each with a
list of elements, is a classic source of bugs: pass an old generator by
mistake and two "random" numbers come out equal. `State StdGen` (from
`mtl`, already a dependency) does the threading: `state (uniformR (lo, hi))`
turns the pure function into an action, and `mapM` runs the actions in
order. The order is therefore part of the meaning: specs in list order,
elements in `elementNames` order. Declaration order of the `.hsg` file
gives the same store for the same seed. `Constant` draws nothing, so a
bias between two weight matrices does not shift the second one's values
(a test checks this).

`uniformR` includes both bounds and accepts them in either order, so
`uniform(1, -1)` means the same as `uniform(-1, 1)`.

`paramValues spec ps` reads one parameter's elements back, in
`elementNames` order: what `--save-params` will need to print a trained
matrix in its declared shape.

### 3.3 Two random streams from one seed

```haskell
data Stream = InitStream | BatchStream

streamGen :: Seed -> Stream -> StdGen
streamGen seed s = case s of
  InitStream -> forInit
  BatchStream -> forBatches
  where
    (forInit, forBatches) = splitGen (mkStdGen seed)
```

One `--seed` serves two purposes: initial parameters and mini-batches.
Taking both from the same generator would couple them: adding a parameter
would shift every batch that follows. `splitGen` turns one generator into
two independent ones, and each purpose gets its own (§4.4).

### 3.4 Box–Muller

```haskell
standardNormal :: State StdGen Double
standardNormal = do
  u1 <- positiveUnit
  u2 <- state (uniformR (0, 1))
  pure (sqrt (-2 * log u1) * cos (2 * pi * u2))
  where
    positiveUnit = do
      u <- state (uniformR (0, 1))
      if u > 0 then pure u else positiveUnit
```

The formula of §1.4, with one detail: `uniformR (0, 1)` can return exactly
`0`, where `log` is `−∞`. That is drawn again (it happens about once in
2⁵³ draws). The transform gives a second normal number for free (`sin`
instead of `cos`); we drop it rather than carry it in the state.

### 3.5 Schedules and the SGD step

```haskell
data LrSchedule
  = ConstantRate             -- const
  | LinearTo !Double         -- linear:END
  | StepDecay !Int !Double   -- step:K:F

data Sgd = Sgd { sgdRate :: !Double, sgdSchedule :: !LrSchedule }

learningRate :: Sgd -> Int -> Int -> Double
learningRate (Sgd base sched) n t = case sched of
  ConstantRate -> base
  LinearTo end
    | n <= 0 -> base
    | otherwise ->
        let s = fromIntegral t / fromIntegral n
         in (1 - s) * base + s * end
  StepDecay k f -> base * f ^ (max 0 t `div` max 1 k)
```

The schedule is a pure function of the step number `t` and the run length
`n`. There is no optimiser state to carry from step to step, so a run
could be replayed or resumed from any step. `LrSchedule` is the flag's
meaning without its syntax: parsing `linear:0.1` is step 15's job.

Two details keep it total and exact. The linear rate is a weighted average
rather than `η₀ + (END − η₀)·s`, because in floating point the latter gives
`0.09999999999999998` instead of `0.1` at `t = n` (a test caught this).
`StepDecay` with `K < 1` would divide by zero; the CLI will reject it, and
the function treats it as `K = 1` instead of crashing.

```haskell
sgdStep :: Double -> Map Name Double -> Params -> Params
sgdStep eta grads = Map.mapWithKey move
  where
    move n p = p - eta * Map.findWithDefault 0 n grads
```

The update of §1.1 over the whole store. Backprop leaves out exactly the
parameters the loss does not depend on (step 5), whose gradient is `0`
(warning W0203), so a missing gradient leaves the parameter alone.
Gradients of names that are not in the store are ignored: the update can
never add a parameter.

### 3.6 Feeds and setups

```haskell
data Feed = Feed
  { feedName :: !Text          -- for error messages
  , feedRows :: ![[Double]]    -- the dataset: one list of column values per row
  , feedSlots :: ![[Name]]     -- the batch: per slot, one leaf name per column
  }
```

A `Feed` is the minimal interface between data and training. Rows are
plain numbers, one per **column**: for the schema `{ x : Vec 2, y : Real }`
the columns are `x[0]`, `x[1]`, `y`. Reading them from CSV or inline `rows`
is `Data.*`'s job (step 13). The slots are the input leaves the compiler
created for the dataset's comprehension, with their names in the same
column order. Binding slot `i` to row `r` is `zip` of the two lists, so
**Train knows nothing about how slots are named** (§4.1). The number of
slots is the batch size `B`.

```haskell
data Setup = Setup
  { setupGraph :: !Graph
  , setupLoss :: !NodeId
  , setupInputs :: !(Map Name Double)     -- inputs that are not slots
  , setupFeeds :: ![Feed]
  , setupAccuracy :: ![(NodeId, NodeId)]  -- per slot: prediction, label
  }

data TrainConfig = TrainConfig
  { trainSteps :: !Int, trainSgd :: !Sgd, trainSeed :: !Seed }
```

D11 again: `Setup` is *what* is trained (it comes from the file),
`TrainConfig` is *how* (it comes from the flags). `setupInputs` holds the
values of ordinary `input` declarations (defaults and `--set`), the same
in every step. `setupAccuracy` lists, per slot, the model's prediction node
and the label's input leaf; an empty list means no accuracy (a regression
model, or an `intro.hsg`-style file without a model).

### 3.7 Sampling a mini-batch

```haskell
sampleBatch :: Int -> Int -> State StdGen [Int]
sampleBatch n b
  | b >= n = pure [0 .. n - 1]
  | otherwise = go 0 IntMap.empty
  where
    go i moved
      | i >= b = pure []
      | otherwise = do
          j <- state (uniformR (i, n - 1))
          let at k = IntMap.findWithDefault k k moved
              moved' = IntMap.insert j (at i) moved
          (at j :) <$> go (i + 1) moved'
```

The full batch is every row in order, drawing nothing. A mini-batch is the
first `b` steps of a **Fisher–Yates shuffle** of the array
`a = [0 .. n−1]`: at step `i`, pick `j` uniformly from `[i, n−1]`, swap
`a[i]` and `a[j]`, and `a[i]` is the next sample. Each step picks among the
rows not chosen yet, so the samples are distinct and every subset (and
order) is equally likely. That is micrograd's
`permutation(n)[:batch_size]`, without shuffling the rows that are thrown
away.

The array is never built. `moved` records only the positions whose
content differs from the identity (`at k` is `a[k]`). After the swap the
sample `a[i]` is never read again, since later steps only look at
positions above `i`, so only the new `a[j]` (the old `a[i]`) is recorded.
A batch of `b` rows costs `O(b log b)`, whatever the dataset size.

### 3.8 The loop

```haskell
train :: TrainConfig -> Setup -> Params -> Run
train cfg setup params0 = case mapM checkFeed (setupFeeds setup) of
  Left e -> Failed e
  Right feeds -> go feeds 0 params0 (streamGen (trainSeed cfg) BatchStream)
  where
    n = trainSteps cfg
    loss = setupLoss setup
    pairs = setupAccuracy setup
    sched = schedule (setupGraph setup) (loss : concat [[p, l] | (p, l) <- pairs])

    go feeds t ps gen
      | t >= n = Done ps
      | otherwise = case forwardBackward sched env loss of
          Left e -> Failed (StepFailed t e)
          Right (vals, grads) ->
            let eta = learningRate (trainSgd cfg) n t
                ps' = sgdStep eta (paramGrads (leafGrads sched grads)) ps
                rep = Report t eta (valueAt vals loss) (accuracy vals pairs)
             in Step rep (go feeds (t + 1) ps' gen')
      where
        (env, gen') = runState (bindFeeds feeds base) gen
        base = Env {envInputs = setupInputs setup, envParams = ps}
```

Before the first step, every feed is checked (`checkFeed`): it has rows
(E0405), it has no more slots than rows (E0405, `--batch` larger than the
dataset), and every row is as wide as the slots. A bad feed ends the run
before it starts. The rows go into a `Data.Sequence` for fast indexing by
the sampled row numbers.

The schedule is built once. Its outputs are the loss and, for the
accuracy, every prediction and label node. They are all part of the loss
in practice, but listing them guarantees that their values are computed
(`backward` handles a schedule with several outputs, step 5).

Then each step `t`, with the store `ps` and the generator `gen`:

1. **bind**: start from the fixed inputs and the store, and bind every
   feed's slots to a batch of its rows (`bindFeeds` runs `sampleBatch`
   for each feed in turn, in `State StdGen`, and returns the next
   generator `gen'`);
2. **forward and backward**: `forwardBackward`. Any `GradError` (a
   non-finite value or gradient, an unbound leaf) ends the run with the
   step number;
3. **update**: `sgdStep` with the step's rate and the parameters'
   gradients by name;
4. **report**: the step number, the rate, the loss value and the accuracy,
   all from this step's forward pass, so they describe the parameters
   *before* the update, as micrograd's printout does.

The generator is threaded through the steps by hand (`gen` in, `gen'`
out), and the store too. Inside a step, `State` does the threading where
there are many draws in a row; across steps, two explicit arguments are
clearer than a monad stack (§4.3).

### 3.9 Runs: a lazy stream of reports

```haskell
data Report = Report
  { reportStep :: !Int, reportRate :: !Double
  , reportLoss :: !Double, reportAccuracy :: !(Maybe Double) }

data Run
  = Step !Report Run
  | Done !Params
  | Failed !TrainError

runReports :: Run -> [Report]
runResult  :: Run -> Either TrainError Params
everyNth   :: Int -> [Report] -> [Report]
```

`Run` is a list of reports with two possible ends: the final store, or the
error that stopped training. Like a list it is **lazy**: `train` returns at
once, and step `t + 1` is computed only when someone looks past step `t`.
So the driver (step 15) can print progress while training runs, from a
pure function:

```haskell
mapM_ printReport (everyNth reportEvery (runReports run))
case runResult run of …
```

Each report is printed as soon as its step is done, and `runResult` then
walks the already computed steps to the end. A test, on the other hand,
just takes the list. `everyNth n` keeps steps `0, n, 2n, …`: what
`--report-every n` will show.

Strictness keeps the stream from piling up work. `Report`'s fields are
strict, so the loss is computed when the step is, not when it is printed.
The new store `ps'` is a thunk when the step is returned, but the next
step forces it (binding it into the environment and evaluating), and
`Data.Map.Strict` forces every value on the way. There is never more than
one step's worth of unevaluated work.

## 4. Design decisions

### 4.1 Train knows no slot names, but it does know element names

Slots and parameters are both leaves whose names the compiler (step 14)
invents, and Train must find them. They are handled differently on
purpose.

- **Slots** are passed in by name (`feedSlots`). How many there are and how
  the compiler names them (`moons#3.x[1]`) depends on how comprehensions
  are unrolled, and the compiler made them, so it can simply hand them
  over. Train then has nothing to keep consistent.
- **Parameter element names** are computed by Train (`elementNames`),
  because the store must exist *before* any graph does: `initParams` only
  sees the declarations. The naming convention is fixed by
  `01-language.md` §8, and step 14 must follow it (§6). A mismatch cannot
  go unnoticed: the first forward pass reports `UnboundLeaves` for every
  parameter leaf without a value.

### 4.2 A pure loop with a lazy result, not callbacks in `IO`

The loop could take a `Report -> IO ()` callback and print as it goes.
Instead it is a pure function whose result is consumed incrementally.
Laziness gives the "print while training" behaviour for free, the tests
need no `IO` and no mutable state to collect reports, and the driver is
free to filter, format or stop early (taking 10 reports runs 10 steps).

`Run` is its own type rather than `[Either TrainError Report]` or
`([Report], Either TrainError Params)`. The first allows reports after an
error; the second, consumed lazily, makes it easy to hold on to every
report while looking for the result. `Run` says exactly "some reports,
then one end".

### 4.3 `State` inside a step, explicit arguments across steps

Random draws are threaded with `State StdGen`, where there are many in a
row: initialising a store, sampling a batch, binding several feeds. The
loop itself carries two things from step to step, the store and the
generator, and returns a lazy structure. Running it in `State` as well
would hide the generator in a monad whose result we then need to be lazy,
which `State`'s strict variant is not. Two explicit arguments of `go` say
everything.

### 4.4 Two independent streams from one seed

With one generator for everything, the initial parameters would be the
first numbers and the batches the rest. Then the batches would change when
a parameter is added or a matrix grows, so two experiments that differ
only in the model would also differ in their data order. `splitGen` gives
two generators that are independent of each other; parameters always use
the first, batches the second (`streamGen`).

The numbers depend on `random`'s algorithm (SplitMix). The index state in
`cabal.project` pins `random` 1.3.1, so a seed means the same model on
every machine until that line is bumped.

### 4.5 `linear:END` reaches END after the last step

`linear:END` could also end exactly at `END` on the last step,
`t = n − 1`. We follow micrograd instead, whose demo is the end-to-end
test of the project (`00-architecture.md` §7): with
`--lr 1.0 --lr-schedule linear:0.1 --steps 100` the rates are exactly
micrograd's `1.0 − 0.9·k/100` (to the last bit or two; a test compares
them), from `1.0` down to `0.109`.

### 4.6 Without replacement; the full batch in order

Sampling without replacement matches micrograd and never wastes a slot on a
row the batch already has. The full batch (`B = n`) is the dataset in
order, with no randomness at all: a full-batch run is plain deterministic
gradient descent, and does not depend on the batch stream (a test runs
the same initial parameters with two different seeds and gets identical
runs).

### 4.7 Accuracy on the batch, before the update

The accuracy is measured on the slots of the current step, from the
forward pass that also gave the gradients: no second pass, no separate
evaluation set. With a full batch that is the training accuracy of the
parameters before the update, which is what micrograd prints. A test
evaluates the trained XOR net again with the final parameters to check the
end result separately.

### 4.8 The store is a plain `Map Name Double`

A newtype would protect the store from being confused with other maps.
But the store's whole job is to be the `envParams` of the environment and
to be updated by name with `paramGrads`. Both are `Map Name Double`, and
a synonym keeps that visible, as `Values` and `Grads` are synonyms
(`docs/05-core-eval.md` §3.6).

### 4.9 Errors: before the run, or at a step

Problems with the feeds are found before training starts and reported
once, with the dataset's name (`EmptyFeed`, `BatchTooLarge`,
`RowWidthMismatch`). Problems in a step (`StepFailed t e`) keep the
reports before them, so the user sees the loss rising towards the
non-finite value, and they carry the step's `GradError` unchanged: the
driver maps it to a diagnostic at the node's source span, as for
`hsgrad eval`.

## 5. Tests

`test/HsGrad/Train/ParamSpec.hs`, `OptimSpec.hs` and `LoopSpec.hs`:

- **Parameters**: element names of scalars, vectors and matrices
  (row-major, `W1[1,2]`); `shapeSize` distinct names for every shape;
  every element gets a value; `initParams` is deterministic for a seed and
  differs between seeds; uniform values lie between the bounds in either
  order; constants are exact and draw nothing; earlier parameters do not
  depend on later ones; 20,000 normal samples have the requested mean and
  standard deviation (within `0.02`, about six standard errors of the
  mean) and about 68.3% of standard normal samples lie within one
  deviation; `paramValues` in order; the two streams differ.
- **Optimiser**: each schedule at chosen steps; `linear` equals
  micrograd's `1.0 − 0.9·k/100` within `10⁻¹²`, and hits both ends
  exactly; `sgdStep` moves each parameter by exactly `−η·g`, leaves
  parameters without a gradient alone and ignores unknown names; "moves
  against the gradient" for random values; and, with gradients from
  backprop on `(w·x − y)² + (v − 1)²`, a small step lowers the loss at
  random points (§1.1), plus one step worked out by hand.
- **Mini-batches**: `sampleBatch` gives `b` distinct rows in range, is
  deterministic, returns every row in order for a full batch without
  touching the generator, picks every row about equally often (4000
  batches of 2 from 4 rows), and depends on the seed. A run whose loss is
  the value of slot 0 shows that each step binds the rows `sampleBatch`
  drew, a new batch per step. Mini-batch runs are deterministic per seed
  and differ between seeds; full-batch runs do not depend on the seed.
- **The loop**: the loss is reported before the update (`(w − 3)²` from
  `w = 0` with `η = 0.25` reports `9`, then `2.25`); the reported rates
  are the schedule's; fixed inputs are used and the run converges; zero
  steps return the store unchanged; no accuracy pairs, no accuracy.
- **Errors**: empty dataset, more slots than rows, a row of the wrong
  width, a parameter without a value (step 0), and `log w` overshooting to
  `w = −1.5`: the run stops at step 1 with `NonFiniteValue` at the `log`
  node, keeping step 0's report.

### 5.1 The milestone: XOR

`xor.hsg` (`01-language.md` §1.3), built by hand in `LoopSpec` the way the
compiler will build it: parameters `W1 : Mat 4 2 ~ uniform(-1, 1)`,
`b1 = 0`, `w2 : Vec 4 ~ uniform(-1, 1)`, `b2 = 0`; four slots
`xor#i.x[0]`, `xor#i.x[1]`, `xor#i.y`; per slot the network
`dot(w2, tanh(W1 @ x + b1)) + b2` and the squared error; the loss is their
mean. Trained with seed 1337, `η = 0.1`, full batch, 200 steps:

| Step | Loss | Accuracy |
|---|---|---|
| 0 | 1.2026 | 75% |
| 50 | 0.8025 | 100% |
| 100 | 0.1121 | 100% |
| 150 | 0.0022 | 100% |
| 199 | 0.00003 | 100% |

The tests check that the loss falls between every pair of these
checkpoints and ends below `10⁻³`, that the last report has accuracy 1 and
the first does not, that a fresh forward pass with the final parameters
puts all four points on the right side of zero, and that seeds 1 to 5 all
reach accuracy 1 as well. The whole run takes a few milliseconds.

### 5.2 Testing the tests

Deliberate bugs, each tried on its own:

| Bug | Failing tests |
|---|---|
| `p + η·g` instead of `p − η·g` | 11 |
| sample `j` from `[0, n−1]` (with replacement) | 2 |
| Fisher–Yates without recording the swap | 2 |
| sample even for a full batch | 2 |
| one stream for both purposes | 1 |
| `normal` ignores `std` | 2 |
| learning rate of step `t + 1` | 1 |
| row values bound to the slot's columns in reverse | 3 |
| accuracy compares with the wrong sign | 2 |
| the same batch generator in every step | 1 |
| `step:K:F` decays one step too early | 2 |

The second-to-last was the one survivor of the first round, and is why
the "a new batch in every step" test exists. One mutation survives and is
harmless: `cos(πu₂)` instead of `cos(2πu₂)` in Box–Muller has the same
distribution (both are the x coordinate of a uniformly random angle,
folded), so no test can, or should, tell them apart.

The full suite has 308 tests and runs in under 3 seconds.

## 6. What later steps build on

- **Step 13 (`Data.*`)**: produce rows as `[[Double]]`, one value per
  column, columns in schema order with vector fields expanded (`x[0]`,
  `x[1]`, `y`). That and the dataset's name is the data half of a `Feed`.
  Empty datasets and `--batch` larger than the dataset are E0405; `train`
  also checks both (`EmptyFeed`, `BatchTooLarge`) and row widths
  (`RowWidthMismatch`), which would be internal errors once Data checks
  rows against the schema (E0404).
- **Step 14 (`Lang.Compile`)**: name parameter leaves exactly as
  `elementNames` does (`b`, `w[i]`, `W[i,j]`, row-major). If Lang wants to
  call the function rather than repeat the convention, move it to a Core
  module instead of importing Train from Lang. For every comprehension over
  a dataset, return its slots as `[[Name]]` (per slot, the leaf names in
  column order): the other half of a `Feed`. For `--report-every`'s
  accuracy, return per slot the model call's result node and the label
  field's leaf (the last field of the pattern, when the model returns
  `Real`). `LoopSpec`'s hand-built XOR net is what compiling `xor.hsg`
  with 4 slots should give, up to spans and labels.
- **Step 15 (`Driver`, `app/Main`)**: `train` is `initParams seed specs`
  (specs in declaration order, from the `param` declarations), then
  `train (TrainConfig steps (Sgd lr schedule) seed) setup params`, then
  `mapM_` over `everyNth n (runReports run)` to print, then `runResult` for
  the final store or the error. Parse `--lr-schedule` into `LrSchedule`
  (`const`, `linear:END`, `step:K:F` with `K ≥ 1`). Map `StepFailed t e`
  like `eval`'s errors (E0501/E0502, and the non-finite gradient code step
  5 asks for), mentioning the step; an unbound *parameter* is an internal
  error. `--save-params` prints each spec with `paramValues`; `predict
  --params` reads such a file back into a `Params`.
- **Later**: momentum or Adam would add optimiser state; it would live
  next to the store in `go` and be updated by a function like `sgdStep`.
  Evaluation on a held-out set would be a second schedule over a second
  copy of the slots.
